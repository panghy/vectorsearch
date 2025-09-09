package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.fdb.FdbVectorIndex;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Deterministic compaction scheduling tests.
 *
 * <p>Verifies that {@link FdbVectorIndex#requestCompaction(java.util.List)} produces a
 * deterministic idempotent queue key regardless of the input segment-id order and results in a
 * single visible compaction task.</p>
 */
public class DeterministicCompactionTest {
  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-compact-det", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(1)
        .localWorkerThreads(0)
        .localMaintenanceWorkerThreads(0)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
  }

  @AfterEach
  public void teardown() {
    if (index != null) index.close();
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  public void requestCompaction_is_order_invariant_and_idempotent() throws Exception {
    // Create two sealed segments deterministically via SegmentBuildService
    index.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS); // seg0 vec0
    index.add(new float[] {5, 6, 7, 8}, null).get(5, TimeUnit.SECONDS); // seg1 vec0
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(
            VectorIndexConfig.builder(db, root)
                .dimension(4)
                .pqM(2)
                .pqK(4)
                .graphDegree(4)
                .build(),
            dirs)
        .build(0)
        .get(5, TimeUnit.SECONDS);
    new SegmentBuildService(
            VectorIndexConfig.builder(db, root)
                .dimension(4)
                .pqM(2)
                .pqK(4)
                .graphDegree(4)
                .build(),
            dirs)
        .build(1)
        .get(5, TimeUnit.SECONDS);

    // Enqueue compaction twice with different input orders
    FdbVectorIndex impl = (FdbVectorIndex) index;
    impl.requestCompaction(Arrays.asList(1, 0)).get(5, TimeUnit.SECONDS);
    impl.requestCompaction(Arrays.asList(0, 1)).get(5, TimeUnit.SECONDS);

    // Construct maintenance queue binding identical to the index's maintenance queue
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .build();
    var mq = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);

    // There should be exactly one compact task to claim
    var claim = mq.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    assertThat(claim).isNotNull();
    MaintenanceTask t = claim.task();
    assertThat(t.hasCompact()).isTrue();
    assertThat(t.getCompact().getSegIdsList()).containsExactly(0, 1);
    // Complete the compact task
    claim.complete().get(5, TimeUnit.SECONDS);

    // No second task should exist; await emptiness to ensure idempotency
    mq.awaitQueueEmpty(db).get(5, TimeUnit.SECONDS);
  }
}
