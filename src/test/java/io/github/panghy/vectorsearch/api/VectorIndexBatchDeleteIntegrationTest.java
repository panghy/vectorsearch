package io.github.panghy.vectorsearch.api;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.tasks.MaintenanceWorker;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Integration test for {@link VectorIndex#deleteAll(int[][])}.
 *
 * <p>Ensures that batch deletion removes results from query output and that a maintenance
 * vacuum task can be processed immediately (threshold set to 0.0 for the test).</p>
 */
public class VectorIndexBatchDeleteIntegrationTest {

  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-delall", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .pqM(4)
        .pqK(16)
        .graphDegree(8)
        .maxSegmentSize(50)
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
        .vacuumMinDeletedRatio(0.0) // always enqueue after deleteAll
        .localWorkerThreads(1)
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
  public void batchDeleteRemovesAndEnqueuesVacuum() throws Exception {
    // Insert
    int[][] ids = new int[12][2];
    for (int i = 0; i < 12; i++) {
      float[] v = new float[8];
      for (int d = 0; d < 8; d++) v[d] = (float) Math.cos(0.01 * i + d);
      ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
    }
    // Delete 3 vectors via deleteAll
    int[][] toDel = new int[][] {ids[2], ids[5], ids[7]};
    index.deleteAll(toDel).get(5, TimeUnit.SECONDS);

    // Verify they do not appear in results
    var res = index.query(new float[] {1, 0, 0, 0, 0, 0, 0, 0}, 10).get(5, TimeUnit.SECONDS);
    for (int[] id : toDel) {
      assertThat(res.stream().noneMatch(r -> r.segmentId() == id[0] && r.vectorId() == id[1]))
          .isTrue();
    }

    // Run a maintenance worker once; with ratio=0.0, vacuum task should be enqueued
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<String, MaintenanceTask> tqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
        .defaultThrottle(Duration.ofMillis(25))
        .taskNameExtractor(mt ->
            mt.hasVacuum() ? ("vacuum-segment:" + mt.getVacuum().getSegId()) : "compact")
        .build();
    var mq = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    boolean processed = new MaintenanceWorker(
            VectorIndexConfig.builder(db, root).dimension(8).build(), dirs, mq)
        .runOnce()
        .get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();
  }
}
