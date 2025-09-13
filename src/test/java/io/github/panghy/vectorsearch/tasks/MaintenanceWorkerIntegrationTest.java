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
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * End-to-end test for {@link io.github.panghy.vectorsearch.tasks.MaintenanceWorker}.
 *
 * <p>Scenario: insert a few vectors, delete one, rely on VectorIndex auto-enqueue logic to
 * schedule a vacuum, then run a worker once against the maintenance queue and assert the task
 * is processed (no sleeps or polling).</p>
 */
public class MaintenanceWorkerIntegrationTest {
  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-mw", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .pqM(4)
        .pqK(16)
        .graphDegree(8)
        .maxSegmentSize(10)
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
        .localWorkerThreads(1)
        .vacuumMinDeletedRatio(0.15)
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
  public void runsVacuumTask() throws Exception {
    // Insert few vectors then delete one
    long[] ids = new long[5];
    for (int i = 0; i < 5; i++) {
      float[] v = new float[8];
      for (int d = 0; d < 8; d++) v[d] = (float) Math.sin(0.02 * i + d);
      ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
    }
    index.delete(ids[0]).get(5, TimeUnit.SECONDS);

    // Auto-enqueue from delete should have happened (ratio ~0.2 >= 0.15)

    // Build a dedicated maintenance worker and process exactly one task
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
        .defaultThrottle(Duration.ofMillis(50))
        .taskNameExtractor(mt ->
            mt.hasVacuum() ? ("vacuum-segment:" + mt.getVacuum().getSegId()) : "compact")
        .build();
    var mq = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    MaintenanceWorker worker = new MaintenanceWorker(
        VectorIndexConfig.builder(db, root).dimension(8).build(), dirs, mq);
    boolean processed = worker.runOnce().get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();
  }

  @Test
  public void throttlingZeroDisablesCompaction() throws Exception {
    // Create a small index and seal two segments so they are candidates for compaction.
    VectorIndexConfig sealCfg = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .pqM(4)
        .pqK(16)
        .graphDegree(8)
        .maxSegmentSize(10)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    // Insert into seg 0, build; then seg 1, build.
    VectorIndex tmp = VectorIndex.createOrOpen(sealCfg).get(5, TimeUnit.SECONDS);
    tmp.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    tmp.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(sealCfg, dirs).build(0).get(5, TimeUnit.SECONDS);
    tmp.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(sealCfg, dirs).build(1).get(5, TimeUnit.SECONDS);
    tmp.close();

    // Build maintenance queue with maxConcurrentCompactions = 0 and enqueue a find-candidates task.
    VectorIndexConfig cfgNoCompact = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .maxConcurrentCompactions(0)
        .build();
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    queue.enqueue(
            "find-candidates:0",
            io.github.panghy.vectorsearch.proto.MaintenanceTask.newBuilder()
                .setFindCandidates(
                    io.github.panghy.vectorsearch.proto.MaintenanceTask.FindCompactionCandidates
                        .newBuilder()
                        .setAnchorSegId(0)
                        .build())
                .build())
        .get(5, TimeUnit.SECONDS);

    // Run the worker once; with throttling at 0, it should not mark any segments COMPACTING.
    boolean processed =
        new MaintenanceWorker(cfgNoCompact, dirs, queue).runOnce().get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();
    int inflight = new MaintenanceService(cfgNoCompact, dirs)
        .countInFlightCompactions()
        .get(5, TimeUnit.SECONDS);
    assertThat(inflight).isEqualTo(0);
  }
}
