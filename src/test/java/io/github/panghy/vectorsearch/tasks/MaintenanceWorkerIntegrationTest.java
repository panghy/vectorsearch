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
    int[][] ids = new int[5][2];
    for (int i = 0; i < 5; i++) {
      float[] v = new float[8];
      for (int d = 0; d < 8; d++) v[d] = (float) Math.sin(0.02 * i + d);
      ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
    }
    index.delete(ids[0][0], ids[0][1]).get(5, TimeUnit.SECONDS);

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
}
