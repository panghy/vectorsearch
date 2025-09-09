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
 * Integration test to validate compaction task wiring and worker processing.
 * Compaction is a no-op skeleton that logs intent; we only assert the task gets processed.
 */
public class CompactionWorkerIntegrationTest {
  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-compact", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
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
  public void schedules_and_runs_compaction_task() throws Exception {
    // Force at least two segments
    index.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {5, 6, 7, 8}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {9, 0, 1, 2}, null).get(5, TimeUnit.SECONDS); // rotates to seg 1

    // Enqueue a compaction request for segIds [0,1]
    ((io.github.panghy.vectorsearch.fdb.FdbVectorIndex) index)
        .requestCompaction(List.of(0, 1))
        .get(5, TimeUnit.SECONDS);

    // Build maintenance queue + worker and process once
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
        VectorIndexConfig.builder(db, root).dimension(4).build(), dirs, mq);
    boolean processed = worker.runOnce().get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();
  }
}
