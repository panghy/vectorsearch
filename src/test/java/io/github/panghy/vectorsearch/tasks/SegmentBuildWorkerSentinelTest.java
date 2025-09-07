package io.github.panghy.vectorsearch.tasks;

/**
 * Tests the sentinel wake-up behavior for SegmentBuildWorkerPool shutdown.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.BuildTask;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

class SegmentBuildWorkerSentinelTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-sentinel", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void worker_handles_sentinel_task() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(2).build();
    // Enqueue sentinel task
    queue.enqueueIfNotExists(
            "build-segment:-1:test",
            BuildTask.newBuilder().setSegId(-1).build())
        .get(5, TimeUnit.SECONDS);
    // Worker should claim and complete it
    SegmentBuildWorker w = new SegmentBuildWorker(cfg, dirs, queue);
    Boolean processed = w.runOnce().get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();
  }
}
