package io.github.panghy.vectorsearch.tasks;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

/**
 * Lifecycle smoke test for {@link MaintenanceWorkerPool}.
 *
 * <p>Verifies that starting and closing the pool does not throw and that sentinel
 * wake-up logic is exercised without relying on timing.
 */
class MaintenanceWorkerPoolTest {
  Database db;
  com.apple.foundationdb.directory.DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-mpool", UUID.randomUUID().toString()),
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
  void start_and_close() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(4).build();

    MaintenanceWorkerPool pool = new MaintenanceWorkerPool(cfg, dirs, queue);
    pool.start(1);
    pool.close();
  }
}
