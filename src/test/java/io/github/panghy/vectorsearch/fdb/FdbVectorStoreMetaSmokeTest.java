package io.github.panghy.vectorsearch.fdb;

/**
 * Smoke tests for FdbVectorStore metadata: index/segment keys exist and maxSegment increments on rotation.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

class FdbVectorStoreMetaSmokeTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-smoke", UUID.randomUUID().toString()),
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
  void meta_and_current_segment_are_readable() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(3).build();
    FdbVectorStore store = new FdbVectorStore(cfg, dirs, queue);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    int cur = store.getCurrentSegment().get(5, TimeUnit.SECONDS);
    assertThat(cur).isEqualTo(0);
    var meta = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(meta.getSegmentId()).isEqualTo(0);
  }

  @Test
  void max_segment_increments_on_rotation() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .maxSegmentSize(1) // force rotation on second insert
        .build();
    FdbVectorStore store = new FdbVectorStore(cfg, dirs, queue);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Insert two vectors: second triggers rotation to seg 1
    store.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS);
    store.add(new float[] {4, 3, 2, 1}, null).get(5, TimeUnit.SECONDS);

    byte[] maxBytes = db.readAsync(tr -> tr.get(dirs.maxSegmentKey())).get(5, TimeUnit.SECONDS);
    long maxSeg = Tuple.fromBytes(maxBytes).getLong(0);
    assertThat(maxSeg).isGreaterThanOrEqualTo(1);
    int current = store.getCurrentSegment().get(5, TimeUnit.SECONDS);
    assertThat(current).isGreaterThanOrEqualTo(1);
  }
}
