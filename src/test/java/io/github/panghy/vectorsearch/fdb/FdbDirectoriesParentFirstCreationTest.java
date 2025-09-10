package io.github.panghy.vectorsearch.fdb;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Regression test: ensure parent-first creation of segment directories so that later callers can
 * {@code open()} child subspaces reliably without races.
 */
class FdbDirectoriesParentFirstCreationTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-parent-first", UUID.randomUUID().toString()),
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
  void segment_children_are_openable_after_createOrOpenIndex() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(3).build();
    // Ensure index + initial segment created
    new FdbVectorStore(cfg, dirs, queue).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Now verify that opening child subspaces works without createOrOpen (i.e., they exist)
    db.readAsync(tr -> dirs.segmentsDir().open(tr, List.of("0"))).get(5, TimeUnit.SECONDS);
    // Child: vectors
    db.readAsync(tr -> dirs.segmentsDir().open(tr, List.of("0", FdbPathUtil.VECTORS)))
        .get(5, TimeUnit.SECONDS);
    // Child: pq and pq/codes
    DirectorySubspace pq = db.readAsync(tr -> dirs.segmentsDir().open(tr, List.of("0", FdbPathUtil.PQ)))
        .get(5, TimeUnit.SECONDS);
    assertThat(pq).isNotNull();
    db.readAsync(tr -> pq.open(tr, List.of(FdbPathUtil.CODES))).get(5, TimeUnit.SECONDS);
    // Child: graph
    db.readAsync(tr -> dirs.segmentsDir().open(tr, List.of("0", FdbPathUtil.GRAPH)))
        .get(5, TimeUnit.SECONDS);

    // Also sanity check that meta key resolves using segmentMetaKey helper
    byte[] meta = db.readAsync(tr -> dirs.segmentMetaKey(tr, 0).thenCompose(mk -> tr.get(mk)))
        .get(5, TimeUnit.SECONDS);
    assertThat(meta).isNotNull();
  }
}
