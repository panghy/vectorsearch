package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

class SegmentBuildServiceIdempotentTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-build", UUID.randomUUID().toString()),
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
  void build_twice_is_safe() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(2)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);

    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    SegmentBuildService svc = new SegmentBuildService(cfg, dirs);
    svc.build(0).get(5, TimeUnit.SECONDS);
    // Second call should no-op on seal path; still returns successfully
    svc.build(0).get(5, TimeUnit.SECONDS);

    // Query to ensure artifacts readable
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    index.close();
  }
}
