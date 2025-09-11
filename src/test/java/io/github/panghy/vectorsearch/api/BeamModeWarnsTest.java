package io.github.panghy.vectorsearch.api;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BeamModeWarnsTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-beam-warn", UUID.randomUUID().toString()),
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
  void beam_mode_executes_and_returns_results() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    try {
      for (int i = 0; i < 5; i++)
        index.add(new float[] {i, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
      var params = SearchParams.of(32, 8, 4, 256, false, SearchParams.Mode.BEAM);
      var res = index.query(new float[] {1, 0, 0, 0}, 3, params).get(5, TimeUnit.SECONDS);
      assertThat(res.size()).isBetween(1, 3);
    } finally {
      index.close();
    }
  }
}
