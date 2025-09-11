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

/** Simple coverage test for gid -> (segment, vector) resolution helper. */
class VectorIndexResolveIdsTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-resolve", UUID.randomUUID().toString()),
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
  void resolve_roundtrips_gids() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(3)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    try {
      long g0 = index.add(new float[] {1, 0, 0}, null).get(5, TimeUnit.SECONDS);
      long g1 = index.add(new float[] {0, 1, 0}, null).get(5, TimeUnit.SECONDS);
      long g2 = index.add(new float[] {0, 0, 1}, null).get(5, TimeUnit.SECONDS); // rotates to seg1
      long[] gids = new long[] {g0, g1, g2};
      int[][] pairs = index.resolveIds(gids).get(5, TimeUnit.SECONDS);
      assertThat(pairs.length).isEqualTo(3);
      // Expect strict-cap: first two land in seg0 ids 0,1; third rotates to seg1 id0
      assertThat(pairs[0][0]).isEqualTo(0);
      assertThat(pairs[0][1]).isEqualTo(0);
      assertThat(pairs[1][0]).isEqualTo(0);
      assertThat(pairs[1][1]).isEqualTo(1);
      assertThat(pairs[2][0]).isEqualTo(1);
      assertThat(pairs[2][1]).isEqualTo(0);
    } finally {
      index.close();
    }
  }
}
