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

/**
 * Exercises the resolveIds fallback path when a gid has no mapping entry.
 */
class ResolveIdsFallbackTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-resolve-fallback", UUID.randomUUID().toString()),
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
  void returns_unknown_when_mapping_missing() throws Exception {
    VectorIndex index = VectorIndex.createOrOpen(VectorIndexConfig.builder(db, root)
            .dimension(4)
            .localWorkerThreads(0)
            .build())
        .get(10, TimeUnit.SECONDS);
    try {
      long fake = 1234567890123L; // arbitrary gid with no mapping
      int[][] pairs = index.resolveIds(new long[] {fake}).get(5, TimeUnit.SECONDS);
      assertThat(pairs).hasDimensions(1, 2);
      assertThat(pairs[0][0]).isEqualTo(-1);
      assertThat(pairs[0][1]).isEqualTo(-1);
    } finally {
      index.close();
    }
  }
}
