package io.github.panghy.vectorsearch.fdb;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies that delete within a caller transaction removes gid mappings (both directions).
 */
class DeleteTransactionalClearsMappingsTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-del-tx", UUID.randomUUID().toString()),
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
  void deleteAll_tx_clears_gid_map_and_rev() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    try {
      long gid = index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
      int[][] loc = index.resolveIds(new long[] {gid}).get(5, TimeUnit.SECONDS);
      int seg = loc[0][0];
      int vec = loc[0][1];

      // Transactional delete
      db.run(tr -> {
        index.delete(tr, gid).join();
        return null;
      });

      // gid mappings should be cleared
      FdbDirectories.IndexDirectories dirs =
          FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      byte[] map = db.readAsync(tr -> tr.get(dirs.gidMapDir().pack(Tuple.from(gid))))
          .get(5, TimeUnit.SECONDS);
      byte[] rev = db.readAsync(tr -> tr.get(dirs.gidRevDir().pack(Tuple.from(seg, vec))))
          .get(5, TimeUnit.SECONDS);
      assertThat(map).isNull();
      assertThat(rev).isNull();
    } finally {
      index.close();
    }
  }
}
