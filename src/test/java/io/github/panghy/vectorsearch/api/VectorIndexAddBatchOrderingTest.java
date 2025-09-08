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
 * Verifies that addAll/addBatch preserves input ordering when assigning (segId, vecId)
 * across rotations and internal transaction chunking. For strict-cap rotation, the
 * expected mapping for the i-th vector is (i / maxSegmentSize, i % maxSegmentSize).
 */
class VectorIndexAddBatchOrderingTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-ordering", UUID.randomUUID().toString()),
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
  void addAll_preserves_order_across_rotations_and_chunks() throws Exception {
    final int dim = 4;
    final int maxSeg = 50;
    final int n = 123; // spans multiple segments

    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(dim)
        .maxSegmentSize(maxSeg)
        .localWorkerThreads(0)
        // keep defaults for txn soft-limit; ordering must still hold
        .build();

    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    float[][] data = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int d = 0; d < dim; d++) data[i][d] = i * 0.001f + d;
    }

    List<int[]> ids = index.addAll(data, null).get(30, TimeUnit.SECONDS);
    assertThat(ids).hasSize(n);

    for (int i = 0; i < n; i++) {
      int expSeg = i / maxSeg;
      int expVec = i % maxSeg;
      assertThat(ids.get(i)[0]).as("segId at i=" + i).isEqualTo(expSeg);
      assertThat(ids.get(i)[1]).as("vecId at i=" + i).isEqualTo(expVec);
    }

    index.close();
  }
}
