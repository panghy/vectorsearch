package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import java.time.Duration;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SegmentCodebookRegistryTest {
  private Database db;
  private DirectorySubspace dir;
  private VectorIndexKeys keys;

  @BeforeEach
  void setUp() {
    db = FDB.selectAPIVersion(730).open();
    String ns = "test/vectorsearch/segment_codebook/" + UUID.randomUUID();
    db.run(tr -> {
      dir = DirectoryLayer.getDefault()
          .createOrOpen(tr, java.util.List.of(ns.split("/")))
          .join();
      return null;
    });
    keys = new VectorIndexKeys(dir);
  }

  @AfterEach
  void tearDown() {
    if (db != null && dir != null) {
      db.run(tr -> {
        DirectoryLayer.getDefault().removeIfExists(tr, dir.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  @DisplayName("store and load segment codebooks into ProductQuantizer")
  void testStoreLoad() throws Exception {
    int D = 8;
    int m = 4;
    int subDim = D / m;
    long seg = 99L;

    // Create simple deterministic codebooks: [m][256][subDim]
    float[][][] codebooks = new float[m][256][subDim];
    for (int s = 0; s < m; s++) {
      for (int c = 0; c < 256; c++) {
        for (int d = 0; d < subDim; d++) {
          codebooks[s][c][d] = s + c * 0.001f + d * 0.01f;
        }
      }
    }

    SegmentCodebookRegistry registry = new SegmentCodebookRegistry(
        db,
        keys,
        D,
        m,
        DistanceMetrics.Metric.L2,
        100,
        Duration.ofMinutes(5),
        java.util.concurrent.ForkJoinPool.commonPool());

    SegmentCodebookRegistry.TrainingStats stats = new SegmentCodebookRegistry.TrainingStats(7, 1000, null);
    db.run(tr -> {
      registry.storeSegmentCodebooks(seg, codebooks, stats).join();
      return null;
    });

    ProductQuantizer pq = registry.getProductQuantizer(seg).get(5, TimeUnit.SECONDS);
    assertThat(pq).isNotNull();
    assertThat(pq.getDimension()).isEqualTo(D);
    assertThat(pq.getNumSubvectors()).isEqualTo(m);
    assertThat(pq.getCodebookVersion()).isEqualTo(7);

    // Verify codeword values survived fp16 round-trip within tolerance
    float[][][] loaded = pq.getCodebooks();
    for (int s = 0; s < m; s++) {
      for (int c = 0; c < 256; c++) {
        for (int d = 0; d < subDim; d++) {
          assertThat(loaded[s][c][d])
              .isCloseTo(codebooks[s][c][d], org.assertj.core.data.Offset.offset(1e-2f));
        }
      }
    }
  }
}
