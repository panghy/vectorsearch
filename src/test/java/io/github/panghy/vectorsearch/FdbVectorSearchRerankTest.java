package io.github.panghy.vectorsearch;

import static com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt;
import static io.github.panghy.vectorsearch.pq.DistanceMetrics.Metric;
import static io.github.panghy.vectorsearch.pq.DistanceMetrics.distance;
import static java.util.concurrent.CompletableFuture.completedFuture;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.search.SearchResult;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.OriginalVectorStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import java.lang.reflect.Field;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Verifies that FdbVectorSearch.search reranks approximate results using original vectors.
 */
class FdbVectorSearchRerankTest {

  private Database db;
  private DirectorySubspace dir;
  private VectorIndexKeys keys;

  @BeforeEach
  void setUp() {
    db = FDB.selectAPIVersion(730).open();
    String ns = "test/vectorsearch/rerank/" + UUID.randomUUID();
    db.run(tr -> {
      DirectoryLayer dl = DirectoryLayer.getDefault();
      dir = dl.createOrOpen(tr, List.of(ns.split("/"))).join();
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
  @DisplayName("search() reranks using original vectors for exact distances")
  void testSearchRerankingWithOriginalVectors() throws Exception {
    // Create index (will initialize storages)
    VectorSearchConfig config = VectorSearchConfig.builder(db, dir)
        .dimension(4)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .build();
    FdbVectorSearch index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Mark an active codebook version so ensureLatestCodebooksLoaded() succeeds
    // Use the same collection subspace the index uses (its meta subspace)
    DirectorySubspace indexMeta = (DirectorySubspace) getField(index, "metaSubspace");
    VectorIndexKeys indexKeys = new VectorIndexKeys(indexMeta);
    db.run(tr -> {
      tr.set(indexKeys.activeCodebookVersionKey(), encodeInt(1));
      return null;
    });

    // Mock ProductQuantizer and Codebook/Beam/OriginalVector storages
    ProductQuantizer mockPq = mock(ProductQuantizer.class);
    when(mockPq.getCodebookVersion()).thenReturn(1);

    CodebookStorage mockCodebooks = mock(CodebookStorage.class);
    when(mockCodebooks.getLatestProductQuantizer()).thenReturn(completedFuture(mockPq));
    when(mockCodebooks.getProductQuantizer(anyLong())).thenReturn(completedFuture(mockPq));

    BeamSearchEngine mockBeam = mock(BeamSearchEngine.class);
    // Approximate candidate IDs (distances ignored in rerank)
    List<SearchResult> approx =
        List.of(new SearchResult(1L, 10f), new SearchResult(2L, 20f), new SearchResult(3L, 30f));
    when(mockBeam.search(any(), any(float[].class), anyInt(), anyInt(), anyInt(), any(ProductQuantizer.class)))
        .thenReturn(completedFuture(approx));

    OriginalVectorStorage mockOriginals = mock(OriginalVectorStorage.class);
    // Query and stored vectors such that exact L2 distances rank 2 < 1 < 3
    float[] q = new float[] {1f, 0f, 0f, 0f};
    float[] v1 = new float[] {0.9f, 0f, 0f, 0f}; // dist ~ 0.01
    float[] v2 = new float[] {1.0f, 0f, 0f, 0f}; // dist 0.0 (best)
    float[] v3 = new float[] {2.0f, 0f, 0f, 0f}; // dist 1.0 (worst)
    when(mockOriginals.readVector(any(), eq(1L))).thenReturn(completedFuture(v1));
    when(mockOriginals.readVector(any(), eq(2L))).thenReturn(completedFuture(v2));
    when(mockOriginals.readVector(any(), eq(3L))).thenReturn(completedFuture(v3));

    // Inject mocks via reflection
    setField(index, "codebookStorage", mockCodebooks);
    setField(index, "beamSearchEngine", mockBeam);
    setField(index, "originalVectorStorage", mockOriginals);

    // Execute search with k=2 (should rerank to [2, 1])
    List<SearchResult> results = index.search(q, 2).get(10, java.util.concurrent.TimeUnit.SECONDS);

    assertThat(results).hasSize(2);
    assertThat(results.get(0).getNodeId()).isEqualTo(2L);
    assertThat(results.get(1).getNodeId()).isEqualTo(1L);

    // Validate distances computed via DistanceMetrics
    float d2 = distance(q, v2, Metric.L2);
    float d1 = distance(q, v1, Metric.L2);
    assertThat(results.get(0).getDistance()).isEqualTo(d2);
    assertThat(results.get(1).getDistance()).isEqualTo(d1);

    index.shutdown();
  }

  private static void setField(Object target, String name, Object value) throws Exception {
    Field f = target.getClass().getDeclaredField(name);
    f.setAccessible(true);
    f.set(target, value);
  }

  private static Object getField(Object target, String name) throws Exception {
    Field f = target.getClass().getDeclaredField(name);
    f.setAccessible(true);
    return f.get(target);
  }
}
