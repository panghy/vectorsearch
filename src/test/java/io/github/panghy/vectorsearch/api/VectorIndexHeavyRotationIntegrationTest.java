package io.github.panghy.vectorsearch.api;

/**
 * Integration stress test inserting many vectors with small segment size to force rotation,
 * and running parallel queries to assert recall under heavy rotation.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Integration test that stresses heavy rotation and parallel search.
 */
class VectorIndexHeavyRotationIntegrationTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-heavy", UUID.randomUUID().toString()),
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
  void inserts_10k_vectors_and_parallel_queries_have_high_recall() throws Exception {
    final int dim = 8;
    final int n = Integer.getInteger("VS_HEAVY_N", 1_000);
    final int segSize = 200; // aggressive rotation (~50 segments)
    final int qCount = 100;
    final int k = 10;

    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(dim)
        .pqM(4)
        .pqK(16)
        .graphDegree(16)
        .maxSegmentSize(segSize)
        .localWorkerThreads(2)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    Random rnd = new Random(1234);
    List<float[]> inserted = new ArrayList<>(n);
    List<int[]> ids = new ArrayList<>(n);
    int batchSize = Integer.getInteger("VS_HEAVY_BATCH", 200);
    for (int start = 0; start < n; start += batchSize) {
      int end = Math.min(n, start + batchSize);
      float[][] batch = new float[end - start][dim];
      for (int i = start; i < end; i++) {
        float[] v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = (float) rnd.nextGaussian();
        batch[i - start] = v;
        inserted.add(v);
      }
      var got = index.addAll(batch, null).get(120, TimeUnit.SECONDS);
      ids.addAll(got);
    }

    // Pick qCount random vectors to query in parallel
    List<Integer> picks = new ArrayList<>();
    for (int i = 0; i < qCount; i++) picks.add(rnd.nextInt(n));

    System.out.println("Inserts done; waiting for indexing...");

    index.awaitIndexingComplete().get(120, TimeUnit.SECONDS);

    System.out.println("Indexing done; querying...");

    List<CompletableFuture<List<SearchResult>>> futures = new ArrayList<>();
    for (int i : picks) futures.add(index.query(inserted.get(i), k));
    CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).get(60, TimeUnit.SECONDS);

    System.out.println("Queries done; computing recall...");

    // Compute recall@k: fraction of queries whose exact (segId, vecId) appears in the result set
    int hits = 0;
    for (int qi = 0; qi < picks.size(); qi++) {
      int idx = picks.get(qi);
      int[] gt = ids.get(idx);
      List<SearchResult> res = futures.get(qi).get(5, TimeUnit.SECONDS);
      boolean found = res.stream().anyMatch(r -> r.segmentId() == gt[0] && r.vectorId() == gt[1]);
      if (found) hits++;
    }
    double recall = hits / (double) qCount;
    System.out.println("Recall@k: " + recall);
    // With all segments discoverable and sealed search tuned, recall for self-queries should be high.
    assertThat(recall).isGreaterThanOrEqualTo(0.9);
    index.close();
  }
}
