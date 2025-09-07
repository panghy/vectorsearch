package io.github.panghy.vectorsearch.api;

/**
 * Smoke test that BEST_FIRST defaults achieve high self-recall on a small random dataset
 * once segments are sealed and indexing complete.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
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

class VectorIndexBestFirstRecallTest {
  Database db;
  DirectorySubspace root;
  OpenTelemetrySdk sdk;
  InMemoryMetricReader reader;

  @BeforeEach
  void setup() throws Exception {
    reader = InMemoryMetricReader.create();
    SdkMeterProvider mp =
        SdkMeterProvider.builder().registerMetricReader(reader).build();
    sdk = OpenTelemetrySdk.builder().setMeterProvider(mp).build();
    GlobalOpenTelemetry.resetForTest();
    GlobalOpenTelemetry.set(sdk);
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-bestfirst", UUID.randomUUID().toString()),
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
    if (sdk != null) sdk.getSdkMeterProvider().close();
    GlobalOpenTelemetry.resetForTest();
  }

  @Test
  void best_first_default_has_high_self_recall() throws Exception {
    final int dim = 8;
    final int n = 200;
    final int k = 10;
    final int qCount = 30;

    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(dim)
        .pqM(4)
        .pqK(16)
        .graphDegree(16)
        .maxSegmentSize(50)
        .localWorkerThreads(1) // seal as we go
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    Random rnd = new Random(42);
    List<float[]> data = new ArrayList<>(n);
    for (int i = 0; i < n; i++) {
      float[] v = new float[dim];
      for (int d = 0; d < dim; d++) v[d] = (float) rnd.nextGaussian();
      data.add(v);
    }

    List<int[]> ids = index.addAll(data.toArray(new float[0][]), null).get(30, TimeUnit.SECONDS);
    index.awaitIndexingComplete().get(20, TimeUnit.SECONDS);

    // Sample queries and compute recall that the true self (segId,vecId) appears in results
    List<Integer> picks = new ArrayList<>();
    for (int i = 0; i < qCount; i++) picks.add(rnd.nextInt(n));
    List<CompletableFuture<List<SearchResult>>> futures = new ArrayList<>();
    for (int idx : picks) futures.add(index.query(data.get(idx), k));
    CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).get(30, TimeUnit.SECONDS);

    int hits = 0;
    for (int i = 0; i < picks.size(); i++) {
      int idx = picks.get(i);
      int[] gt = ids.get(idx);
      List<SearchResult> res = futures.get(i).get(5, TimeUnit.SECONDS);
      boolean found = res.stream().anyMatch(r -> r.segmentId() == gt[0] && r.vectorId() == gt[1]);
      if (found) hits++;
    }
    double recall = hits / (double) qCount;
    assertThat(recall).isGreaterThanOrEqualTo(0.9);
    // Basic metrics presence
    assertThat(reader.collectAllMetrics().stream().anyMatch(m -> m.getName().equals("vectorsearch.cache.size")))
        .isTrue();
    index.close();
  }
}
