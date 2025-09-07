package io.github.panghy.vectorsearch.api;

/**
 * Prefetch smoke test: verifies query-time codebook prefetch (fire-and-forget) warms
 * the codebook cache when multiple segments are SEALED.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.tasks.SegmentBuildService;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.LongPointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorIndexPrefetchSmokeTest {
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
                List.of("vs-prefetch", UUID.randomUUID().toString()),
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
  void codebook_prefetch_warms_cache_for_sealed_segments() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(2)
        .maxSegmentSize(1) // every insert seals previous
        .localWorkerThreads(0) // we will seal manually
        .build();

    var index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);

    // Insert and seal 5 segments (ids 0..4)
    for (int i = 0; i < 5; i++) {
      index.add(new float[] {i, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
      new SegmentBuildService(cfg, dirs).build(i).get(5, TimeUnit.SECONDS);
    }

    // Cache should be empty before query
    long before = index.getCodebookCacheSize();
    assertThat(before).isLessThanOrEqualTo(1L);

    // Trigger a query to fire prefetch (non-blocking)
    index.query(new float[] {0, 0, 0, 0}, 1).get(5, TimeUnit.SECONDS);

    // Wait for prefetch to warm the cache (best-effort)
    long deadline = System.currentTimeMillis() + 2_000;
    long size;
    do {
      Thread.sleep(50);
      size = index.getCodebookCacheSize();
      if (size >= 5) break;
    } while (System.currentTimeMillis() < deadline);

    assertThat(index.getCodebookCacheSize()).isGreaterThanOrEqualTo(3L); // best-effort warm

    // Assert via metrics as well
    long cbSize = -1;
    for (MetricData m : reader.collectAllMetrics()) {
      if (!"vectorsearch.cache.size".equals(m.getName())) continue;
      for (LongPointData p : m.getLongGaugeData().getPoints()) {
        String cache = p.getAttributes().get(AttributeKey.stringKey("cache"));
        if ("codebook".equals(cache)) cbSize = Math.max(cbSize, p.getValue());
      }
    }
    assertThat(cbSize).isGreaterThanOrEqualTo(3L);
    index.close();
  }
}
