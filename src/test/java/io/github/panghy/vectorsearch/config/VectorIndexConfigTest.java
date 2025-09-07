package io.github.panghy.vectorsearch.config;

/**
 * Unit tests for builder defaults and getters in VectorIndexConfig (with real FDB + DirectoryLayer).
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorIndexConfigTest {

  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-cfg", UUID.randomUUID().toString()),
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
  void builderDefaultsAndOverrides() {
    var cfg = VectorIndexConfig.builder(db, root)
        .dimension(1024)
        .metric(VectorIndexConfig.Metric.COSINE)
        .maxSegmentSize(200_000)
        .pqM(32)
        .pqK(512)
        .graphDegree(96)
        .oversample(3)
        .estimatedWorkerCount(4)
        .defaultTtl(Duration.ofMinutes(10))
        .defaultThrottle(Duration.ofSeconds(2))
        .codebookBatchLoadSize(1234)
        .adjacencyBatchLoadSize(5678)
        .prefetchCodebooksEnabled(false)
        .buildTxnLimitBytes(5L * 1024 * 1024)
        .buildTxnSoftLimitRatio(0.8)
        .buildSizeCheckEvery(17)
        .metricAttribute("env", "test")
        .build();

    assertThat(cfg.getDatabase()).isSameAs(db);
    assertThat(cfg.getIndexDir()).isSameAs(root);
    assertThat(cfg.getDimension()).isEqualTo(1024);
    assertThat(cfg.getMetric()).isEqualTo(VectorIndexConfig.Metric.COSINE);
    assertThat(cfg.getMaxSegmentSize()).isEqualTo(200_000);
    assertThat(cfg.getPqM()).isEqualTo(32);
    assertThat(cfg.getPqK()).isEqualTo(512);
    assertThat(cfg.getGraphDegree()).isEqualTo(96);
    assertThat(cfg.getOversample()).isEqualTo(3);
    assertThat(cfg.getEstimatedWorkerCount()).isEqualTo(4);
    assertThat(cfg.getDefaultTtl()).isEqualTo(Duration.ofMinutes(10));
    assertThat(cfg.getDefaultThrottle()).isEqualTo(Duration.ofSeconds(2));
    assertThat(cfg.getCodebookBatchLoadSize()).isEqualTo(1234);
    assertThat(cfg.getAdjacencyBatchLoadSize()).isEqualTo(5678);
    assertThat(cfg.isPrefetchCodebooksEnabled()).isFalse();
    assertThat(cfg.getBuildTxnLimitBytes()).isEqualTo(5L * 1024 * 1024);
    assertThat(cfg.getBuildTxnSoftLimitRatio()).isEqualTo(0.8);
    assertThat(cfg.getBuildSizeCheckEvery()).isEqualTo(17);
    assertThat(cfg.getMetricAttributes()).containsEntry("env", "test");
  }

  @Test
  void builderValidation() {
    assertThatThrownBy(
            () -> VectorIndexConfig.builder(db, root).dimension(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() ->
            VectorIndexConfig.builder(db, root).maxSegmentSize(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root).pqM(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root).pqK(1).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> VectorIndexConfig.builder(db, root).graphDegree(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> VectorIndexConfig.builder(db, root).oversample(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .estimatedWorkerCount(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .defaultTtl(Duration.ZERO)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .defaultThrottle(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .buildTxnLimitBytes(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .buildTxnSoftLimitRatio(0.0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .buildSizeCheckEvery(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
  }
}
