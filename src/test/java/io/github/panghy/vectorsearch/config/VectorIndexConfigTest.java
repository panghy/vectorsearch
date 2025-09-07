package io.github.panghy.vectorsearch.config;

/**
 * Unit tests for builder defaults and getters in VectorIndexConfig.
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class VectorIndexConfigTest {

  @Test
  void builderDefaultsAndOverrides() {
    Database db = mock(Database.class);
    DirectorySubspace root = mock(DirectorySubspace.class);

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
  }

  @Test
  void builderValidation() {
    Database db = mock(Database.class);
    DirectorySubspace root = mock(DirectorySubspace.class);

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
  }
}
