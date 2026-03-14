package io.github.panghy.vectorsearch.config;

/**
 * Validation tests for VectorIndexConfig to ensure invalid parameters are rejected.
 */
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

class VectorIndexConfigValidationTest {
  static Database db;
  static com.apple.foundationdb.directory.DirectorySubspace root;

  @BeforeAll
  static void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vic-validate", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterAll
  static void teardown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void builder_validations_throw() {
    // pqM <= 0
    assertThatThrownBy(() ->
            VectorIndexConfig.builder(db, root).dimension(4).pqM(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    // pqK <= 1
    assertThatThrownBy(() ->
            VectorIndexConfig.builder(db, root).dimension(4).pqK(1).build())
        .isInstanceOf(IllegalArgumentException.class);
    // graphDegree <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .graphDegree(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // oversample <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .oversample(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // estimatedWorkerCount <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .estimatedWorkerCount(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // localWorkerThreads < 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .localWorkerThreads(-1)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // defaultTtl non-positive
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .defaultTtl(Duration.ZERO)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // defaultThrottle negative
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .defaultThrottle(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  // ---- WorkerConfig.Builder validation tests ----

  @Test
  void workerConfig_rejectsInvalidEstimatedWorkerCount() {
    assertThatThrownBy(() -> WorkerConfig.builder().estimatedWorkerCount(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("estimatedWorkerCount");
    assertThatThrownBy(() -> WorkerConfig.builder().estimatedWorkerCount(-1).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidDefaultTtl() {
    assertThatThrownBy(
            () -> WorkerConfig.builder().defaultTtl(Duration.ZERO).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultTtl");
    assertThatThrownBy(() -> WorkerConfig.builder()
            .defaultTtl(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> WorkerConfig.builder().defaultTtl(null).build())
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void workerConfig_rejectsNegativeDefaultThrottle() {
    assertThatThrownBy(() -> WorkerConfig.builder()
            .defaultThrottle(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultThrottle");
  }

  @Test
  void workerConfig_rejectsNegativeMaxConcurrentCompactions() {
    assertThatThrownBy(() ->
            WorkerConfig.builder().maxConcurrentCompactions(-1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxConcurrentCompactions");
  }

  @Test
  void workerConfig_rejectsInvalidVacuumCooldown() {
    assertThatThrownBy(() -> WorkerConfig.builder()
            .vacuumCooldown(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vacuumCooldown");
  }

  @Test
  void workerConfig_rejectsInvalidVacuumMinDeletedRatio() {
    assertThatThrownBy(
            () -> WorkerConfig.builder().vacuumMinDeletedRatio(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vacuumMinDeletedRatio");
    assertThatThrownBy(
            () -> WorkerConfig.builder().vacuumMinDeletedRatio(1.1).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidBuildTxnLimitBytes() {
    assertThatThrownBy(() -> WorkerConfig.builder().buildTxnLimitBytes(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("buildTxnLimitBytes");
    assertThatThrownBy(() -> WorkerConfig.builder().buildTxnLimitBytes(-1).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidBuildTxnSoftLimitRatio() {
    assertThatThrownBy(
            () -> WorkerConfig.builder().buildTxnSoftLimitRatio(0.0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("buildTxnSoftLimitRatio");
    assertThatThrownBy(
            () -> WorkerConfig.builder().buildTxnSoftLimitRatio(1.0).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidBuildSizeCheckEvery() {
    assertThatThrownBy(() -> WorkerConfig.builder().buildSizeCheckEvery(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("buildSizeCheckEvery");
  }

  @Test
  void workerConfig_rejectsInvalidBatchLoadSizes() {
    assertThatThrownBy(() -> WorkerConfig.builder().codebookBatchLoadSize(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("codebookBatchLoadSize");
    assertThatThrownBy(
            () -> WorkerConfig.builder().adjacencyBatchLoadSize(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("adjacencyBatchLoadSize");
  }

  @Test
  void workerConfig_rejectsInvalidCompactionMinSegments() {
    assertThatThrownBy(() -> WorkerConfig.builder().compactionMinSegments(1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionMinSegments");
  }

  @Test
  void workerConfig_rejectsInvalidCompactionMaxSegments() {
    assertThatThrownBy(() -> WorkerConfig.builder()
            .compactionMinSegments(4)
            .compactionMaxSegments(3)
            .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionMaxSegments");
  }

  @Test
  void workerConfig_rejectsInvalidCompactionMinFragmentation() {
    assertThatThrownBy(() ->
            WorkerConfig.builder().compactionMinFragmentation(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionMinFragmentation");
    assertThatThrownBy(() ->
            WorkerConfig.builder().compactionMinFragmentation(1.1).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsNegativeCompactionBiasWeights() {
    assertThatThrownBy(() ->
            WorkerConfig.builder().compactionAgeBiasWeight(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionAgeBiasWeight");
    assertThatThrownBy(() ->
            WorkerConfig.builder().compactionSizeBiasWeight(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionSizeBiasWeight");
    assertThatThrownBy(() ->
            WorkerConfig.builder().compactionFragBiasWeight(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("compactionFragBiasWeight");
  }

  @Test
  void workerConfig_rejectsNullInstantSource() {
    assertThatThrownBy(() -> WorkerConfig.builder().instantSource(null).build())
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void workerConfig_defaultsAreValid() {
    // Default builder should produce a valid WorkerConfig without throwing
    WorkerConfig cfg = WorkerConfig.builder().build();
    org.assertj.core.api.Assertions.assertThat(cfg.getEstimatedWorkerCount())
        .isEqualTo(1);
    org.assertj.core.api.Assertions.assertThat(cfg.getDefaultTtl()).isEqualTo(Duration.ofMinutes(5));
  }

  // ---- VectorIndexConfig null Duration checks ----

  @Test
  void vectorIndexConfig_rejectsNullVacuumCooldown() {
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .vacuumCooldown(null)
            .build())
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("vacuumCooldown");
  }

  @Test
  void vectorIndexConfig_rejectsNullDefaultThrottle() {
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .defaultThrottle(null)
            .build())
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("defaultThrottle");
  }

  // ---- WorkerConfig data-format fallback default validation tests ----

  @Test
  void workerConfig_rejectsInvalidDefaultMaxSegmentSize() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultMaxSegmentSize(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultMaxSegmentSize");
    assertThatThrownBy(
            () -> WorkerConfig.builder().defaultMaxSegmentSize(-1).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidDefaultPqM() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultPqM(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultPqM");
  }

  @Test
  void workerConfig_rejectsInvalidDefaultPqK() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultPqK(1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultPqK");
    assertThatThrownBy(() -> WorkerConfig.builder().defaultPqK(0).build())
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void workerConfig_rejectsInvalidDefaultGraphDegree() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultGraphDegree(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultGraphDegree");
  }

  @Test
  void workerConfig_rejectsInvalidDefaultOversample() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultOversample(0).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultOversample");
  }

  @Test
  void workerConfig_rejectsInvalidDefaultGraphBuildBreadth() {
    // defaultGraphBuildBreadth must be >= defaultGraphDegree
    assertThatThrownBy(() -> WorkerConfig.builder()
            .defaultGraphDegree(64)
            .defaultGraphBuildBreadth(32)
            .build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultGraphBuildBreadth");
  }

  @Test
  void workerConfig_rejectsNegativeDefaultGraphAlpha() {
    assertThatThrownBy(() -> WorkerConfig.builder().defaultGraphAlpha(-0.1).build())
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultGraphAlpha");
  }

  @Test
  void vectorIndexConfig_rejectsInvalidPrebuiltWorkerConfig() {
    // When a pre-built WorkerConfig with invalid values is supplied to VectorIndexConfig,
    // the WorkerConfig.Builder.build() should reject it before it reaches VectorIndexConfig
    assertThatThrownBy(() -> {
          WorkerConfig bad =
              WorkerConfig.builder().estimatedWorkerCount(0).build();
          VectorIndexConfig.builder(db, root)
              .dimension(4)
              .workerConfig(bad)
              .build();
        })
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("estimatedWorkerCount");

    assertThatThrownBy(() -> {
          WorkerConfig bad =
              WorkerConfig.builder().buildTxnSoftLimitRatio(0.0).build();
          VectorIndexConfig.builder(db, root)
              .dimension(4)
              .workerConfig(bad)
              .build();
        })
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("buildTxnSoftLimitRatio");
  }
}
