package io.github.panghy.vectorsearch;

import static org.assertj.core.api.AssertionsForClassTypes.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.vectorsearch.proto.Config;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for VectorSearchConfig validation and builder logic.
 */
class VectorSearchConfigTest {

  private Database db;
  private DirectorySubspace directory;

  @BeforeEach
  void setUp() throws ExecutionException, InterruptedException, TimeoutException {
    db = FDB.selectAPIVersion(730).open();
    directory = db.runAsync(tr -> {
          DirectoryLayer layer = DirectoryLayer.getDefault();
          return layer.createOrOpen(
              tr,
              List.of("test", UUID.randomUUID().toString()),
              "vector_search".getBytes(StandardCharsets.UTF_8));
        })
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    if (db != null) {
      db.run(tr -> {
        directory.remove(tr);
        return null;
      });
      db.close();
    }
  }

  // ============= Builder Default Values Tests =============

  @Test
  @DisplayName("Builder should set correct default values")
  void testBuilderDefaults() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(128) // Required
        .build();

    // Core defaults
    assertEquals(128, config.getDimension());
    assertEquals(VectorSearchConfig.DistanceMetric.L2, config.getDistanceMetric());

    // PQ defaults
    assertTrue(config.isPqEnabled());
    assertEquals(64, config.getPqSubvectors()); // dimension/2
    assertEquals(8, config.getPqNbits());
    assertEquals(512, config.getCodesPerBlock());

    // Graph defaults
    assertEquals(64, config.getGraphDegree());
    assertEquals(16, config.getDefaultSearchList());
    assertEquals(1500, config.getMaxSearchVisits());
    assertEquals(32, config.getEntryPointCount());
    assertEquals(Duration.ofHours(1), config.getEntryPointRefreshInterval());

    // Cache defaults
    assertEquals(1024L * 1024 * 1024, config.getPqBlockCacheSize()); // 1GB
    assertEquals(100_000, config.getAdjacencyCacheSize());
    assertTrue(config.isKeepCodebooksInMemory());

    // Worker defaults
    assertEquals(4, config.getLinkWorkerCount());
    assertEquals(2, config.getUnlinkWorkerCount());
    assertEquals(ForkJoinPool.commonPool(), config.getBackgroundExecutor());

    // Maintenance defaults
    assertTrue(config.isAutoRepairEnabled());
    assertEquals(Duration.ofHours(6), config.getGraphRepairInterval());
    assertEquals(0.95, config.getMinConnectivityThreshold());

    // Storage defaults
    assertFalse(config.isUseBlockStriping());
    assertEquals(4, config.getStripeCount());
    assertTrue(config.isStoreVectorSketches());
  }

  @Test
  @DisplayName("Builder should auto-configure PQ subvectors when not set")
  void testPqSubvectorsAutoConfiguration() {
    // Test dimension 768 (BERT)
    VectorSearchConfig config768 =
        VectorSearchConfig.builder(db, directory).dimension(768).build();
    assertEquals(384, config768.getPqSubvectors()); // 768/2

    // Test dimension 1536 (OpenAI)
    VectorSearchConfig config1536 =
        VectorSearchConfig.builder(db, directory).dimension(1536).build();
    assertEquals(768, config1536.getPqSubvectors()); // 1536/2

    // Test odd dimension that doesn't divide evenly
    VectorSearchConfig config100 =
        VectorSearchConfig.builder(db, directory).dimension(100).build();
    assertEquals(50, config100.getPqSubvectors()); // 100/2

    // Test prime dimension
    VectorSearchConfig config97 =
        VectorSearchConfig.builder(db, directory).dimension(97).build();
    assertEquals(1, config97.getPqSubvectors()); // Falls back to 1 when can't divide evenly
  }

  // ============= Custom Configuration Tests =============

  @Test
  @DisplayName("Builder should accept custom values")
  void testBuilderCustomValues() {
    InstantSource customSource = InstantSource.fixed(Instant.EPOCH);

    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(256)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .pqEnabled(true)
        .pqSubvectors(32)
        .pqNbits(8)
        .codesPerBlock(256)
        .graphDegree(128)
        .defaultSearchList(32)
        .maxSearchVisits(2000)
        .entryPointCount(64)
        .entryPointRefreshInterval(Duration.ofMinutes(30))
        .pqBlockCacheSize(2L * 1024 * 1024 * 1024) // 2GB
        .adjacencyCacheSize(200_000)
        .keepCodebooksInMemory(false)
        .linkWorkerCount(8)
        .unlinkWorkerCount(4)
        .autoRepairEnabled(false)
        .graphRepairInterval(Duration.ofHours(12))
        .minConnectivityThreshold(0.9)
        .useBlockStriping(true)
        .stripeCount(8)
        .storeVectorSketches(false)
        .instantSource(customSource)
        .build();

    assertEquals(256, config.getDimension());
    assertEquals(VectorSearchConfig.DistanceMetric.COSINE, config.getDistanceMetric());
    assertEquals(32, config.getPqSubvectors());
    assertEquals(256, config.getCodesPerBlock());
    assertEquals(128, config.getGraphDegree());
    assertEquals(32, config.getDefaultSearchList());
    assertEquals(2000, config.getMaxSearchVisits());
    assertEquals(64, config.getEntryPointCount());
    assertEquals(Duration.ofMinutes(30), config.getEntryPointRefreshInterval());
    assertEquals(2L * 1024 * 1024 * 1024, config.getPqBlockCacheSize());
    assertEquals(200_000, config.getAdjacencyCacheSize());
    assertFalse(config.isKeepCodebooksInMemory());
    assertEquals(8, config.getLinkWorkerCount());
    assertEquals(4, config.getUnlinkWorkerCount());
    assertFalse(config.isAutoRepairEnabled());
    assertEquals(Duration.ofHours(12), config.getGraphRepairInterval());
    assertEquals(0.9, config.getMinConnectivityThreshold());
    assertTrue(config.isUseBlockStriping());
    assertEquals(8, config.getStripeCount());
    assertFalse(config.isStoreVectorSketches());
    assertEquals(customSource, config.getInstantSource());
  }

  // ============= Validation Tests =============

  @Test
  @DisplayName("Should throw exception when dimension is not set")
  void testMissingDimension() {
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorSearchConfig.builder(db, directory).build(),
        "dimension must be set and positive");
  }

  @Test
  @DisplayName("Should throw exception for invalid dimension values")
  void testInvalidDimension() {
    // Zero dimension
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorSearchConfig.builder(db, directory).dimension(0).build());

    // Negative dimension
    assertThrows(
        IllegalArgumentException.class,
        () -> VectorSearchConfig.builder(db, directory).dimension(-1).build());

    // Too large dimension
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(5000) // > 4096
        .build());
  }

  @Test
  @DisplayName("Should throw exception for invalid PQ configuration")
  void testInvalidPqConfiguration() {
    // PQ subvectors > dimension
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .pqSubvectors(256)
        .build());

    // PQ subvectors doesn't divide dimension evenly
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(100)
        .pqSubvectors(33)
        .build());

    // Invalid nbits (only 8 supported)
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .pqNbits(16)
        .build());

    // Invalid codes per block
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .codesPerBlock(0)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .codesPerBlock(2000) // > 1024
        .build());
  }

  @Test
  @DisplayName("Should throw exception for invalid graph configuration")
  void testInvalidGraphConfiguration() {
    // Invalid graph degree
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .graphDegree(0)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .graphDegree(300) // > 256
        .build());

    // Invalid search list
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .defaultSearchList(0)
        .build());

    // Invalid max visits
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .maxSearchVisits(0)
        .build());

    // Invalid entry point count
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .entryPointCount(0)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .entryPointCount(300) // > 256
        .build());
  }

  @Test
  @DisplayName("Should throw exception for invalid worker configuration")
  void testInvalidWorkerConfiguration() {
    // Invalid link worker count
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .linkWorkerCount(0)
        .build());

    // Invalid unlink worker count
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .unlinkWorkerCount(0)
        .build());
  }

  @Test
  @DisplayName("Should throw exception for invalid maintenance configuration")
  void testInvalidMaintenanceConfiguration() {
    // Invalid connectivity threshold
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .minConnectivityThreshold(-0.1)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .minConnectivityThreshold(1.1)
        .build());

    // Invalid repair interval
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .graphRepairInterval(Duration.ZERO)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .graphRepairInterval(Duration.ofHours(-1))
        .build());
  }

  @Test
  @DisplayName("Should throw exception for invalid striping configuration")
  void testInvalidStripingConfiguration() {
    // Invalid stripe count when striping enabled
    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .useBlockStriping(true)
        .stripeCount(0)
        .build());

    assertThrows(IllegalArgumentException.class, () -> VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .useBlockStriping(true)
        .stripeCount(20) // > 16
        .build());
  }

  // ============= Proto Conversion Tests =============

  @Test
  @DisplayName("Should correctly convert to protobuf Config")
  void testToProtoConfig() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .pqSubvectors(96)
        .pqNbits(8)
        .graphDegree(64)
        .codesPerBlock(512)
        .defaultSearchList(32)
        .maxSearchVisits(2000)
        .build();

    Config protoConfig = config.toProtoConfig();

    assertEquals(768, protoConfig.getDimension());
    assertEquals("COSINE", protoConfig.getMetric());
    assertEquals(96, protoConfig.getPqSubvectors());
    assertEquals(8, protoConfig.getPqNbits());
    assertEquals(64, protoConfig.getGraphDegree());
    assertEquals(512, protoConfig.getCodesPerBlock());
    assertEquals(32, protoConfig.getDefaultSearchList());
    assertEquals(2000, protoConfig.getMaxSearchVisits());
  }

  @Test
  @DisplayName("Should handle all distance metrics in proto conversion")
  void testDistanceMetricProtoConversion() {
    // L2
    VectorSearchConfig l2Config = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .build();
    assertEquals("L2", l2Config.toProtoConfig().getMetric());

    // Inner Product
    VectorSearchConfig ipConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .distanceMetric(VectorSearchConfig.DistanceMetric.INNER_PRODUCT)
        .build();
    assertEquals("IP", ipConfig.toProtoConfig().getMetric());

    // Cosine
    VectorSearchConfig cosineConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .build();
    assertEquals("COSINE", cosineConfig.toProtoConfig().getMetric());
  }

  // ============= Validation Against Stored Config Tests =============

  @Test
  @DisplayName("Should validate successfully when configs match")
  void testValidateAgainstStoredSuccess() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .pqSubvectors(96)
        .pqNbits(8)
        .graphDegree(64)
        .codesPerBlock(512)
        .build();

    Config storedConfig = Config.newBuilder()
        .setDimension(768)
        .setMetric("COSINE")
        .setPqSubvectors(96)
        .setPqNbits(8)
        .setGraphDegree(64)
        .setCodesPerBlock(512)
        .build();

    // Should not throw
    assertDoesNotThrow(() -> config.validateAgainstStored(storedConfig));
  }

  @Test
  @DisplayName("Should throw exception when dimension doesn't match stored")
  void testValidateAgainstStoredDimensionMismatch() {
    VectorSearchConfig config =
        VectorSearchConfig.builder(db, directory).dimension(768).build();

    Config storedConfig = Config.newBuilder()
        .setDimension(512) // Different dimension
        .setMetric("L2")
        .setGraphDegree(64)
        .setCodesPerBlock(512)
        .build();

    assertThatThrownBy(() -> config.validateAgainstStored(storedConfig))
        .isInstanceOf(IndexConfigurationException.class)
        .hasMessageContaining("Dimension mismatch");
  }

  @Test
  @DisplayName("Should throw exception when distance metric doesn't match stored")
  void testValidateAgainstStoredMetricMismatch() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .build();

    Config storedConfig = Config.newBuilder()
        .setDimension(768)
        .setMetric("L2") // Different metric
        .setGraphDegree(64)
        .setCodesPerBlock(512)
        .build();

    assertThatThrownBy(() -> config.validateAgainstStored(storedConfig))
        .isInstanceOf(IndexConfigurationException.class)
        .hasMessageContaining("Distance metric mismatch");
  }

  @Test
  @DisplayName("Should throw exception when PQ configuration doesn't match stored")
  void testValidateAgainstStoredPqMismatch() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .pqSubvectors(96)
        .build();

    Config storedConfig = Config.newBuilder()
        .setDimension(768)
        .setMetric("L2")
        .setPqSubvectors(64) // Different PQ subvectors
        .setPqNbits(8)
        .setGraphDegree(64)
        .setCodesPerBlock(512)
        .build();

    assertThatThrownBy(() -> config.validateAgainstStored(storedConfig))
        .isInstanceOf(IndexConfigurationException.class)
        .hasMessageContaining("PQ subvectors mismatch");
  }

  @Test
  @DisplayName("Should throw exception when graph degree doesn't match stored")
  void testValidateAgainstStoredGraphDegreeMismatch() throws Exception {
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .graphDegree(64)
        .build();

    Config storedConfig = Config.newBuilder()
        .setDimension(768)
        .setMetric("L2")
        .setPqSubvectors(384) // Match the auto-calculated value for 768
        .setPqNbits(8)
        .setGraphDegree(128) // Different graph degree
        .setCodesPerBlock(512)
        .build();

    assertThatThrownBy(() -> config.validateAgainstStored(storedConfig))
        .isInstanceOf(IndexConfigurationException.class)
        .hasMessageContaining("Graph degree mismatch");
  }

  @Test
  @DisplayName("Should allow different mutable values when validating against stored")
  void testValidateAgainstStoredMutableValues() throws Exception {
    // Config with different mutable values (cache sizes, worker counts, etc.)
    VectorSearchConfig config = VectorSearchConfig.builder(db, directory)
        .dimension(768)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .graphDegree(64)
        // These are mutable and can differ
        .defaultSearchList(32) // Different from stored
        .maxSearchVisits(3000) // Different from stored
        .linkWorkerCount(16) // Different from stored
        .pqBlockCacheSize(4L * 1024 * 1024 * 1024) // Different from stored
        .build();

    Config storedConfig = Config.newBuilder()
        .setDimension(768)
        .setMetric("L2")
        .setPqSubvectors(384) // Match the auto-calculated value for 768
        .setPqNbits(8)
        .setGraphDegree(64)
        .setCodesPerBlock(512)
        // These are stored but can be overridden
        .setDefaultSearchList(16)
        .setMaxSearchVisits(1500)
        .build();

    // Should not throw - mutable values can differ
    assertDoesNotThrow(() -> config.validateAgainstStored(storedConfig));
  }

  // ============= Edge Cases Tests =============

  @Test
  @DisplayName("Should handle null values correctly")
  void testNullValues() {
    assertThrows(
        NullPointerException.class,
        () -> VectorSearchConfig.builder(null, directory).dimension(128).build());

    assertThrows(
        NullPointerException.class,
        () -> VectorSearchConfig.builder(db, null).dimension(128).build());
  }

  @Test
  @DisplayName("Should handle boundary values correctly")
  void testBoundaryValues() {
    // Minimum valid dimension
    VectorSearchConfig minConfig =
        VectorSearchConfig.builder(db, directory).dimension(1).build();
    assertEquals(1, minConfig.getDimension());

    // Maximum valid dimension
    VectorSearchConfig maxConfig =
        VectorSearchConfig.builder(db, directory).dimension(4096).build();
    assertEquals(4096, maxConfig.getDimension());

    // Minimum connectivity threshold
    VectorSearchConfig minThreshold = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .minConnectivityThreshold(0.0)
        .build();
    assertEquals(0.0, minThreshold.getMinConnectivityThreshold());

    // Maximum connectivity threshold
    VectorSearchConfig maxThreshold = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .minConnectivityThreshold(1.0)
        .build();
    assertEquals(1.0, maxThreshold.getMinConnectivityThreshold());
  }

  @Test
  @DisplayName("Test taskQueueConfig getter and builder")
  void testTaskQueueConfig() throws Exception {
    // Test with null taskQueueConfig (default)
    VectorSearchConfig configWithDefault =
        VectorSearchConfig.builder(db, directory).dimension(128).build();
    assertNull(configWithDefault.getTaskQueueConfig());

    // Test with custom taskQueueConfig using the builder method
    DirectorySubspace queueDir =
        directory.createOrOpen(db, List.of("test_queue")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<Long, Long> customTaskQueueConfig = TaskQueueConfig.builder(
            db,
            queueDir,
            new FdbVectorSearchIndex.LongSerializer(),
            new FdbVectorSearchIndex.LongSerializer())
        .build();

    VectorSearchConfig.Builder builder =
        VectorSearchConfig.builder(db, directory).dimension(128);

    // Use the taskQueueConfig method to set it
    builder.taskQueueConfig(customTaskQueueConfig);

    VectorSearchConfig configWithCustom = builder.build();
    assertNotNull(configWithCustom.getTaskQueueConfig());
    assertEquals(customTaskQueueConfig, configWithCustom.getTaskQueueConfig());
  }

  @Test
  @DisplayName("Test backgroundExecutor builder")
  void testBackgroundExecutor() {
    // Test with default executor
    VectorSearchConfig configWithDefault =
        VectorSearchConfig.builder(db, directory).dimension(128).build();
    assertEquals(ForkJoinPool.commonPool(), configWithDefault.getBackgroundExecutor());

    // Test with custom executor
    ForkJoinPool customExecutor = new ForkJoinPool(2);
    try {
      VectorSearchConfig configWithCustom = VectorSearchConfig.builder(db, directory)
          .dimension(128)
          .backgroundExecutor(customExecutor)
          .build();
      assertEquals(customExecutor, configWithCustom.getBackgroundExecutor());
    } finally {
      customExecutor.shutdown();
    }
  }
}
