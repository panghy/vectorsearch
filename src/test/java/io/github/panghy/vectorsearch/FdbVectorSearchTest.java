package io.github.panghy.vectorsearch;

import static com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt;
import static com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt;
import static com.apple.foundationdb.tuple.Tuple.from;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.proto.Config;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for FdbVectorSearch.
 */
class FdbVectorSearchTest {

  private Database db;
  private DirectorySubspace directory;
  private VectorSearchConfig config;
  private FdbVectorSearch index;

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

    // Create a default configuration for testing
    config = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .pqSubvectors(16)
        .graphDegree(32)
        .build();
  }

  @AfterEach
  void tearDown() throws InterruptedException {
    if (index != null) {
      // Shutdown the index gracefully
      index.shutdown(true, 5, TimeUnit.SECONDS);
    }

    if (db != null) {
      db.run(tr -> {
        directory.remove(tr);
        return null;
      });
      db.close();
    }
  }

  // ============= Creation and Initialization Tests =============

  @Test
  @DisplayName("Should create new index successfully")
  void testCreateNewIndex() throws ExecutionException, InterruptedException, TimeoutException {
    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(config, db);
    index = future.get(10, TimeUnit.SECONDS);

    assertNotNull(index);

    // Verify configuration was stored
    Config storedConfig = db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] configBytes = tr.get(metaDir.pack(from("config"))).join();
      assertNotNull(configBytes);
      try {
        return Config.parseFrom(configBytes);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });

    assertEquals(128, storedConfig.getDimension());
    assertEquals("L2", storedConfig.getMetric());
    assertEquals(16, storedConfig.getPqSubvectors());
    assertEquals(32, storedConfig.getGraphDegree());
  }

  @Test
  @DisplayName("Should open existing index successfully")
  void testOpenExistingIndex() throws ExecutionException, InterruptedException, TimeoutException {
    // Create the index first
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);

    // Close it
    index.shutdown();

    // Open it again with same config
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);
  }

  @Test
  @DisplayName("Should validate configuration when opening existing index")
  void testValidateConfigurationOnOpen() throws ExecutionException, InterruptedException, TimeoutException {
    // Create the index first
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);

    // Close it
    index.shutdown();
    index = null;

    // Try to open with different immutable config (should fail)
    VectorSearchConfig differentConfig = VectorSearchConfig.builder(db, directory)
        .dimension(256) // Different dimension
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .build();

    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(differentConfig, db);

    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IndexConfigurationException.class)
        .hasRootCauseMessage(
            "Dimension mismatch: configured=256, stored=128. Dimension cannot be changed after collection"
                + " creation.");
  }

  @Test
  @DisplayName("Should allow different mutable configuration when reopening")
  void testAllowMutableConfigChanges() throws ExecutionException, InterruptedException, TimeoutException {
    // Create the index first
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);

    // Close it
    index.shutdown();
    index = null;

    // Open with different mutable config (should succeed)
    VectorSearchConfig differentConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128) // Same dimension
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2) // Same metric
        .pqSubvectors(16) // Same PQ config
        .graphDegree(32) // Same graph degree
        .linkWorkerCount(8) // Different worker count (mutable)
        .pqBlockCacheSize(2L * 1024 * 1024 * 1024) // Different cache size (mutable)
        .build();

    index = FdbVectorSearch.createOrOpen(differentConfig, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);
  }

  // ============= Subspace Creation Tests =============

  @Test
  @DisplayName("Should create all required subspaces")
  void testSubspaceCreation() throws ExecutionException, InterruptedException, TimeoutException {
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Verify all subspaces were created
    db.run(tr -> {
      // Check meta subspace
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      assertNotNull(metaDir);

      // Check PQ subspaces
      DirectorySubspace codebookDir =
          directory.createOrOpen(tr, List.of("pq", "codebook")).join();
      assertNotNull(codebookDir);
      DirectorySubspace pqBlockDir =
          directory.createOrOpen(tr, List.of("pq", "block")).join();
      assertNotNull(pqBlockDir);

      // Check graph subspaces
      DirectorySubspace graphNodeDir =
          directory.createOrOpen(tr, List.of("graph", "node")).join();
      assertNotNull(graphNodeDir);
      DirectorySubspace graphMetaDir =
          directory.createOrOpen(tr, List.of("graph", "meta")).join();
      assertNotNull(graphMetaDir);

      // Check other subspaces
      DirectorySubspace sketchDir =
          directory.createOrOpen(tr, List.of("sketch")).join();
      assertNotNull(sketchDir);
      DirectorySubspace entryDir =
          directory.createOrOpen(tr, List.of("entry")).join();
      assertNotNull(entryDir);

      // Check task queue directories
      DirectorySubspace linkQueueDir =
          directory.createOrOpen(tr, List.of("queue", "link")).join();
      assertNotNull(linkQueueDir);
      DirectorySubspace unlinkQueueDir =
          directory.createOrOpen(tr, List.of("queue", "unlink")).join();
      assertNotNull(unlinkQueueDir);

      return null;
    });
  }

  // ============= Codebook Version Tests =============

  @Test
  @DisplayName("Should initialize codebook version to 0")
  void testInitialCodebookVersion() throws ExecutionException, InterruptedException, TimeoutException {
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Check codebook version in FDB
    long version = db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] cbvBytes = tr.get(metaDir.pack(from("cbv_active"))).join();
      assertNotNull(cbvBytes);
      return decodeInt(cbvBytes);
    });

    assertEquals(0L, version);
  }

  @Test
  @DisplayName("Should preserve codebook version when reopening")
  void testPreserveCodebookVersion() throws ExecutionException, InterruptedException, TimeoutException {
    // Create index
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Manually update codebook version to test preservation
    db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] cbvKey = metaDir.pack(from("cbv_active"));
      tr.set(cbvKey, encodeInt(5));
      return null;
    });

    // Close and reopen
    index.shutdown();
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Verify version was preserved
    long version = db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] cbvBytes = tr.get(metaDir.pack(from("cbv_active"))).join();
      return decodeInt(cbvBytes);
    });

    assertEquals(5L, version);
  }

  // ============= Maintenance Task Tests =============

  @Test
  @DisplayName("Should schedule maintenance tasks when auto-repair is enabled")
  void testMaintenanceTaskScheduling() throws ExecutionException, InterruptedException, TimeoutException {
    VectorSearchConfig configWithRepair = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .autoRepairEnabled(true)
        .graphRepairInterval(Duration.ofMinutes(1))
        .entryPointRefreshInterval(Duration.ofMinutes(1))
        .build();

    index = FdbVectorSearch.createOrOpen(configWithRepair, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);

    // Tasks should be scheduled but we can't easily verify this without exposing internals
    // At least verify the index was created successfully with repair enabled
  }

  @Test
  @DisplayName("Should not schedule maintenance tasks when auto-repair is disabled")
  void testNoMaintenanceTasksWhenDisabled() throws ExecutionException, InterruptedException, TimeoutException {
    VectorSearchConfig configNoRepair = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .autoRepairEnabled(false)
        .build();

    index = FdbVectorSearch.createOrOpen(configNoRepair, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);

    // Tasks should not be scheduled
    // At least verify the index was created successfully with repair disabled
  }

  // ============= Shutdown Tests =============

  @Test
  @DisplayName("Should shutdown gracefully with timeout")
  void testGracefulShutdown() throws ExecutionException, InterruptedException, TimeoutException {
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    boolean terminated = index.shutdown(true, 5, TimeUnit.SECONDS);
    assertTrue(terminated);
  }

  @Test
  @DisplayName("Should shutdown immediately without waiting")
  void testImmediateShutdown() throws ExecutionException, InterruptedException, TimeoutException {
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    index.shutdown(); // Immediate shutdown

    // Should complete without throwing
  }

  @Test
  @DisplayName("Should handle multiple shutdown calls")
  void testMultipleShutdowns() throws ExecutionException, InterruptedException, TimeoutException {
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // First shutdown
    boolean terminated1 = index.shutdown(true, 1, TimeUnit.SECONDS);
    assertTrue(terminated1);

    // Second shutdown should still work (scheduler already terminated)
    boolean terminated2 = index.shutdown(true, 1, TimeUnit.SECONDS);
    assertTrue(terminated2);
  }

  // ============= Configuration Validation Tests =============

  @Test
  @DisplayName("Should reject incompatible dimension when reopening")
  void testRejectIncompatibleDimension() throws ExecutionException, InterruptedException, TimeoutException {
    // Create with dimension 128
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    index.shutdown();
    index = null;

    // Try to open with dimension 256
    VectorSearchConfig incompatibleConfig =
        VectorSearchConfig.builder(db, directory).dimension(256).build();

    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(incompatibleConfig, db);

    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IndexConfigurationException.class)
        .hasRootCauseMessage(
            "Dimension mismatch: configured=256, stored=128. Dimension cannot be changed after collection"
                + " creation.");
  }

  @Test
  @DisplayName("Should reject incompatible metric when reopening")
  void testRejectIncompatibleMetric() throws ExecutionException, InterruptedException, TimeoutException {
    // Create with L2 metric
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    index.shutdown();
    index = null;

    // Try to open with COSINE metric
    VectorSearchConfig incompatibleConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .build();

    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(incompatibleConfig, db);

    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IndexConfigurationException.class)
        .hasRootCauseMessage(
            "Distance metric mismatch: configured=COSINE, stored=L2. Distance metric cannot be changed"
                + " after collection creation.");
  }

  @Test
  @DisplayName("Should reject incompatible PQ configuration when reopening")
  void testRejectIncompatiblePqConfig() throws ExecutionException, InterruptedException, TimeoutException {
    // Create with PQ subvectors 16
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    index.shutdown();
    index = null;

    // Try to open with PQ subvectors 32
    VectorSearchConfig incompatibleConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .pqSubvectors(32)
        .build();

    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(incompatibleConfig, db);
    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IndexConfigurationException.class)
        .hasRootCauseMessage(
            "PQ subvectors mismatch: configured=32, stored=16. PQ configuration cannot be changed after"
                + " collection creation.");
  }

  @Test
  @DisplayName("Should reject incompatible graph degree when reopening")
  void testRejectIncompatibleGraphDegree() throws ExecutionException, InterruptedException, TimeoutException {
    // Create with graph degree 32
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    index.shutdown();
    index = null;

    // Try to open with graph degree 64
    VectorSearchConfig incompatibleConfig = VectorSearchConfig.builder(db, directory)
        .dimension(128)
        .pqSubvectors(16)
        .graphDegree(64)
        .build();

    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(incompatibleConfig, db);

    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IndexConfigurationException.class)
        .hasRootCauseMessage(
            "Graph degree mismatch: configured=64, stored=32. Graph degree cannot be changed after"
                + " collection creation (would require graph rebuild).");
  }

  // ============= Error Handling Tests =============

  @Test
  @DisplayName("Should handle corrupted config gracefully")
  void testHandleCorruptedConfig() throws ExecutionException, InterruptedException, TimeoutException {
    // Create index first
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    index.shutdown();
    index = null;

    // Corrupt the stored config
    db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] configKey = metaDir.pack(from("config"));
      tr.set(configKey, "corrupted data".getBytes());
      return null;
    });

    // Try to open - should fail with proper error
    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(config, db);

    assertThatThrownBy(() -> future.get(10, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(InvalidProtocolBufferException.InvalidWireTypeException.class);
  }

  @Test
  @DisplayName("Should handle missing codebook version gracefully")
  void testHandleMissingCodebookVersion() throws ExecutionException, InterruptedException, TimeoutException {
    // Create index
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);

    // Delete the codebook version
    db.run(tr -> {
      DirectorySubspace metaDir =
          directory.createOrOpen(tr, List.of("meta")).join();
      byte[] cbvKey = metaDir.pack(from("cbv_active"));
      tr.clear(cbvKey);
      return null;
    });

    // Close and reopen - should handle missing version gracefully
    index.shutdown();
    index = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
    assertNotNull(index);
  }

  // ============= Concurrent Access Tests =============

  @Test
  @DisplayName("Should handle concurrent index creation")
  void testConcurrentCreation() throws InterruptedException {
    // Launch multiple threads trying to create the same index
    int threadCount = 5;
    CompletableFuture<?>[] futures = new CompletableFuture[threadCount];
    FdbVectorSearch[] indices = new FdbVectorSearch[threadCount];

    for (int i = 0; i < threadCount; i++) {
      final int idx = i;
      futures[i] = CompletableFuture.supplyAsync(() -> {
        try {
          indices[idx] = FdbVectorSearch.createOrOpen(config, db).get(10, TimeUnit.SECONDS);
          return indices[idx];
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      });
    }

    // Wait for all to complete
    CompletableFuture.allOf(futures).join();

    // All should have succeeded and created valid indices
    for (int i = 0; i < threadCount; i++) {
      assertNotNull(indices[i]);
      if (i == 0) {
        index = indices[i]; // Keep one for cleanup
      } else {
        indices[i].shutdown();
      }
    }
  }

  @Test
  @DisplayName("Test LongSerializer serialization and deserialization")
  void testLongSerializer() {
    FdbVectorSearch.LongSerializer serializer = new FdbVectorSearch.LongSerializer();

    // Test positive number
    Long value1 = 12345L;
    com.google.protobuf.ByteString serialized1 = serializer.serialize(value1);
    Long deserialized1 = serializer.deserialize(serialized1);
    assertEquals(value1, deserialized1);

    // Test zero
    Long value2 = 0L;
    com.google.protobuf.ByteString serialized2 = serializer.serialize(value2);
    Long deserialized2 = serializer.deserialize(serialized2);
    assertEquals(value2, deserialized2);

    // Test negative number
    Long value3 = -99999L;
    com.google.protobuf.ByteString serialized3 = serializer.serialize(value3);
    Long deserialized3 = serializer.deserialize(serialized3);
    assertEquals(value3, deserialized3);

    // Test max value
    Long value4 = Long.MAX_VALUE;
    com.google.protobuf.ByteString serialized4 = serializer.serialize(value4);
    Long deserialized4 = serializer.deserialize(serialized4);
    assertEquals(value4, deserialized4);

    // Test min value
    Long value5 = Long.MIN_VALUE;
    com.google.protobuf.ByteString serialized5 = serializer.serialize(value5);
    Long deserialized5 = serializer.deserialize(serialized5);
    assertEquals(value5, deserialized5);
  }

  @Test
  @DisplayName("Test ProtoSerializer with LinkTask")
  void testProtoSerializerWithLinkTask() {
    FdbVectorSearch.ProtoSerializer<io.github.panghy.vectorsearch.proto.LinkTask> serializer =
        new FdbVectorSearch.ProtoSerializer<>(io.github.panghy.vectorsearch.proto.LinkTask.parser());

    // Create a test LinkTask
    io.github.panghy.vectorsearch.proto.LinkTask task = io.github.panghy.vectorsearch.proto.LinkTask.newBuilder()
        .setNodeId(12345L)
        .setCodebookVersion(1)
        .setPqCode(com.google.protobuf.ByteString.copyFrom(new byte[] {1, 2, 3, 4, 5}))
        .build();

    // Serialize
    com.google.protobuf.ByteString serialized = serializer.serialize(task);
    assertNotNull(serialized);
    assertTrue(serialized.size() > 0);

    // Deserialize
    io.github.panghy.vectorsearch.proto.LinkTask deserialized = serializer.deserialize(serialized);
    assertNotNull(deserialized);
    assertEquals(task.getNodeId(), deserialized.getNodeId());
    assertEquals(task.getCodebookVersion(), deserialized.getCodebookVersion());
    assertEquals(task.getPqCode(), deserialized.getPqCode());
  }

  @Test
  @DisplayName("Test ProtoSerializer with invalid data")
  void testProtoSerializerWithInvalidData() {
    FdbVectorSearch.ProtoSerializer<io.github.panghy.vectorsearch.proto.LinkTask> serializer =
        new FdbVectorSearch.ProtoSerializer<>(io.github.panghy.vectorsearch.proto.LinkTask.parser());

    // Test with invalid protobuf data
    com.google.protobuf.ByteString invalidData =
        com.google.protobuf.ByteString.copyFromUtf8("not a valid protobuf");

    assertThatThrownBy(() -> serializer.deserialize(invalidData))
        .isInstanceOf(RuntimeException.class)
        .hasMessageContaining("Failed to parse protobuf message")
        .hasCauseInstanceOf(InvalidProtocolBufferException.class);
  }

  @Test
  @DisplayName("Test ProtoSerializer with empty ByteString")
  void testProtoSerializerWithEmptyData() {
    FdbVectorSearch.ProtoSerializer<io.github.panghy.vectorsearch.proto.LinkTask> serializer =
        new FdbVectorSearch.ProtoSerializer<>(io.github.panghy.vectorsearch.proto.LinkTask.parser());

    // Test with empty ByteString - should parse to default instance
    com.google.protobuf.ByteString emptyData = com.google.protobuf.ByteString.EMPTY;
    io.github.panghy.vectorsearch.proto.LinkTask deserialized = serializer.deserialize(emptyData);
    assertNotNull(deserialized);
    // Should have default values
    assertEquals(0, deserialized.getNodeId());
    assertEquals(0, deserialized.getCodebookVersion());
  }

  @Test
  @DisplayName("Test CodebookCacheKey record")
  void testCodebookCacheKey() {
    // Test creation and fields
    FdbVectorSearch.CodebookCacheKey key1 = new FdbVectorSearch.CodebookCacheKey(1, 5);
    assertEquals(1, key1.version());
    assertEquals(5, key1.subspace());

    // Test equality
    FdbVectorSearch.CodebookCacheKey key2 = new FdbVectorSearch.CodebookCacheKey(1, 5);
    assertEquals(key1, key2);
    assertEquals(key1.hashCode(), key2.hashCode());

    // Test inequality
    FdbVectorSearch.CodebookCacheKey key3 = new FdbVectorSearch.CodebookCacheKey(2, 5);
    assertNotEquals(key1, key3);

    FdbVectorSearch.CodebookCacheKey key4 = new FdbVectorSearch.CodebookCacheKey(1, 6);
    assertNotEquals(key1, key4);

    // Test toString
    assertNotNull(key1.toString());
    assertTrue(key1.toString().contains("1"));
    assertTrue(key1.toString().contains("5"));
  }

  @Test
  @DisplayName("Test PqBlockCacheKey record")
  void testPqBlockCacheKey() {
    // Test creation and fields
    FdbVectorSearch.PqBlockCacheKey key1 = new FdbVectorSearch.PqBlockCacheKey(1, 1000L);
    assertEquals(1, key1.version());
    assertEquals(1000L, key1.blockNumber());

    // Test equality
    FdbVectorSearch.PqBlockCacheKey key2 = new FdbVectorSearch.PqBlockCacheKey(1, 1000L);
    assertEquals(key1, key2);
    assertEquals(key1.hashCode(), key2.hashCode());

    // Test inequality
    FdbVectorSearch.PqBlockCacheKey key3 = new FdbVectorSearch.PqBlockCacheKey(2, 1000L);
    assertNotEquals(key1, key3);

    FdbVectorSearch.PqBlockCacheKey key4 = new FdbVectorSearch.PqBlockCacheKey(1, 2000L);
    assertNotEquals(key1, key4);

    // Test toString
    assertNotNull(key1.toString());
    assertTrue(key1.toString().contains("1"));
    assertTrue(key1.toString().contains("1000"));
  }

  @Test
  @DisplayName("Test refreshEntryPoints method")
  void testRefreshEntryPoints() throws Exception {
    // Create index
    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(config, db);
    index = future.get(10, TimeUnit.SECONDS);

    index.refreshEntryPoints().join();
  }

  @Test
  @DisplayName("Test checkAndRepairConnectivity method")
  void testCheckAndRepairConnectivity() throws Exception {
    // Create index
    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(config, db);
    index = future.get(10, TimeUnit.SECONDS);

    // Use reflection to access private method
    index.checkAndRepairConnectivity().join();
  }

  @Test
  @DisplayName("Test PqBlockCache weigher")
  void testPqBlockCacheWeigher() throws Exception {
    // Create index
    CompletableFuture<FdbVectorSearch> future = FdbVectorSearch.createOrOpen(config, db);
    index = future.get(10, TimeUnit.SECONDS);

    // Get the cache
    var cache = index.getPqBlockCache();
    assertNotNull(cache);

    // Create a test PqCodesBlock
    io.github.panghy.vectorsearch.proto.PqCodesBlock block =
        io.github.panghy.vectorsearch.proto.PqCodesBlock.newBuilder()
            .setCodes(com.google.protobuf.ByteString.copyFrom(new byte[1000]))
            .build();

    // Put it in the cache
    FdbVectorSearch.PqBlockCacheKey key = new FdbVectorSearch.PqBlockCacheKey(1, 100L);
    cache.put(key, block);

    // Verify the cache contains the entry
    assertEquals(block, cache.getIfPresent(key));

    // Verify stats are recorded (since recordStats() was called in builder)
    var stats = cache.stats();
    assertNotNull(stats);
  }
}
