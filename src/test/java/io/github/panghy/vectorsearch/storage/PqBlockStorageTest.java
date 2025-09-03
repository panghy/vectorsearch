package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class PqBlockStorageTest {

  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private PqBlockStorage storage;
  private String testCollectionName;
  private static final int CODES_PER_BLOCK = 256;
  private static final int PQ_SUBVECTORS = 8;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "pq_block_storage", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace);
    storage = new PqBlockStorage(
        db,
        keys,
        CODES_PER_BLOCK,
        PQ_SUBVECTORS,
        java.time.InstantSource.system(),
        1000,
        java.time.Duration.ofMinutes(10));
  }

  @AfterEach
  void tearDown() {
    if (db != null && testSpace != null) {
      db.run(tr -> {
        DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
        directoryLayer.removeIfExists(tr, testSpace.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  void testStoreSinglePqCode() throws ExecutionException, InterruptedException {
    long nodeId = 42L;
    byte[] pqCode = createRandomPqCode();
    int codebookVersion = 1;

    storage.storePqCode(nodeId, pqCode, codebookVersion).get();

    byte[] loaded = storage.loadPqCode(nodeId, codebookVersion).get();

    assertThat(loaded).isNotNull();
    assertThat(loaded).isEqualTo(pqCode);
  }

  @Test
  void testStoreMultiplePqCodesInSameBlock() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Store multiple codes in the same block (block 0)
    for (long nodeId = 0; nodeId < 10; nodeId++) {
      byte[] pqCode = createRandomPqCode();
      storage.storePqCode(nodeId, pqCode, codebookVersion).get();
    }

    // Load and verify each code
    for (long nodeId = 0; nodeId < 10; nodeId++) {
      byte[] loaded = storage.loadPqCode(nodeId, codebookVersion).get();
      assertThat(loaded).isNotNull();
      assertThat(loaded.length).isEqualTo(PQ_SUBVECTORS);
    }
  }

  @Test
  void testStoreAcrossMultipleBlocks() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Store codes across multiple blocks
    long[] nodeIds = {0L, 255L, 256L, 257L, 512L};
    byte[][] pqCodes = new byte[nodeIds.length][];

    for (int i = 0; i < nodeIds.length; i++) {
      pqCodes[i] = createRandomPqCode();
      storage.storePqCode(nodeIds[i], pqCodes[i], codebookVersion).get();
    }

    // Verify each code
    for (int i = 0; i < nodeIds.length; i++) {
      byte[] loaded = storage.loadPqCode(nodeIds[i], codebookVersion).get();
      assertThat(loaded).isEqualTo(pqCodes[i]);
    }
  }

  @Test
  void testBatchStore() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    List<Long> nodeIds = Arrays.asList(10L, 20L, 30L, 300L, 400L);
    List<byte[]> pqCodes = new ArrayList<>();
    for (int i = 0; i < nodeIds.size(); i++) {
      pqCodes.add(createRandomPqCode());
    }

    storage.batchStorePqCodes(nodeIds, pqCodes, codebookVersion).get();

    // Verify all codes were stored
    for (int i = 0; i < nodeIds.size(); i++) {
      byte[] loaded = storage.loadPqCode(nodeIds.get(i), codebookVersion).get();
      assertThat(loaded).isEqualTo(pqCodes.get(i));
    }
  }

  @Test
  void testBatchLoad() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Store some codes
    List<Long> nodeIds = Arrays.asList(5L, 15L, 25L, 260L, 520L);
    List<byte[]> pqCodes = new ArrayList<>();
    for (Long nodeId : nodeIds) {
      byte[] code = createRandomPqCode();
      pqCodes.add(code);
      storage.storePqCode(nodeId, code, codebookVersion).get();
    }

    // Batch load
    List<byte[]> loaded = storage.batchLoadPqCodes(nodeIds, codebookVersion).get();

    assertThat(loaded).hasSize(nodeIds.size());
    for (int i = 0; i < nodeIds.size(); i++) {
      assertThat(loaded.get(i)).isEqualTo(pqCodes.get(i));
    }
  }

  @Test
  void testBatchLoadWithMissingNodes() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Store only some codes
    storage.storePqCode(10L, createRandomPqCode(), codebookVersion).get();
    storage.storePqCode(30L, createRandomPqCode(), codebookVersion).get();

    // Try to load including missing nodes
    List<Long> nodeIds = Arrays.asList(10L, 20L, 30L);
    List<byte[]> loaded = storage.batchLoadPqCodes(nodeIds, codebookVersion).get();

    assertThat(loaded).hasSize(3);
    assertThat(loaded.get(0)).isNotNull(); // 10L exists
    assertThat(loaded.get(1)).isNull(); // 20L doesn't exist (null)
    assertThat(loaded.get(2)).isNotNull(); // 30L exists
  }

  @Test
  void testLoadNonExistentCode() throws ExecutionException, InterruptedException {
    byte[] loaded = storage.loadPqCode(999L, 1).get();
    assertThat(loaded).isNull();
  }

  @Test
  void testDeletePqCode() throws ExecutionException, InterruptedException {
    long nodeId = 100L;
    byte[] pqCode = createRandomPqCode();
    int codebookVersion = 1;

    // Store code
    storage.storePqCode(nodeId, pqCode, codebookVersion).get();
    assertThat(storage.loadPqCode(nodeId, codebookVersion).get()).isNotNull();

    // Delete code
    storage.deletePqCode(nodeId, codebookVersion).get();

    // Verify deleted (returns empty/zero code)
    byte[] deleted = storage.loadPqCode(nodeId, codebookVersion).get();
    assertThat(deleted).isEqualTo(new byte[PQ_SUBVECTORS]);
  }

  @Test
  void testDeleteVersion() throws ExecutionException, InterruptedException {
    int version1 = 1;
    int version2 = 2;

    // Store codes in two versions (use different nodes to avoid any interference)
    byte[] code1 = createRandomPqCode();
    byte[] code2 = createRandomPqCode();

    storage.storePqCode(10L, code1, version1).get();
    storage.storePqCode(11L, code2, version2).get();

    // Verify both are stored correctly first
    assertThat(storage.loadPqCode(10L, version1).get()).isNotNull();
    assertThat(storage.loadPqCode(11L, version2).get()).isNotNull();

    // Delete version 1
    storage.deleteVersion(version1).get();

    // Version 1 should be gone, version 2 should remain
    byte[] deletedCode = storage.loadPqCode(10L, version1).get();
    assertThat(deletedCode).isNull();

    // Version 2 should still be there
    byte[] remainingCode = storage.loadPqCode(11L, version2).get();
    assertThat(remainingCode).isNotNull().isEqualTo(code2);
  }

  @Test
  void testMigrateVersion() throws ExecutionException, InterruptedException {
    int fromVersion = 1;
    int toVersion = 2;

    // Store codes in version 1
    List<Long> nodeIds = Arrays.asList(10L, 20L, 30L);
    List<byte[]> originalCodes = new ArrayList<>();
    for (Long nodeId : nodeIds) {
      byte[] code = createRandomPqCode();
      originalCodes.add(code);
      storage.storePqCode(nodeId, code, fromVersion).get();
    }

    // Migrate with a simple re-encoder (adds 1 to each byte)
    PqBlockStorage.CodeReencoder reencoder = (nodeId, oldCode) -> {
      byte[] newCode = new byte[oldCode.length];
      for (int i = 0; i < oldCode.length; i++) {
        newCode[i] = (byte) (oldCode[i] + 1);
      }
      return newCode;
    };

    storage.migrateVersion(fromVersion, toVersion, reencoder).get();

    // Verify migration
    for (int i = 0; i < nodeIds.size(); i++) {
      byte[] migrated = storage.loadPqCode(nodeIds.get(i), toVersion).get();
      assertThat(migrated).isNotNull();

      // Check re-encoding was applied
      byte[] original = originalCodes.get(i);
      for (int j = 0; j < PQ_SUBVECTORS; j++) {
        assertThat(migrated[j]).isEqualTo((byte) (original[j] + 1));
      }
    }
  }

  @Test
  void testGetStats() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Store multiple codes
    int numCodes = 50;
    for (long nodeId = 0; nodeId < numCodes; nodeId++) {
      storage.storePqCode(nodeId, createRandomPqCode(), codebookVersion).get();
    }

    PqBlockStorage.StorageStats stats = storage.getStats(codebookVersion).get();

    assertThat(stats.blockCount()).isGreaterThan(0);
    assertThat(stats.totalCodes()).isGreaterThanOrEqualTo(numCodes);
    assertThat(stats.storageBytes()).isGreaterThan(0);
  }

  @Test
  void testGetStatsEmptyVersion() throws ExecutionException, InterruptedException {
    PqBlockStorage.StorageStats stats = storage.getStats(999).get();

    assertThat(stats.blockCount()).isEqualTo(0);
    assertThat(stats.totalCodes()).isEqualTo(0);
    assertThat(stats.storageBytes()).isEqualTo(0);
  }

  @Test
  void testCaching() throws ExecutionException, InterruptedException {
    long nodeId = 50L;
    byte[] pqCode = createRandomPqCode();
    int codebookVersion = 1;

    // Store code
    storage.storePqCode(nodeId, pqCode, codebookVersion).get();

    // First load - from DB
    byte[] loaded1 = storage.loadPqCode(nodeId, codebookVersion).get();

    // Second load - should be from cache (fast)
    byte[] loaded2 = storage.loadPqCode(nodeId, codebookVersion).get();

    assertThat(loaded1).isEqualTo(loaded2);
    assertThat(loaded2).isEqualTo(pqCode);

    // Clear cache
    storage.clearCache();

    // Load again - from DB
    byte[] loaded3 = storage.loadPqCode(nodeId, codebookVersion).get();
    assertThat(loaded3).isEqualTo(pqCode);
  }

  @Test
  void testInvalidPqCodeLength() {
    long nodeId = 1L;
    byte[] invalidCode = new byte[PQ_SUBVECTORS + 1]; // Wrong length
    int codebookVersion = 1;

    assertThatThrownBy(() -> storage.storePqCode(nodeId, invalidCode, codebookVersion))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessage("PQ code length mismatch: expected " + PQ_SUBVECTORS + ", got " + (PQ_SUBVECTORS + 1));
  }

  @Test
  void testBatchStoreCountMismatch() {
    List<Long> nodeIds = Arrays.asList(1L, 2L, 3L);
    List<byte[]> pqCodes = Arrays.asList(createRandomPqCode(), createRandomPqCode());

    assertThatThrownBy(() -> storage.batchStorePqCodes(nodeIds, pqCodes, 1).get())
        .hasCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Node IDs and PQ codes count mismatch");
  }

  @Test
  void testUpdateExistingCode() throws ExecutionException, InterruptedException {
    long nodeId = 75L;
    byte[] originalCode = createRandomPqCode();
    byte[] updatedCode = createRandomPqCode();
    int codebookVersion = 1;

    // Store original
    storage.storePqCode(nodeId, originalCode, codebookVersion).get();
    assertThat(storage.loadPqCode(nodeId, codebookVersion).get()).isEqualTo(originalCode);

    // Update with new code
    storage.storePqCode(nodeId, updatedCode, codebookVersion).get();
    assertThat(storage.loadPqCode(nodeId, codebookVersion).get()).isEqualTo(updatedCode);
  }

  @Test
  void testLargeBlockNumber() throws ExecutionException, InterruptedException {
    // Test with a very large node ID (large block number)
    long largeNodeId = 1_000_000L;
    byte[] pqCode = createRandomPqCode();
    int codebookVersion = 1;

    storage.storePqCode(largeNodeId, pqCode, codebookVersion).get();
    byte[] loaded = storage.loadPqCode(largeNodeId, codebookVersion).get();

    assertThat(loaded).isEqualTo(pqCode);
  }

  @Test
  void testBatchOperationsAcrossBlocks() throws ExecutionException, InterruptedException {
    int codebookVersion = 1;

    // Create node IDs that span multiple blocks
    List<Long> nodeIds = new ArrayList<>();
    List<byte[]> pqCodes = new ArrayList<>();

    // Block 0
    nodeIds.add(10L);
    nodeIds.add(50L);
    // Block 1
    nodeIds.add(260L);
    nodeIds.add(300L);
    // Block 2
    nodeIds.add(520L);

    for (int i = 0; i < nodeIds.size(); i++) {
      pqCodes.add(createRandomPqCode());
    }

    // Batch store across blocks
    storage.batchStorePqCodes(nodeIds, pqCodes, codebookVersion).get();

    // Batch load across blocks
    List<byte[]> loaded = storage.batchLoadPqCodes(nodeIds, codebookVersion).get();

    assertThat(loaded).hasSize(nodeIds.size());
    for (int i = 0; i < nodeIds.size(); i++) {
      assertThat(loaded.get(i)).isEqualTo(pqCodes.get(i));
    }
  }

  @Test
  void testDeleteNonExistentVersion() throws ExecutionException, InterruptedException {
    // Should not throw
    storage.deleteVersion(999).get();
  }

  @Test
  void testMigrateEmptyVersion() throws ExecutionException, InterruptedException {
    PqBlockStorage.CodeReencoder reencoder = (nodeId, oldCode) -> oldCode;

    // Should not throw
    storage.migrateVersion(999, 1000, reencoder).get();
  }

  @Test
  void testClearEmptyCache() {
    // Should not throw
    storage.clearCache();
  }

  // Helper method to create random PQ code
  private byte[] createRandomPqCode() {
    Random random = new Random();
    byte[] code = new byte[PQ_SUBVECTORS];
    random.nextBytes(code);
    return code;
  }
}
