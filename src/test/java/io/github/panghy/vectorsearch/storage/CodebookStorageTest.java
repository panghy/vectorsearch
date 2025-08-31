package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CodebookStorageTest {

  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private CodebookStorage storage;
  private String testCollectionName;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "codebook_storage", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace);
    storage = new CodebookStorage(db, keys);
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
  void testStoreAndLoadCodebooks() throws ExecutionException, InterruptedException {
    // Create test codebooks
    int numSubvectors = 4;
    int numCentroids = 256;
    int subDimension = 8;
    float[][][] codebooks = createRandomCodebooks(numSubvectors, numCentroids, subDimension);

    // Store codebooks
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(10000L, 0.05);

    storage.storeCodebooks(1, codebooks, stats).get();

    // Load codebooks
    float[][][] loaded = storage.loadCodebooks(1).get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.length).isEqualTo(numSubvectors);

    // Check values are approximately preserved (fp16 conversion)
    for (int i = 0; i < numSubvectors; i++) {
      assertThat(loaded[i].length).isEqualTo(numCentroids);
      for (int j = 0; j < numCentroids; j++) {
        assertThat(loaded[i][j].length).isEqualTo(subDimension);
        for (int k = 0; k < subDimension; k++) {
          assertThat(loaded[i][j][k]).isCloseTo(codebooks[i][j][k], within(0.01f));
        }
      }
    }
  }

  @Test
  void testLoadNonExistentVersion() throws ExecutionException, InterruptedException {
    float[][][] loaded = storage.loadCodebooks(999).get();
    assertThat(loaded).isNull();
  }

  @Test
  void testActiveVersion() throws ExecutionException, InterruptedException {
    // Initially no active version
    int version = storage.getActiveVersion().get();
    assertThat(version).isEqualTo(-1);

    // Set active version
    storage.setActiveVersion(5).get();

    // Get active version
    version = storage.getActiveVersion().get();
    assertThat(version).isEqualTo(5);

    // Update active version
    storage.setActiveVersion(7).get();
    version = storage.getActiveVersion().get();
    assertThat(version).isEqualTo(7);
  }

  @Test
  void testListVersions() throws ExecutionException, InterruptedException {
    // Store multiple versions
    float[][][] codebooks = createRandomCodebooks(2, 256, 4);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(1000L, null);

    storage.storeCodebooks(1, codebooks, stats).get();
    storage.storeCodebooks(3, codebooks, stats).get();
    storage.storeCodebooks(5, codebooks, stats).get();

    // List versions
    List<Integer> versions = storage.listVersions().get();

    assertThat(versions).containsExactly(1, 3, 5);
  }

  @Test
  void testDeleteVersion() throws ExecutionException, InterruptedException {
    // Store codebooks
    float[][][] codebooks = createRandomCodebooks(2, 256, 4);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(1000L, 0.1);

    storage.storeCodebooks(1, codebooks, stats).get();
    storage.storeCodebooks(2, codebooks, stats).get();

    // Verify both exist
    assertThat(storage.loadCodebooks(1).get()).isNotNull();
    assertThat(storage.loadCodebooks(2).get()).isNotNull();

    // Delete version 1
    storage.deleteVersion(1).get();

    // Verify version 1 is deleted, version 2 still exists
    assertThat(storage.loadCodebooks(1).get()).isNull();
    assertThat(storage.loadCodebooks(2).get()).isNotNull();

    // List should only show version 2
    List<Integer> versions = storage.listVersions().get();
    assertThat(versions).containsExactly(2);
  }

  @Test
  void testCaching() throws ExecutionException, InterruptedException {
    float[][][] codebooks = createRandomCodebooks(2, 256, 4);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(1000L, null);

    // Store and load to populate cache
    storage.storeCodebooks(1, codebooks, stats).get();
    float[][][] loaded1 = storage.loadCodebooks(1).get();

    // Load again - should come from cache (same object reference)
    float[][][] loaded2 = storage.loadCodebooks(1).get();
    assertThat(loaded2).isSameAs(loaded1);

    // Clear cache
    storage.clearCache();

    // Load again - should load from DB (different object)
    float[][][] loaded3 = storage.loadCodebooks(1).get();
    assertThat(loaded3).isNotSameAs(loaded1);

    // But values should be the same
    assertThat(loaded3.length).isEqualTo(loaded1.length);
  }

  @Test
  void testCacheStats() throws ExecutionException, InterruptedException {
    // Initially empty
    CodebookStorage.CacheStats stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(0);
    assertThat(stats.estimatedSizeBytes).isEqualTo(0L);

    // Store some codebooks
    float[][][] codebooks = createRandomCodebooks(4, 256, 8);
    storage.storeCodebooks(1, codebooks, new CodebookStorage.TrainingStats(1000L, null))
        .get();

    // Check cache stats
    stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(1);
    assertThat(stats.estimatedSizeBytes).isGreaterThan(0L);

    // Store another version
    storage.storeCodebooks(2, codebooks, new CodebookStorage.TrainingStats(2000L, null))
        .get();

    stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(2);
  }

  @Test
  void testTrainingStatsWithQuantizationError() throws ExecutionException, InterruptedException {
    float[][][] codebooks = createRandomCodebooks(2, 256, 4);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(5000L, 0.123);

    storage.storeCodebooks(1, codebooks, stats).get();

    // Verify stats were stored (can't directly read them without accessing proto)
    float[][][] loaded = storage.loadCodebooks(1).get();
    assertThat(loaded).isNotNull();
  }

  @Test
  void testMultipleSubspacesStorage() throws ExecutionException, InterruptedException {
    // Test with many subspaces
    int numSubvectors = 16;
    int numCentroids = 256;
    int subDimension = 8;
    float[][][] codebooks = createRandomCodebooks(numSubvectors, numCentroids, subDimension);

    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(50000L, 0.02);

    storage.storeCodebooks(1, codebooks, stats).get();

    float[][][] loaded = storage.loadCodebooks(1).get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.length).isEqualTo(numSubvectors);

    // Verify all subspaces were stored correctly
    for (int i = 0; i < numSubvectors; i++) {
      assertThat(loaded[i].length).isEqualTo(numCentroids);
      assertThat(loaded[i][0].length).isEqualTo(subDimension);
    }
  }

  @Test
  void testVersionOverwrite() throws ExecutionException, InterruptedException {
    // Store initial version
    float[][][] codebooks1 = createRandomCodebooks(2, 256, 4);
    storage.storeCodebooks(1, codebooks1, new CodebookStorage.TrainingStats(1000L, null))
        .get();

    // Overwrite with different codebooks
    float[][][] codebooks2 = createRandomCodebooks(3, 256, 6);
    storage.storeCodebooks(1, codebooks2, new CodebookStorage.TrainingStats(2000L, null))
        .get();

    // Load and verify it's the new version
    float[][][] loaded = storage.loadCodebooks(1).get();
    assertThat(loaded.length).isEqualTo(3);
    assertThat(loaded[0][0].length).isEqualTo(6);
  }

  @Test
  void testFp16ConversionAccuracy() throws ExecutionException, InterruptedException {
    // Create codebooks with specific values to test fp16 conversion
    float[][][] codebooks = new float[1][2][4];
    codebooks[0][0] = new float[] {1.0f, -0.5f, 0.0f, 2.5f};
    codebooks[0][1] = new float[] {-3.25f, 0.125f, 100.0f, -100.0f};

    storage.storeCodebooks(1, codebooks, new CodebookStorage.TrainingStats(100L, null))
        .get();

    float[][][] loaded = storage.loadCodebooks(1).get();

    // Check values are preserved within fp16 precision
    assertThat(loaded[0][0][0]).isCloseTo(1.0f, within(0.001f));
    assertThat(loaded[0][0][1]).isCloseTo(-0.5f, within(0.001f));
    assertThat(loaded[0][0][2]).isCloseTo(0.0f, within(0.001f));
    assertThat(loaded[0][0][3]).isCloseTo(2.5f, within(0.01f));

    assertThat(loaded[0][1][0]).isCloseTo(-3.25f, within(0.01f));
    assertThat(loaded[0][1][1]).isCloseTo(0.125f, within(0.001f));
    assertThat(loaded[0][1][2]).isCloseTo(100.0f, within(0.1f));
    assertThat(loaded[0][1][3]).isCloseTo(-100.0f, within(0.1f));
  }

  @Test
  void testDeleteNonExistentVersion() throws ExecutionException, InterruptedException {
    // Should not throw
    storage.deleteVersion(999).get();

    // Verify nothing was affected
    List<Integer> versions = storage.listVersions().get();
    assertThat(versions).isEmpty();
  }

  @Test
  void testClearEmptyCache() {
    // Should not throw
    storage.clearCache();

    CodebookStorage.CacheStats stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(0);
    assertThat(stats.estimatedSizeBytes).isEqualTo(0L);
  }

  // Helper method to create random codebooks
  private float[][][] createRandomCodebooks(int numSubvectors, int numCentroids, int subDimension) {
    Random random = new Random(42);
    float[][][] codebooks = new float[numSubvectors][numCentroids][subDimension];

    for (int i = 0; i < numSubvectors; i++) {
      for (int j = 0; j < numCentroids; j++) {
        for (int k = 0; k < subDimension; k++) {
          codebooks[i][j][k] = random.nextFloat() * 2 - 1; // Range [-1, 1]
        }
      }
    }

    return codebooks;
  }
}
