package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
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
    storage = new CodebookStorage(db, keys, 64, 4, DistanceMetrics.Metric.L2);
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

    // Try to set active version for non-existent codebook - should fail
    assertThatThrownBy(() -> storage.setActiveVersion(5).get())
        .isInstanceOf(ExecutionException.class)
        .hasCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebook version 5 does not exist");

    // Store codebooks for version 5
    float[][][] codebooks5 = createRandomCodebooks(4, 256, 16);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(100L, 0.1);
    storage.storeCodebooks(5, codebooks5, stats).get();

    // Now set active version should work
    storage.setActiveVersion(5).get();

    // Get active version
    version = storage.getActiveVersion().get();
    assertThat(version).isEqualTo(5);

    // Store codebooks for version 7
    float[][][] codebooks7 = createRandomCodebooks(4, 256, 16);
    storage.storeCodebooks(7, codebooks7, stats).get();

    // Update active version
    storage.setActiveVersion(7).get();
    version = storage.getActiveVersion().get();
    assertThat(version).isEqualTo(7);
  }

  @Test
  void testListVersions() throws ExecutionException, InterruptedException {
    // Store multiple versions
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);
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
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);
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
    float[][][] codebooks = createRandomCodebooks(4, 256, 16); // 4 subvectors, 16 dims each (64/4=16)
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(1000L, null);

    // Store codebooks
    storage.storeCodebooks(1, codebooks, stats).get();

    // Test ProductQuantizer caching (this is what's actually cached)
    var pq1 = storage.getProductQuantizer(1).get();
    var pq2 = storage.getProductQuantizer(1).get();

    // Should be the same cached object
    assertThat(pq2).isSameAs(pq1);

    // Clear cache
    storage.clearCache();

    // Get ProductQuantizer again - should be a different object
    var pq3 = storage.getProductQuantizer(1).get();
    assertThat(pq3).isNotSameAs(pq1);

    // But should have the same version
    assertThat(pq3.getCodebookVersion()).isEqualTo(pq1.getCodebookVersion());

    // Test that raw codebooks are not cached (always new arrays)
    float[][][] loaded1 = storage.loadCodebooks(1).get();
    float[][][] loaded2 = storage.loadCodebooks(1).get();
    assertThat(loaded2).isNotSameAs(loaded1); // Different objects

    // But values should be the same
    assertThat(loaded2.length).isEqualTo(loaded1.length);
  }

  @Test
  void testCacheStats() throws ExecutionException, InterruptedException {
    // Initially empty
    CodebookStorage.CacheStats stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(0);
    assertThat(stats.hitCount).isEqualTo(0L);
    assertThat(stats.missCount).isEqualTo(0L);

    // Store some codebooks
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);
    storage.storeCodebooks(1, codebooks, new CodebookStorage.TrainingStats(1000L, null))
        .get();

    // Get ProductQuantizer to populate cache
    storage.getProductQuantizer(1).get();

    // Check cache stats
    stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(1);
    assertThat(stats.hitCount).isGreaterThanOrEqualTo(0L);
    assertThat(stats.missCount).isGreaterThanOrEqualTo(0L);

    // Store another version
    storage.storeCodebooks(2, codebooks, new CodebookStorage.TrainingStats(2000L, null))
        .get();

    // Get ProductQuantizer for second version to populate cache
    storage.getProductQuantizer(2).get();

    stats = storage.getCacheStats();
    assertThat(stats.entries).isEqualTo(2);
  }

  @Test
  void testTrainingStatsWithQuantizationError() throws ExecutionException, InterruptedException {
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);
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
    float[][][] codebooks1 = createRandomCodebooks(4, 256, 16);
    storage.storeCodebooks(1, codebooks1, new CodebookStorage.TrainingStats(1000L, null))
        .get();

    // Delete version 1 first, then store new version
    storage.deleteVersion(1).get();

    // Store new codebooks
    float[][][] codebooks2 = createRandomCodebooks(4, 256, 16);
    storage.storeCodebooks(1, codebooks2, new CodebookStorage.TrainingStats(2000L, null))
        .get();

    // Load and verify it's the new version (should be able to load successfully)
    float[][][] loaded = storage.loadCodebooks(1).get();
    assertThat(loaded).isNotNull();
    assertThat(loaded.length).isEqualTo(4); // 4 subvectors as configured
    assertThat(loaded[0][0].length).isEqualTo(16); // 16 dimensions per subvector
  }

  @Test
  void testFp16ConversionAccuracy() throws ExecutionException, InterruptedException {
    // Create codebooks with specific values to test fp16 conversion (4 subvectors, 256 centroids, 16 dims each)
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);

    // Override some specific values to test fp16 conversion precision
    codebooks[0][0] = new float[] {
      1.0f, -0.5f, 0.0f, 2.5f, -3.25f, 0.125f, 100.0f, -100.0f, 0.25f, -1.75f, 0.5f, -0.25f, 42.0f, -42.0f, 3.14f,
      -3.14f
    };

    storage.storeCodebooks(1, codebooks, new CodebookStorage.TrainingStats(100L, null))
        .get();

    float[][][] loaded = storage.loadCodebooks(1).get();

    // Check specific values are preserved within fp16 precision
    assertThat(loaded[0][0][0]).isCloseTo(1.0f, within(0.001f));
    assertThat(loaded[0][0][1]).isCloseTo(-0.5f, within(0.001f));
    assertThat(loaded[0][0][2]).isCloseTo(0.0f, within(0.001f));
    assertThat(loaded[0][0][3]).isCloseTo(2.5f, within(0.01f));
    assertThat(loaded[0][0][4]).isCloseTo(-3.25f, within(0.01f));
    assertThat(loaded[0][0][5]).isCloseTo(0.125f, within(0.001f));
    assertThat(loaded[0][0][6]).isCloseTo(100.0f, within(0.1f));
    assertThat(loaded[0][0][7]).isCloseTo(-100.0f, within(0.1f));
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
    assertThat(stats.hitCount).isEqualTo(0L);
    assertThat(stats.missCount).isEqualTo(0L);
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

  @Test
  void testCodebookStorageWithMultipleVersions() throws Exception {
    CodebookStorage storage = new CodebookStorage(db, keys, 64, 4, DistanceMetrics.Metric.L2);

    // Train and store multiple versions
    float[][][] codebooks1 = createRandomCodebooks(4, 256, 16);
    float[][][] codebooks2 = createRandomCodebooks(4, 256, 16);

    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(100L, 0.1);

    // Store version 1
    storage.storeCodebooks(1, codebooks1, stats).get();

    // Store version 2
    storage.storeCodebooks(2, codebooks2, stats).get();

    // Try to store version 1 again - should fail
    assertThatThrownBy(() -> storage.storeCodebooks(1, codebooks1, stats).get())
        .hasCauseInstanceOf(IllegalStateException.class)
        .hasMessageContaining("already exists");

    // List versions
    List<Integer> versions = storage.listVersions().get();
    assertThat(versions).containsExactlyInAnyOrder(1, 2);

    // Delete version 1
    storage.deleteVersion(1).get();

    // List versions again
    versions = storage.listVersions().get();
    assertThat(versions).containsExactly(2);

    // Now we can store version 1 again
    storage.storeCodebooks(1, codebooks1, stats).get();

    // Test cache stats
    CodebookStorage.CacheStats cacheStats = storage.getCacheStats();
    assertThat(cacheStats).isNotNull();
    assertThat(cacheStats.entries).isGreaterThanOrEqualTo(0);

    // Clear cache
    storage.clearCache();
    cacheStats = storage.getCacheStats();
    assertThat(cacheStats.entries).isEqualTo(0);
  }

  @Test
  void testCodebookStorageActiveVersion() throws Exception {
    CodebookStorage storage = new CodebookStorage(db, keys, 64, 4, DistanceMetrics.Metric.L2);

    // Initially no active version
    Integer activeVersion = storage.getActiveVersion().get();
    assertThat(activeVersion).isEqualTo(-1);

    // Store a codebook
    float[][][] codebooks = createRandomCodebooks(4, 256, 16);
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(100L, null);
    storage.storeCodebooks(1, codebooks, stats).get();

    // Set active version
    storage.setActiveVersion(1).get();

    // Check active version
    activeVersion = storage.getActiveVersion().get();
    assertThat(activeVersion).isEqualTo(1);

    // Get latest ProductQuantizer
    ProductQuantizer pq = storage.getLatestProductQuantizer().get();
    assertThat(pq).isNotNull();
    assertThat(pq.getCodebookVersion()).isEqualTo(1);
  }
}
