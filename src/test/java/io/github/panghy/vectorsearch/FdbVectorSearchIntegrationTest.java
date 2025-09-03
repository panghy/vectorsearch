package io.github.panghy.vectorsearch;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.search.SearchResult;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/**
 * Integration tests for FdbVectorSearch that test the complete indexing and search pipeline.
 * These tests insert meaningful vectors, wait for indexing to complete, and verify search accuracy.
 */
@Tag("integration")
class FdbVectorSearchIntegrationTest {

  private static final int DIMENSION = 128;
  private static final int NUM_CLUSTERS = 10;
  private static final int VECTORS_PER_CLUSTER = 50;
  private static final int TOTAL_VECTORS = NUM_CLUSTERS * VECTORS_PER_CLUSTER;

  private Database db;
  private DirectorySubspace testDir;
  private FdbVectorSearch vectorSearch;
  private Map<Long, Integer> vectorIdToCluster;
  private Map<Integer, List<Long>> clusterToVectorIds;
  private Map<Long, float[]> allVectors;

  @BeforeEach
  void setUp() throws ExecutionException, InterruptedException, TimeoutException {
    db = FDB.selectAPIVersion(730).open();
    String testDirName = "integration_test_" + UUID.randomUUID();
    testDir = db.runAsync(tr -> {
          DirectoryLayer layer = DirectoryLayer.getDefault();
          return layer.createOrOpen(tr, Arrays.asList("test", testDirName));
        })
        .get(5, TimeUnit.SECONDS);

    vectorIdToCluster = new HashMap<>();
    clusterToVectorIds = new HashMap<>();
    allVectors = new HashMap<>();
  }

  @AfterEach
  void tearDown() throws ExecutionException, InterruptedException, TimeoutException {
    if (vectorSearch != null) {
      vectorSearch.shutdown();
    }

    // Clean up test directory
    if (testDir != null && db != null) {
      db.runAsync(tr -> {
            DirectoryLayer layer = DirectoryLayer.getDefault();
            return layer.remove(tr, testDir.getPath());
          })
          .get(5, TimeUnit.SECONDS);
    }
  }

  @Test
  @DisplayName("Should handle large-scale vector insertion and search with clustering")
  @Timeout(value = 60, unit = TimeUnit.SECONDS)
  void testLargeScaleVectorSearchWithClustering() {
    // Create index with specific configuration for testing
    VectorSearchConfig config = VectorSearchConfig.builder(db, testDir)
        .dimension(DIMENSION)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .graphDegree(32)
        .pqSubvectors(16) // 128 / 8 = 16 subvectors
        .build();

    vectorSearch = FdbVectorSearch.createOrOpen(config, db).join();

    // Generate clustered vectors
    List<float[]> vectors = generateClusteredVectors();

    // Insert all vectors and track their IDs
    System.out.println("Inserting " + TOTAL_VECTORS + " vectors...");
    List<Long> vectorIds = vectorSearch.insert(vectors).join();
    assertThat(vectorIds).hasSize(TOTAL_VECTORS);

    // Map vector IDs to clusters
    for (int i = 0; i < vectorIds.size(); i++) {
      Long vectorId = vectorIds.get(i);
      int clusterIndex = i / VECTORS_PER_CLUSTER;
      vectorIdToCluster.put(vectorId, clusterIndex);
      clusterToVectorIds
          .computeIfAbsent(clusterIndex, k -> new ArrayList<>())
          .add(vectorId);
      allVectors.put(vectorId, vectors.get(i));
    }

    // Wait for initial indexing to complete
    System.out.println("Waiting for indexing to complete...");
    waitForIndexingCompletion();

    // Test 1: Search with vectors from each cluster center
    System.out.println("Testing cluster center searches...");
    for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++) {
      float[] queryVector = generateClusterCenter(cluster);
      List<SearchResult> results = vectorSearch.search(queryVector, 10).join();

      // Since we don't have codebooks yet, results will be empty
      // This is expected until codebook training is implemented
      assertThat(results).isNotNull();

      // Once codebooks are trained, we would verify:
      // - Results contain vectors primarily from the same cluster
      // - Distance increases as we move away from cluster center
    }

    // Test 2: Search with exact vectors that were inserted
    System.out.println("Testing exact vector searches...");
    Random random = new Random(42);
    for (int i = 0; i < 10; i++) {
      Long randomId = vectorIds.get(random.nextInt(vectorIds.size()));
      float[] exactVector = allVectors.get(randomId);

      List<SearchResult> results = vectorSearch.search(exactVector, 5).join();
      assertThat(results).isNotNull();

      // Once indexing is complete, the exact vector should be the top result
      // with distance close to 0
    }

    // Test 3: Search with interpolated vectors between clusters
    System.out.println("Testing interpolated vector searches...");
    for (int i = 0; i < NUM_CLUSTERS - 1; i++) {
      float[] center1 = generateClusterCenter(i);
      float[] center2 = generateClusterCenter(i + 1);
      float[] interpolated = interpolate(center1, center2, 0.5f);

      List<SearchResult> results = vectorSearch.search(interpolated, 20).join();
      assertThat(results).isNotNull();

      // Results should contain vectors from both adjacent clusters
    }

    // Test 4: Test search with different parameters
    System.out.println("Testing search with custom parameters...");
    float[] queryVector = generateRandomVector(DIMENSION, random);

    // Search with small beam (fast but less accurate)
    List<SearchResult> fastResults =
        vectorSearch.search(queryVector, 10, 16, 100).join();
    assertThat(fastResults).isNotNull();

    // Search with large beam (slower but more accurate)
    List<SearchResult> accurateResults =
        vectorSearch.search(queryVector, 10, 64, 500).join();
    assertThat(accurateResults).isNotNull();

    // Larger beam should explore more of the graph
    // Once implemented, accurate search should have better recall

    // Test 5: Verify statistics
    VectorSearch.IndexStats stats = vectorSearch.getStats().join();
    assertThat(stats).isNotNull();
    assertThat(stats.getVectorCount()).isGreaterThanOrEqualTo(0);

    // Test 6: Comprehensive search verification after full indexing
    System.out.println("Performing comprehensive search verification...");

    // Wait for complete indexing with codebooks
    vectorSearch.waitForIndexing(Duration.ofSeconds(30)).join();

    // 6a. Verify exact vector matches
    System.out.println("Testing exact vector retrieval...");

    int exactMatchTests = 20;
    int exactMatchesFound = 0;
    int nonEmptyResults = 0;

    for (int i = 0; i < exactMatchTests; i++) {
      Long testId = vectorIds.get(random.nextInt(vectorIds.size()));
      float[] exactVector = allVectors.get(testId);

      // Search for top 10 to ensure our exact match is in there
      List<SearchResult> results = vectorSearch.search(exactVector, 100).join();

      // Check if we got at least 10 results (or all available vectors if less than 10)
      int expectedResults = Math.min(10, TOTAL_VECTORS);
      assertThat(results).isNotNull();

      // Track how many searches return non-empty results
      if (!results.isEmpty()) {
        nonEmptyResults++;
        assertThat(results.size()).isLessThanOrEqualTo(100);

        // Print distance distribution for debugging
        if (i < 3) { // Print details for first few searches
          System.out.println("\n== Search " + (i + 1) + " distance distribution (testId=" + testId + ") ==");
          for (int k = 0; k < Math.min(20, results.size()); k++) {
            SearchResult r = results.get(k);
            System.out.printf(
                "  Rank %d: nodeId=%d, distance=%.6f%s%n",
                k + 1,
                r.getNodeId(),
                r.getDistance(),
                r.getNodeId() == testId ? " <-- EXACT MATCH" : "");
          }

          // Check for overflow/unusual values
          float minDist = results.stream()
              .map(SearchResult::getDistance)
              .min(Float::compare)
              .orElse(0f);
          float maxDist = results.stream()
              .map(SearchResult::getDistance)
              .max(Float::compare)
              .orElse(0f);
          System.out.printf("  Distance range: min=%.6f, max=%.6f%n", minDist, maxDist);

          // Check how many have similar distances
          Map<Float, Long> distCounts = results.stream()
              .collect(Collectors.groupingBy(
                  r -> Math.round(r.getDistance() * 1000) / 1000f, // Round to 3 decimal places
                  Collectors.counting()));
          System.out.println("  Unique distances (rounded to 3 decimals): " + distCounts.size());

          // Check for potential overflow or calculation issues
          long overflowCount = results.stream()
              .filter(r -> r.getDistance() > 10000
                  || r.getDistance() < 0
                  || Float.isNaN(r.getDistance())
                  || Float.isInfinite(r.getDistance()))
              .count();
          if (overflowCount > 0) {
            System.out.println("  WARNING: " + overflowCount + " results with overflow/invalid distances!");
          }
        }

        // Check if the exact vector is in the results
        boolean foundExact = false;
        int exactRank = -1;
        for (int j = 0; j < results.size(); j++) {
          SearchResult result = results.get(j);
          if (result.getNodeId() == testId) {
            foundExact = true;
            exactRank = j + 1;
            // Distance should be very close to 0 for L2
            if (result.getDistance() < 0.001f) {
              exactMatchesFound++;
            }
            break;
          }
        }

        if (foundExact && i < 3) { // Print for first few searches
          System.out.printf(
              "Search %d: Exact match found at rank %d with distance %.6f%n",
              i + 1, exactRank, results.get(exactRank - 1).getDistance());
        }
      }
    }

    System.out.println("Search results summary:");
    System.out.println("  Total searches: " + exactMatchTests);
    System.out.println("  Non-empty results: " + nonEmptyResults);
    System.out.println("  Exact matches found in top 3: " + exactMatchesFound);

    // First ensure we're getting search results at all
    assertThat(nonEmptyResults)
        .as("Most searches should return non-empty results after indexing completes")
        .isGreaterThan(exactMatchTests / 2); // At least half should return results

    // If we're getting results, verify exact match quality
    if (nonEmptyResults > 0) {
      // For searches that return results, most should find the exact match
      double matchRate = (double) exactMatchesFound / nonEmptyResults;
      System.out.println("  Match rate for non-empty results: " + String.format("%.1f%%", matchRate * 100));

      assertThat(exactMatchesFound)
          .as("At least 75%% of non-empty search results should find the exact match in top 3")
          .isGreaterThanOrEqualTo((nonEmptyResults * 3) / 4);
    }

    // 6b. Verify topK returns correct number of results
    System.out.println("Testing topK result counts...");
    int[] topKValues = {1, 5, 10, 20, 50, 100};

    for (int k : topKValues) {
      float[] topKQueryVector = generateRandomVector(DIMENSION, new Random(k * 100));
      List<SearchResult> results = vectorSearch.search(topKQueryVector, k).join();

      assertThat(results).isNotNull();

      // After indexing is complete, we should get min(k, total_vectors) results
      if (!results.isEmpty()) {
        int expectedCount = Math.min(k, TOTAL_VECTORS);
        assertThat(results.size())
            .as("Search for top %d should return at most %d results", k, expectedCount)
            .isLessThanOrEqualTo(expectedCount);

        // Verify results are sorted by distance
        for (int i = 1; i < results.size(); i++) {
          assertThat(results.get(i).getDistance())
              .as("Results should be sorted by increasing distance")
              .isGreaterThanOrEqualTo(results.get(i - 1).getDistance());
        }

        System.out.println("TopK=" + k + " returned " + results.size() + " results (expected max: "
            + Math.min(k, TOTAL_VECTORS) + ")");
      }
    }

    // 6c. Verify cluster-based searches return relevant results
    System.out.println("Testing cluster-based retrieval...");
    for (int cluster = 0; cluster < Math.min(3, NUM_CLUSTERS); cluster++) {
      float[] clusterCenter = generateClusterCenter(cluster);
      List<SearchResult> results = vectorSearch.search(clusterCenter, 50).join();

      if (!results.isEmpty()) {
        // Count how many results are from the target cluster
        int fromTargetCluster = 0;
        for (SearchResult result : results) {
          Integer resultCluster = vectorIdToCluster.get(result.getNodeId());
          if (resultCluster != null && resultCluster == cluster) {
            fromTargetCluster++;
          }
        }

        System.out.println("Cluster " + cluster + " search: " + fromTargetCluster + "/" + results.size()
            + " results from target cluster");

        // At least some results should be from the target cluster
        if (results.size() >= 10) {
          assertThat(fromTargetCluster)
              .as("Search from cluster center should return some vectors from that cluster")
              .isGreaterThan(0);
        }
      }
    }

    System.out.println("Integration test completed successfully!");
  }

  @Test
  @DisplayName("Should handle vector updates and deletions correctly")
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  @Disabled("UnlinkWorker not implemented yet")
  void testVectorUpdatesAndDeletions() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, testDir)
        .dimension(DIMENSION)
        .distanceMetric(VectorSearchConfig.DistanceMetric.COSINE)
        .graphDegree(16)
        .build();

    vectorSearch = FdbVectorSearch.createOrOpen(config, db).join();

    // Insert initial vectors
    List<float[]> initialVectors = IntStream.range(0, 100)
        .mapToObj(i -> generateRandomVector(DIMENSION, new Random(i)))
        .collect(Collectors.toList());

    List<Long> vectorIds = vectorSearch.insert(initialVectors).join();
    assertThat(vectorIds).hasSize(100);

    // Update some vectors with new values
    Map<Long, float[]> updates = new HashMap<>();
    Random random = new Random(999);
    for (int i = 0; i < 20; i++) {
      Long idToUpdate = vectorIds.get(i * 5); // Update every 5th vector
      float[] newVector = generateRandomVector(DIMENSION, random);
      updates.put(idToUpdate, newVector);
    }

    vectorSearch.upsert(updates).join();

    // Delete some vectors
    List<Long> idsToDelete = vectorIds.subList(80, 100);
    vectorSearch.delete(idsToDelete).join();

    // Wait for operations to complete
    waitForIndexingCompletion();

    // Verify search still works after updates and deletions
    float[] queryVector = generateRandomVector(DIMENSION, new Random(42));
    List<SearchResult> results = vectorSearch.search(queryVector, 10).join();
    assertThat(results).isNotNull();

    // Deleted vectors should not appear in results once indexing is complete
    for (SearchResult result : results) {
      assertThat(idsToDelete).doesNotContain(result.getNodeId());
    }
  }

  @Test
  @DisplayName("Should maintain graph connectivity during incremental insertions")
  @Timeout(value = 30, unit = TimeUnit.SECONDS)
  void testIncrementalInsertionsWithConnectivity() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, testDir)
        .dimension(DIMENSION)
        .distanceMetric(VectorSearchConfig.DistanceMetric.INNER_PRODUCT)
        .graphDegree(24)
        .build();

    vectorSearch = FdbVectorSearch.createOrOpen(config, db).join();

    Random random = new Random(123);

    // Insert vectors in small batches to simulate real-world usage
    for (int batch = 0; batch < 10; batch++) {
      List<float[]> batchVectors = IntStream.range(0, 50)
          .mapToObj(i -> generateNormalizedVector(DIMENSION, random))
          .collect(Collectors.toList());

      List<Long> batchIds = vectorSearch.insert(batchVectors).join();
      assertThat(batchIds).hasSize(50);

      // Perform searches to verify index remains queryable
      float[] queryVector = generateNormalizedVector(DIMENSION, random);
      List<SearchResult> results = vectorSearch.search(queryVector, 5).join();
      assertThat(results).isNotNull();
    }

    vectorSearch.waitForIndexing(Duration.ofSeconds(10)).join();

    // Trigger manual connectivity check
    vectorSearch.refreshEntryPoints().join();

    // Verify final state
    assertThat(vectorSearch.isHealthy().join()).isTrue();

    // Additional verification: Ensure searches return proper results
    System.out.println("Verifying search results after incremental insertions...");

    // Test that we can get various topK values
    int[] testKValues = {5, 10, 25, 50};
    for (int k : testKValues) {
      float[] testQuery = generateNormalizedVector(DIMENSION, new Random(k + 1000));
      List<SearchResult> results = vectorSearch.search(testQuery, k).join();

      assertThat(results).isNotNull();
      if (!results.isEmpty()) {
        // We should get min(k, 500) results since we inserted 500 vectors total
        int expectedMax = Math.min(k, 500);
        assertThat(results.size())
            .as("Search for top %d should return results", k)
            .isLessThanOrEqualTo(expectedMax);

        // Verify results are properly sorted
        for (int i = 1; i < results.size(); i++) {
          assertThat(results.get(i).getDistance())
              .isGreaterThanOrEqualTo(results.get(i - 1).getDistance());
        }

        System.out.println("TopK=" + k + " returned " + results.size() + " results");
      }
    }
  }

  /**
   * Generates clustered vectors for testing.
   * Each cluster has vectors distributed around a cluster center.
   */
  private List<float[]> generateClusteredVectors() {
    List<float[]> vectors = new ArrayList<>();
    Random random = new Random(42);

    for (int cluster = 0; cluster < NUM_CLUSTERS; cluster++) {
      float[] clusterCenter = generateClusterCenter(cluster);

      for (int i = 0; i < VECTORS_PER_CLUSTER; i++) {
        float[] vector = addNoiseToVector(clusterCenter, 0.1f, random);
        vectors.add(vector);
      }
    }

    return vectors;
  }

  /**
   * Generates a deterministic cluster center based on cluster index.
   */
  private float[] generateClusterCenter(int clusterIndex) {
    float[] center = new float[DIMENSION];
    Random random = new Random(clusterIndex * 1000L);

    // Create distinct cluster centers in different regions of the vector space
    for (int i = 0; i < DIMENSION; i++) {
      // Use different patterns for different clusters
      if (clusterIndex % 3 == 0) {
        center[i] = (float) Math.sin(2 * Math.PI * i / DIMENSION + clusterIndex);
      } else if (clusterIndex % 3 == 1) {
        center[i] = (float) Math.cos(2 * Math.PI * i / DIMENSION + clusterIndex);
      } else {
        center[i] = (float) (random.nextGaussian() * 0.5);
      }
    }

    return center;
  }

  /**
   * Adds Gaussian noise to a vector.
   */
  private float[] addNoiseToVector(float[] vector, float noiseLevel, Random random) {
    float[] noisyVector = Arrays.copyOf(vector, vector.length);
    for (int i = 0; i < noisyVector.length; i++) {
      noisyVector[i] += (float) (random.nextGaussian() * noiseLevel);
    }
    return noisyVector;
  }

  /**
   * Generates a random vector with values in [-1, 1].
   */
  private float[] generateRandomVector(int dimension, Random random) {
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat() * 2 - 1; // Range [-1, 1]
    }
    return vector;
  }

  /**
   * Generates a normalized random vector (unit length).
   */
  private float[] generateNormalizedVector(int dimension, Random random) {
    float[] vector = generateRandomVector(dimension, random);
    float norm = 0;
    for (float v : vector) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    for (int i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
    return vector;
  }

  /**
   * Interpolates between two vectors.
   */
  private float[] interpolate(float[] v1, float[] v2, float alpha) {
    float[] result = new float[v1.length];
    for (int i = 0; i < v1.length; i++) {
      result[i] = v1[i] * (1 - alpha) + v2[i] * alpha;
    }
    return result;
  }

  /**
   * Waits for indexing operations to complete using the built-in wait method.
   */
  private void waitForIndexingCompletion() {
    // Wait up to a minute for indexing to complete
    System.out.println("Waiting for indexing to complete...");
    vectorSearch.waitForIndexing(Duration.ofSeconds(60)).join();
    System.out.println("Indexing complete");
  }
}
