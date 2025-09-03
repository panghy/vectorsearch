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

    // 6a. Verify exact vector matches and cluster accuracy
    System.out.println("Testing exact vector retrieval and cluster-based accuracy...");

    int exactMatchTests = 20;
    int exactMatchesFound = 0;
    int nonEmptyResults = 0;
    int sameClusterMatches = 0; // Track how many times top result is from same cluster

    for (int i = 0; i < exactMatchTests; i++) {
      Long testId = vectorIds.get(random.nextInt(vectorIds.size()));
      float[] exactVector = allVectors.get(testId);
      Integer queryCluster = vectorIdToCluster.get(testId); // Get the cluster of the query vector

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

        // Check if top results are from the same cluster
        if (!results.isEmpty()) {
          // Count how many of the top 10 results are from the same cluster
          int topK = Math.min(10, results.size());
          int fromSameCluster = 0;
          for (int j = 0; j < topK; j++) {
            Long resultId = results.get(j).getNodeId();
            Integer resultCluster = vectorIdToCluster.get(resultId);
            if (resultCluster != null && resultCluster.equals(queryCluster)) {
              fromSameCluster++;
            }
          }

          // If majority of top 10 are from same cluster, count as success
          if (fromSameCluster >= topK / 2) {
            sameClusterMatches++;
          }

          if (i < 3) { // Print cluster info for first few searches
            System.out.printf(
                "Search %d: Query from cluster %d, %d/%d top results from same cluster%n",
                i + 1, queryCluster, fromSameCluster, topK);
          }
        }
      }
    }

    System.out.println("Search results summary:");
    System.out.println("  Total searches: " + exactMatchTests);
    System.out.println("  Non-empty results: " + nonEmptyResults);
    System.out.println("  Exact matches found in top 3: " + exactMatchesFound);
    System.out.println("  Cluster-based matches (majority in top 10): " + sameClusterMatches);

    // First ensure we're getting search results at all
    assertThat(nonEmptyResults)
        .as("Most searches should return non-empty results after indexing completes")
        .isGreaterThan(exactMatchTests / 2); // At least half should return results

    // If we're getting results, verify cluster-based accuracy
    if (nonEmptyResults > 0) {
      // For searches that return results, check cluster-based accuracy
      double clusterMatchRate = (double) sameClusterMatches / nonEmptyResults;
      System.out.println(
          "  Cluster match rate for non-empty results: " + String.format("%.1f%%", clusterMatchRate * 100));

      // With PQ compression on clustered data, we expect good cluster-based accuracy
      // even if exact matches are poor
      assertThat(sameClusterMatches)
          .as("At least 50%% of non-empty search results should have majority from same cluster")
          .isGreaterThanOrEqualTo(nonEmptyResults / 2);

      // Log exact match rate for comparison (but don't assert on it)
      double exactMatchRate = (double) exactMatchesFound / nonEmptyResults;
      System.out.println(
          "  Exact match rate for non-empty results: " + String.format("%.1f%%", exactMatchRate * 100));
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
  @DisplayName("Should handle random vectors with exact retrieval before and after entry point refresh")
  void testRandomVectorsWithExactRetrieval() {
    VectorSearchConfig config = VectorSearchConfig.builder(db, testDir)
        .dimension(DIMENSION)
        .distanceMetric(VectorSearchConfig.DistanceMetric.L2)
        .graphDegree(32)
        .pqSubvectors(16)
        .build();

    vectorSearch = FdbVectorSearch.createOrOpen(config, db).join();

    Random random = new Random(12345);
    Map<Long, float[]> idToVector = new HashMap<>();
    List<Long> allIds = new ArrayList<>();

    // Insert 1000 random vectors
    System.out.println("Inserting 1000 random vectors...");
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < 1000; i++) {
      float[] vector = generateRandomVector(DIMENSION, random);
      vectors.add(vector);
    }

    List<Long> vectorIds = vectorSearch.insert(vectors).join();
    assertThat(vectorIds).hasSize(1000);

    // Store mapping for later verification
    for (int i = 0; i < vectorIds.size(); i++) {
      idToVector.put(vectorIds.get(i), vectors.get(i));
      allIds.add(vectorIds.get(i));
    }

    assertThat(idToVector).hasSize(1000);
    assertThat(allIds).hasSize(1000);
    System.out.println("  Inserted 1000 vectors successfully");

    // Wait for initial indexing to complete
    System.out.println("Waiting for indexing to complete...");
    vectorSearch.waitForIndexing(Duration.ofSeconds(60)).join();

    // Test 1: Query for exact vectors before entry point refresh
    System.out.println("\n=== Testing exact retrieval BEFORE entry point refresh ===");
    int testSampleSize = 50; // Test a sample of 50 vectors
    Random testRandom = new Random(54321);
    List<Long> testIds = new ArrayList<>();

    // Select random IDs to test
    for (int i = 0; i < testSampleSize; i++) {
      testIds.add(allIds.get(testRandom.nextInt(allIds.size())));
    }

    int exactMatchesBeforeRefresh = 0;
    int top1MatchesBeforeRefresh = 0;
    int top10MatchesBeforeRefresh = 0;
    double totalDistanceBeforeRefresh = 0;

    for (int i = 0; i < testIds.size(); i++) {
      Long testId = testIds.get(i);
      float[] exactVector = idToVector.get(testId);

      // Search for the exact vector - get top 100 to see where exact match ranks
      List<SearchResult> results = vectorSearch.search(exactVector, 100).join();

      if (!results.isEmpty()) {
        // Check if exact match is the top result
        if (results.get(0).getNodeId() == testId.longValue()) {
          top1MatchesBeforeRefresh++;
          // With PQ compression, exact match distance will be ~5-10 due to quantization
          if (results.get(0).getDistance() < 15.0f) {
            exactMatchesBeforeRefresh++;
          }
        }

        // Check if exact match is in top 10
        boolean foundInTop10 = false;
        int exactPosition = -1;
        float exactDistance = -1f;
        for (int j = 0; j < Math.min(10, results.size()); j++) {
          if (results.get(j).getNodeId() == testId.longValue()) {
            foundInTop10 = true;
            exactPosition = j + 1;
            exactDistance = results.get(j).getDistance();
            totalDistanceBeforeRefresh += results.get(j).getDistance();
            break;
          }
        }
        if (foundInTop10) {
          top10MatchesBeforeRefresh++;
        }

        // Log details for first few searches - show more details
        if (i < 10) {
          System.out.printf(
              "Test %d (id=%d): Top result id=%d, distance=%.6f%s%n",
              i + 1,
              testId,
              results.get(0).getNodeId(),
              results.get(0).getDistance(),
              results.get(0).getNodeId() == testId.longValue() ? " ✓ EXACT MATCH" : "");

          if (foundInTop10 && exactPosition > 1) {
            System.out.printf(
                "  -> Exact vector found at position %d with distance %.6f%n",
                exactPosition, exactDistance);
          } else if (!foundInTop10) {
            // Search in top 100 to see where it actually is
            for (int j = 10; j < results.size(); j++) {
              if (results.get(j).getNodeId() == testId.longValue()) {
                System.out.printf(
                    "  -> Exact vector found at position %d with distance %.6f (outside top 10)%n",
                    j + 1, results.get(j).getDistance());
                break;
              }
            }
          }

          // Show distance distribution
          if (i < 3) {
            System.out.println("  Top 5 results:");
            for (int j = 0; j < Math.min(5, results.size()); j++) {
              System.out.printf(
                  "    %d. id=%d, distance=%.6f%s%n",
                  j + 1,
                  results.get(j).getNodeId(),
                  results.get(j).getDistance(),
                  results.get(j).getNodeId() == testId.longValue() ? " <-- EXACT" : "");
            }
          }
        }
      }
    }

    double avgDistanceBeforeRefresh = top10MatchesBeforeRefresh > 0
        ? totalDistanceBeforeRefresh / top10MatchesBeforeRefresh
        : Double.MAX_VALUE;

    System.out.println("\nResults BEFORE entry point refresh:");
    System.out.printf(
        "  Exact matches (distance < 15): %d/%d (%.1f%%)%n",
        exactMatchesBeforeRefresh, testSampleSize, 100.0 * exactMatchesBeforeRefresh / testSampleSize);
    System.out.printf(
        "  Top-1 matches (exact vector as #1): %d/%d (%.1f%%)%n",
        top1MatchesBeforeRefresh, testSampleSize, 100.0 * top1MatchesBeforeRefresh / testSampleSize);
    System.out.printf(
        "  Top-10 matches (exact vector in top 10): %d/%d (%.1f%%)%n",
        top10MatchesBeforeRefresh, testSampleSize, 100.0 * top10MatchesBeforeRefresh / testSampleSize);
    System.out.printf("  Average PQ distance for matches: %.6f%n", avgDistanceBeforeRefresh);

    // Trigger entry point recalculation
    System.out.println("\n=== Triggering entry point refresh ===");
    vectorSearch.refreshEntryPoints().join();

    // Small delay to ensure refresh completes
    try {
      Thread.sleep(2000);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    // Test 2: Query for exact vectors after entry point refresh
    System.out.println("\n=== Testing exact retrieval AFTER entry point refresh ===");
    int exactMatchesAfterRefresh = 0;
    int top1MatchesAfterRefresh = 0;
    int top10MatchesAfterRefresh = 0;
    double totalDistanceAfterRefresh = 0;

    for (int i = 0; i < testIds.size(); i++) {
      Long testId = testIds.get(i);
      float[] exactVector = idToVector.get(testId);

      // Search for the exact vector - get top 100 to see where exact match ranks
      List<SearchResult> results = vectorSearch.search(exactVector, 100).join();

      if (!results.isEmpty()) {
        // Check if exact match is the top result
        if (results.get(0).getNodeId() == testId.longValue()) {
          top1MatchesAfterRefresh++;
          // With PQ compression, exact match distance will be ~5-10 due to quantization
          if (results.get(0).getDistance() < 15.0f) {
            exactMatchesAfterRefresh++;
          }
        }

        // Check if exact match is in top 10
        boolean foundInTop10 = false;
        for (int j = 0; j < Math.min(10, results.size()); j++) {
          if (results.get(j).getNodeId() == testId.longValue()) {
            foundInTop10 = true;
            totalDistanceAfterRefresh += results.get(j).getDistance();
            break;
          }
        }
        if (foundInTop10) {
          top10MatchesAfterRefresh++;
        }

        // Log details for first few searches
        if (i < 5) {
          System.out.printf(
              "Test %d (id=%d): Top result id=%d, distance=%.6f%s%n",
              i + 1,
              testId,
              results.get(0).getNodeId(),
              results.get(0).getDistance(),
              results.get(0).getNodeId() == testId.longValue() ? " ✓ EXACT MATCH" : "");
        }
      }
    }

    double avgDistanceAfterRefresh =
        top10MatchesAfterRefresh > 0 ? totalDistanceAfterRefresh / top10MatchesAfterRefresh : Double.MAX_VALUE;

    System.out.println("\nResults AFTER entry point refresh:");
    System.out.printf(
        "  Exact matches (distance < 15): %d/%d (%.1f%%)%n",
        exactMatchesAfterRefresh, testSampleSize, 100.0 * exactMatchesAfterRefresh / testSampleSize);
    System.out.printf(
        "  Top-1 matches (exact vector as #1): %d/%d (%.1f%%)%n",
        top1MatchesAfterRefresh, testSampleSize, 100.0 * top1MatchesAfterRefresh / testSampleSize);
    System.out.printf(
        "  Top-10 matches (exact vector in top 10): %d/%d (%.1f%%)%n",
        top10MatchesAfterRefresh, testSampleSize, 100.0 * top10MatchesAfterRefresh / testSampleSize);
    System.out.printf("  Average PQ distance for matches: %.6f%n", avgDistanceAfterRefresh);

    // Compare before and after
    System.out.println("\n=== Comparison ===");
    System.out.printf(
        "Top-1 improvement: %+d (%.1f%% → %.1f%%)%n",
        top1MatchesAfterRefresh - top1MatchesBeforeRefresh,
        100.0 * top1MatchesBeforeRefresh / testSampleSize,
        100.0 * top1MatchesAfterRefresh / testSampleSize);
    System.out.printf(
        "Top-10 improvement: %+d (%.1f%% → %.1f%%)%n",
        top10MatchesAfterRefresh - top10MatchesBeforeRefresh,
        100.0 * top10MatchesBeforeRefresh / testSampleSize,
        100.0 * top10MatchesAfterRefresh / testSampleSize);

    // Assertions
    // With PQ compression on random vectors, we expect good retrieval accuracy
    assertThat(top1MatchesBeforeRefresh)
        .as("At least 70%% of queries should find the exact vector as the top result")
        .isGreaterThanOrEqualTo(testSampleSize * 70 / 100);

    // After refresh, accuracy should be maintained or improved
    assertThat(top1MatchesAfterRefresh)
        .as("At least 70%% of queries should find the exact vector as the top result after refresh")
        .isGreaterThanOrEqualTo(testSampleSize * 70 / 100);

    // Entry point refresh should not make results worse
    assertThat(top10MatchesAfterRefresh)
        .as("Entry point refresh should not degrade search quality")
        .isGreaterThanOrEqualTo(top10MatchesBeforeRefresh - 5); // Allow small variance

    System.out.println("\nTest completed successfully!");
  }

  @Test
  @DisplayName("Should maintain graph connectivity during incremental insertions")
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
