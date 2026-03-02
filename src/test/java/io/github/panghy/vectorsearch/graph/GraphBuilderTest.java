package io.github.panghy.vectorsearch.graph;

/**
 * Unit tests for GraphBuilder: basic L2 neighbor construction, ordering, degree handling,
 * and Vamana incremental graph construction.
 */
import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

class GraphBuilderTest {
  @Test
  void builds_knn_neighbors() {
    float[][] v = new float[][] {{0f, 0f}, {1f, 0f}, {2f, 0f}};
    int[][] n = GraphBuilder.buildL2Neighbors(v, 1);
    assertThat(n.length).isEqualTo(3);
    // middle should link to either 0 or 2; ends link to middle
    assertThat(n[0]).containsExactly(1);
    assertThat(n[2]).containsExactly(1);
  }

  @Test
  void vamana_empty_and_single_vector() {
    assertThat(GraphBuilder.buildVamanaGraph(new float[0][], 4, 8, 1.2)).isEmpty();
    int[][] single = GraphBuilder.buildVamanaGraph(new float[][] {{1f, 2f}}, 4, 8, 1.2);
    assertThat(single.length).isEqualTo(1);
    assertThat(single[0]).isEmpty();
  }

  @Test
  void vamana_three_points_on_line() {
    // Three points on a line: each should connect to its nearest neighbor(s)
    float[][] v = new float[][] {{0f, 0f}, {1f, 0f}, {2f, 0f}};
    int[][] n = GraphBuilder.buildVamanaGraph(v, 2, 4, 1.2);
    assertThat(n.length).isEqualTo(3);
    // Each node should have at least 1 neighbor
    for (int[] neighbors : n) {
      assertThat(neighbors.length).isGreaterThanOrEqualTo(1);
    }
    // Node 0 should have node 1 as a neighbor (nearest)
    assertThat(n[0]).contains(1);
    // Node 2 should have node 1 as a neighbor (nearest)
    assertThat(n[2]).contains(1);
  }

  @Test
  void vamana_respects_degree_limit() {
    // 10 points, degree=3 — no node should exceed degree
    float[][] v = new float[10][4];
    Random rng = new Random(42);
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 4; j++) v[i][j] = rng.nextFloat();
    }
    int[][] n = GraphBuilder.buildVamanaGraph(v, 3, 8, 1.2);
    assertThat(n.length).isEqualTo(10);
    for (int i = 0; i < 10; i++) {
      assertThat(n[i].length).isLessThanOrEqualTo(3);
      // No self-loops
      for (int nb : n[i]) {
        assertThat(nb).isNotEqualTo(i);
      }
    }
  }

  @Test
  void vamana_medoid_selection() {
    // Centroid of {0,0}, {2,0}, {0,2}, {2,2} is {1,1}
    // All equidistant, so medoid should be one of them
    float[][] v = new float[][] {{0f, 0f}, {2f, 0f}, {0f, 2f}, {2f, 2f}};
    int medoid = GraphBuilder.findMedoid(v);
    assertThat(medoid).isBetween(0, 3);
  }

  @Test
  void vamana_recall_test_100_vectors() {
    // Generate 100 random 8-dim vectors
    int n = 100;
    int dim = 8;
    int degree = 16;
    int lBuild = 64;
    double alpha = 1.2;
    Random rng = new Random(12345);
    float[][] vectors = new float[n][dim];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < dim; j++) vectors[i][j] = rng.nextFloat();
    }

    int[][] vamanaGraph = GraphBuilder.buildVamanaGraph(vectors, degree, lBuild, alpha);

    // Compute brute-force top-10 for each query vector (using the dataset itself as queries)
    int k = 10;
    int numQueries = 20; // test 20 random queries
    int totalHits = 0;
    int totalExpected = 0;

    for (int q = 0; q < numQueries; q++) {
      int queryIdx = rng.nextInt(n);
      float[] query = vectors[queryIdx];

      // Brute-force top-k (excluding self)
      Set<Integer> bruteForceTopK = bruteForceTopK(vectors, query, k, queryIdx);

      // Graph search: BFS/greedy from medoid
      Set<Integer> graphResults = graphSearch(vectors, vamanaGraph, query, k, lBuild);
      graphResults.remove(queryIdx);

      // Count overlap
      for (int id : graphResults) {
        if (bruteForceTopK.contains(id)) totalHits++;
      }
      totalExpected += bruteForceTopK.size();
    }

    double recall = (double) totalHits / totalExpected;
    assertThat(recall).as("Vamana recall@%d should be >= 0.80", k).isGreaterThanOrEqualTo(0.80);
  }

  /** Brute-force top-k nearest neighbors (excluding excludeIdx). */
  private Set<Integer> bruteForceTopK(float[][] vectors, float[] query, int k, int excludeIdx) {
    int n = vectors.length;
    double[] dists = new double[n];
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) {
      indices[i] = i;
      dists[i] = l2(vectors[i], query);
    }
    Arrays.sort(indices, (a, b) -> Double.compare(dists[a], dists[b]));
    Set<Integer> result = new HashSet<>();
    for (int idx : indices) {
      if (idx == excludeIdx) continue;
      result.add(idx);
      if (result.size() >= k) break;
    }
    return result;
  }

  /** Simple greedy graph search for testing recall. */
  private Set<Integer> graphSearch(float[][] vectors, int[][] graph, float[] query, int k, int lSearch) {
    int n = vectors.length;
    int start = GraphBuilder.findMedoid(vectors);
    boolean[] visited = new boolean[n];
    // Use a sorted candidate list
    TreeMap<Double, Integer> candidates = new TreeMap<>();
    TreeMap<Double, Integer> results = new TreeMap<>();

    double startDist = l2(vectors[start], query);
    candidates.put(startDist + start * 1e-12, start); // tie-break with id
    visited[start] = true;

    while (!candidates.isEmpty()) {
      var entry = candidates.pollFirstEntry();
      int cur = entry.getValue();
      double curDist = l2(vectors[cur], query);
      results.put(curDist + cur * 1e-12, cur);
      if (results.size() > lSearch) {
        results.pollLastEntry();
      }

      // If current is worse than worst result and results full, stop
      if (results.size() >= lSearch) {
        double worstResult = l2(vectors[results.lastEntry().getValue()], query);
        if (curDist > worstResult) break;
      }

      for (int nb : graph[cur]) {
        if (!visited[nb]) {
          visited[nb] = true;
          double dist = l2(vectors[nb], query);
          candidates.put(dist + nb * 1e-12, nb);
        }
      }
    }

    Set<Integer> topK = new HashSet<>();
    for (var e : results.entrySet()) {
      topK.add(e.getValue());
      if (topK.size() >= k) break;
    }
    return topK;
  }

  private static double l2(float[] a, float[] b) {
    double s = 0.0;
    for (int i = 0; i < a.length; i++) {
      double d = (double) a[i] - b[i];
      s += d * d;
    }
    return s;
  }
}
