package io.github.panghy.vectorsearch.graph;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

/**
 * Tests for graph pruning behavior in both brute-force and Vamana graph builders.
 */
class GraphBuilderPruningTest {
  @Test
  void prunes_when_alpha_small_and_breadth_large() {
    // Four points on a line: 0,1,2,3
    float[][] v = new float[][] {{0f}, {1f}, {2f}, {3f}};
    int degree = 2;
    int lbuild = 3;
    double alpha = 1.1; // enable pruning
    int[][] neigh = GraphBuilder.buildPrunedNeighbors(v, degree, lbuild, alpha);
    // For point 0: nearest are 1 and 2; with pruning vs 0, point 2 should be pruned due to proximity to 1
    assertThat(neigh[0].length).isEqualTo(1);
    assertThat(neigh[0][0]).isEqualTo(1);
  }

  @Test
  void disables_pruning_when_alpha_leq_one() {
    float[][] v = new float[][] {{0f}, {1f}, {2f}};
    int[][] neigh = GraphBuilder.buildPrunedNeighbors(v, 2, 2, 1.0);
    // Without pruning, should take top-2 nearest by distance
    assertThat(neigh[0].length).isEqualTo(2);
    assertThat(neigh[0][0]).isEqualTo(1);
    assertThat(neigh[0][1]).isEqualTo(2);
  }

  @Test
  void vamana_prunes_collinear_points() {
    // Four points on a line: 0,1,2,3 — with alpha=1.2, pruning should remove redundant edges
    float[][] v = new float[][] {{0f}, {1f}, {2f}, {3f}};
    int[][] neigh = GraphBuilder.buildVamanaGraph(v, 2, 4, 1.2);
    // Each node should have at most 2 neighbors
    for (int i = 0; i < 4; i++) {
      assertThat(neigh[i].length).isLessThanOrEqualTo(2);
      // No self-loops
      for (int nb : neigh[i]) {
        assertThat(nb).isNotEqualTo(i);
      }
    }
    // Node 0 should connect to node 1 (nearest neighbor)
    assertThat(neigh[0]).contains(1);
    // Node 3 should connect to node 2 (nearest neighbor)
    assertThat(neigh[3]).contains(2);
  }

  @Test
  void vamana_produces_connected_graph() {
    // 8 points in 2D — the graph should be connected (every node reachable from medoid)
    float[][] v = new float[][] {
      {0f, 0f}, {1f, 0f}, {0f, 1f}, {1f, 1f},
      {2f, 0f}, {2f, 1f}, {0.5f, 0.5f}, {1.5f, 0.5f}
    };
    int[][] neigh = GraphBuilder.buildVamanaGraph(v, 4, 8, 1.2);
    int medoid = GraphBuilder.findMedoid(v);

    // BFS from medoid to check connectivity
    boolean[] reached = new boolean[v.length];
    java.util.Queue<Integer> queue = new java.util.LinkedList<>();
    queue.add(medoid);
    reached[medoid] = true;
    while (!queue.isEmpty()) {
      int cur = queue.poll();
      for (int nb : neigh[cur]) {
        if (!reached[nb]) {
          reached[nb] = true;
          queue.add(nb);
        }
      }
    }
    for (int i = 0; i < v.length; i++) {
      assertThat(reached[i])
          .as("Node %d should be reachable from medoid", i)
          .isTrue();
    }
  }

  @Test
  void vamana_reverse_edges_present() {
    // With reverse edge updates, if u→v exists, v should also have some path back
    float[][] v = new float[][] {{0f, 0f}, {1f, 0f}, {2f, 0f}, {3f, 0f}, {4f, 0f}};
    int[][] neigh = GraphBuilder.buildVamanaGraph(v, 2, 4, 1.2);
    // Check that for at least some edges u→v, v→u also exists (reverse edges)
    int reverseCount = 0;
    int totalEdges = 0;
    for (int u = 0; u < v.length; u++) {
      for (int nb : neigh[u]) {
        totalEdges++;
        boolean hasReverse = false;
        for (int rnb : neigh[nb]) {
          if (rnb == u) {
            hasReverse = true;
            break;
          }
        }
        if (hasReverse) reverseCount++;
      }
    }
    // At least some edges should be bidirectional due to reverse edge updates
    assertThat(reverseCount).as("Some edges should be bidirectional").isGreaterThan(0);
  }
}
