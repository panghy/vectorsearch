package io.github.panghy.vectorsearch.graph;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

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
}
