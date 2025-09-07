package io.github.panghy.vectorsearch.graph;

/**
 * Unit tests for GraphBuilder: basic L2 neighbor construction, ordering, and degree handling.
 */
import static org.assertj.core.api.Assertions.assertThat;

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
}
