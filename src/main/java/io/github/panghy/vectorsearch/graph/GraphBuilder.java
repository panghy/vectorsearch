package io.github.panghy.vectorsearch.graph;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Builds a simple k-nearest neighbor adjacency per vector using L2 distance.
 *
 * <p>This is an O(n^2 * d) baseline used for small segments and testing. It selects the closest
 * {@code degree} neighbors for each vector independently without symmetric pruning.</p>
 */
public final class GraphBuilder {
  private GraphBuilder() {}

  /**
   * Computes per-vector neighbor lists.
   *
   * @param vectors array of vectors [n][d]
   * @param degree  desired out-degree (neighbors per vector)
   * @return neighbors[n][] where each entry contains up to {@code degree} distinct indices (not including self)
   */
  public static int[][] buildL2Neighbors(float[][] vectors, int degree) {
    int n = vectors.length;
    int[][] neigh = new int[n][];
    for (int i = 0; i < n; i++) {
      // compute distances to all j != i
      Integer[] idx = new Integer[n - 1];
      int p = 0;
      for (int j = 0; j < n; j++) if (j != i) idx[p++] = j;
      final int ii = i;
      Arrays.sort(idx, Comparator.comparingDouble(j -> l2(vectors[ii], vectors[j])));
      int take = Math.min(degree, n - 1);
      neigh[i] = new int[take];
      for (int k = 0; k < take; k++) neigh[i][k] = idx[k];
    }
    return neigh;
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
