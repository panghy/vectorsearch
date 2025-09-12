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

  /**
   * Builds neighbors with simple Vamana-style pruning.
   *
   * <p>Algorithm:
   * 1) For each node i, compute distances to all j != i and take the top L_build by distance.
   * 2) Greedily add candidates in order, pruning a candidate u if there exists a kept neighbor p
   *    such that dist(u, p) <= alpha * dist(u, i). Set alpha <= 1 to disable pruning.
   */
  public static int[][] buildPrunedNeighbors(float[][] vectors, int degree, int lBuild, double alpha) {
    int n = vectors.length;
    int[][] neigh = new int[n][];
    boolean prune = alpha > 1.0;
    for (int i = 0; i < n; i++) {
      Integer[] idx = new Integer[n - 1];
      int p = 0;
      for (int j = 0; j < n; j++) if (j != i) idx[p++] = j;
      final int ii = i;
      Arrays.sort(idx, Comparator.comparingDouble(j -> l2(vectors[ii], vectors[j])));
      int limit = Math.max(0, Math.min(lBuild, n - 1));
      int[] selected = new int[Math.min(degree, limit)];
      int s = 0;
      outer:
      for (int k = 0; k < limit && s < selected.length; k++) {
        int u = idx[k];
        if (!prune) {
          selected[s++] = u;
          continue;
        }
        double diu = l2(vectors[ii], vectors[u]);
        for (int t = 0; t < s; t++) {
          int pnb = selected[t];
          double dup = l2(vectors[u], vectors[pnb]);
          if (dup <= alpha * diu) {
            continue outer; // pruned
          }
        }
        selected[s++] = u;
      }
      if (s < selected.length) {
        int[] shrink = new int[s];
        System.arraycopy(selected, 0, shrink, 0, s);
        neigh[i] = shrink;
      } else {
        neigh[i] = selected;
      }
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
