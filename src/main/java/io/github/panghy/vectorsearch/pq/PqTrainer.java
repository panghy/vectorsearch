package io.github.panghy.vectorsearch.pq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Trains product quantization (PQ) codebooks by running k-means independently on each subspace.
 *
 * <p>Use small K and M for tests; production parameters should be tuned.</p>
 */
public final class PqTrainer {
  private PqTrainer() {}

  /**
   * Trains centroids for each of M subspaces.
   *
   * @param vectors   list of float[] vectors of length D
   * @param dimension full vector dimension D
   * @param m         number of subspaces (must divide D)
   * @param k         number of centroids per subspace (K <= 256 recommended)
   * @param iterations number of k-means iterations
   * @param seed      RNG seed for deterministic initialization
   * @return centroids[M][K][subDim]
   */
  public static float[][][] train(List<float[]> vectors, int dimension, int m, int k, int iterations, long seed) {
    if (m <= 0 || k <= 0 || dimension <= 0) {
      throw new IllegalArgumentException("Invalid PQ params (m,k,dimension)");
    }
    if (dimension % m != 0) {
      throw new IllegalArgumentException("dimension must be divisible by m");
    }
    int subDim = dimension / m;
    float[][][] centroids = new float[m][k][subDim];
    Random rnd = new Random(seed);

    for (int s = 0; s < m; s++) {
      // Extract subspace data
      List<float[]> data = new ArrayList<>(vectors.size());
      for (float[] v : vectors) {
        float[] sub = Arrays.copyOfRange(v, s * subDim, (s + 1) * subDim);
        data.add(sub);
      }
      // Initialize centroids by sampling without replacement (or cycling if fewer points than k)
      for (int ci = 0; ci < k; ci++) {
        int idx = data.isEmpty() ? 0 : rnd.nextInt(data.size());
        centroids[s][ci] = Arrays.copyOf(data.get(idx), subDim);
      }
      // Lloyd's iterations
      int n = data.size();
      int[] assign = new int[n];
      for (int it = 0; it < iterations; it++) {
        // Assignment step
        for (int i = 0; i < n; i++) {
          float[] x = data.get(i);
          int best = 0;
          double bestDist = Double.POSITIVE_INFINITY;
          for (int ci = 0; ci < k; ci++) {
            double d = l2(x, centroids[s][ci]);
            if (d < bestDist) {
              bestDist = d;
              best = ci;
            }
          }
          assign[i] = best;
        }
        // Update step
        float[][] newC = new float[k][subDim];
        int[] counts = new int[k];
        for (int i = 0; i < n; i++) {
          int a = assign[i];
          float[] x = data.get(i);
          for (int d = 0; d < subDim; d++) newC[a][d] += x[d];
          counts[a]++;
        }
        for (int ci = 0; ci < k; ci++) {
          if (counts[ci] == 0) {
            // Reinitialize empty centroid to a random point
            int idx = rnd.nextInt(data.size());
            newC[ci] = Arrays.copyOf(data.get(idx), subDim);
          } else {
            for (int d = 0; d < subDim; d++) newC[ci][d] /= counts[ci];
          }
        }
        centroids[s] = newC;
      }
    }
    return centroids;
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
