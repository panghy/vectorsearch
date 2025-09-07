package io.github.panghy.vectorsearch.pq;

/**
 * Encodes vectors into PQ codes using trained centroids.
 */
public final class PqEncoder {
  private PqEncoder() {}

  /**
   * Encodes a vector into M subspace codes (one byte each) using L2 distance.
   *
   * @param centroids centroids[M][K][subDim]
   * @param vector    full vector of length M*subDim
   * @return byte array of length M with code index per subspace (0..K-1)
   */
  public static byte[] encode(float[][][] centroids, float[] vector) {
    int m = centroids.length;
    int k = centroids[0].length;
    int subDim = centroids[0][0].length;
    byte[] codes = new byte[m];
    for (int s = 0; s < m; s++) {
      int off = s * subDim;
      int best = 0;
      double bestDist = Double.POSITIVE_INFINITY;
      for (int ci = 0; ci < k; ci++) {
        double d = 0.0;
        for (int dIdx = 0; dIdx < subDim; dIdx++) {
          double dd = (double) vector[off + dIdx] - centroids[s][ci][dIdx];
          d += dd * dd;
        }
        if (d < bestDist) {
          bestDist = d;
          best = ci;
        }
      }
      codes[s] = (byte) (best & 0xFF);
    }
    return codes;
  }
}
