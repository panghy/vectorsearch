package io.github.panghy.vectorsearch.util;

/**
 * Basic distance/similarity utilities for vector comparisons.
 */
public final class Distances {
  private Distances() {}

  /**
   * Computes Euclidean (L2) distance between two vectors.
   *
   * <p>Use this when you need the true geometric distance (e.g., for reporting or thresholds
   * expressed in the original metric). For nearest-neighbor comparisons where only relative ordering
   * matters, prefer {@link #l2Squared(float[], float[])} to avoid the sqrt overhead.
   *
   * @param a vector A (must be same length as B)
   * @param b vector B
   * @return L2 distance (non-negative)
   * @see #l2Squared(float[], float[])
   */
  public static double l2(float[] a, float[] b) {
    return Math.sqrt(l2Squared(a, b));
  }

  /**
   * Computes squared Euclidean (L2²) distance between two vectors.
   *
   * <p>This is equivalent to {@code l2(a, b) * l2(a, b)} but avoids the {@link Math#sqrt} call.
   * Because sqrt is monotonic, squared distance preserves the same ordering as true L2 distance,
   * making it ideal for nearest-neighbor comparisons, graph construction, and pruning where only
   * relative ordering matters.
   *
   * @param a vector A (must be same length as B)
   * @param b vector B
   * @return squared L2 distance (non-negative)
   * @see #l2(float[], float[])
   */
  public static double l2Squared(float[] a, float[] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      double d = (double) a[i] - b[i];
      sum += d * d;
    }
    return sum;
  }

  /**
   * Computes dot product between two vectors.
   *
   * @param a vector A
   * @param b vector B
   * @return dot product value
   */
  public static double dot(float[] a, float[] b) {
    double s = 0.0;
    for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
    return s;
  }

  /**
   * Computes Euclidean norm of a vector.
   *
   * @param a vector
   * @return sqrt(sum(x^2))
   */
  public static double norm(float[] a) {
    double s = 0.0;
    for (float v : a) s += (double) v * v;
    return Math.sqrt(s);
  }

  /**
   * Computes cosine similarity for two vectors in R^n.
   *
   * @param a vector A
   * @param b vector B
   * @return cosine similarity in [-1, 1] (0 if either has zero norm)
   */
  public static double cosine(float[] a, float[] b) {
    double n = norm(a) * norm(b);
    if (n == 0.0) return 0.0;
    return dot(a, b) / n;
  }
}
