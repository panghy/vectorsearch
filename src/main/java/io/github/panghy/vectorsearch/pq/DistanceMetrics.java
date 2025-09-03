package io.github.panghy.vectorsearch.pq;

/**
 * Utility class for computing distance metrics between vectors.
 * Supports L2 (Euclidean), Inner Product (IP), and Cosine similarity distances.
 */
public class DistanceMetrics {

  /** Distance metric types supported by the vector search index. */
  public enum Metric {
    L2,
    INNER_PRODUCT,
    COSINE
  }

  /**
   * Computes the L2 (Euclidean) distance between two vectors.
   *
   * @param a first vector
   * @param b second vector
   * @return the L2 distance
   * @throws IllegalArgumentException if vectors have different dimensions
   */
  public static float l2Distance(float[] a, float[] b) {
    if (a.length != b.length) {
      throw new IllegalArgumentException("Vectors must have the same dimension: " + a.length + " != " + b.length);
    }
    float sum = 0;
    for (int i = 0; i < a.length; i++) {
      float diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sum; // Return squared distance (avoid sqrt for efficiency)
  }

  /**
   * Computes the inner product between two vectors.
   * Note: For similarity search, higher values indicate closer vectors.
   *
   * @param a first vector
   * @param b second vector
   * @return the inner product
   * @throws IllegalArgumentException if vectors have different dimensions
   */
  public static float innerProduct(float[] a, float[] b) {
    if (a.length != b.length) {
      throw new IllegalArgumentException("Vectors must have the same dimension: " + a.length + " != " + b.length);
    }
    float sum = 0;
    for (int i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return -sum; // Negate to convert similarity to distance (lower is better)
  }

  /**
   * Computes the cosine distance between two vectors.
   * Cosine distance = 1 - cosine_similarity
   *
   * @param a first vector
   * @param b second vector
   * @return the cosine distance [0, 2]
   * @throws IllegalArgumentException if vectors have different dimensions
   */
  public static float cosineDistance(float[] a, float[] b) {
    if (a.length != b.length) {
      throw new IllegalArgumentException("Vectors must have the same dimension: " + a.length + " != " + b.length);
    }

    float dotProduct = 0;
    float normA = 0;
    float normB = 0;

    for (int i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA == 0 || normB == 0) {
      return 2.0f; // Maximum distance for zero vectors
    }

    float similarity = dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    // Clamp to [-1, 1] to handle numerical errors
    similarity = Math.max(-1.0f, Math.min(1.0f, similarity));
    return 1.0f - similarity; // Convert to distance
  }

  /**
   * Computes distance between two vectors using the specified metric.
   *
   * @param a first vector
   * @param b second vector
   * @param metric the distance metric to use
   * @return the distance
   * @throws IllegalArgumentException if vectors have different dimensions
   */
  public static float distance(float[] a, float[] b, Metric metric) {
    return switch (metric) {
      case L2 -> l2Distance(a, b);
      case INNER_PRODUCT -> innerProduct(a, b);
      case COSINE -> cosineDistance(a, b);
    };
  }

  /**
   * Computes the L2 norm (magnitude) of a vector.
   *
   * @param vector the input vector
   * @return the L2 norm
   */
  public static float norm(float[] vector) {
    float sum = 0;
    for (float v : vector) {
      sum += v * v;
    }
    return (float) Math.sqrt(sum);
  }

  /**
   * Normalizes a vector to unit length (L2 norm = 1).
   * Creates a new array without modifying the input.
   *
   * @param vector the input vector
   * @return normalized vector
   */
  public static float[] normalize(float[] vector) {
    float norm = norm(vector);
    if (norm == 0) {
      return vector.clone(); // Return copy of zero vector
    }
    float[] normalized = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      normalized[i] = vector[i] / norm;
    }
    return normalized;
  }

  /**
   * Normalizes a vector in-place to unit length (L2 norm = 1).
   *
   * @param vector the vector to normalize in-place
   */
  public static void normalizeInPlace(float[] vector) {
    float norm = norm(vector);
    if (norm == 0) {
      return; // Cannot normalize zero vector
    }
    for (int i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }
}
