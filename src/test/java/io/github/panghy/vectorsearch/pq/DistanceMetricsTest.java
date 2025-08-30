package io.github.panghy.vectorsearch.pq;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class DistanceMetricsTest {

  private static final float EPSILON = 1e-6f;

  @Test
  void testL2Distance() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};

    // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9 + 9 + 9) = sqrt(27)
    // But we return squared distance = 27
    float distance = DistanceMetrics.l2Distance(a, b);
    assertThat(distance).isCloseTo(27.0f, within(EPSILON));

    // Same vectors should have distance 0
    assertThat(DistanceMetrics.l2Distance(a, a)).isCloseTo(0.0f, within(EPSILON));
  }

  @Test
  void testL2DistanceDimensionMismatch() {
    float[] a = {1.0f, 2.0f};
    float[] b = {1.0f, 2.0f, 3.0f};

    assertThatThrownBy(() -> DistanceMetrics.l2Distance(a, b))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("same dimension");
  }

  @Test
  void testInnerProduct() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    // Negated for distance = -32
    float distance = DistanceMetrics.innerProduct(a, b);
    assertThat(distance).isCloseTo(-32.0f, within(EPSILON));

    // Orthogonal vectors
    float[] c = {1.0f, 0.0f};
    float[] d = {0.0f, 1.0f};
    assertThat(DistanceMetrics.innerProduct(c, d)).isCloseTo(0.0f, within(EPSILON));
  }

  @Test
  void testCosineDistance() {
    // Parallel vectors (same direction)
    float[] a = {1.0f, 0.0f};
    float[] b = {2.0f, 0.0f};
    assertThat(DistanceMetrics.cosineDistance(a, b)).isCloseTo(0.0f, within(EPSILON));

    // Orthogonal vectors
    float[] c = {1.0f, 0.0f};
    float[] d = {0.0f, 1.0f};
    assertThat(DistanceMetrics.cosineDistance(c, d)).isCloseTo(1.0f, within(EPSILON));

    // Opposite vectors
    float[] e = {1.0f, 0.0f};
    float[] f = {-1.0f, 0.0f};
    assertThat(DistanceMetrics.cosineDistance(e, f)).isCloseTo(2.0f, within(EPSILON));
  }

  @Test
  void testCosineDistanceWithZeroVector() {
    float[] zero = {0.0f, 0.0f, 0.0f};
    float[] nonZero = {1.0f, 2.0f, 3.0f};

    // Zero vector returns maximum distance
    assertThat(DistanceMetrics.cosineDistance(zero, nonZero)).isCloseTo(2.0f, within(EPSILON));
    assertThat(DistanceMetrics.cosineDistance(zero, zero)).isCloseTo(2.0f, within(EPSILON));
  }

  @Test
  void testDistanceWithMetric() {
    float[] a = {1.0f, 2.0f};
    float[] b = {3.0f, 4.0f};

    assertThat(DistanceMetrics.distance(a, b, DistanceMetrics.Metric.L2))
        .isCloseTo(DistanceMetrics.l2Distance(a, b), within(EPSILON));

    assertThat(DistanceMetrics.distance(a, b, DistanceMetrics.Metric.INNER_PRODUCT))
        .isCloseTo(DistanceMetrics.innerProduct(a, b), within(EPSILON));

    assertThat(DistanceMetrics.distance(a, b, DistanceMetrics.Metric.COSINE))
        .isCloseTo(DistanceMetrics.cosineDistance(a, b), within(EPSILON));
  }

  @Test
  void testNorm() {
    float[] vector = {3.0f, 4.0f};
    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    assertThat(DistanceMetrics.norm(vector)).isCloseTo(5.0f, within(EPSILON));

    float[] zero = {0.0f, 0.0f, 0.0f};
    assertThat(DistanceMetrics.norm(zero)).isCloseTo(0.0f, within(EPSILON));
  }

  @Test
  void testNormalize() {
    float[] vector = {3.0f, 4.0f};
    float[] normalized = DistanceMetrics.normalize(vector);

    // Check unit length
    assertThat(DistanceMetrics.norm(normalized)).isCloseTo(1.0f, within(EPSILON));

    // Check direction preserved
    assertThat(normalized[0]).isCloseTo(0.6f, within(EPSILON));
    assertThat(normalized[1]).isCloseTo(0.8f, within(EPSILON));

    // Original vector unchanged
    assertThat(vector[0]).isEqualTo(3.0f);
    assertThat(vector[1]).isEqualTo(4.0f);
  }

  @Test
  void testNormalizeZeroVector() {
    float[] zero = {0.0f, 0.0f, 0.0f};
    float[] normalized = DistanceMetrics.normalize(zero);

    // Zero vector remains zero
    assertThat(normalized).containsExactly(0.0f, 0.0f, 0.0f);
  }

  @Test
  void testNormalizeInPlace() {
    float[] vector = {3.0f, 4.0f};
    DistanceMetrics.normalizeInPlace(vector);

    // Check unit length
    assertThat(DistanceMetrics.norm(vector)).isCloseTo(1.0f, within(EPSILON));

    // Check values modified in place
    assertThat(vector[0]).isCloseTo(0.6f, within(EPSILON));
    assertThat(vector[1]).isCloseTo(0.8f, within(EPSILON));
  }

  @Test
  void testHighDimensionalVectors() {
    // Test with higher dimensional vectors
    float[] a = new float[128];
    float[] b = new float[128];

    for (int i = 0; i < 128; i++) {
      a[i] = i * 0.01f;
      b[i] = (128 - i) * 0.01f;
    }

    // All metrics should work
    float l2 = DistanceMetrics.l2Distance(a, b);
    float ip = DistanceMetrics.innerProduct(a, b);
    float cosine = DistanceMetrics.cosineDistance(a, b);

    assertThat(l2).isGreaterThan(0);
    assertThat(ip).isLessThan(0); // Negated inner product
    assertThat(cosine).isBetween(0.0f, 2.0f);
  }

  @Test
  void testNumericalStability() {
    // Test with very small values - may round to zero
    float[] small1 = {1e-10f, 1e-10f};
    float[] small2 = {2e-10f, 2e-10f};

    float distance = DistanceMetrics.cosineDistance(small1, small2);
    assertThat(distance).isCloseTo(0.0f, within(0.01f)); // Same direction

    // Test with large values
    float[] large1 = {1e10f, 1e10f};
    float[] large2 = {2e10f, 2e10f};

    distance = DistanceMetrics.cosineDistance(large1, large2);
    assertThat(distance).isCloseTo(0.0f, within(0.01f)); // Same direction
  }
}
