package io.github.panghy.vectorsearch.util;

/**
 * Unit tests for SIMD-accelerated Distances (L2, cosine, dot, norm).
 *
 * <p>Parameterized tests exercise various dimensions (1, 3, 7, 16, 128, 1000) to cover both full
 * SIMD lanes and scalar tail handling.
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class DistancesTest {

  @Test
  void l2_and_cosine_behave_reasonably() {
    float[] a = new float[] {1f, 0f, 0f};
    float[] b = new float[] {0f, 1f, 0f};
    float[] c = new float[] {1f, 0f, 0f};

    assertThat(Distances.l2(a, b)).isGreaterThan(1.0);
    assertThat(Distances.l2(a, c)).isEqualTo(0.0);

    double cosAB = Distances.cosine(a, b);
    double cosAC = Distances.cosine(a, c);
    assertThat(cosAB).isCloseTo(0.0, within(1e-6));
    assertThat(cosAC).isCloseTo(1.0, within(1e-6));
  }

  @Test
  void l2Squared_returns_squared_distance() {
    float[] a = new float[] {1f, 2f, 3f};
    float[] b = new float[] {4f, 6f, 3f};

    // squared distance = (4-1)^2 + (6-2)^2 + (3-3)^2 = 9 + 16 + 0 = 25
    assertThat(Distances.l2Squared(a, b)).isCloseTo(25.0, within(1e-9));
    // l2 should be sqrt of l2Squared
    assertThat(Distances.l2(a, b)).isCloseTo(5.0, within(1e-9));

    // identical vectors
    assertThat(Distances.l2Squared(a, a)).isEqualTo(0.0);
  }

  static Stream<Integer> dimensions() {
    return Stream.of(1, 3, 7, 16, 128, 1000);
  }

  @ParameterizedTest
  @MethodSource("dimensions")
  void l2_matches_scalar_reference(int dim) {
    Random rng = new Random(42 + dim);
    float[] a = randomVector(rng, dim);
    float[] b = randomVector(rng, dim);

    double expected = scalarL2(a, b);
    double actual = Distances.l2(a, b);
    assertThat(actual).isCloseTo(expected, within(1e-3));
  }

  @ParameterizedTest
  @MethodSource("dimensions")
  void dot_matches_scalar_reference(int dim) {
    Random rng = new Random(42 + dim);
    float[] a = randomVector(rng, dim);
    float[] b = randomVector(rng, dim);

    double expected = scalarDot(a, b);
    double actual = Distances.dot(a, b);
    assertThat(actual).isCloseTo(expected, within(1e-2));
  }

  @ParameterizedTest
  @MethodSource("dimensions")
  void norm_matches_scalar_reference(int dim) {
    Random rng = new Random(42 + dim);
    float[] a = randomVector(rng, dim);

    double expected = scalarNorm(a);
    double actual = Distances.norm(a);
    assertThat(actual).isCloseTo(expected, within(1e-3));
  }

  @ParameterizedTest
  @MethodSource("dimensions")
  void cosine_matches_scalar_reference(int dim) {
    Random rng = new Random(42 + dim);
    float[] a = randomVector(rng, dim);
    float[] b = randomVector(rng, dim);

    double expected = scalarCosine(a, b);
    double actual = Distances.cosine(a, b);
    assertThat(actual).isCloseTo(expected, within(1e-5));
  }

  @Test
  void l2_identical_vectors_is_zero() {
    float[] v = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
    assertThat(Distances.l2(v, v)).isEqualTo(0.0);
  }

  @Test
  void dot_orthogonal_vectors_is_zero() {
    float[] a = {1f, 0f, 0f, 0f};
    float[] b = {0f, 1f, 0f, 0f};
    assertThat(Distances.dot(a, b)).isEqualTo(0.0);
  }

  @Test
  void norm_unit_vector() {
    float[] v = {1f, 0f, 0f};
    assertThat(Distances.norm(v)).isCloseTo(1.0, within(1e-9));
  }

  @Test
  void cosine_zero_vector_returns_zero() {
    float[] zero = {0f, 0f, 0f};
    float[] v = {1f, 2f, 3f};
    assertThat(Distances.cosine(zero, v)).isEqualTo(0.0);
    assertThat(Distances.cosine(v, zero)).isEqualTo(0.0);
  }

  // --- Scalar reference implementations ---

  private static double scalarL2(float[] a, float[] b) {
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      double d = (double) a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  private static double scalarDot(float[] a, float[] b) {
    double s = 0.0;
    for (int i = 0; i < a.length; i++) {
      s += (double) a[i] * b[i];
    }
    return s;
  }

  private static double scalarNorm(float[] a) {
    double s = 0.0;
    for (float v : a) {
      s += (double) v * v;
    }
    return Math.sqrt(s);
  }

  private static double scalarCosine(float[] a, float[] b) {
    double n = scalarNorm(a) * scalarNorm(b);
    if (n == 0.0) return 0.0;
    return scalarDot(a, b) / n;
  }

  private static float[] randomVector(Random rng, int dim) {
    float[] v = new float[dim];
    for (int i = 0; i < dim; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
    }
    return v;
  }
}
