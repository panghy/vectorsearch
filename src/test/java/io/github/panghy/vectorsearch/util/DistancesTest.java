package io.github.panghy.vectorsearch.util;

/**
 * Unit tests for Distances (L2, cosine, dot, normalization).
 */
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

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

  private static org.assertj.core.data.Offset<Double> within(double d) {
    return org.assertj.core.data.Offset.offset(d);
  }
}
