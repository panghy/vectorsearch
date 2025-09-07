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

  private static org.assertj.core.data.Offset<Double> within(double d) {
    return org.assertj.core.data.Offset.offset(d);
  }
}
