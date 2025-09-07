package io.github.panghy.vectorsearch.pq;

/**
 * Unit tests for PqEncoder: encoding correctness and basic bounds.
 */
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class PqEncoderTest {
  @Test
  void encodes_expected_codes() {
    float[][][] centroids = new float[2][2][2]; // m=2, k=2, subDim=2
    centroids[0][0] = new float[] {0f, 0f};
    centroids[0][1] = new float[] {1f, 1f};
    centroids[1][0] = new float[] {0f, 0f};
    centroids[1][1] = new float[] {2f, 2f};

    byte[] codes = PqEncoder.encode(centroids, new float[] {0.1f, 0.1f, 1.9f, 2.1f});
    assertThat(codes).hasSize(2);
    assertThat(((int) codes[0]) & 0xFF).isEqualTo(0);
    assertThat(((int) codes[1]) & 0xFF).isEqualTo(1);
  }
}
