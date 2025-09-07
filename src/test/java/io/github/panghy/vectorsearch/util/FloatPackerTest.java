package io.github.panghy.vectorsearch.util;

/**
 * Unit tests for FloatPacker: float<->byte conversions and endianness consistency.
 */
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class FloatPackerTest {

  @Test
  void roundTripFloatPacking() {
    float[] in = new float[] {0.0f, -1.5f, 3.14159f, 12345.678f};
    byte[] bytes = FloatPacker.floatsToBytes(in);
    float[] out = FloatPacker.bytesToFloats(bytes);
    assertThat(out).hasSize(in.length);
    for (int i = 0; i < in.length; i++) {
      assertThat(out[i]).isCloseTo(in[i], org.assertj.core.data.Offset.offset(1e-6f));
    }
  }
}
