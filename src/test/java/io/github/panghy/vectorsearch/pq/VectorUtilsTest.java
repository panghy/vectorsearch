package io.github.panghy.vectorsearch.pq;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.Test;

class VectorUtilsTest {

  private static final float EPSILON = 1e-5f;

  @Test
  void testFloat16Conversion() {
    // Test basic conversion
    float[] original = {1.0f, -0.5f, 0.0f, 2.5f, -3.25f};
    byte[] bytes = VectorUtils.toFloat16Bytes(original);
    float[] recovered = VectorUtils.fromFloat16Bytes(bytes);

    assertThat(bytes).hasSize(original.length * 2);
    assertThat(recovered).hasSize(original.length);

    // Check values are approximately preserved (float16 has limited precision)
    for (int i = 0; i < original.length; i++) {
      assertThat(recovered[i]).isCloseTo(original[i], within(0.01f));
    }
  }

  @Test
  void testFloat16SpecialValues() {
    // Test special float values
    float[] special = {0.0f, -0.0f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};

    byte[] bytes = VectorUtils.toFloat16Bytes(special);
    float[] recovered = VectorUtils.fromFloat16Bytes(bytes);

    assertThat(recovered[0]).isEqualTo(0.0f);
    assertThat(recovered[1]).isEqualTo(-0.0f);
    assertThat(recovered[2]).isEqualTo(Float.POSITIVE_INFINITY);
    assertThat(recovered[3]).isEqualTo(Float.NEGATIVE_INFINITY);
    assertThat(recovered[4]).isNaN();
  }

  @Test
  void testFloat16Range() {
    // Test values at different scales
    float[] values = {
      1e-4f, // Small value
      1.0f, // Normal value
      100.0f, // Larger value
      65504f // Near max float16 value
    };

    byte[] bytes = VectorUtils.toFloat16Bytes(values);
    float[] recovered = VectorUtils.fromFloat16Bytes(bytes);

    // Float16 has limited precision, especially for small/large values
    assertThat(recovered[0]).isCloseTo(values[0], within(1e-3f));
    assertThat(recovered[1]).isCloseTo(values[1], within(1e-3f));
    assertThat(recovered[2]).isCloseTo(values[2], within(0.1f));
    assertThat(recovered[3]).isCloseTo(values[3], within(100f));
  }

  @Test
  void testFromFloat16BytesInvalidLength() {
    byte[] oddLengthBytes = new byte[5];

    assertThatThrownBy(() -> VectorUtils.fromFloat16Bytes(oddLengthBytes))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("even");
  }

  @Test
  void testGetSubvectorEvenSplit() {
    float[] vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    int numSubvectors = 3;

    // Each subvector should have 2 elements
    float[] sub0 = VectorUtils.getSubvector(vector, 0, numSubvectors);
    float[] sub1 = VectorUtils.getSubvector(vector, 1, numSubvectors);
    float[] sub2 = VectorUtils.getSubvector(vector, 2, numSubvectors);

    assertThat(sub0).containsExactly(1.0f, 2.0f);
    assertThat(sub1).containsExactly(3.0f, 4.0f);
    assertThat(sub2).containsExactly(5.0f, 6.0f);
  }

  @Test
  void testGetSubvectorUnevenSplit() {
    float[] vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    int numSubvectors = 3;

    // First subvector gets extra element (7/3 = 2 remainder 1)
    float[] sub0 = VectorUtils.getSubvector(vector, 0, numSubvectors);
    float[] sub1 = VectorUtils.getSubvector(vector, 1, numSubvectors);
    float[] sub2 = VectorUtils.getSubvector(vector, 2, numSubvectors);

    assertThat(sub0).containsExactly(1.0f, 2.0f, 3.0f);
    assertThat(sub1).containsExactly(4.0f, 5.0f);
    assertThat(sub2).containsExactly(6.0f, 7.0f);
  }

  @Test
  void testGetSubvectorSingleSubvector() {
    float[] vector = {1.0f, 2.0f, 3.0f};

    float[] sub = VectorUtils.getSubvector(vector, 0, 1);
    assertThat(sub).containsExactly(1.0f, 2.0f, 3.0f);
  }

  @Test
  void testGetSubvectorInvalidIndex() {
    float[] vector = {1.0f, 2.0f, 3.0f};

    assertThatThrownBy(() -> VectorUtils.getSubvector(vector, -1, 2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("out of range");

    assertThatThrownBy(() -> VectorUtils.getSubvector(vector, 2, 2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("out of range");
  }

  @Test
  void testRandomUnitVector() {
    Random random = new Random(42);
    int dimension = 100;

    float[] vector = VectorUtils.randomUnitVector(dimension, random);

    assertThat(vector).hasSize(dimension);

    // Check unit length
    float norm = DistanceMetrics.norm(vector);
    assertThat(norm).isCloseTo(1.0f, within(EPSILON));

    // Check values are not all the same
    float first = vector[0];
    boolean allSame = true;
    for (float v : vector) {
      if (Math.abs(v - first) > EPSILON) {
        allSame = false;
        break;
      }
    }
    assertThat(allSame).isFalse();
  }

  @Test
  void testRandomVector() {
    Random random = new Random(42);
    int dimension = 50;

    float[] vector = VectorUtils.randomVector(dimension, random);

    assertThat(vector).hasSize(dimension);

    // Check all values are in range [-1, 1]
    for (float v : vector) {
      assertThat(v).isBetween(-1.0f, 1.0f);
    }

    // Check randomness
    float sum = 0;
    for (float v : vector) {
      sum += v;
    }
    float mean = sum / dimension;
    assertThat(mean).isCloseTo(0.0f, within(0.2f)); // Should be roughly centered
  }

  @Test
  void testMean() {
    float[][] vectors = {
      {1.0f, 2.0f, 3.0f},
      {4.0f, 5.0f, 6.0f},
      {7.0f, 8.0f, 9.0f}
    };

    float[] mean = VectorUtils.mean(vectors);

    assertThat(mean).containsExactly(4.0f, 5.0f, 6.0f);
  }

  @Test
  void testMeanSingleVector() {
    float[][] vectors = {{1.0f, 2.0f, 3.0f}};

    float[] mean = VectorUtils.mean(vectors);

    assertThat(mean).containsExactly(1.0f, 2.0f, 3.0f);
  }

  @Test
  void testMeanEmptySet() {
    float[][] vectors = new float[0][];

    assertThatThrownBy(() -> VectorUtils.mean(vectors))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("empty");
  }

  @Test
  void testMeanDimensionMismatch() {
    float[][] vectors = {
      {1.0f, 2.0f},
      {3.0f, 4.0f, 5.0f}
    };

    assertThatThrownBy(() -> VectorUtils.mean(vectors))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("same dimension");
  }

  @Test
  void testAdd() {
    float[] a = {1.0f, 2.0f, 3.0f};
    float[] b = {4.0f, 5.0f, 6.0f};

    float[] result = VectorUtils.add(a, b);

    assertThat(result).containsExactly(5.0f, 7.0f, 9.0f);
    // Original vectors unchanged
    assertThat(a).containsExactly(1.0f, 2.0f, 3.0f);
    assertThat(b).containsExactly(4.0f, 5.0f, 6.0f);
  }

  @Test
  void testAddDimensionMismatch() {
    float[] a = {1.0f, 2.0f};
    float[] b = {3.0f, 4.0f, 5.0f};

    assertThatThrownBy(() -> VectorUtils.add(a, b))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("same dimension");
  }

  @Test
  void testScale() {
    float[] vector = {1.0f, 2.0f, 3.0f};
    float scalar = 2.5f;

    float[] result = VectorUtils.scale(vector, scalar);

    assertThat(result).containsExactly(2.5f, 5.0f, 7.5f);
    // Original vector unchanged
    assertThat(vector).containsExactly(1.0f, 2.0f, 3.0f);
  }

  @Test
  void testScaleByZero() {
    float[] vector = {1.0f, 2.0f, 3.0f};

    float[] result = VectorUtils.scale(vector, 0.0f);

    assertThat(result).containsExactly(0.0f, 0.0f, 0.0f);
  }

  @Test
  void testScaleNegative() {
    float[] vector = {1.0f, 2.0f, 3.0f};

    float[] result = VectorUtils.scale(vector, -1.0f);

    assertThat(result).containsExactly(-1.0f, -2.0f, -3.0f);
  }

  @Test
  void testLargeVectorOperations() {
    // Test with large vectors to ensure efficiency
    int dimension = 1024;
    Random random = new Random(42);

    float[] large1 = VectorUtils.randomVector(dimension, random);
    float[] large2 = VectorUtils.randomVector(dimension, random);

    // Test operations complete without error
    float[] sum = VectorUtils.add(large1, large2);
    float[] scaled = VectorUtils.scale(large1, 0.5f);
    byte[] compressed = VectorUtils.toFloat16Bytes(large1);
    float[] decompressed = VectorUtils.fromFloat16Bytes(compressed);

    assertThat(sum).hasSize(dimension);
    assertThat(scaled).hasSize(dimension);
    assertThat(decompressed).hasSize(dimension);
  }
}
