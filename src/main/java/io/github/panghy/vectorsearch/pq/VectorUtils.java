package io.github.panghy.vectorsearch.pq;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

/**
 * Utility class for vector operations including float16 conversion,
 * vector slicing, and random vector generation.
 */
public class VectorUtils {

  /**
   * Converts a float array to float16 (half-precision) bytes.
   * Uses IEEE 754 half-precision format for space-efficient storage.
   *
   * @param floats the input float array
   * @return byte array containing float16 values (2 bytes per float)
   */
  public static byte[] toFloat16Bytes(float[] floats) {
    ByteBuffer buffer = ByteBuffer.allocate(floats.length * 2);
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    for (float f : floats) {
      buffer.putShort(floatToHalf(f));
    }

    return buffer.array();
  }

  /**
   * Converts float16 (half-precision) bytes back to a float array.
   *
   * @param bytes the input byte array containing float16 values
   * @return float array
   */
  public static float[] fromFloat16Bytes(byte[] bytes) {
    if (bytes.length % 2 != 0) {
      throw new IllegalArgumentException("Byte array length must be even for float16 conversion");
    }

    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    float[] floats = new float[bytes.length / 2];
    for (int i = 0; i < floats.length; i++) {
      floats[i] = halfToFloat(buffer.getShort());
    }

    return floats;
  }

  /**
   * Converts a float to IEEE 754 half-precision format.
   * Based on the ARM half-precision conversion algorithm.
   *
   * @param f the float value
   * @return half-precision representation as short
   */
  private static short floatToHalf(float f) {
    int fbits = Float.floatToIntBits(f);
    int sign = (fbits >>> 16) & 0x8000;
    int val = (fbits & 0x7fffffff) + 0x1000;

    if (val >= 0x47800000) {
      if ((fbits & 0x7fffffff) >= 0x47800000) {
        if (val < 0x7f800000) {
          return (short) (sign | 0x7c00);
        }
        return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
      }
      return (short) (sign | 0x7bff);
    }

    if (val >= 0x38800000) {
      return (short) (sign | ((val - 0x38000000) >>> 13));
    }

    if (val < 0x33000000) {
      return (short) sign;
    }

    val = (fbits & 0x7fffffff) >>> 23;
    return (short) (sign | (((fbits & 0x7fffff | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
  }

  /**
   * Converts IEEE 754 half-precision format to float.
   *
   * @param half the half-precision value
   * @return float value
   */
  private static float halfToFloat(short half) {
    int h = half & 0xffff;
    int sign = (h >>> 15) & 0x1;
    int exponent = (h >>> 10) & 0x1f;
    int mantissa = h & 0x3ff;

    if (exponent == 0) {
      if (mantissa == 0) {
        return sign == 0 ? 0.0f : -0.0f;
      } else {
        // Subnormal number
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        exponent++;
        mantissa &= ~0x400;
      }
    } else if (exponent == 31) {
      if (mantissa == 0) {
        return sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
      } else {
        return Float.NaN;
      }
    }

    exponent = exponent + 112;
    mantissa = mantissa << 13;

    int fbits = (sign << 31) | (exponent << 23) | mantissa;
    return Float.intBitsToFloat(fbits);
  }

  /**
   * Extracts a subvector from a larger vector.
   * Used for Product Quantization to split vectors into subspaces.
   *
   * @param vector the input vector
   * @param subvectorIndex the index of the subvector to extract
   * @param numSubvectors the total number of subvectors
   * @return the extracted subvector
   */
  public static float[] getSubvector(float[] vector, int subvectorIndex, int numSubvectors) {
    if (subvectorIndex < 0 || subvectorIndex >= numSubvectors) {
      throw new IllegalArgumentException(
          "Subvector index out of range: " + subvectorIndex + " (0-" + (numSubvectors - 1) + ")");
    }

    int dimension = vector.length;
    int subDimension = dimension / numSubvectors;
    int remainder = dimension % numSubvectors;

    // Handle uneven splits by giving extra dimensions to earlier subvectors
    int startIdx, endIdx;
    if (subvectorIndex < remainder) {
      subDimension += 1;
      startIdx = subvectorIndex * subDimension;
      endIdx = startIdx + subDimension;
    } else {
      startIdx = remainder * (subDimension + 1) + (subvectorIndex - remainder) * subDimension;
      endIdx = startIdx + subDimension;
    }

    float[] subvector = new float[endIdx - startIdx];
    System.arraycopy(vector, startIdx, subvector, 0, subvector.length);
    return subvector;
  }

  /**
   * Generates a random unit vector (normalized to L2 norm = 1).
   * Useful for testing and initialization.
   *
   * @param dimension the dimension of the vector
   * @param random the random number generator
   * @return a random unit vector
   */
  public static float[] randomUnitVector(int dimension, Random random) {
    float[] vector = new float[dimension];

    // Generate Gaussian random values (for uniform distribution on hypersphere)
    for (int i = 0; i < dimension; i++) {
      vector[i] = (float) random.nextGaussian();
    }

    // Normalize to unit length
    DistanceMetrics.normalizeInPlace(vector);
    return vector;
  }

  /**
   * Generates a random vector with values in the range [-1, 1].
   *
   * @param dimension the dimension of the vector
   * @param random the random number generator
   * @return a random vector
   */
  public static float[] randomVector(int dimension, Random random) {
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat() * 2 - 1; // Range [-1, 1]
    }
    return vector;
  }

  /**
   * Computes the mean vector from a set of vectors.
   *
   * @param vectors array of vectors
   * @return the mean vector
   */
  public static float[] mean(float[][] vectors) {
    if (vectors.length == 0) {
      throw new IllegalArgumentException("Cannot compute mean of empty vector set");
    }

    int dimension = vectors[0].length;
    float[] mean = new float[dimension];

    for (float[] vector : vectors) {
      if (vector.length != dimension) {
        throw new IllegalArgumentException("All vectors must have the same dimension");
      }
      for (int i = 0; i < dimension; i++) {
        mean[i] += vector[i];
      }
    }

    for (int i = 0; i < dimension; i++) {
      mean[i] /= vectors.length;
    }

    return mean;
  }

  /**
   * Adds two vectors element-wise.
   *
   * @param a first vector
   * @param b second vector
   * @return result vector
   */
  public static float[] add(float[] a, float[] b) {
    if (a.length != b.length) {
      throw new IllegalArgumentException("Vectors must have the same dimension");
    }
    float[] result = new float[a.length];
    for (int i = 0; i < a.length; i++) {
      result[i] = a[i] + b[i];
    }
    return result;
  }

  /**
   * Scales a vector by a scalar value.
   *
   * @param vector the input vector
   * @param scalar the scaling factor
   * @return scaled vector
   */
  public static float[] scale(float[] vector, float scalar) {
    float[] result = new float[vector.length];
    for (int i = 0; i < vector.length; i++) {
      result[i] = vector[i] * scalar;
    }
    return result;
  }
}
