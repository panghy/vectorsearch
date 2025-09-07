package io.github.panghy.vectorsearch.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Utility for packing/unpacking float32 arrays to little-endian byte arrays.
 *
 * <p>This is used to store embeddings in FoundationDB efficiently and to reconstruct them
 * for query-time computations.</p>
 */
public final class FloatPacker {
  private FloatPacker() {}

  /**
   * Packs a float array into a little-endian byte array (4 bytes per element).
   *
   * @param arr the float array to pack (must not be null)
   * @return a new byte array containing the packed floats
   */
  public static byte[] floatsToBytes(float[] arr) {
    ByteBuffer bb = ByteBuffer.allocate(arr.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float v : arr) bb.putFloat(v);
    return bb.array();
  }

  /**
   * Unpacks a little-endian byte array into a float array.
   *
   * @param bytes the byte array to unpack (length must be a multiple of 4)
   * @return a new float array reconstructed from the bytes
   */
  public static float[] bytesToFloats(byte[] bytes) {
    ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int n = bytes.length / Float.BYTES;
    float[] out = new float[n];
    for (int i = 0; i < n; i++) out[i] = bb.getFloat();
    return out;
  }
}
