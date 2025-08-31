package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Transaction;
import com.google.protobuf.ByteString;
import io.github.panghy.vectorsearch.proto.VectorSketch;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.concurrent.CompletableFuture;

/**
 * Storage handler for vector sketches in FoundationDB.
 * Manages compact vector representations for future codebook retraining.
 */
public class VectorSketchStorage {

  private final VectorIndexKeys keys;

  public VectorSketchStorage(VectorIndexKeys keys) {
    this.keys = keys;
  }

  /**
   * Stores a vector sketch for future codebook retraining.
   * Uses SimHash for compact representation.
   *
   * @param tr the transaction
   * @param nodeId the node ID
   * @param vector the original vector
   * @return future completing when sketch is stored
   */
  public CompletableFuture<Void> storeVectorSketch(Transaction tr, long nodeId, float[] vector) {

    // Generate SimHash sketch (256 bits)
    byte[] sketch = generateSimHash(vector);

    // Store sketch
    byte[] key = keys.vectorSketchKey(nodeId);
    VectorSketch sketchProto = VectorSketch.newBuilder()
        .setNodeId(nodeId)
        .setSketchType(VectorSketch.SketchType.SIMHASH_256)
        .setSketchData(ByteString.copyFrom(sketch))
        .build();

    tr.set(key, sketchProto.toByteArray());
    return completedFuture(null);
  }

  /**
   * Generates a SimHash sketch of a vector.
   */
  private byte[] generateSimHash(float[] vector) {
    // Simple SimHash implementation using float accumulation for better precision
    float[] accumulator = new float[256];

    try {
      MessageDigest md = MessageDigest.getInstance("SHA-256");

      // Hash each dimension with its value
      for (int i = 0; i < vector.length; i++) {
        md.reset();
        md.update(ByteBuffer.allocate(4).putInt(i).array());
        byte[] hash = md.digest();

        // Weight by vector value (keep full float precision)
        float weight = vector[i];
        for (int j = 0; j < 256 && j < hash.length * 8; j++) {
          int byteIdx = j / 8;
          int bitIdx = j % 8;
          // Using little-endian bit ordering within bytes (LSB = bit 0)
          boolean bit = ((hash[byteIdx] >> bitIdx) & 1) == 1;
          accumulator[j] += bit ? weight : -weight;
        }
      }

      // Convert to binary (threshold at 0)
      byte[] sketch = new byte[32]; // 256 bits
      for (int i = 0; i < 256; i++) {
        if (accumulator[i] > 0) {
          int byteIdx = i / 8;
          int bitIdx = i % 8;
          // Using same little-endian bit ordering for consistency
          sketch[byteIdx] |= (byte) (1 << bitIdx);
        }
      }

      return sketch;
    } catch (Exception e) {
      throw new RuntimeException("Failed to generate SimHash", e);
    }
  }
}
