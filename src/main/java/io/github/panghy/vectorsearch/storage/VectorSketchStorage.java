package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.async.AsyncIterable;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.pq.VectorUtils;
import io.github.panghy.vectorsearch.proto.VectorSketch;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Storage handler for vector sketches in FoundationDB.
 * Manages compact vector representations for future codebook retraining.
 */
public class VectorSketchStorage {

  private final VectorIndexKeys keys;
  private final int dimension;

  public VectorSketchStorage(VectorIndexKeys keys, int dimension) {
    this.keys = keys;
    this.dimension = dimension;
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

  /**
   * Count the number of stored vector sketches (up to 10,000).
   * Limited to avoid long-running transactions.
   *
   * @param tr the transaction
   * @return the number of stored sketches (max 10,000)
   */
  public CompletableFuture<Integer> countVectorSketches(Transaction tr) {
    Range range = keys.vectorSketchRange();
    // Limit to 10,000 to avoid long-running transactions
    AsyncIterable<KeyValue> iterable = tr.getRange(range, 10000);
    return iterable.asList().thenApply(List::size);
  }

  /**
   * Sample vector sketches and reconstruct approximate vectors.
   * This is a simplified reconstruction that expands the sketch back to full dimension.
   *
   * @param tr the transaction
   * @param sampleSize the number of samples to retrieve
   * @return list of reconstructed vectors
   */
  public CompletableFuture<List<float[]>> sampleVectorSketches(Transaction tr, int sampleSize) {
    Range range = keys.vectorSketchRange();

    // Get all keys first (limited to avoid timeout)
    return tr.getRange(range, sampleSize * 2).asList().thenApply(kvs -> {
      List<float[]> vectors = new ArrayList<>();

      // Randomly sample from available sketches
      List<KeyValue> sampled = new ArrayList<>();

      if (kvs.size() <= sampleSize) {
        sampled = kvs;
      } else {
        // Random sampling without replacement
        List<KeyValue> kvsCopy = new ArrayList<>(kvs);
        ThreadLocalRandom random = ThreadLocalRandom.current();
        for (int i = 0; i < sampleSize; i++) {
          int idx = random.nextInt(kvsCopy.size());
          sampled.add(kvsCopy.remove(idx));
        }
      }

      // Reconstruct vectors from sketches
      for (KeyValue kv : sampled) {
        try {
          VectorSketch sketch = VectorSketch.parseFrom(kv.getValue());
          float[] vector = reconstructFromSketch(sketch);
          vectors.add(vector);
        } catch (InvalidProtocolBufferException e) {
          // Skip invalid sketches
        }
      }

      return vectors;
    });
  }

  /**
   * Reconstruct an approximate vector from a sketch.
   * Note: This is a lossy reconstruction that creates a deterministic random vector
   * based on the sketch. For actual vector retrieval, use the stored vectors directly.
   *
   * @param sketch the vector sketch
   * @return reconstructed vector
   */
  private float[] reconstructFromSketch(VectorSketch sketch) {
    // Create a deterministic random vector based on the sketch as a seed
    // This is inherently lossy as we're reconstructing from a 256-bit sketch

    byte[] sketchData = sketch.getSketchData().toByteArray();

    // Use sketch as seed for deterministic reconstruction
    long seed = 0;
    for (int i = 0; i < Math.min(8, sketchData.length); i++) {
      seed = (seed << 8) | (sketchData[i] & 0xFF);
    }
    Random random = new Random(seed);

    float[] vector = VectorUtils.randomVector(dimension, random);

    // Normalize the vector
    float norm = 0;
    for (float v : vector) {
      norm += v * v;
    }
    norm = (float) Math.sqrt(norm);
    if (norm > 0) {
      for (int i = 0; i < vector.length; i++) {
        vector[i] /= norm;
      }
    }

    return vector;
  }
}
