package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.ReadTransaction;
import com.apple.foundationdb.Transaction;
import io.github.panghy.vectorsearch.pq.VectorUtils;
import java.util.concurrent.CompletableFuture;

/**
 * Storage for original vectors (fp16-packed) in FoundationDB.
 * Provides simple get/put/delete primitives suitable for reranking and retraining.
 */
public class OriginalVectorStorage {

  private final VectorIndexKeys keys;
  private final int dimension;

  public OriginalVectorStorage(VectorIndexKeys keys, int dimension) {
    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive, got: " + dimension);
    }
    this.keys = keys;
    this.dimension = dimension;
  }

  /**
   * Stores an original vector encoded as fp16 bytes under the nodeId.
   */
  public CompletableFuture<Void> storeVector(Transaction tr, long nodeId, float[] vector) {
    if (vector == null || vector.length != dimension) {
      throw new IllegalArgumentException("Vector must be non-null and length=" + dimension);
    }
    byte[] data = VectorUtils.toFloat16Bytes(vector);
    tr.set(keys.vectorKey(nodeId), data);
    return completedFuture(null);
  }

  /**
   * Reads an original vector, decoding fp16 bytes back to floats.
   * Returns null if not found.
   */
  public CompletableFuture<float[]> readVector(ReadTransaction tr, long nodeId) {
    return tr.get(keys.vectorKey(nodeId)).thenApply(bytes -> {
      if (bytes == null) return null;
      float[] vec = VectorUtils.fromFloat16Bytes(bytes);
      if (vec.length != dimension) {
        // If dimensions mismatch, return as-is to avoid data loss but flag via exception
        throw new IllegalStateException(
            "Stored vector dimension mismatch: expected " + dimension + ", got " + vec.length);
      }
      return vec;
    });
  }

  /**
   * Deletes an original vector for a node if present.
   */
  public CompletableFuture<Void> deleteVector(Transaction tr, long nodeId) {
    tr.clear(keys.vectorKey(nodeId));
    return completedFuture(null);
  }
}
