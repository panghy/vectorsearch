package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.ReadTransaction;
import com.apple.foundationdb.Transaction;
import java.util.concurrent.CompletableFuture;

/**
 * Stores mapping from nodeId -> segmentId for use during query scoring.
 */
public class NodeSegmentStorage {
  private final VectorIndexKeys keys;

  public NodeSegmentStorage(VectorIndexKeys keys) {
    this.keys = keys;
  }

  public CompletableFuture<Void> store(Transaction tr, long nodeId, long segmentId) {
    tr.set(keys.nodeSegmentKey(nodeId), com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt(segmentId));
    return CompletableFuture.completedFuture(null);
  }

  public CompletableFuture<Long> get(ReadTransaction tr, long nodeId) {
    return tr.get(keys.nodeSegmentKey(nodeId))
        .thenApply(bytes -> bytes == null ? 0L : com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt(bytes));
  }
}
