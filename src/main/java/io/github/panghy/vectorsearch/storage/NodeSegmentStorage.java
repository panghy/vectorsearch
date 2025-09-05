package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.ReadTransaction;
import com.apple.foundationdb.Transaction;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

/**
 * Stores mapping from nodeId -> segmentId for use during query scoring.
 * Includes a Caffeine cache for hot lookups.
 */
public class NodeSegmentStorage {
  private final Database db;
  private final VectorIndexKeys keys;
  private final AsyncLoadingCache<Long, Long> cache;

  public NodeSegmentStorage(Database db, VectorIndexKeys keys, int maxCacheSize, Duration cacheTtl) {
    this.db = db;
    this.keys = keys;
    this.cache = Caffeine.newBuilder()
        .maximumSize(maxCacheSize)
        .expireAfterAccess(cacheTtl)
        .buildAsync((nodeId, executor) -> db.read(tr -> get(tr, nodeId)));
  }

  public CompletableFuture<Void> store(Transaction tr, long nodeId, long segmentId) {
    tr.set(keys.nodeSegmentKey(nodeId), com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt(segmentId));
    cache.synchronous().invalidate(nodeId);
    return CompletableFuture.completedFuture(null);
  }

  public CompletableFuture<Long> get(ReadTransaction tr, long nodeId) {
    return tr.get(keys.nodeSegmentKey(nodeId))
        .thenApply(bytes -> bytes == null ? 0L : com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt(bytes));
  }

  public CompletableFuture<Long> getAsync(long nodeId) {
    return cache.get(nodeId);
  }

  public void clearCache() {
    cache.synchronous().invalidateAll();
  }
}
