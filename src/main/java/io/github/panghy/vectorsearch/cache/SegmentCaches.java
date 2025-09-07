package io.github.panghy.vectorsearch.cache;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.github.benmanes.caffeine.cache.AsyncCacheLoader;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.Adjacency;
import io.github.panghy.vectorsearch.proto.PQCodebook;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.metrics.Meter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/**
 * Caffeine-based caches for per-segment artifacts (e.g., decoded PQ codebooks, adjacency).
 */
public final class SegmentCaches {
  private final AsyncLoadingCache<Integer, float[][][]> codebooks;
  private final AsyncLoadingCache<Long, int[]> adjacency;

  public SegmentCaches(VectorIndexConfig config, FdbDirectories.IndexDirectories dirs) {
    Database db = config.getDatabase();
    this.codebooks = Caffeine.newBuilder()
        .expireAfterAccess(10, TimeUnit.MINUTES)
        .buildAsync(new AsyncCacheLoader<>() {
          @Override
          public CompletableFuture<float[][][]> asyncLoad(Integer segId, Executor ex) {
            return db.readAsync(tr ->
                    tr.get(dirs.segmentKeys(segStr(segId)).pqCodebookKey()))
                .thenApply(bytes -> bytes == null ? null : decodeCodebook(bytes));
          }

          @Override
          public CompletableFuture<Map<Integer, float[][][]>> asyncLoadAll(
              Set<? extends Integer> keys, Executor executor) {
            List<Set<? extends Integer>> parts = chunk(keys, config.getCodebookBatchLoadSize());
            List<CompletableFuture<Map<Integer, float[][][]>>> fs = new ArrayList<>();
            for (Set<? extends Integer> part : parts) {
              fs.add(db.readAsync(tr -> {
                Map<Integer, float[][][]> out = new HashMap<>();
                for (Integer sid : part) {
                  byte[] b = tr.get(dirs.segmentKeys(segStr(sid))
                          .pqCodebookKey())
                      .join();
                  if (b != null) out.put(sid, decodeCodebook(b));
                }
                return completedFuture(out);
              }));
            }
            return CompletableFuture.allOf(fs.toArray(CompletableFuture[]::new))
                .thenApply(v -> {
                  Map<Integer, float[][][]> merged = new HashMap<>(keys.size());
                  for (var f : fs) merged.putAll(f.join());
                  return merged;
                });
          }
        });

    this.adjacency = Caffeine.newBuilder()
        .maximumSize(100_000)
        .expireAfterAccess(10, TimeUnit.MINUTES)
        .buildAsync(new AsyncCacheLoader<>() {
          @Override
          public CompletableFuture<int[]> asyncLoad(Long key, Executor ex) {
            int segId = (int) (key >> 32);
            int vecId = (int) (key & 0xffffffffL);
            return db.readAsync(tr ->
                    tr.get(dirs.segmentKeys(segStr(segId)).graphKey(vecId)))
                .thenApply(SegmentCaches::toAdj);
          }

          @Override
          public CompletableFuture<Map<Long, int[]>> asyncLoadAll(
              Set<? extends Long> keys, Executor executor) {
            List<Set<? extends Long>> parts = chunk(keys, config.getAdjacencyBatchLoadSize());
            List<CompletableFuture<Map<Long, int[]>>> fs = new ArrayList<>(parts.size());
            for (Set<? extends Long> part : parts) {
              fs.add(db.readAsync(tr -> {
                Map<Long, int[]> out = new HashMap<>(part.size());
                for (Long key : part) {
                  int segId = (int) (key >> 32);
                  int vecId = (int) (key & 0xffffffffL);
                  byte[] b = tr.get(dirs.segmentKeys(segStr(segId))
                          .graphKey(vecId))
                      .join();
                  out.put(key, toAdj(b));
                }
                return completedFuture(out);
              }));
            }
            return CompletableFuture.allOf(fs.toArray(CompletableFuture[]::new))
                .thenApply(v -> {
                  Map<Long, int[]> merged = new HashMap<>(keys.size());
                  for (var f : fs) merged.putAll(f.join());
                  return merged;
                });
          }
        });
  }

  // Visible for tests: allow injecting custom async caches
  public SegmentCaches(AsyncLoadingCache<Integer, float[][][]> codebooks, AsyncLoadingCache<Long, int[]> adjacency) {
    this.codebooks = codebooks;
    this.adjacency = adjacency;
  }

  public AsyncLoadingCache<Integer, float[][][]> getCodebookCacheAsync() {
    return codebooks;
  }

  public AsyncLoadingCache<Long, int[]> getAdjacencyCacheAsync() {
    return adjacency;
  }

  public Cache<Long, int[]> getAdjacencyCache() {
    return adjacency.synchronous();
  }

  public static long adjKey(int segId, int vecId) {
    return (((long) segId) << 32) | (vecId & 0xffffffffL);
  }

  private static String segStr(int segId) {
    return String.format("%06d", segId);
  }

  private static float[][][] decodeCodebook(byte[] bytes) {
    try {
      PQCodebook cb = PQCodebook.parseFrom(bytes);
      int m = cb.getM();
      int k = cb.getK();
      float[][][] centroids = new float[m][k][];
      for (int s = 0; s < m; s++) {
        byte[] b = cb.getCentroids(s).toByteArray();
        int subDim = b.length / (k * 4);
        centroids[s] = new float[k][subDim];
        ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        for (int ci = 0; ci < k; ci++) {
          for (int di = 0; di < subDim; di++) {
            centroids[s][ci][di] = bb.getFloat();
          }
        }
      }
      return centroids;
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  /** Registers OTel observable gauges for cache sizes and statistics. */
  public void registerMetrics(VectorIndexConfig cfg) {
    Meter meter = GlobalOpenTelemetry.getMeter("io.github.panghy.vectorsearch");
    Attributes base = buildAttrs(cfg);
    // Codebook cache
    meter.gaugeBuilder("vectorsearch.cache.size")
        .ofLongs()
        .setDescription("Estimated cache size")
        .setUnit("entries")
        .buildWithCallback(obs -> obs.record(
            codebooks.synchronous().estimatedSize(),
            base.toBuilder().put("cache", "codebook").build()));
    meter.gaugeBuilder("vectorsearch.cache.hit_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            codebooks.synchronous().stats().hitCount(),
            base.toBuilder().put("cache", "codebook").build()));
    meter.gaugeBuilder("vectorsearch.cache.miss_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            codebooks.synchronous().stats().missCount(),
            base.toBuilder().put("cache", "codebook").build()));
    meter.gaugeBuilder("vectorsearch.cache.load_success_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            codebooks.synchronous().stats().loadSuccessCount(),
            base.toBuilder().put("cache", "codebook").build()));
    meter.gaugeBuilder("vectorsearch.cache.load_failure_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            codebooks.synchronous().stats().loadFailureCount(),
            base.toBuilder().put("cache", "codebook").build()));

    // Adjacency cache
    meter.gaugeBuilder("vectorsearch.cache.size")
        .ofLongs()
        .setDescription("Estimated cache size")
        .setUnit("entries")
        .buildWithCallback(obs -> obs.record(
            adjacency.synchronous().estimatedSize(),
            base.toBuilder().put("cache", "adjacency").build()));
    meter.gaugeBuilder("vectorsearch.cache.hit_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            adjacency.synchronous().stats().hitCount(),
            base.toBuilder().put("cache", "adjacency").build()));
    meter.gaugeBuilder("vectorsearch.cache.miss_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            adjacency.synchronous().stats().missCount(),
            base.toBuilder().put("cache", "adjacency").build()));
    meter.gaugeBuilder("vectorsearch.cache.load_success_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            adjacency.synchronous().stats().loadSuccessCount(),
            base.toBuilder().put("cache", "adjacency").build()));
    meter.gaugeBuilder("vectorsearch.cache.load_failure_count")
        .ofLongs()
        .buildWithCallback(obs -> obs.record(
            adjacency.synchronous().stats().loadFailureCount(),
            base.toBuilder().put("cache", "adjacency").build()));
  }

  private static Attributes buildAttrs(VectorIndexConfig cfg) {
    AttributesBuilder b = Attributes.builder();
    for (var e : cfg.getMetricAttributes().entrySet()) b.put(e.getKey(), e.getValue());
    return b.build();
  }

  private static int[] toAdj(byte[] bytes) {
    if (bytes == null) return new int[0];
    try {
      Adjacency adj = Adjacency.parseFrom(bytes);
      int[] arr = new int[adj.getNeighborIdsCount()];
      for (int i = 0; i < arr.length; i++) arr[i] = adj.getNeighborIds(i);
      return arr;
    } catch (InvalidProtocolBufferException e) {
      throw new RuntimeException(e);
    }
  }

  private static <T> List<Set<? extends T>> chunk(Set<? extends T> keys, int size) {
    List<Set<? extends T>> out = new ArrayList<>();
    Iterator<? extends T> it = keys.iterator();
    while (it.hasNext()) {
      Set<T> part = new HashSet<>();
      for (int i = 0; i < size && it.hasNext(); i++) part.add((T) it.next());
      out.add(part);
    }
    return out;
  }
}
