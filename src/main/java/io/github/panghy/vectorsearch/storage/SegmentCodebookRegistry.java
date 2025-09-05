package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Range;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.pq.VectorUtils;
import io.github.panghy.vectorsearch.proto.CodebookSub;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Segment-scoped codebook registry with ProductQuantizer cache.
 */
public class SegmentCodebookRegistry {
  private static final Logger LOGGER = LoggerFactory.getLogger(SegmentCodebookRegistry.class);

  private final Database db;
  private final VectorIndexKeys keys;
  private final int dimension;
  private final int numSubvectors;
  private final DistanceMetrics.Metric metric;
  private final AsyncLoadingCache<Long, ProductQuantizer> pqCache;

  public SegmentCodebookRegistry(
      Database db, VectorIndexKeys keys, int dimension, int numSubvectors, DistanceMetrics.Metric metric) {
    this(db, keys, dimension, numSubvectors, metric, 100, Duration.ofMinutes(10), ForkJoinPool.commonPool());
  }

  public SegmentCodebookRegistry(
      Database db,
      VectorIndexKeys keys,
      int dimension,
      int numSubvectors,
      DistanceMetrics.Metric metric,
      int maxCacheSize,
      Duration cacheTtl,
      Executor executor) {
    this.db = db;
    this.keys = keys;
    this.dimension = dimension;
    this.numSubvectors = numSubvectors;
    this.metric = metric;
    this.pqCache = Caffeine.newBuilder()
        .maximumSize(maxCacheSize)
        .expireAfterAccess(cacheTtl)
        .executor(executor)
        .buildAsync(this::loadProductQuantizer);
  }

  public CompletableFuture<Void> storeSegmentCodebooks(long segmentId, float[][][] codebooks, TrainingStats stats) {
    return db.runAsync(tx -> {
      for (int s = 0; s < codebooks.length; s++) {
        byte[] key = keys.segmentCodebookKey(segmentId, s);
        byte[] fp16 = VectorUtils.toFloat16Bytes(flatten(codebooks[s]));
        CodebookSub.Builder builder = CodebookSub.newBuilder()
            .setSubspaceIndex(s)
            .setVersion(stats.version)
            .setCodewordsFp16(ByteString.copyFrom(fp16))
            .setTrainedOnVectors(stats.trainedOnVectors)
            .setCreatedAt(now());
        if (stats.quantizationError != null) builder.setQuantizationError(stats.quantizationError);
        tx.set(key, builder.build().toByteArray());
      }
      pqCache.synchronous().invalidate(segmentId);
      LOGGER.info("Stored {} codebook subspaces for segment {}", codebooks.length, segmentId);
      return completedFuture(null);
    });
  }

  public CompletableFuture<ProductQuantizer> getProductQuantizer(long segmentId) {
    return pqCache.get(segmentId);
  }

  private CompletableFuture<ProductQuantizer> loadProductQuantizer(Long segmentId, Executor executor) {
    return loadCodebooks(segmentId).thenApply(result -> {
      if (result == null) return null;
      int version = (int) result.version;
      return new ProductQuantizer(dimension, numSubvectors, metric, version, result.codebooks);
    });
  }

  record Loaded(long version, float[][][] codebooks) {}

  CompletableFuture<Loaded> loadCodebooks(long segmentId) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.segmentCodebookPrefix(segmentId);
      Range range = Range.startsWith(prefix);
      return StorageTransactionUtils.readProtoRange(tx, range, CodebookSub.parser())
          .thenApply(subs -> {
            if (subs.isEmpty()) return null;
            int m = subs.size();
            float[][][] codebooks = new float[m][][];
            long version = 0;
            for (CodebookSub sub : subs) {
              int idx = sub.getSubspaceIndex();
              byte[] fp16 = sub.getCodewordsFp16().toByteArray();
              float[] flat = VectorUtils.fromFloat16Bytes(fp16);
              int numCentroids = 256;
              int subDim = flat.length / numCentroids;
              codebooks[idx] = reshape(flat, numCentroids, subDim);
              version = sub.getVersion();
            }
            return new Loaded(version, codebooks);
          });
    });
  }

  private static float[] flatten(float[][] array) {
    int rows = array.length;
    int cols = array[0].length;
    float[] flat = new float[rows * cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(array[i], 0, flat, i * cols, cols);
    }
    return flat;
  }

  private static float[][] reshape(float[] flat, int rows, int cols) {
    float[][] arr = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      System.arraycopy(flat, i * cols, arr[i], 0, cols);
    }
    return arr;
  }

  private static Timestamp now() {
    long ms = System.currentTimeMillis();
    return Timestamp.newBuilder()
        .setSeconds(ms / 1000)
        .setNanos((int) ((ms % 1000) * 1_000_000))
        .build();
  }

  public record TrainingStats(long version, long trainedOnVectors, Double quantizationError) {}
}
