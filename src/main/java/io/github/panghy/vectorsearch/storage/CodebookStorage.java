package io.github.panghy.vectorsearch.storage;

import static com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt;
import static com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt;

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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Storage handler for PQ codebooks in FoundationDB.
 * Manages versioned codebooks with atomic rotation support.
 */
public class CodebookStorage {
  private static final Logger LOGGER = LoggerFactory.getLogger(CodebookStorage.class);

  private final Database db;
  private final VectorIndexKeys keys;
  private final int dimension;
  private final int numSubvectors;
  private final DistanceMetrics.Metric metric;
  private final AsyncLoadingCache<Long, ProductQuantizer> productQuantizerCache;

  public CodebookStorage(
      Database db, VectorIndexKeys keys, int dimension, int numSubvectors, DistanceMetrics.Metric metric) {
    this(
        db,
        keys,
        dimension,
        numSubvectors,
        metric,
        100, // Default max cache size
        Duration.ofMinutes(10), // Default cache expiry
        ForkJoinPool.commonPool());
  }

  public CodebookStorage(
      Database db,
      VectorIndexKeys keys,
      int dimension,
      int numSubvectors,
      DistanceMetrics.Metric metric,
      int maxCacheSize,
      Duration cacheExpiry,
      Executor executor) {
    this.db = db;
    this.keys = keys;
    this.dimension = dimension;
    this.numSubvectors = numSubvectors;
    this.metric = metric;

    // Initialize Caffeine cache for ProductQuantizer instances
    this.productQuantizerCache = Caffeine.newBuilder()
        .maximumSize(maxCacheSize)
        .expireAfterAccess(cacheExpiry)
        .executor(executor)
        .buildAsync(this::loadProductQuantizer);
  }

  /**
   * Stores a complete set of codebooks for a version.
   *
   * @param version       codebook version
   * @param codebooks     array of codebooks [numSubvectors][numCentroids][subDimension]
   * @param trainingStats training statistics (vectors used, error, etc.)
   * @return future completing when all codebooks are stored
   * @throws IllegalStateException if the version already exists
   */
  public CompletableFuture<Void> storeCodebooks(long version, float[][][] codebooks, TrainingStats trainingStats) {
    return db.runAsync(tx -> {
      // Check if this version already exists. FDB's transaction isolation ensures this
      // check-then-set operation is atomic within the transaction. If two transactions
      // try to insert the same version, one will succeed and the other will retry and
      // see the existing version.
      byte[] firstKey = keys.codebookKey(version, 0);
      return tx.get(firstKey).thenCompose(existingValue -> {
        if (existingValue != null) {
          throw new IllegalStateException(
              "Codebook version " + version + " already exists. Cannot overwrite existing codebooks.");
        }

        // Store all codebook subspaces
        for (int i = 0; i < codebooks.length; i++) {
          byte[] key = keys.codebookKey(version, i);

          // Convert to fp16 for storage efficiency
          byte[] fp16Data = VectorUtils.toFloat16Bytes(flatten(codebooks[i]));

          // Build protobuf message
          CodebookSub.Builder builder = CodebookSub.newBuilder()
              .setSubspaceIndex(i)
              .setVersion(version)
              .setCodewordsFp16(ByteString.copyFrom(fp16Data))
              .setTrainedOnVectors(trainingStats.trainedOnVectors)
              .setCreatedAt(currentTimestamp());

          if (trainingStats.quantizationError != null) {
            builder.setQuantizationError(trainingStats.quantizationError);
          }

          tx.set(key, builder.build().toByteArray());
        }

        // Invalidate cache for this version to force reload
        productQuantizerCache.synchronous().invalidate(version);

        LOGGER.info("Stored {} codebook subspaces for version {}", codebooks.length, version);

        return CompletableFuture.completedFuture(null);
      });
    });
  }

  /**
   * Gets a ProductQuantizer for the specified version.
   * Uses cache for efficiency.
   *
   * @param version codebook version
   * @return ProductQuantizer instance or null if version doesn't exist
   */
  public CompletableFuture<ProductQuantizer> getProductQuantizer(long version) {
    return productQuantizerCache.get(version);
  }

  /**
   * Gets the ProductQuantizer for the latest/active version.
   *
   * @return ProductQuantizer for active version or null if no active version
   */
  public CompletableFuture<ProductQuantizer> getLatestProductQuantizer() {
    return getActiveVersion().thenCompose(activeVersion -> {
      if (activeVersion < 0) {
        return CompletableFuture.completedFuture(null);
      }
      return getProductQuantizer(activeVersion);
    });
  }

  /**
   * Loads a ProductQuantizer for a specific version.
   * This is the cache loader function.
   */
  private CompletableFuture<ProductQuantizer> loadProductQuantizer(Long version, Executor executor) {
    return loadCodebooks(version).thenApply(codebooks -> {
      if (codebooks == null) {
        return null;
      }
      return new ProductQuantizer(dimension, numSubvectors, metric, version.intValue(), codebooks);
    });
  }

  /**
   * Loads all codebooks for a specific version.
   *
   * @param version codebook version to load
   * @return array of codebooks or null if version doesn't exist
   */
  CompletableFuture<float[][][]> loadCodebooks(long version) {
    return db.runAsync(tx -> {
      // Read all codebook subspaces for this version
      byte[] prefix = keys.codebookPrefixForVersion(version);
      Range range = Range.startsWith(prefix);

      return StorageTransactionUtils.readProtoRange(tx, range, CodebookSub.parser())
          .thenApply(codebookSubs -> {
            if (codebookSubs.isEmpty()) {
              return null;
            }

            // Reconstruct codebooks array
            int numSubvectors = codebookSubs.size();
            float[][][] codebooks = new float[numSubvectors][][];

            for (CodebookSub sub : codebookSubs) {
              int idx = sub.getSubspaceIndex();
              if (idx >= numSubvectors) {
                throw new IllegalStateException(
                    "Invalid subspace index: " + idx + " >= " + numSubvectors);
              }

              // Convert from fp16 back to float
              byte[] fp16Data = sub.getCodewordsFp16().toByteArray();
              float[] flat = VectorUtils.fromFloat16Bytes(fp16Data);

              // Reshape to [numCentroids][subDimension]
              int numCentroids = 256; // Always 256 for 8-bit PQ
              int subDimension = flat.length / numCentroids;

              codebooks[idx] = reshape(flat, numCentroids, subDimension);
            }

            LOGGER.debug("Loaded {} codebook subspaces for version {}", numSubvectors, version);

            return codebooks;
          });
    });
  }

  /**
   * Gets the active codebook version.
   *
   * @return active version or -1 if not set
   */
  public CompletableFuture<Integer> getActiveVersion() {
    return db.runAsync(tx -> {
      byte[] key = keys.activeCodebookVersionKey();
      return tx.get(key).thenApply(value -> {
        if (value == null) {
          return -1;
        }
        // Decode as long integer
        return (int) decodeInt(value);
      });
    });
  }

  /**
   * Sets the active codebook version atomically.
   * Verifies that the codebook version exists before setting it as active.
   *
   * @param version the version to activate
   * @return future completing when version is set
   * @throws IllegalArgumentException if the codebook version does not exist
   */
  public CompletableFuture<Void> setActiveVersion(int version) {
    return db.runAsync(tx -> {
      // First check if the codebook version exists by checking for the first subvector
      byte[] codebookKey = keys.codebookKey(version, 0);
      return tx.get(codebookKey).thenCompose(codebookValue -> {
        if (codebookValue == null) {
          throw new IllegalArgumentException("Codebook version " + version + " does not exist");
        }

        // Set the active version
        byte[] key = keys.activeCodebookVersionKey();
        tx.set(key, encodeInt(version));

        LOGGER.info("Set active codebook version to {}", version);
        return CompletableFuture.completedFuture(null);
      });
    });
  }

  /**
   * Lists all available codebook versions.
   *
   * @return list of versions in ascending order
   */
  public CompletableFuture<List<Integer>> listVersions() {
    return db.runAsync(tx -> {
      byte[] prefix = keys.allCodebooksPrefix();
      Range range = Range.startsWith(prefix);

      return StorageTransactionUtils.readRange(tx, range, 1000).thenApply(kvs -> {
        // Extract unique versions from keys
        List<Integer> versions = new ArrayList<>();
        int lastVersion = -1;

        for (var kv : kvs) {
          // Parse version from key structure
          // Key format: .../pq/codebook/{version}/{subspace}
          byte[] key = kv.getKey();
          int version = parseVersionFromKey(key, prefix);

          if (version != lastVersion) {
            versions.add(version);
            lastVersion = version;
          }
        }

        return versions;
      });
    });
  }

  /**
   * Deletes a specific codebook version.
   *
   * @param version the version to delete
   * @return future completing when deletion is done
   */
  public CompletableFuture<Void> deleteVersion(long version) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.codebookPrefixForVersion(version);
      Range range = Range.startsWith(prefix);

      StorageTransactionUtils.clearRange(tx, range);

      // Remove from cache
      productQuantizerCache.synchronous().invalidate(version);

      LOGGER.info("Deleted codebook version {}", version);

      return CompletableFuture.completedFuture(null);
    });
  }

  /**
   * Clears the in-memory ProductQuantizer cache.
   */
  public void clearCache() {
    productQuantizerCache.synchronous().invalidateAll();
    LOGGER.debug("Cleared ProductQuantizer cache");
  }

  /**
   * Gets cache statistics.
   */
  public CacheStats getCacheStats() {
    var stats = productQuantizerCache.synchronous().stats();
    return new CacheStats(
        (int) productQuantizerCache.synchronous().estimatedSize(),
        stats.hitCount(),
        stats.missCount(),
        stats.hitRate());
  }

  /**
   * Gets the ProductQuantizer cache for testing.
   */
  AsyncLoadingCache<Long, ProductQuantizer> getProductQuantizerCache() {
    return productQuantizerCache;
  }

  // Helper methods

  private float[] flatten(float[][] array) {
    int rows = array.length;
    int cols = array[0].length;
    float[] flat = new float[rows * cols];

    for (int i = 0; i < rows; i++) {
      System.arraycopy(array[i], 0, flat, i * cols, cols);
    }

    return flat;
  }

  private float[][] reshape(float[] flat, int rows, int cols) {
    float[][] array = new float[rows][cols];

    for (int i = 0; i < rows; i++) {
      System.arraycopy(flat, i * cols, array[i], 0, cols);
    }

    return array;
  }

  private int parseVersionFromKey(byte[] key, byte[] prefix) {
    // The key structure after prefix is: {version}/{subspace}
    // We need to extract the version from the tuple
    try {
      // Get the part after the prefix
      byte[] tupleBytes = new byte[key.length - prefix.length];
      System.arraycopy(key, prefix.length, tupleBytes, 0, tupleBytes.length);

      // Parse the tuple
      com.apple.foundationdb.tuple.Tuple tuple = com.apple.foundationdb.tuple.Tuple.fromBytes(tupleBytes);
      if (tuple.size() > 0) {
        return (int) tuple.getLong(0);
      }
    } catch (Exception e) {
      // Ignore parsing errors
    }
    return -1;
  }

  private Timestamp currentTimestamp() {
    long millis = System.currentTimeMillis();
    return Timestamp.newBuilder()
        .setSeconds(millis / 1000)
        .setNanos((int) ((millis % 1000) * 1_000_000))
        .build();
  }

  /**
   * Training statistics for codebook creation.
   */
  public static class TrainingStats {
    public final long trainedOnVectors;
    public final Double quantizationError;

    public TrainingStats(long trainedOnVectors, Double quantizationError) {
      this.trainedOnVectors = trainedOnVectors;
      this.quantizationError = quantizationError;
    }
  }

  /**
   * Cache statistics.
   */
  public static class CacheStats {
    public final int entries;
    public final long hitCount;
    public final long missCount;
    public final double hitRate;

    public CacheStats(int entries, long hitCount, long missCount, double hitRate) {
      this.entries = entries;
      this.hitCount = hitCount;
      this.missCount = missCount;
      this.hitRate = hitRate;
    }
  }
}
