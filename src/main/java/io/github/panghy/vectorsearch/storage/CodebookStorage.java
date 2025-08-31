package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Range;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.pq.VectorUtils;
import io.github.panghy.vectorsearch.proto.CodebookSub;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
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
  private final Map<Long, float[][][]> codebookCache;

  public CodebookStorage(Database db, VectorIndexKeys keys) {
    this.db = db;
    this.keys = keys;
    this.codebookCache = new ConcurrentHashMap<>();
  }

  /**
   * Stores a complete set of codebooks for a version.
   *
   * @param version       codebook version
   * @param codebooks     array of codebooks [numSubvectors][numCentroids][subDimension]
   * @param trainingStats training statistics (vectors used, error, etc.)
   * @return future completing when all codebooks are stored
   */
  public CompletableFuture<Void> storeCodebooks(long version, float[][][] codebooks, TrainingStats trainingStats) {
    return db.runAsync(tx -> {
      for (int i = 0; i < codebooks.length; i++) {
        final int subspaceIndex = i;
        byte[] key = keys.codebookKey(version, subspaceIndex);

        // Convert to fp16 for storage efficiency
        byte[] fp16Data = VectorUtils.toFloat16Bytes(flatten(codebooks[i]));

        // Build protobuf message
        CodebookSub.Builder builder = CodebookSub.newBuilder()
            .setSubspaceIndex(subspaceIndex)
            .setVersion(version)
            .setCodewordsFp16(ByteString.copyFrom(fp16Data))
            .setTrainedOnVectors(trainingStats.trainedOnVectors)
            .setCreatedAt(currentTimestamp());

        if (trainingStats.quantizationError != null) {
          builder.setQuantizationError(trainingStats.quantizationError);
        }

        tx.set(key, builder.build().toByteArray());
      }

      // Update cache
      codebookCache.put(version, codebooks);

      LOGGER.info("Stored {} codebook subspaces for version {}", codebooks.length, version);

      return CompletableFuture.completedFuture(null);
    });
  }

  /**
   * Loads all codebooks for a specific version.
   *
   * @param version codebook version to load
   * @return array of codebooks or null if version doesn't exist
   */
  public CompletableFuture<float[][][]> loadCodebooks(long version) {
    // Check cache first
    float[][][] cached = codebookCache.get(version);
    if (cached != null) {
      return CompletableFuture.completedFuture(cached);
    }

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

            // Cache for future use
            codebookCache.put(version, codebooks);

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
        // Store as 4-byte integer
        return bytesToInt(value);
      });
    });
  }

  /**
   * Sets the active codebook version atomically.
   *
   * @param version the version to activate
   * @return future completing when version is set
   */
  public CompletableFuture<Void> setActiveVersion(int version) {
    return db.runAsync(tx -> {
      byte[] key = keys.activeCodebookVersionKey();
      tx.set(key, intToBytes(version));

      LOGGER.info("Set active codebook version to {}", version);
      return CompletableFuture.completedFuture(null);
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
      codebookCache.remove(version);

      LOGGER.info("Deleted codebook version {}", version);

      return CompletableFuture.completedFuture(null);
    });
  }

  /**
   * Clears the in-memory codebook cache.
   */
  public void clearCache() {
    codebookCache.clear();
    LOGGER.debug("Cleared codebook cache");
  }

  /**
   * Gets cache statistics.
   */
  public CacheStats getCacheStats() {
    return new CacheStats(codebookCache.size(), estimateCacheSize());
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

  private byte[] intToBytes(int value) {
    // Little-endian: least significant byte first (consistent with FDB's atomic operations)
    return new byte[] {(byte) value, (byte) (value >>> 8), (byte) (value >>> 16), (byte) (value >>> 24)};
  }

  private int bytesToInt(byte[] bytes) {
    // Little-endian: least significant byte first (consistent with FDB's atomic operations)
    return (bytes[0] & 0xFF) | ((bytes[1] & 0xFF) << 8) | ((bytes[2] & 0xFF) << 16) | ((bytes[3] & 0xFF) << 24);
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

  private long estimateCacheSize() {
    long size = 0;
    for (float[][][] codebooks : codebookCache.values()) {
      for (float[][] codebook : codebooks) {
        for (float[] centroids : codebook) {
          size += centroids.length * 4; // 4 bytes per float
        }
      }
    }
    return size;
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
    public final long estimatedSizeBytes;

    public CacheStats(int entries, long estimatedSizeBytes) {
      this.entries = entries;
      this.estimatedSizeBytes = estimatedSizeBytes;
    }
  }
}
