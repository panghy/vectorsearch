package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.allOf;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Range;
import com.github.benmanes.caffeine.cache.AsyncCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.PqCodesBlock;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Storage handler for PQ-encoded vectors in FoundationDB.
 * Manages blocked storage of PQ codes with versioning and caching.
 */
public class PqBlockStorage {
  private static final Logger LOGGER = LoggerFactory.getLogger(PqBlockStorage.class);

  private final Database db;
  private final VectorIndexKeys keys;
  private final int codesPerBlock;
  private final int pqSubvectors;

  // Cache for frequently accessed blocks using Caffeine
  private final AsyncCache<String, PqCodesBlock> blockCache;
  private static final int MAX_CACHE_SIZE = 1000;
  private static final Duration CACHE_TTL = Duration.ofMinutes(10);

  public PqBlockStorage(Database db, VectorIndexKeys keys, int codesPerBlock, int pqSubvectors) {
    this.db = db;
    this.keys = keys;
    this.codesPerBlock = codesPerBlock;
    this.pqSubvectors = pqSubvectors;
    this.blockCache = Caffeine.newBuilder()
        .maximumSize(MAX_CACHE_SIZE)
        .expireAfterWrite(CACHE_TTL)
        .buildAsync();
  }

  /**
   * Stores a single PQ-encoded vector.
   *
   * @param nodeId          the node/vector ID
   * @param pqCode          the PQ code bytes
   * @param codebookVersion version of codebook used for encoding
   * @return future completing when stored
   */
  public CompletableFuture<Void> storePqCode(long nodeId, byte[] pqCode, int codebookVersion) {
    if (pqCode.length != pqSubvectors) {
      throw new IllegalArgumentException(
          "PQ code length mismatch: expected " + pqSubvectors + ", got " + pqCode.length);
    }

    long blockNumber = VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
    int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

    return db.runAsync(tx -> {
      byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);

      return StorageTransactionUtils.readProto(tx, blockKey, PqCodesBlock.parser())
          .thenApply(existingBlock -> {
            PqCodesBlock.Builder blockBuilder;
            byte[] codes;

            if (existingBlock == null) {
              // Create new block
              long blockFirstNid = blockNumber * codesPerBlock;
              codes = new byte[codesPerBlock * pqSubvectors];

              blockBuilder = PqCodesBlock.newBuilder()
                  .setBlockFirstNid(blockFirstNid)
                  .setCodesInBlock(blockOffset + 1)
                  .setCodebookVersion(codebookVersion)
                  .setBlockVersion(1)
                  .setUpdatedAt(currentTimestamp());
            } else {
              // Update existing block
              codes = existingBlock.getCodes().toByteArray();

              // Ensure block has sufficient capacity
              if (codes.length < codesPerBlock * pqSubvectors) {
                byte[] newCodes = new byte[codesPerBlock * pqSubvectors];
                System.arraycopy(codes, 0, newCodes, 0, codes.length);
                codes = newCodes;
              }

              blockBuilder = existingBlock.toBuilder()
                  .setCodesInBlock(Math.max(existingBlock.getCodesInBlock(), blockOffset + 1))
                  .setBlockVersion(existingBlock.getBlockVersion() + 1)
                  .setUpdatedAt(currentTimestamp());
            }

            // Write PQ code at the correct offset
            System.arraycopy(pqCode, 0, codes, blockOffset * pqSubvectors, pqSubvectors);

            blockBuilder.setCodes(ByteString.copyFrom(codes));
            PqCodesBlock updatedBlock = blockBuilder.build();

            tx.set(blockKey, updatedBlock.toByteArray());

            // Update cache
            blockCache.put(
                blockCacheKey(codebookVersion, blockNumber),
                CompletableFuture.completedFuture(updatedBlock));

            return null;
          });
    });
  }

  /**
   * Batch stores multiple PQ codes.
   *
   * @param nodeIds         list of node IDs
   * @param pqCodes         list of PQ code arrays
   * @param codebookVersion codebook version
   * @return future completing when all stored
   */
  public CompletableFuture<Void> batchStorePqCodes(List<Long> nodeIds, List<byte[]> pqCodes, int codebookVersion) {
    if (nodeIds.size() != pqCodes.size()) {
      throw new IllegalArgumentException("Node IDs and PQ codes count mismatch");
    }

    // Group by block for efficiency
    Map<Long, List<Integer>> blockGroups = groupByBlocks(nodeIds);

    return db.runAsync(tx -> {
      List<CompletableFuture<Void>> futures = new ArrayList<>();

      for (Map.Entry<Long, List<Integer>> entry : blockGroups.entrySet()) {
        long blockNumber = entry.getKey();
        List<Integer> indices = entry.getValue();

        byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);

        futures.add(StorageTransactionUtils.readProto(tx, blockKey, PqCodesBlock.parser())
            .thenApply(existingBlock -> {
              long blockFirstNid = blockNumber * codesPerBlock;
              byte[] codes;
              int maxOffset = 0;

              if (existingBlock == null) {
                codes = new byte[codesPerBlock * pqSubvectors];
              } else {
                codes = existingBlock.getCodes().toByteArray();
                if (codes.length < codesPerBlock * pqSubvectors) {
                  byte[] newCodes = new byte[codesPerBlock * pqSubvectors];
                  System.arraycopy(codes, 0, newCodes, 0, codes.length);
                  codes = newCodes;
                }
                maxOffset = existingBlock.getCodesInBlock();
              }

              // Write all codes for this block
              for (int idx : indices) {
                long nodeId = nodeIds.get(idx);
                byte[] pqCode = pqCodes.get(idx);
                int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

                System.arraycopy(pqCode, 0, codes, blockOffset * pqSubvectors, pqSubvectors);
                maxOffset = Math.max(maxOffset, blockOffset + 1);
              }

              PqCodesBlock.Builder blockBuilder = PqCodesBlock.newBuilder()
                  .setBlockFirstNid(blockFirstNid)
                  .setCodesInBlock(maxOffset)
                  .setCodes(ByteString.copyFrom(codes))
                  .setCodebookVersion(codebookVersion)
                  .setBlockVersion(existingBlock != null ? existingBlock.getBlockVersion() + 1 : 1)
                  .setUpdatedAt(currentTimestamp());

              PqCodesBlock updatedBlock = blockBuilder.build();
              tx.set(blockKey, updatedBlock.toByteArray());

              // Update cache
              blockCache.put(
                  blockCacheKey(codebookVersion, blockNumber),
                  CompletableFuture.completedFuture(updatedBlock));

              return null;
            }));
      }
      return allOf(futures.toArray(CompletableFuture[]::new));
    });
  }

  private Map<Long, List<Integer>> groupByBlocks(List<Long> nodeIds) {
    Map<Long, List<Integer>> blockGroups = new ConcurrentHashMap<>();
    for (int i = 0; i < nodeIds.size(); i++) {
      long blockNumber = VectorIndexKeys.blockNumber(nodeIds.get(i), codesPerBlock);
      blockGroups.computeIfAbsent(blockNumber, k -> new ArrayList<>()).add(i);
    }
    return blockGroups;
  }

  /**
   * Loads a single PQ code.
   *
   * @param nodeId          the node ID
   * @param codebookVersion codebook version
   * @return PQ code bytes or null if not found
   */
  public CompletableFuture<byte[]> loadPqCode(long nodeId, int codebookVersion) {
    long blockNumber = VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
    int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

    // Check cache first
    String cacheKey = blockCacheKey(codebookVersion, blockNumber);
    CompletableFuture<PqCodesBlock> cachedFuture = blockCache.getIfPresent(cacheKey);
    if (cachedFuture != null) {
      return cachedFuture.thenApply(cached -> {
        if (cached != null && blockOffset < cached.getCodesInBlock()) {
          return extractCode(cached, blockOffset);
        }
        return null;
      });
    }

    return db.runAsync(tx -> {
      byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);

      return StorageTransactionUtils.readProto(tx, blockKey, PqCodesBlock.parser())
          .thenApply(block -> {
            if (block == null || blockOffset >= block.getCodesInBlock()) {
              return null;
            }

            // Cache the block
            blockCache.put(cacheKey, CompletableFuture.completedFuture(block));

            return extractCode(block, blockOffset);
          });
    });
  }

  /**
   * Batch loads multiple PQ codes.
   *
   * @param nodeIds         list of node IDs
   * @param codebookVersion codebook version
   * @return list of PQ codes (null for missing nodes)
   */
  public CompletableFuture<List<byte[]>> batchLoadPqCodes(List<Long> nodeIds, int codebookVersion) {

    // Group by block
    Map<Long, List<Integer>> blockGroups = groupByBlocks(nodeIds);

    return db.runAsync(tx -> {
      Map<Long, CompletableFuture<PqCodesBlock>> blockFutures = new ConcurrentHashMap<>();

      // Load unique blocks
      for (long blockNumber : blockGroups.keySet()) {
        String cacheKey = blockCacheKey(codebookVersion, blockNumber);
        CompletableFuture<PqCodesBlock> cachedFuture = blockCache.getIfPresent(cacheKey);

        if (cachedFuture != null) {
          blockFutures.put(blockNumber, cachedFuture);
        } else {
          byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);
          CompletableFuture<PqCodesBlock> future = StorageTransactionUtils.readProto(
                  tx, blockKey, PqCodesBlock.parser())
              .thenApply(block -> {
                if (block != null) {
                  // Cache the block
                  blockCache.put(cacheKey, CompletableFuture.completedFuture(block));
                }
                return block;
              });
          blockFutures.put(blockNumber, future);
        }
      }

      // Extract codes from blocks
      List<byte[]> results = new ArrayList<>(nodeIds.size());
      for (int i = 0; i < nodeIds.size(); i++) {
        results.add(null); // Initialize with nulls
      }

      List<CompletableFuture<Void>> extractFutures = new ArrayList<>();

      for (Map.Entry<Long, List<Integer>> entry : blockGroups.entrySet()) {
        long blockNumber = entry.getKey();
        List<Integer> indices = entry.getValue();

        extractFutures.add(blockFutures.get(blockNumber).thenAccept(block -> {
          if (block != null) {
            for (int idx : indices) {
              long nodeId = nodeIds.get(idx);
              int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

              if (blockOffset < block.getCodesInBlock()) {
                results.set(idx, extractCode(block, blockOffset));
              }
            }
          }
        }));
      }

      return allOf(extractFutures.toArray(CompletableFuture[]::new)).thenApply(v -> results);
    });
  }

  /**
   * Deletes a PQ code.
   *
   * @param nodeId          the node ID
   * @param codebookVersion codebook version
   * @return future completing when deleted
   */
  public CompletableFuture<Void> deletePqCode(long nodeId, int codebookVersion) {
    // For now, we just zero out the code in the block
    // A more sophisticated implementation would track deleted slots for reuse
    byte[] emptyCode = new byte[pqSubvectors];
    return storePqCode(nodeId, emptyCode, codebookVersion);
  }

  /**
   * Migrates PQ codes from one codebook version to another.
   *
   * @param fromVersion source codebook version
   * @param toVersion   target codebook version
   * @param reencoder   function to re-encode codes
   * @return future completing when migration is done
   */
  public CompletableFuture<Void> migrateVersion(int fromVersion, int toVersion, CodeReencoder reencoder) {
    return db.runAsync(tx -> {
          byte[] prefix = keys.pqBlockPrefixForVersion(fromVersion);
          Range range = Range.startsWith(prefix);

          return StorageTransactionUtils.readProtoRange(tx, range, PqCodesBlock.parser())
              .thenAccept(blocks -> {
                for (PqCodesBlock block : blocks) {
                  byte[] oldCodes = block.getCodes().toByteArray();
                  byte[] newCodes = new byte[oldCodes.length];

                  // Re-encode each code in the block
                  for (int i = 0; i < block.getCodesInBlock(); i++) {
                    byte[] oldCode = new byte[pqSubvectors];
                    System.arraycopy(oldCodes, i * pqSubvectors, oldCode, 0, pqSubvectors);

                    byte[] newCode = reencoder.reencode(block.getBlockFirstNid() + i, oldCode);

                    System.arraycopy(newCode, 0, newCodes, i * pqSubvectors, pqSubvectors);
                  }

                  // Write new block
                  PqCodesBlock newBlock = block.toBuilder()
                      .setCodes(ByteString.copyFrom(newCodes))
                      .setCodebookVersion(toVersion)
                      .setBlockVersion(1)
                      .setUpdatedAt(currentTimestamp())
                      .build();

                  byte[] newKey =
                      keys.pqBlockKey(toVersion, block.getBlockFirstNid() / codesPerBlock);

                  tx.set(newKey, newBlock.toByteArray());
                }
              });
        })
        .thenRun(() -> LOGGER.info("Migrated PQ codes from version {} to {}", fromVersion, toVersion));
  }

  /**
   * Deletes all PQ blocks for a specific version.
   *
   * @param version codebook version
   * @return future completing when deleted
   */
  public CompletableFuture<Void> deleteVersion(int version) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.pqBlockPrefixForVersion(version);
      Range range = Range.startsWith(prefix);

      StorageTransactionUtils.clearRange(tx, range);

      // Clear cache entries for this version
      // Note: Caffeine doesn't provide direct access to keys, so we invalidate all
      // In production, might want to track keys separately or use a more sophisticated approach
      blockCache.synchronous().invalidateAll();

      LOGGER.info("Deleted all PQ blocks for version {}", version);

      return CompletableFuture.completedFuture(null);
    });
  }

  /**
   * Gets statistics for PQ block storage.
   *
   * @param version codebook version
   * @return storage statistics
   */
  public CompletableFuture<StorageStats> getStats(int version) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.pqBlockPrefixForVersion(version);
      Range range = Range.startsWith(prefix);

      return StorageTransactionUtils.countRange(tx, range).thenCompose(blockCount -> {
        // Sample first block to estimate storage
        return StorageTransactionUtils.readRange(tx, range, 1).thenApply(sample -> {
          long totalCodes = 0;
          long storageBytes = 0;

          if (!sample.isEmpty()) {
            try {
              PqCodesBlock firstBlock =
                  PqCodesBlock.parseFrom(sample.get(0).getValue());
              storageBytes = blockCount * sample.get(0).getValue().length;

              // Estimate total codes (assuming uniform distribution)
              totalCodes = blockCount * firstBlock.getCodesInBlock();
            } catch (Exception e) {
              LOGGER.warn("Failed to parse sample block", e);
            }
          }

          return new StorageStats(blockCount, totalCodes, storageBytes);
        });
      });
    });
  }

  /**
   * Clears the block cache.
   */
  public void clearCache() {
    blockCache.synchronous().invalidateAll();
    LOGGER.debug("Cleared PQ block cache");
  }

  // Helper methods

  private byte[] extractCode(PqCodesBlock block, int blockOffset) {
    byte[] codes = block.getCodes().toByteArray();
    byte[] pqCode = new byte[pqSubvectors];
    System.arraycopy(codes, blockOffset * pqSubvectors, pqCode, 0, pqSubvectors);
    return pqCode;
  }

  private String blockCacheKey(int version, long blockNumber) {
    return version + ":" + blockNumber;
  }

  // updateCache method removed - Caffeine handles eviction automatically

  private Timestamp currentTimestamp() {
    long millis = System.currentTimeMillis();
    return Timestamp.newBuilder()
        .setSeconds(millis / 1000)
        .setNanos((int) ((millis % 1000) * 1_000_000))
        .build();
  }

  /**
   * Interface for re-encoding PQ codes during migration.
   */
  public interface CodeReencoder {
    byte[] reencode(long nodeId, byte[] oldCode);
  }

  /**
   * Storage statistics.
   */
  public record StorageStats(long blockCount, long totalCodes, long storageBytes) {}
}
