package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;
import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProtoRange;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.Transaction;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.ByteString;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.PqCodesBlock;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.BitSet;
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
  private final InstantSource instantSource;

  // Cache for frequently accessed blocks using Caffeine
  private final AsyncLoadingCache<BlockCacheKey, PqCodesBlock> blockCache;

  public PqBlockStorage(
      Database db,
      VectorIndexKeys keys,
      int codesPerBlock,
      int pqSubvectors,
      InstantSource instantSource,
      int maxCacheSize,
      Duration cacheTtl) {
    this.db = db;
    this.keys = keys;
    this.codesPerBlock = codesPerBlock;
    this.pqSubvectors = pqSubvectors;
    this.instantSource = instantSource;
    this.blockCache = Caffeine.newBuilder()
        .maximumSize(maxCacheSize)
        .refreshAfterWrite(cacheTtl)
        .buildAsync((key, executor) -> loadBlock(key));
  }

  /**
   * Gets the block number for a given node ID.
   *
   * @param nodeId the node ID
   * @return the block number
   */
  public long getBlockNumber(long nodeId) {
    return VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
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
    return db.runAsync(tx -> storePqCode(tx, nodeId, pqCode, codebookVersion));
  }

  /**
   * Stores a single PQ-encoded vector within an existing transaction.
   *
   * @param tx              the transaction to use
   * @param nodeId          the node/vector ID
   * @param pqCode          the PQ code bytes
   * @param codebookVersion version of codebook used for encoding
   * @return future completing when stored
   */
  public CompletableFuture<Void> storePqCode(Transaction tx, long nodeId, byte[] pqCode, int codebookVersion) {
    // For single stores, delegate to batch method to ensure proper synchronization
    return batchStorePqCodesInTransaction(tx, List.of(nodeId), List.of(pqCode), codebookVersion);
  }

  /**
   * Batch stores multiple PQ codes within a transaction.
   * This method ensures proper handling when multiple codes map to the same block.
   */
  public CompletableFuture<Void> batchStorePqCodesInTransaction(
      Transaction tx, List<Long> nodeIds, List<byte[]> pqCodes, int codebookVersion) {
    if (nodeIds.size() != pqCodes.size()) {
      throw new IllegalArgumentException("Node IDs and PQ codes count mismatch");
    }

    // Group by block for efficiency
    Map<Long, List<Integer>> blockGroups = groupByBlocks(nodeIds);

    // Process each block only once
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (Map.Entry<Long, List<Integer>> entry : blockGroups.entrySet()) {
      long blockNumber = entry.getKey();
      List<Integer> indices = entry.getValue();

      byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);

      futures.add(readProto(tx, blockKey, PqCodesBlock.parser()).thenApply(existingBlock -> {
        long blockFirstNid = blockNumber * codesPerBlock;
        byte[] codes;
        BitSet writtenOffsets;
        int maxOffset = 0;

        if (existingBlock == null) {
          codes = new byte[codesPerBlock * pqSubvectors];
          writtenOffsets = new BitSet(codesPerBlock);
        } else {
          codes = existingBlock.getCodes().toByteArray();
          if (codes.length < codesPerBlock * pqSubvectors) {
            byte[] newCodes = new byte[codesPerBlock * pqSubvectors];
            System.arraycopy(codes, 0, newCodes, 0, codes.length);
            codes = newCodes;
          }
          maxOffset = existingBlock.getCodesInBlock();

          // Load existing bitmap
          if (existingBlock.hasWrittenOffsets()
              && !existingBlock.getWrittenOffsets().isEmpty()) {
            writtenOffsets =
                BitSet.valueOf(existingBlock.getWrittenOffsets().toByteArray());
          } else {
            writtenOffsets = new BitSet(codesPerBlock);
          }
        }

        // Write all codes for this block
        for (int idx : indices) {
          long nodeId = nodeIds.get(idx);
          byte[] pqCode = pqCodes.get(idx);
          int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

          System.arraycopy(pqCode, 0, codes, blockOffset * pqSubvectors, pqSubvectors);
          writtenOffsets.set(blockOffset);
          maxOffset = Math.max(maxOffset, blockOffset + 1);
        }

        PqCodesBlock.Builder blockBuilder = PqCodesBlock.newBuilder()
            .setBlockFirstNid(blockFirstNid)
            .setCodesInBlock(maxOffset)
            .setCodes(ByteString.copyFrom(codes))
            .setWrittenOffsets(ByteString.copyFrom(writtenOffsets.toByteArray()))
            .setCodebookVersion(codebookVersion)
            .setBlockVersion(existingBlock != null ? existingBlock.getBlockVersion() + 1 : 1)
            .setUpdatedAt(currentTimestamp());

        PqCodesBlock updatedBlock = blockBuilder.build();
        tx.set(blockKey, updatedBlock.toByteArray());

        // Update cache
        blockCache.synchronous().invalidate(new BlockCacheKey(codebookVersion, blockNumber));

        return null;
      }));
    }
    return allOf(futures.toArray(CompletableFuture[]::new));
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
    return db.runAsync(tx -> batchStorePqCodesInTransaction(tx, nodeIds, pqCodes, codebookVersion));
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

    // Use AsyncLoadingCache which automatically loads if not present
    BlockCacheKey cacheKey = new BlockCacheKey(codebookVersion, blockNumber);
    return blockCache.get(cacheKey).thenApply(block -> {
      if (block == null || blockOffset >= block.getCodesInBlock()) {
        return null;
      }
      return extractCode(block, blockOffset);
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

    // Load all unique blocks using the cache
    Map<Long, CompletableFuture<PqCodesBlock>> blockFutures = new ConcurrentHashMap<>();
    for (long blockNumber : blockGroups.keySet()) {
      BlockCacheKey cacheKey = new BlockCacheKey(codebookVersion, blockNumber);
      blockFutures.put(blockNumber, blockCache.get(cacheKey));
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
          // Load the written offsets bitmap
          BitSet writtenOffsets;
          if (block.hasWrittenOffsets() && !block.getWrittenOffsets().isEmpty()) {
            writtenOffsets =
                BitSet.valueOf(block.getWrittenOffsets().toByteArray());
          } else {
            // If no bitmap is stored, assume all codes up to codesInBlock are written
            writtenOffsets = new BitSet(codesPerBlock);
            for (int i = 0; i < block.getCodesInBlock(); i++) {
              writtenOffsets.set(i);
            }
          }

          // Debug logging for first few blocks
          if (blockNumber < 3) {
            byte[] rawBytes = block.getWrittenOffsets().toByteArray();
            LOGGER.debug(
                "Block {}: hasWrittenOffsets={}, isEmpty={}, rawBytesLen={}, cardinality={},"
                    + " codesInBlock={}, firstBytes={}",
                blockNumber,
                block.hasWrittenOffsets(),
                block.hasWrittenOffsets()
                    ? block.getWrittenOffsets().isEmpty()
                    : "N/A",
                rawBytes.length,
                writtenOffsets.cardinality(),
                block.getCodesInBlock(),
                rawBytes.length > 0
                    ? String.format(
                        "[%d, %d, ...]",
                        Byte.toUnsignedInt(rawBytes[0]),
                        rawBytes.length > 1 ? Byte.toUnsignedInt(rawBytes[1]) : -1)
                    : "[]");
          }

          for (int idx : indices) {
            long nodeId = nodeIds.get(idx);
            int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);

            // Check if this offset has been written and extract if so
            if (writtenOffsets.get(blockOffset) && blockOffset < block.getCodesInBlock()) {
              byte[] extractedCode = extractCode(block, blockOffset);
              results.set(idx, extractedCode);
            }
          }
        }
      }));
    }

    return allOf(extractFutures.toArray(CompletableFuture[]::new)).thenApply(v -> results);
  }

  /**
   * Deletes a PQ code.
   *
   * @param nodeId          the node ID
   * @param codebookVersion codebook version
   * @return future completing when deleted
   */
  public CompletableFuture<Void> deletePqCode(long nodeId, int codebookVersion) {
    long blockNumber = VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
    int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);
    byte[] blockKey = keys.pqBlockKey(codebookVersion, blockNumber);

    return db.runAsync(tx -> readProto(tx, blockKey, PqCodesBlock.parser()).thenApply(existingBlock -> {
      if (existingBlock == null) {
        // Block doesn't exist, nothing to delete
        return null;
      }

      // Load existing bitmap
      BitSet writtenOffsets =
          BitSet.valueOf(existingBlock.getWrittenOffsets().toByteArray());

      // Clear the bit for this offset
      writtenOffsets.clear(blockOffset);

      // Zero out the code in the block
      byte[] codes = existingBlock.getCodes().toByteArray();
      byte[] emptyCode = new byte[pqSubvectors];
      System.arraycopy(emptyCode, 0, codes, blockOffset * pqSubvectors, pqSubvectors);

      // Update the block
      PqCodesBlock updatedBlock = existingBlock.toBuilder()
          .setCodes(ByteString.copyFrom(codes))
          .setWrittenOffsets(ByteString.copyFrom(writtenOffsets.toByteArray()))
          .setBlockVersion(existingBlock.getBlockVersion() + 1)
          .setUpdatedAt(currentTimestamp())
          .build();

      tx.set(blockKey, updatedBlock.toByteArray());

      // Invalidate cache
      blockCache.synchronous().invalidate(new BlockCacheKey(codebookVersion, blockNumber));
      return null;
    }));
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
          return readProtoRange(tx, range, PqCodesBlock.parser()).thenAccept(blocks -> {
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

              byte[] newKey = keys.pqBlockKey(toVersion, block.getBlockFirstNid() / codesPerBlock);

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

      return completedFuture(null);
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

  // AsyncLoadingCache loader function
  private CompletableFuture<PqCodesBlock> loadBlock(BlockCacheKey key) {
    byte[] blockKey = keys.pqBlockKey(key.version(), key.blockNumber());
    return db.runAsync(tx -> readProto(tx, blockKey, PqCodesBlock.parser()));
  }

  private byte[] extractCode(PqCodesBlock block, int blockOffset) {
    byte[] codes = block.getCodes().toByteArray();
    byte[] pqCode = new byte[pqSubvectors];
    int sourcePos = blockOffset * pqSubvectors;
    System.arraycopy(codes, sourcePos, pqCode, 0, pqSubvectors);
    return pqCode;
  }

  private Timestamp currentTimestamp() {
    Instant now = instantSource.instant();
    return Timestamp.newBuilder()
        .setSeconds(now.getEpochSecond())
        .setNanos(now.getNano())
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

  /**
   * Cache key for block lookups.
   */
  private record BlockCacheKey(int version, long blockNumber) {}
}
