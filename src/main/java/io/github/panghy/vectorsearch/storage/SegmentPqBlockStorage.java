package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;
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
 * Segment-scoped PQ block storage. Stores PQ codes under /segment/{segId}/pq/block/{blockNo}.
 */
public class SegmentPqBlockStorage {
  private static final Logger LOGGER = LoggerFactory.getLogger(SegmentPqBlockStorage.class);

  private final Database db;
  private final VectorIndexKeys keys;
  private final int codesPerBlock;
  private final int pqSubvectors;
  private final InstantSource instantSource;

  private final AsyncLoadingCache<BlockCacheKey, PqCodesBlock> blockCache;

  public SegmentPqBlockStorage(
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

  public long getBlockNumber(long nodeId) {
    return VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
  }

  public CompletableFuture<Void> storePqCode(Transaction tx, long segmentId, long nodeId, byte[] pqCode) {
    return batchStorePqCodesInTransaction(tx, segmentId, List.of(nodeId), List.of(pqCode));
  }

  public CompletableFuture<Void> storePqCode(long segmentId, long nodeId, byte[] pqCode) {
    return db.runAsync(tx -> storePqCode(tx, segmentId, nodeId, pqCode));
  }

  public CompletableFuture<Void> batchStorePqCodesInTransaction(
      Transaction tx, long segmentId, List<Long> nodeIds, List<byte[]> pqCodes) {
    if (nodeIds.size() != pqCodes.size()) {
      throw new IllegalArgumentException("Node IDs and PQ codes count mismatch");
    }

    Map<Long, List<Integer>> blockGroups = groupByBlocks(nodeIds);
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (Map.Entry<Long, List<Integer>> entry : blockGroups.entrySet()) {
      long blockNumber = entry.getKey();
      List<Integer> indices = entry.getValue();

      byte[] blockKey = keys.segmentPqBlockKey(segmentId, blockNumber);
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

          if (existingBlock.hasWrittenOffsets()
              && !existingBlock.getWrittenOffsets().isEmpty()) {
            writtenOffsets =
                BitSet.valueOf(existingBlock.getWrittenOffsets().toByteArray());
          } else {
            writtenOffsets = new BitSet(codesPerBlock);
          }
        }

        for (int idx : indices) {
          long nodeId = nodeIds.get(idx);
          byte[] pqCode = pqCodes.get(idx);
          int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);
          System.arraycopy(pqCode, 0, codes, blockOffset * pqSubvectors, pqSubvectors);
          writtenOffsets.set(blockOffset);
          maxOffset = Math.max(maxOffset, blockOffset + 1);
        }

        PqCodesBlock updated = PqCodesBlock.newBuilder()
            .setBlockFirstNid(blockFirstNid)
            .setCodesInBlock(maxOffset)
            .setCodes(ByteString.copyFrom(codes))
            .setWrittenOffsets(ByteString.copyFrom(writtenOffsets.toByteArray()))
            .setBlockVersion(existingBlock != null ? existingBlock.getBlockVersion() + 1 : 1)
            .setUpdatedAt(currentTimestamp())
            .build();

        tx.set(blockKey, updated.toByteArray());
        blockCache.synchronous().invalidate(new BlockCacheKey(segmentId, blockNumber));
        return null;
      }));
    }
    return allOf(futures.toArray(CompletableFuture[]::new));
  }

  public CompletableFuture<Void> batchStorePqCodes(long segmentId, List<Long> nodeIds, List<byte[]> pqCodes) {
    return db.runAsync(tx -> batchStorePqCodesInTransaction(tx, segmentId, nodeIds, pqCodes));
  }

  public CompletableFuture<byte[]> loadPqCode(long segmentId, long nodeId) {
    long blockNumber = VectorIndexKeys.blockNumber(nodeId, codesPerBlock);
    int blockOffset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);
    BlockCacheKey cacheKey = new BlockCacheKey(segmentId, blockNumber);
    return blockCache.get(cacheKey).thenApply(block -> {
      if (block == null || blockOffset >= block.getCodesInBlock()) return null;
      return extractCode(block, blockOffset);
    });
  }

  public CompletableFuture<List<byte[]>> batchLoadPqCodes(long segmentId, List<Long> nodeIds) {
    Map<Long, List<Integer>> blockGroups = groupByBlocks(nodeIds);
    Map<Long, CompletableFuture<PqCodesBlock>> blockFutures = new ConcurrentHashMap<>();
    for (long blockNumber : blockGroups.keySet()) {
      blockFutures.put(blockNumber, blockCache.get(new BlockCacheKey(segmentId, blockNumber)));
    }
    List<byte[]> results = new ArrayList<>(nodeIds.size());
    for (int i = 0; i < nodeIds.size(); i++) results.add(null);

    List<CompletableFuture<Void>> futures = new ArrayList<>();
    for (Map.Entry<Long, List<Integer>> e : blockGroups.entrySet()) {
      long blockNo = e.getKey();
      CompletableFuture<PqCodesBlock> bf = blockFutures.get(blockNo);
      futures.add(bf.thenAccept(block -> {
        if (block == null) return;
        for (int idx : e.getValue()) {
          long nodeId = nodeIds.get(idx);
          int offset = VectorIndexKeys.blockOffset(nodeId, codesPerBlock);
          if (offset < block.getCodesInBlock()) {
            results.set(idx, extractCode(block, offset));
          }
        }
      }));
    }
    return allOf(futures.toArray(CompletableFuture[]::new)).thenApply(v -> results);
  }

  public CompletableFuture<Void> deleteSegment(long segmentId) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.segmentPqBlockPrefix(segmentId);
      Range range = Range.startsWith(prefix);
      StorageTransactionUtils.clearRange(tx, range);
      blockCache.synchronous().invalidateAll();
      return completedFuture(null);
    });
  }

  public CompletableFuture<StorageStats> getStats(long segmentId) {
    return db.runAsync(tx -> {
      byte[] prefix = keys.segmentPqBlockPrefix(segmentId);
      Range range = Range.startsWith(prefix);
      return StorageTransactionUtils.countRange(tx, range)
          .thenCompose(blockCount -> StorageTransactionUtils.readRange(tx, range, 1)
              .thenApply(sample -> {
                long totalCodes = 0;
                long storageBytes = 0;
                if (!sample.isEmpty()) {
                  try {
                    PqCodesBlock first = PqCodesBlock.parseFrom(
                        sample.get(0).getValue());
                    storageBytes =
                        blockCount * sample.get(0).getValue().length;
                    totalCodes = blockCount * first.getCodesInBlock();
                  } catch (Exception ignored) {
                  }
                }
                return new StorageStats(blockCount, totalCodes, storageBytes);
              }));
    });
  }

  private CompletableFuture<PqCodesBlock> loadBlock(BlockCacheKey key) {
    return db.runAsync(tx -> StorageTransactionUtils.readProto(
        tx, keys.segmentPqBlockKey(key.segmentId, key.blockNumber), PqCodesBlock.parser()));
  }

  private byte[] extractCode(PqCodesBlock block, int blockOffset) {
    byte[] codes = block.getCodes().toByteArray();
    byte[] out = new byte[this.pqSubvectors];
    System.arraycopy(codes, blockOffset * this.pqSubvectors, out, 0, this.pqSubvectors);
    return out;
  }

  private Map<Long, List<Integer>> groupByBlocks(List<Long> nodeIds) {
    Map<Long, List<Integer>> blockGroups = new ConcurrentHashMap<>();
    for (int i = 0; i < nodeIds.size(); i++) {
      long blockNumber = VectorIndexKeys.blockNumber(nodeIds.get(i), codesPerBlock);
      blockGroups.computeIfAbsent(blockNumber, k -> new ArrayList<>()).add(i);
    }
    return blockGroups;
  }

  private Timestamp currentTimestamp() {
    long millis = instantSource.millis();
    return Timestamp.newBuilder()
        .setSeconds(millis / 1000)
        .setNanos((int) ((millis % 1000) * 1_000_000))
        .build();
  }

  record BlockCacheKey(long segmentId, long blockNumber) {}

  public record StorageStats(long blockCount, long totalCodes, long storageBytes) {}
}
