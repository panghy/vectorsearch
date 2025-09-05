package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.ReadTransaction;
import com.apple.foundationdb.Transaction;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Registry for segment metadata and active segment management.
 */
public class SegmentRegistry {

  private final VectorIndexKeys keys;

  public SegmentRegistry(VectorIndexKeys keys) {
    this.keys = keys;
  }

  public CompletableFuture<Void> createSegment(Transaction tr, long segId, int codebookVersion) {
    SegmentMeta meta = SegmentMeta.newBuilder()
        .setSegmentId(segId)
        .setCodebookVersion(codebookVersion)
        .setStatus(SegmentMeta.Status.ACTIVE)
        .setCreatedAt(nowTs())
        .build();
    tr.set(keys.segmentMetaKey(segId), meta.toByteArray());
    return CompletableFuture.completedFuture(null);
  }

  public CompletableFuture<Void> sealSegment(Transaction tr, long segId) {
    return readSegmentMeta(tr, segId).thenAccept(existing -> {
      if (existing == null) return;
      SegmentMeta updated =
          existing.toBuilder().setStatus(SegmentMeta.Status.SEALED).build();
      tr.set(keys.segmentMetaKey(segId), updated.toByteArray());
    });
  }

  public CompletableFuture<SegmentMeta> readSegmentMeta(ReadTransaction tr, long segId) {
    return tr.get(keys.segmentMetaKey(segId)).thenApply(bytes -> {
      if (bytes == null) return null;
      try {
        return SegmentMeta.parseFrom(bytes);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
  }

  public CompletableFuture<Void> setActiveSegment(Transaction tr, long segId) {
    tr.set(keys.activeSegmentKey(), com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt(segId));
    return CompletableFuture.completedFuture(null);
  }

  public CompletableFuture<Long> getActiveSegment(ReadTransaction tr) {
    return tr.get(keys.activeSegmentKey())
        .thenApply(bytes -> bytes == null ? 0L : com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt(bytes));
  }

  // Simple list by range scan over meta prefix. Assumes small segment count.
  public CompletableFuture<List<SegmentMeta>> listSegments(ReadTransaction tr) {
    byte[] prefix = keys.segmentMetaKey(0L);
    com.apple.foundationdb.Range range = VectorIndexKeys.prefixRange(keys.segmentMetaKey(0L))[0] == null
        ? new com.apple.foundationdb.Range(prefix, com.apple.foundationdb.tuple.ByteArrayUtil.strinc(prefix))
        : new com.apple.foundationdb.Range(prefix, com.apple.foundationdb.tuple.ByteArrayUtil.strinc(prefix));
    return tr.getRange(range).asList().thenApply(kvs -> {
      List<SegmentMeta> list = new ArrayList<>();
      for (com.apple.foundationdb.KeyValue kv : kvs) {
        try {
          list.add(SegmentMeta.parseFrom(kv.getValue()));
        } catch (Exception ignored) {
        }
      }
      return list;
    });
  }

  private static Timestamp nowTs() {
    long ms = System.currentTimeMillis();
    return Timestamp.newBuilder()
        .setSeconds(ms / 1000)
        .setNanos((int) ((ms % 1000) * 1_000_000))
        .build();
  }
}
