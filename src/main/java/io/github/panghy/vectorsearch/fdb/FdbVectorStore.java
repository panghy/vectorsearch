package io.github.panghy.vectorsearch.fdb;

import static com.apple.foundationdb.async.AsyncUtil.tag;
import static com.apple.foundationdb.async.AsyncUtil.whileTrue;
import static com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt;
import static com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.tuple.ByteArrayUtil;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.IndexMeta;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.util.FloatPacker;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Directory-based storage operations for ACTIVE segment CRUD and rotation.
 *
 * <p>Provides async methods that operate on DirectoryLayer subspaces. All transactions
 * respect FoundationDB limits by keeping write sets small and short-lived.</p>
 */
public final class FdbVectorStore {

  private static final Logger LOGGER = LoggerFactory.getLogger(FdbVectorStore.class);

  private final VectorIndexConfig config;
  private final FdbDirectories.IndexDirectories indexDirs;
  private final TaskQueue<String, BuildTask> taskQueue;

  public FdbVectorStore(
      VectorIndexConfig config,
      FdbDirectories.IndexDirectories indexDirs,
      TaskQueue<String, BuildTask> taskQueue) {
    this.config = config;
    this.indexDirs = requireNonNull(indexDirs, "indexDirs");
    this.taskQueue = requireNonNull(taskQueue, "taskQueue");
  }

  /**
   * @return the index directories helper for this store (already opened)
   */
  public CompletableFuture<FdbDirectories.IndexDirectories> openIndexDirs() {
    return completedFuture(indexDirs);
  }

  /**
   * Creates directories and initializes index metadata and the first ACTIVE segment if missing.
   * Validates persisted {@link IndexMeta} matches {@link VectorIndexConfig} on reopen.
   *
   * @return future that completes when the index is ready for use
   */
  public CompletableFuture<Void> createOrOpenIndex() {
    Database db = config.getDatabase();
    return db.runAsync(tr -> {
      byte[] metaK = indexDirs.metaKey();
      byte[] curSegK = indexDirs.currentSegmentKey();
      byte[] maxSegK = indexDirs.maxSegmentKey();
      return tr.get(metaK).thenCompose(metaBytes -> {
        if (metaBytes == null) {
          IndexMeta meta = IndexMeta.newBuilder()
              .setDimension(config.getDimension())
              .setMetric(
                  config.getMetric() == VectorIndexConfig.Metric.COSINE
                      ? IndexMeta.Metric.METRIC_COSINE
                      : IndexMeta.Metric.METRIC_L2)
              .setMaxSegmentSize(config.getMaxSegmentSize())
              .setPqM(config.getPqM())
              .setPqK(config.getPqK())
              .setGraphDegree(config.getGraphDegree())
              .setOversample(config.getOversample())
              .build();
          tr.set(metaK, meta.toByteArray());
        } else {
          try {
            IndexMeta existing = IndexMeta.parseFrom(metaBytes);
            validateIndexMeta(existing);
          } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Corrupted IndexMeta", e);
          }
        }
        return tr.get(curSegK).thenCompose(curBytes -> {
          if (curBytes != null) return completedFuture(null);
          int segId = 0;
          tr.set(curSegK, encodeInt(segId));
          tr.set(maxSegK, encodeInt(segId));
          SegmentMeta sm = SegmentMeta.newBuilder()
              .setSegmentId(segId)
              .setState(SegmentMeta.State.ACTIVE)
              .setCount(0)
              .setCreatedAtMs(Instant.now().toEpochMilli())
              .setDeletedCount(0)
              .build();
          return indexDirs.segmentKeys(tr, segId).thenApply(sk -> {
            tr.set(sk.metaKey(), sm.toByteArray());
            tr.set(indexDirs.segmentsIndexKey(segId), new byte[0]);
            return null;
          });
        });
      });
    });
  }

  private void validateIndexMeta(IndexMeta existing) {
    // Name is not validated; index directory is provided by caller.
    if (existing.getDimension() != config.getDimension()) {
      throw new IllegalArgumentException(
          "Dimension mismatch: existing=" + existing.getDimension() + ", requested=" + config.getDimension());
    }
    IndexMeta.Metric expectedMetric = (config.getMetric() == VectorIndexConfig.Metric.COSINE)
        ? IndexMeta.Metric.METRIC_COSINE
        : IndexMeta.Metric.METRIC_L2;
    if (existing.getMetric() != expectedMetric) {
      throw new IllegalArgumentException(
          "Metric mismatch: existing=" + existing.getMetric() + ", requested=" + expectedMetric);
    }
    if (existing.getMaxSegmentSize() != config.getMaxSegmentSize()) {
      throw new IllegalArgumentException("maxSegmentSize mismatch: existing=" + existing.getMaxSegmentSize()
          + ", requested=" + config.getMaxSegmentSize());
    }
    if (existing.getPqM() != config.getPqM()) {
      throw new IllegalArgumentException(
          "pqM mismatch: existing=" + existing.getPqM() + ", requested=" + config.getPqM());
    }
    if (existing.getPqK() != config.getPqK()) {
      throw new IllegalArgumentException(
          "pqK mismatch: existing=" + existing.getPqK() + ", requested=" + config.getPqK());
    }
    if (existing.getGraphDegree() != config.getGraphDegree()) {
      throw new IllegalArgumentException("graphDegree mismatch: existing=" + existing.getGraphDegree()
          + ", requested=" + config.getGraphDegree());
    }
    if (existing.getOversample() != config.getOversample()) {
      throw new IllegalArgumentException("oversample mismatch: existing=" + existing.getOversample()
          + ", requested=" + config.getOversample());
    }
    // vacuumCooldown is runtime-only and not persisted in IndexMeta; no validation.
  }

  /**
   * @return future with the current ACTIVE segment id
   */
  public CompletableFuture<Integer> getCurrentSegment() {
    Database db = config.getDatabase();
    return db.readAsync(tr -> tr.get(indexDirs.currentSegmentKey())
        .thenApply(ByteArrayUtil::decodeInt)
        .thenApply(Math::toIntExact));
  }

  /**
   * Inserts one vector using the store's multi-transaction batching path.
   *
   * @param embedding vector values (length must equal configured dimension)
   * @param payload   optional payload bytes (nullable)
   * @return future with {@code [segId, vecId]}
   */
  public CompletableFuture<int[]> add(float[] embedding, byte[] payload) {
    return addBatch(new float[][] {embedding}, new byte[][] {payload}).thenApply(ids -> ids.get(0));
  }

  /**
   * Inserts one vector inside a caller-supplied transaction.
   *
   * @param tr        FDB transaction
   * @param embedding vector values (length must equal configured dimension)
   * @param payload   optional payload bytes (nullable)
   * @return future with {@code [segId, vecId]}
   */
  public CompletableFuture<int[]> add(Transaction tr, float[] embedding, byte[] payload) {
    return addBatch(tr, new float[][] {embedding}, new byte[][] {payload}).thenApply(ids -> ids.get(0));
  }

  /**
   * Inserts many vectors with internal batching and rotation across multiple transactions.
   * Missing payload entries (when {@code payloads == null} or shorter than embeddings) are treated
   * as {@code null} payloads; extra payload entries are ignored.
   *
   * @param embeddings vectors to insert
   * @param payloads   optional per-vector payloads
   * @return future with assigned ids per vector in the same order
   */
  public CompletableFuture<List<int[]>> addBatch(float[][] embeddings, byte[][] payloads) {
    Database db = config.getDatabase();
    List<int[]> assigned = new ArrayList<>(embeddings.length);
    AtomicInteger pos = new AtomicInteger(0);

    return tag(
        whileTrue(() -> {
          if (pos.get() >= embeddings.length) return completedFuture(false);
          return writeOnce(db, embeddings, payloads, pos).thenApply(batch -> {
            assigned.addAll(batch);
            pos.addAndGet(batch.size());
            return true;
          });
        }),
        assigned);
  }

  /**
   * Inserts many vectors inside a single caller-supplied transaction. Rotates within the same
   * transaction as capacity is reached.
   *
   * @param tr         FDB transaction
   * @param embeddings vectors to insert
   * @param payloads   optional per-vector payloads (may be {@code null} or shorter than embeddings)
   * @return future with assigned ids per vector in the same order
   */
  public CompletableFuture<List<int[]>> addBatch(Transaction tr, float[][] embeddings, byte[][] payloads) {
    List<int[]> assigned = new ArrayList<>(embeddings.length);
    byte[] curSegK = indexDirs.currentSegmentKey();
    return tr.get(curSegK).thenCompose(curSegV -> {
      int segId = Math.toIntExact(decodeInt(curSegV));
      return addAllInTxn(tr, segId, 0, embeddings, payloads, assigned).thenApply(v -> assigned);
    });
  }

  /**
   * Marks a single vector deleted and updates segment counters.
   *
   * @param segId segment id
   * @param vecId vector id within the segment
   * @return future that completes when persisted
   */
  public CompletableFuture<Void> delete(int segId, int vecId) {
    return deleteBatch(new int[][] {{segId, vecId}});
  }

  /**
   * Single-transaction delete.
   *
   * @param tr    FDB transaction
   * @param segId segment id
   * @param vecId vector id within the segment
   * @return future that completes when persisted
   */
  public CompletableFuture<Void> delete(Transaction tr, int segId, int vecId) {
    return deleteBatch(tr, new int[][] {{segId, vecId}});
  }

  /**
   * Batch delete across one or more transactions; groups by segment for efficiency.
   *
   * @param ids array of {@code [segId, vecId]} pairs; {@code null} or empty is a no-op
   * @return future that completes when all mutations are persisted
   */
  public CompletableFuture<Void> deleteBatch(int[][] ids) {
    if (ids == null || ids.length == 0) return completedFuture(null);
    Database db = config.getDatabase();
    // Group by segment to update SegmentMeta efficiently
    Map<Integer, List<Integer>> bySeg = new HashMap<>();
    for (int[] id : ids)
      bySeg.computeIfAbsent(id[0], k -> new ArrayList<>()).add(id[1]);
    List<CompletableFuture<Void>> writes = new ArrayList<>();
    for (var e : bySeg.entrySet()) {
      int segId = e.getKey();
      List<Integer> vecIds = e.getValue();
      writes.add(db.runAsync(tr -> indexDirs.segmentKeys(tr, segId).thenCompose(sk -> {
        // Load meta
        return tr.get(sk.metaKey()).thenCompose(metaBytes -> {
          try {
            SegmentMeta sm = SegmentMeta.parseFrom(metaBytes);
            // Gather reads for the ids
            List<CompletableFuture<byte[]>> reads = new ArrayList<>();
            for (int vId : vecIds) reads.add(tr.get(sk.vectorKey(vId)));
            return allOf(reads.toArray(CompletableFuture[]::new)).thenApply($ -> {
              int dec = 0;
              int incDel = 0;
              for (int i = 0; i < vecIds.size(); i++) {
                byte[] bytes = reads.get(i).getNow(null);
                if (bytes == null) continue;
                try {
                  VectorRecord rec = VectorRecord.parseFrom(bytes);
                  if (!rec.getDeleted()) {
                    VectorRecord updated =
                        rec.toBuilder().setDeleted(true).build();
                    tr.set(sk.vectorKey(vecIds.get(i)), updated.toByteArray());
                    // Clear gid mappings
                    byte[] revK = indexDirs
                        .gidRevDir()
                        .pack(com.apple.foundationdb.tuple.Tuple.from(segId, vecIds.get(i)));
                    byte[] gb = tr.get(revK).join();
                    if (gb != null) {
                      long gid = com.apple.foundationdb.tuple.Tuple.fromBytes(gb)
                          .getLong(0);
                      byte[] mapK = indexDirs
                          .gidMapDir()
                          .pack(com.apple.foundationdb.tuple.Tuple.from(gid));
                      tr.clear(mapK);
                    }
                    tr.clear(revK);
                    dec++;
                    incDel++;
                  }
                } catch (InvalidProtocolBufferException ex) {
                  throw new RuntimeException(ex);
                }
              }
              // Update meta with computed deltas
              SegmentMeta updated = sm.toBuilder()
                  .setCount(Math.max(0, sm.getCount() - dec))
                  .setDeletedCount(Math.max(0, sm.getDeletedCount() + incDel))
                  .build();
              tr.set(sk.metaKey(), updated.toByteArray());
              return null;
            });
          } catch (InvalidProtocolBufferException ex) {
            throw new RuntimeException(ex);
          }
        });
      })));
    }
    return allOf(writes.toArray(CompletableFuture[]::new));
  }

  /**
   * Batch delete inside a single caller-supplied transaction.
   *
   * @param tr  FDB transaction
   * @param ids array of {@code [segId, vecId]} pairs; {@code null} or empty is a no-op
   * @return future that completes when persisted
   */
  public CompletableFuture<Void> deleteBatch(Transaction tr, int[][] ids) {
    if (ids == null || ids.length == 0) return completedFuture(null);
    // Group by segment to update SegmentMeta efficiently
    Map<Integer, List<Integer>> bySeg = new HashMap<>();
    for (int[] id : ids)
      bySeg.computeIfAbsent(id[0], k -> new ArrayList<>()).add(id[1]);
    CompletableFuture<Void> chain = completedFuture(null);
    for (var e : bySeg.entrySet()) {
      int segId = e.getKey();
      List<Integer> vecIds = e.getValue();
      chain = chain.thenCompose(v -> indexDirs.segmentKeys(tr, segId).thenCompose(sk -> tr.get(sk.metaKey())
          .thenCompose(metaBytes -> {
            try {
              SegmentMeta sm = SegmentMeta.parseFrom(metaBytes);
              List<CompletableFuture<byte[]>> reads = new ArrayList<>();
              for (int vId : vecIds) reads.add(tr.get(sk.vectorKey(vId)));
              return allOf(reads.toArray(CompletableFuture[]::new))
                  .thenApply($ -> {
                    int dec = 0;
                    int incDel = 0;
                    for (int i = 0; i < vecIds.size(); i++) {
                      byte[] bytes = reads.get(i).getNow(null);
                      if (bytes == null) continue;
                      try {
                        VectorRecord rec = VectorRecord.parseFrom(bytes);
                        if (!rec.getDeleted()) {
                          VectorRecord updated = rec.toBuilder()
                              .setDeleted(true)
                              .build();
                          tr.set(sk.vectorKey(vecIds.get(i)), updated.toByteArray());
                          dec++;
                          incDel++;
                        }
                      } catch (InvalidProtocolBufferException ex) {
                        throw new RuntimeException(ex);
                      }
                    }
                    SegmentMeta updated = sm.toBuilder()
                        .setCount(Math.max(0, sm.getCount() - dec))
                        .setDeletedCount(Math.max(0, sm.getDeletedCount() + incDel))
                        .build();
                    tr.set(sk.metaKey(), updated.toByteArray());
                    return null;
                  });
            } catch (InvalidProtocolBufferException ex) {
              throw new RuntimeException(ex);
            }
          })));
    }
    return chain;
  }

  private CompletableFuture<List<int[]>> writeOnce(
      Database db, float[][] embeddings, byte[][] payloads, AtomicInteger pos) {
    return db.runAsync(tr -> {
      List<int[]> batch = new ArrayList<>();
      byte[] curSegK = indexDirs.currentSegmentKey();
      return tr.get(curSegK).thenCompose(curSegV -> {
        int segId = Math.toIntExact(decodeInt(curSegV));
        return indexDirs.segmentKeys(tr, segId).thenCompose(sk -> tr.get(sk.metaKey())
            .thenCompose(segMetaBytes -> {
              SegmentMeta sm;
              try {
                if (segMetaBytes == null) {
                  sm = SegmentMeta.newBuilder()
                      .setSegmentId(segId)
                      .setState(SegmentMeta.State.ACTIVE)
                      .setCount(0)
                      .setCreatedAtMs(Instant.now().toEpochMilli())
                      .build();
                  tr.set(sk.metaKey(), sm.toByteArray());
                } else {
                  sm = SegmentMeta.parseFrom(segMetaBytes);
                }
              } catch (InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
              if (sm.getState() != SegmentMeta.State.ACTIVE) {
                throw new IllegalStateException("Current segment not ACTIVE: " + sm.getState());
              }

              int remaining = embeddings.length - pos.get();
              int count = sm.getCount();
              int capacity = config.getMaxSegmentSize() - count;

              if (capacity <= 0 && remaining > 0) {
                // Rotate without writing in this txn
                rotateToNextActive(tr, sk, curSegK, sm, segId, count);
                LOGGER.debug(
                    "Rotated segment: {} -> {} (sealed PENDING seg {}), enqueuing build task",
                    segId,
                    segId + 1,
                    segId);
                return enqueueBuildTask(tr, segId).thenApply($ -> batch);
              }

              int toWrite = Math.min(remaining, Math.max(0, capacity));
              long softLimit =
                  (long) (config.getBuildTxnLimitBytes() * config.getBuildTxnSoftLimitRatio());
              int checkEvery = config.getBuildSizeCheckEvery();
              return writeSomeVectors(
                      tr,
                      sk,
                      segId,
                      count,
                      pos.get(),
                      toWrite,
                      embeddings,
                      payloads,
                      checkEvery,
                      softLimit,
                      batch)
                  .thenCompose(wroteN -> updateAfterWriteAndMaybeRotate(
                          tr, sk, curSegK, sm, segId, count, wroteN)
                      .thenApply($ -> batch));
            }));
      });
    });
  }

  private CompletableFuture<Void> updateAfterWriteAndMaybeRotate(
      Transaction tr,
      FdbDirectories.SegmentKeys sk,
      byte[] curSegK,
      SegmentMeta sm,
      int segId,
      int startCount,
      int wroteN) {
    // Update meta with actual written count.
    SegmentMeta updated = sm.toBuilder().setCount(startCount + wroteN).build();
    tr.set(sk.metaKey(), updated.toByteArray());
    boolean willRotate = updated.getCount() >= config.getMaxSegmentSize();
    LOGGER.debug("Writing {} vectors to segment {} (rotate: {})", wroteN, segId, willRotate);
    if (!willRotate) return completedFuture(null);
    return rotateToNextActive(tr, sk, curSegK, sm, segId, Math.max(sm.getCount(), startCount + wroteN))
        .thenCompose($ -> enqueueBuildTask(tr, segId).thenApply(zz -> null));
  }

  private CompletableFuture<Void> rotateToNextActive(
      Transaction tr,
      FdbDirectories.SegmentKeys sk,
      byte[] curSegK,
      SegmentMeta sm,
      int segId,
      int pendingCount) {
    SegmentMeta pending = sm.toBuilder()
        .setState(SegmentMeta.State.PENDING)
        .setCount(pendingCount)
        .build();
    tr.set(sk.metaKey(), pending.toByteArray());
    int nextSeg = segId + 1;
    tr.set(curSegK, encodeInt(nextSeg));
    tr.set(indexDirs.maxSegmentKey(), encodeInt(nextSeg));
    SegmentMeta nextMeta = SegmentMeta.newBuilder()
        .setSegmentId(nextSeg)
        .setState(SegmentMeta.State.ACTIVE)
        .setCount(0)
        .setCreatedAtMs(Instant.now().toEpochMilli())
        .build();
    // Pre-create next segment subspaces and meta to avoid races in builders/search.
    tr.set(indexDirs.segmentsIndexKey(nextSeg), new byte[0]);
    return indexDirs.segmentKeys(tr, nextSeg).thenApply(nsk -> {
      tr.set(nsk.metaKey(), nextMeta.toByteArray());
      return null;
    });
  }

  private CompletableFuture<Void> addAllInTxn(
      Transaction tr, int segId, int startIdx, float[][] embeddings, byte[][] payloads, List<int[]> outIds) {
    if (startIdx >= embeddings.length) return completedFuture(null);
    final byte[] curSegK = indexDirs.currentSegmentKey();
    final AtomicInteger curSeg = new AtomicInteger(segId);
    final AtomicInteger idx = new AtomicInteger(startIdx);

    return whileTrue(() -> {
          if (idx.get() >= embeddings.length) return completedFuture(false);
          int seg = curSeg.get();
          return indexDirs.segmentKeys(tr, seg).thenCompose(sk -> {
            return tr.get(sk.metaKey()).thenCompose(metaBytes -> {
              SegmentMeta sm;
              try {
                sm = SegmentMeta.parseFrom(metaBytes);
              } catch (InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
              if (sm.getState() != SegmentMeta.State.ACTIVE) {
                throw new IllegalStateException("Current segment not ACTIVE: " + sm.getState());
              }
              int max = config.getMaxSegmentSize();
              int count = sm.getCount();
              int remaining = embeddings.length - idx.get();
              int capacity = Math.max(0, max - count);
              if (capacity <= 0 && remaining > 0) {
                int prev = seg;
                return rotateToNextActive(tr, sk, curSegK, sm, seg, count)
                    .thenCompose($ -> {
                      curSeg.set(prev + 1);
                      return enqueueBuildTask(tr, prev).thenApply(xx -> true);
                    });
              }
              int toWrite = Math.min(capacity, remaining);
              for (int i = 0; i < toWrite; i++) {
                int vecId = count + i;
                int j = idx.get() + i;
                float[] e = embeddings[j];
                byte[] p = (payloads != null && j < payloads.length) ? payloads[j] : null;
                VectorRecord rec = VectorRecord.newBuilder()
                    .setSegId(seg)
                    .setVecId(vecId)
                    .setEmbedding(ByteString.copyFrom(FloatPacker.floatsToBytes(e)))
                    .setDeleted(false)
                    .setPayload(p == null ? ByteString.EMPTY : ByteString.copyFrom(p))
                    .build();
                tr.set(sk.vectorKey(vecId), rec.toByteArray());
                outIds.add(new int[] {seg, vecId});
              }
              int newCount = count + toWrite;
              tr.set(
                  sk.metaKey(),
                  sm.toBuilder().setCount(newCount).build().toByteArray());
              idx.addAndGet(toWrite);
              if (newCount < max) {
                return completedFuture(idx.get() < embeddings.length);
              }
              return rotateToNextActive(tr, sk, curSegK, sm, seg, Math.max(sm.getCount(), newCount))
                  .thenCompose($ -> {
                    curSeg.set(seg + 1);
                    return enqueueBuildTask(tr, seg).thenApply(xx -> true);
                  });
            });
          });
        })
        .thenApply(v -> null);
  }

  private CompletableFuture<Integer> writeSomeVectors(
      Transaction tr,
      FdbDirectories.SegmentKeys sk,
      int segId,
      int startVecCount,
      int inputStartIdx,
      int toWrite,
      float[][] embeddings,
      byte[][] payloads,
      int checkEvery,
      long softLimit,
      List<int[]> outIds) {
    return writeSomeVectorsRec(
        tr,
        sk,
        segId,
        startVecCount,
        inputStartIdx,
        0,
        toWrite,
        embeddings,
        payloads,
        checkEvery,
        softLimit,
        outIds);
  }

  private CompletableFuture<Integer> writeSomeVectorsRec(
      Transaction tr,
      FdbDirectories.SegmentKeys sk,
      int segId,
      int startVecCount,
      int inputStartIdx,
      int wroteSoFar,
      int toWrite,
      float[][] embeddings,
      byte[][] payloads,
      int checkEvery,
      long softLimit,
      List<int[]> outIds) {
    if (wroteSoFar >= toWrite) return completedFuture(wroteSoFar);
    int i = wroteSoFar;
    for (; i < toWrite; i++) {
      int idx = inputStartIdx + i;
      float[] e = embeddings[idx];
      byte[] p = (payloads != null && idx < payloads.length) ? payloads[idx] : null;
      VectorRecord rec = VectorRecord.newBuilder()
          .setSegId(segId)
          .setVecId(startVecCount + i)
          .setEmbedding(ByteString.copyFrom(FloatPacker.floatsToBytes(e)))
          .setDeleted(false)
          .setPayload(p == null ? ByteString.EMPTY : ByteString.copyFrom(p))
          .build();
      tr.set(sk.vectorKey(startVecCount + i), rec.toByteArray());
      outIds.add(new int[] {segId, startVecCount + i});
      if (((i + 1) % Math.max(1, checkEvery)) == 0) {
        int nextCount = i + 1;
        return tr.getApproximateSize().thenCompose(sz -> {
          if (sz != null && sz >= softLimit) {
            return completedFuture(nextCount);
          }
          return writeSomeVectorsRec(
              tr,
              sk,
              segId,
              startVecCount,
              inputStartIdx,
              nextCount,
              toWrite,
              embeddings,
              payloads,
              checkEvery,
              softLimit,
              outIds);
        });
      }
    }
    return completedFuture(i);
  }

  private CompletableFuture<Void> enqueueBuildTask(Transaction txn, int segId) {
    String key = "build-segment:" + segId;
    BuildTask task = BuildTask.newBuilder().setSegId(segId).build();
    return taskQueue.enqueueIfNotExists(txn, key, task).thenApply(x -> null);
  }

  /**
   * Reads {@link SegmentMeta} for a given segment id. Visible for testing.
   */
  CompletableFuture<SegmentMeta> getSegmentMeta(int segId) {
    Database db = config.getDatabase();
    return db.readAsync(tr -> indexDirs
        .segmentKeys(tr, segId)
        .thenCompose(sk -> tr.get(sk.metaKey()))
        .thenApply(bytes -> {
          try {
            return SegmentMeta.parseFrom(bytes);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }));
  }
}
