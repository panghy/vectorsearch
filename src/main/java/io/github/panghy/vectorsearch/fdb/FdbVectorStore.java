package io.github.panghy.vectorsearch.fdb;

import static com.apple.foundationdb.async.AsyncUtil.tag;
import static com.apple.foundationdb.async.AsyncUtil.whileTrue;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.tuple.Tuple;
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
import java.util.List;
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
   * Opens or creates the index directories under the configured app root.
   *
   * @return future with the opened {@link FdbDirectories.IndexDirectories}
   */
  public CompletableFuture<FdbDirectories.IndexDirectories> openIndexDirs() {
    return completedFuture(indexDirs);
  }

  /**
   * Creates directories and initializes index meta and the first ACTIVE segment if missing.
   * If the index exists, validates persisted {@link IndexMeta}
   * against the provided {@link VectorIndexConfig} and throws on mismatches.
   *
   * @return future that completes when the index is ready
   */
  public CompletableFuture<Void> createOrOpenIndex() {
    Database db = config.getDatabase();
    return db.runAsync(tr -> {
      byte[] metaK = indexDirs.metaKey();
      byte[] curSegK = indexDirs.currentSegmentKey();
      byte[] maxSegK = indexDirs.maxSegmentKey();
      CompletableFuture<byte[]> metaV = tr.get(metaK);
      CompletableFuture<byte[]> curSegV = tr.get(curSegK);
      CompletableFuture<byte[]> maxSegV = tr.get(maxSegK);
      return CompletableFuture.allOf(metaV, curSegV, maxSegV).thenCompose(v -> {
        if (metaV.join() == null) {
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
          // Validate existing meta matches configuration
          try {
            IndexMeta existing = IndexMeta.parseFrom(metaV.join());
            validateIndexMeta(existing);
          } catch (InvalidProtocolBufferException e) {
            throw new IllegalStateException("Corrupted IndexMeta", e);
          }
        }
        if (curSegV.join() == null) {
          // initialize first segment 0 as ACTIVE
          int segId = 0;
          tr.set(curSegK, Tuple.from(segId).pack());
          tr.set(maxSegK, Tuple.from(segId).pack());
          SegmentMeta sm = SegmentMeta.newBuilder()
              .setSegmentId(segId)
              .setState(SegmentMeta.State.ACTIVE)
              .setCount(0)
              .setCreatedAtMs(Instant.now().toEpochMilli())
              .setDeletedCount(0)
              .build();
          tr.set(indexDirs.segmentKeys(segIdStr(segId)).metaKey(), sm.toByteArray());
        }
        return completedFuture(null);
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
  }

  /**
   * Returns the current ACTIVE segment id.
   */
  public CompletableFuture<Integer> getCurrentSegment() {
    Database db = config.getDatabase();
    return db.readAsync(tr -> tr.get(indexDirs.currentSegmentKey())
        .thenApply(v -> (int) Tuple.fromBytes(v).getLong(0)));
  }

  /**
   * Adds one vector to the ACTIVE segment.
   * Rotates to a new ACTIVE segment (marking the current as PENDING) when the threshold is exceeded,
   * and enqueues a background BuildTask for the sealed segment.
   *
   * @param embedding vector values (length must equal configured dimension)
   * @param payload   optional payload bytes
   * @return future with [segmentId, vectorId]
   */
  public CompletableFuture<int[]> add(float[] embedding, byte[] payload) {
    return addBatch(new float[][] {embedding}, new byte[][] {payload}).thenApply(ids -> ids.get(0));
  }

  /**
   * Adds many vectors efficiently by batching writes per transaction and reducing contention
   * on the current segment pointer and segment meta keys.
   *
   * <p>Writes up to the remaining capacity of the ACTIVE segment in one transaction, rotates if
   * needed, enqueues a build task for the sealed segment, and continues until all vectors are
   * written. Respects FoundationDB limits by implicitly chunking work across transactions.</p>
   *
   * @param embeddings vectors to add (each of length = dimension)
   * @param payloads   optional payload bytes per vector (may be null or shorter than embeddings)
   * @return future with assigned ids per vector in the same order [segId, vecId]
   */
  public CompletableFuture<List<int[]>> addBatch(float[][] embeddings, byte[][] payloads) {
    Database db = config.getDatabase();
    List<int[]> assigned = new ArrayList<>(embeddings.length);
    List<int[]> batch = new ArrayList<>();
    AtomicInteger pos = new AtomicInteger(0);

    return tag(
        whileTrue(() -> {
          if (pos.get() >= embeddings.length) return completedFuture(false);
          return db.runAsync(tr -> {
                batch.clear();
                byte[] curSegK = indexDirs.currentSegmentKey();
                return tr.get(curSegK).thenCompose(curSegV -> {
                  int segId = (int) Tuple.fromBytes(curSegV).getLong(0);
                  String segStr = segIdStr(segId);
                  FdbDirectories.SegmentKeys sk = indexDirs.segmentKeys(segStr);
                  return tr.get(sk.metaKey()).thenCompose(segMetaBytes -> {
                    SegmentMeta sm;
                    try {
                      sm = SegmentMeta.parseFrom(segMetaBytes);
                    } catch (InvalidProtocolBufferException e) {
                      throw new RuntimeException(e);
                    }
                    if (sm.getState() != SegmentMeta.State.ACTIVE) {
                      throw new IllegalStateException(
                          "Current segment not ACTIVE: " + sm.getState());
                    }

                    int remaining = embeddings.length - pos.get();
                    int count = sm.getCount();
                    int capacity = config.getMaxSegmentSize() - count;
                    boolean rotate = false;
                    int wrote = 0;

                    if (capacity <= 0 && remaining > 0) {
                      rotate = true;
                    } else {
                      int toWrite = Math.min(remaining, Math.max(capacity, 0));
                      for (int i = 0; i < toWrite; i++) {
                        int idx = pos.get() + i;
                        float[] e = embeddings[idx];
                        byte[] p = (payloads != null && idx < payloads.length)
                            ? payloads[idx]
                            : null;
                        VectorRecord rec = VectorRecord.newBuilder()
                            .setSegId(segId)
                            .setVecId(count + i)
                            .setEmbedding(ByteString.copyFrom(FloatPacker.floatsToBytes(e)))
                            .setDeleted(false)
                            .setPayload(
                                p == null ? ByteString.EMPTY : ByteString.copyFrom(p))
                            .build();
                        tr.set(sk.vectorKey(count + i), rec.toByteArray());
                        batch.add(new int[] {segId, count + i});
                      }
                      wrote = toWrite;
                      SegmentMeta updated = sm.toBuilder()
                          .setCount(count + toWrite)
                          .build();
                      tr.set(sk.metaKey(), updated.toByteArray());
                      if (updated.getCount() >= config.getMaxSegmentSize()) rotate = true;
                    }

                    pos.addAndGet(wrote);
                    LOGGER.debug(
                        "Writing {} vectors to segment {} (rotate: {})", wrote, segId, rotate);
                    if (rotate) {
                      // Seal current and open next
                      SegmentMeta pending = sm.toBuilder()
                          .setState(SegmentMeta.State.PENDING)
                          .setCount(Math.max(sm.getCount(), count + wrote))
                          .build();
                      tr.set(sk.metaKey(), pending.toByteArray());
                      int nextSeg = segId + 1;
                      tr.set(curSegK, Tuple.from(nextSeg).pack());
                      // Advance the known maximum segment id so queries discover all segments.
                      tr.set(
                          indexDirs.maxSegmentKey(),
                          Tuple.from(nextSeg).pack());
                      String nextStr = segIdStr(nextSeg);
                      SegmentMeta nextMeta = SegmentMeta.newBuilder()
                          .setSegmentId(nextSeg)
                          .setState(SegmentMeta.State.ACTIVE)
                          .setCount(0)
                          .setCreatedAtMs(
                              Instant.now().toEpochMilli())
                          .build();
                      tr.set(
                          indexDirs
                              .segmentKeys(nextStr)
                              .metaKey(),
                          nextMeta.toByteArray());
                      LOGGER.debug(
                          "Rotated segment: {} -> {} (sealed PENDING seg {}), enqueuing"
                              + " build task",
                          segId,
                          nextSeg,
                          segId);
                      return enqueueBuildTask(tr, segId).thenApply($ -> true);
                    }
                    return completedFuture(true);
                  });
                });
              })
              .whenComplete((v, ex) -> {
                if (ex == null) {
                  assigned.addAll(batch);
                }
              });
        }),
        assigned);
  }

  private CompletableFuture<Void> enqueueBuildTask(Transaction txn, int segId) {
    String key = "build-segment:" + segId;
    BuildTask task = BuildTask.newBuilder().setSegId(segId).build();
    return taskQueue.enqueueIfNotExists(txn, key, task).thenApply(x -> null);
  }

  /**
   * Reads {@link SegmentMeta} for a given segment id.
   */
  public CompletableFuture<SegmentMeta> getSegmentMeta(int segId) {
    Database db = config.getDatabase();
    return db.readAsync(
        tr -> tr.get(indexDirs.segmentKeys(segIdStr(segId)).metaKey()).thenApply(bytes -> {
          try {
            return SegmentMeta.parseFrom(bytes);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }));
  }

  // listSegmentIds intentionally omitted in v1; DirectoryLayer listing is non-trivial and not needed yet.

  private static String segIdStr(int segId) {
    return String.format("%06d", segId);
  }
}
