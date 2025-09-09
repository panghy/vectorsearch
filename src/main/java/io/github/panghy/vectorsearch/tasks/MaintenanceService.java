package io.github.panghy.vectorsearch.tasks;

import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Maintenance operations for a VectorIndex.
 *
 * <p>This service provides two primary capabilities:
 * <ul>
 *   <li><b>Vacuum</b>: Physically removes rows that have been logically deleted (tombstoned)
 *       from an individual segment. Vacuum clears the raw vector record, the PQ code, and the
 *       graph adjacency entry, and then decrements {@code deleted_count} in {@code SegmentMeta}.
 *       Writes are chunked using the same approximate-transaction-size controls used by the
 *       segment build path, so it respects FoundationDB's 10MB/5s constraints.</li>
 *   <li><b>Compaction (skeleton)</b>: Placeholder method for future segment merge/compaction.
 *       Currently a no-op with informative logging.</li>
 * </ul>
 *
 * <p>The class is intentionally stateless aside from injected configuration and directory handles,
 * and all methods are non-blocking (async) to align with project guidelines.</p>
 */
public final class MaintenanceService {
  private static final Logger LOG = LoggerFactory.getLogger(MaintenanceService.class);

  private final VectorIndexConfig config;
  private final FdbDirectories.IndexDirectories indexDirs;
  private CompletableFuture<TaskQueue<String, MaintenanceTask>> maintQueue;

  public MaintenanceService(VectorIndexConfig cfg, FdbDirectories.IndexDirectories dirs) {
    this.config = requireNonNull(cfg, "config");
    this.indexDirs = requireNonNull(dirs, "indexDirs");
  }

  /**
   * Vacuum a single segment: remove tombstoned vector records, PQ codes, and adjacency entries.
   *
   * <p>If {@code minDeletedRatio} is greater than zero, the method first checks the
   * ratio {@code deleted_count / (deleted_count + count)} from {@code SegmentMeta} and returns
   * early when below the threshold.</p>
   *
   * @param segId segment identifier
   * @param minDeletedRatio minimum deletion ratio required to perform the vacuum (0 to always run)
   * @return a future completing when vacuum is finished or a no-op if below threshold
   */
  public CompletableFuture<Void> vacuumSegment(int segId, double minDeletedRatio) {
    Database db = config.getDatabase();
    var sk = indexDirs.segmentKeys(segId);
    return db.readAsync(tr -> tr.get(sk.metaKey())).thenCompose(metaBytes -> {
      if (metaBytes == null) return completedFuture(null);
      SegmentMeta sm;
      try {
        sm = SegmentMeta.parseFrom(metaBytes);
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
      long live = sm.getCount();
      long del = sm.getDeletedCount();
      double ratio = (live + del) == 0 ? 0.0 : ((double) del) / ((double) (live + del));
      if (minDeletedRatio > 0.0 && ratio < minDeletedRatio) {
        LOG.debug("vacuum skipped segId={} ratio={} < threshold={}", segId, ratio, minDeletedRatio);
        return completedFuture(null);
      }
      // Scan vectors
      Subspace vprefix = new Subspace(indexDirs.segmentsDir().pack(Tuple.from(segId, "vectors")));
      Range vr = vprefix.range();
      return db.readAsync(tr -> tr.getRange(vr).asList())
          .thenCompose(kvs -> deleteTombstones(db, sk, kvs))
          .thenCompose(removed -> updateMetaAfterVacuum(db, sk, sm, removed));
    });
  }

  private CompletableFuture<Integer> deleteTombstones(
      Database db, FdbDirectories.SegmentKeys sk, List<KeyValue> kvs) {
    final long soft = (long) (config.getBuildTxnLimitBytes() * config.getBuildTxnSoftLimitRatio());
    final int checkEvery = config.getBuildSizeCheckEvery();
    return deleteSome(db, sk, kvs, 0, 0, checkEvery, soft).thenApply(res -> res.removed);
  }

  private static final class Res {
    final int next;
    final int removed;

    Res(int n, int r) {
      this.next = n;
      this.removed = r;
    }
  }

  private CompletableFuture<Res> deleteSome(
      Database db,
      FdbDirectories.SegmentKeys sk,
      List<KeyValue> kvs,
      int j,
      int removedAcc,
      int checkEvery,
      long softLimit) {
    if (j >= kvs.size()) return completedFuture(new Res(kvs.size(), removedAcc));
    return db.runAsync(tr -> {
          int wrote = 0;
          int removedThis = 0;
          final int cadence = Math.max(1, checkEvery);
          int i;
          for (i = j; i < kvs.size(); i++) {
            KeyValue kv = kvs.get(i);
            try {
              VectorRecord rec = VectorRecord.parseFrom(kv.getValue());
              var tup = indexDirs.segmentsDir().unpack(kv.getKey());
              int vecId = (int) tup.getLong(tup.size() - 1);
              if (rec.getDeleted()) {
                tr.clear(sk.vectorKey(vecId));
                tr.clear(sk.pqCodeKey(vecId));
                tr.clear(sk.graphKey(vecId));
                removedThis++;
              }
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
            wrote++;
            if ((wrote % cadence) == 0) {
              Long size = tr.getApproximateSize().join();
              if (size != null && size >= softLimit) {
                return completedFuture(new Res(i + 1, removedAcc + removedThis));
              }
            }
          }
          return completedFuture(new Res(i, removedAcc + removedThis));
        })
        .thenCompose(res -> deleteSome(db, sk, kvs, res.next, res.removed, checkEvery, softLimit));
  }

  private CompletableFuture<Void> updateMetaAfterVacuum(
      Database db, FdbDirectories.SegmentKeys sk, SegmentMeta sm, int removed) {
    long now = config.getInstantSource().instant().toEpochMilli();
    int maxSize = config.getMaxSegmentSize();
    return db.runAsync(tr -> {
          SegmentMeta updated = sm.toBuilder()
              .setDeletedCount(Math.max(0, sm.getDeletedCount() - Math.max(0, removed)))
              .setLastVacuumAtMs(now)
              .build();
          tr.set(sk.metaKey(), updated.toByteArray());
          return completedFuture(updated);
        })
        .thenCompose(updated -> {
          // If segment is small (<50% of max), enqueue a FindCompactionCandidates task.
          if (config.isAutoFindCompactionCandidates() && updated.getCount() < (maxSize / 2)) {
            MaintenanceTask.FindCompactionCandidates fcc =
                MaintenanceTask.FindCompactionCandidates.newBuilder()
                    .setAnchorSegId(updated.getSegmentId())
                    .build();
            MaintenanceTask mt = MaintenanceTask.newBuilder()
                .setFindCandidates(fcc)
                .build();
            String key = "find-candidates:" + updated.getSegmentId();
            return getOrCreateMaintQueue(db)
                .thenCompose(q -> db.runAsync(tr -> q.enqueueIfNotExists(tr, key, mt)))
                .thenApply(x -> null);
          }
          return completedFuture(null);
        });
  }

  private CompletableFuture<TaskQueue<String, MaintenanceTask>> getOrCreateMaintQueue(Database db) {
    if (maintQueue != null) return maintQueue;
    maintQueue = indexDirs
        .tasksDir()
        .createOrOpen(db, java.util.List.of("maint"))
        .thenCompose(maintDir -> {
          TaskQueueConfig<String, MaintenanceTask> tqc = TaskQueueConfig.builder(
                  db,
                  maintDir,
                  new ProtoSerializers.StringSerializer(),
                  new ProtoSerializers.MaintenanceTaskSerializer())
              .estimatedWorkerCount(config.getEstimatedWorkerCount())
              .defaultTtl(config.getDefaultTtl())
              .defaultThrottle(config.getDefaultThrottle())
              .taskNameExtractor(t -> t.hasVacuum()
                  ? ("vacuum-segment:" + t.getVacuum().getSegId())
                  : (t.hasCompact() ? "compact" : "find-candidates"))
              .build();
          return TaskQueues.createTaskQueue(tqc);
        });
    return maintQueue;
  }

  /**
   * Compaction skeleton: currently a no-op placeholder that logs intent.
   *
   * <p>Future work: merge multiple small sealed segments into a larger target to reclaim
   * graph/PQ overhead and improve search efficiency.</p>
   *
   * @param segIds candidate segments to compact (order/selection policy TBD)
   * @return a completed future after logging the request
   */
  public CompletableFuture<Void> compactSegments(List<Integer> segIds) {
    LOG.info("Compaction invoked for segIds={}", segIds);
    Database db = config.getDatabase();
    // No-op if index not initialized
    return db.readAsync(tr -> tr.get(indexDirs.maxSegmentKey())).thenCompose(maxB -> {
      if (maxB == null) return completedFuture(null);
      // 1) Reserve a new segment id and initialize meta as WRITING (invisible to queries)
      CompletableFuture<Integer> reserve = db.runAsync(tr -> {
        int nextSeg = Math.toIntExact(com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt(
                tr.get(indexDirs.maxSegmentKey()).join()))
            + 1;
        tr.set(indexDirs.maxSegmentKey(), com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt(nextSeg));
        SegmentMeta meta = SegmentMeta.newBuilder()
            .setSegmentId(nextSeg)
            .setState(SegmentMeta.State.WRITING)
            .setCount(0)
            .setCreatedAtMs(config.getInstantSource().instant().toEpochMilli())
            .build();
        tr.set(indexDirs.segmentKeys(nextSeg).metaKey(), meta.toByteArray());
        return completedFuture(nextSeg);
      });

      return reserve.thenCompose(newSegId -> {
        // 2) Read all vectors from source segments
        return db.readAsync(tr -> {
              java.util.List<KeyValue> all = new java.util.ArrayList<>();
              for (int sid : segIds) {
                Subspace vprefix =
                    new Subspace(indexDirs.segmentsDir().pack(Tuple.from(sid, "vectors")));
                all.addAll(tr.getRange(vprefix.range()).asList().join());
              }
              return completedFuture(all);
            })
            .thenCompose(kvs -> {
              // 3) Write combined vectors into new segment in batches
              final int batchSize = 1000;
              java.util.List<float[]> vectors = new java.util.ArrayList<>(kvs.size());
              java.util.List<byte[]> payloads = new java.util.ArrayList<>(kvs.size());
              for (KeyValue kv : kvs) {
                try {
                  VectorRecord rec = VectorRecord.parseFrom(kv.getValue());
                  if (rec.getDeleted()) continue;
                  vectors.add(io.github.panghy.vectorsearch.util.FloatPacker.bytesToFloats(
                      rec.getEmbedding().toByteArray()));
                  payloads.add(rec.getPayload().toByteArray());
                } catch (InvalidProtocolBufferException e) {
                  throw new RuntimeException(e);
                }
              }
              java.util.concurrent.CompletableFuture<Void> w = completedFuture(null);
              for (int i = 0; i < vectors.size(); i += batchSize) {
                final int start = i;
                final int end = Math.min(vectors.size(), i + batchSize);
                w = w.thenCompose(vv -> db.runAsync(tr -> {
                  for (int j = start; j < end; j++) {
                    int vecId = j;
                    VectorRecord rec = VectorRecord.newBuilder()
                        .setSegId(newSegId)
                        .setVecId(vecId)
                        .setEmbedding(com.google.protobuf.ByteString.copyFrom(
                            io.github.panghy.vectorsearch.util.FloatPacker.floatsToBytes(
                                vectors.get(j))))
                        .setDeleted(false)
                        .setPayload(com.google.protobuf.ByteString.copyFrom(payloads.get(j)))
                        .build();
                    tr.set(indexDirs.segmentKeys(newSegId).vectorKey(vecId), rec.toByteArray());
                  }
                  // Update meta with running count while WRITING; remains invisible to queries
                  SegmentMeta meta = SegmentMeta.newBuilder()
                      .setSegmentId(newSegId)
                      .setState(SegmentMeta.State.WRITING)
                      .setCount(end)
                      .setCreatedAtMs(config.getInstantSource()
                          .instant()
                          .toEpochMilli())
                      .build();
                  tr.set(indexDirs.segmentKeys(newSegId).metaKey(), meta.toByteArray());
                  return completedFuture(null);
                }));
              }
              return w.thenApply(vv -> newSegId);
            })
            .thenCompose(segId -> new SegmentBuildService(config, indexDirs)
                .build(segId)
                .thenApply(v -> segId))
            .thenCompose(segId -> db.runAsync(tr -> {
              // 4) Registry swap and old segments cleanup atomically
              tr.set(indexDirs.segmentsIndexKey(segId), new byte[0]);
              for (int sid : segIds) {
                // Clear ranges and keys
                Subspace vprefix =
                    new Subspace(indexDirs.segmentsDir().pack(Tuple.from(sid, "vectors")));
                tr.clear(vprefix.range().begin, vprefix.range().end);
                Subspace cprefix =
                    new Subspace(indexDirs.segmentsDir().pack(Tuple.from(sid, "pq", "codes")));
                tr.clear(cprefix.range().begin, cprefix.range().end);
                Subspace gprefix =
                    new Subspace(indexDirs.segmentsDir().pack(Tuple.from(sid, "graph")));
                tr.clear(gprefix.range().begin, gprefix.range().end);
                tr.clear(indexDirs.segmentKeys(sid).pqCodebookKey());
                tr.clear(indexDirs.segmentKeys(sid).metaKey());
                tr.clear(indexDirs.segmentsIndexKey(sid));
              }
              return completedFuture(null);
            }));
      });
    });
  }

  /**
   * Finds a small set of candidate segments to compact together with the anchor.
   * Heuristic: choose SEALED segments with the smallest counts first until their
   * combined live counts reach at least ~80% of maxSegmentSize or up to 4 segments.
   */
  public CompletableFuture<java.util.List<Integer>> findCompactionCandidates(int anchorSegId) {
    Database db = config.getDatabase();
    int maxSize = config.getMaxSegmentSize();
    com.apple.foundationdb.subspace.Subspace reg = indexDirs.segmentsIndexSubspace();
    return db.readAsync(tr -> tr.getRange(reg.range()).asList()).thenCompose(kvs -> {
      java.util.List<Integer> ids = new java.util.ArrayList<>(kvs.size());
      for (com.apple.foundationdb.KeyValue kv : kvs) {
        int sid = Math.toIntExact(reg.unpack(kv.getKey()).getLong(0));
        ids.add(sid);
      }
      java.util.List<java.util.concurrent.CompletableFuture<byte[]>> metas = new java.util.ArrayList<>();
      for (int sid : ids)
        metas.add(db.readAsync(tr -> tr.get(indexDirs.segmentKeys(sid).metaKey())));
      return java.util.concurrent.CompletableFuture.allOf(
              metas.toArray(java.util.concurrent.CompletableFuture[]::new))
          .thenApply(v -> {
            java.util.List<int[]> sealed = new java.util.ArrayList<>(); // [segId, count]
            for (int i = 0; i < ids.size(); i++) {
              byte[] b = metas.get(i).join();
              if (b == null) continue;
              try {
                var sm = io.github.panghy.vectorsearch.proto.SegmentMeta.parseFrom(b);
                if (sm.getState() == io.github.panghy.vectorsearch.proto.SegmentMeta.State.SEALED) {
                  sealed.add(new int[] {ids.get(i), sm.getCount()});
                }
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }
            sealed.sort(java.util.Comparator.comparingInt(a -> a[1]));
            int budget = (int) Math.max(1, Math.round(0.8 * maxSize));
            int sum = 0;
            java.util.List<Integer> pick = new java.util.ArrayList<>();
            // Ensure anchor is present (if sealed)
            for (int[] pair : sealed)
              if (pair[0] == anchorSegId) {
                pick.add(anchorSegId);
                sum += pair[1];
                break;
              }
            for (int[] pair : sealed) {
              if (pick.contains(pair[0])) continue;
              if (pick.size() >= 4) break;
              pick.add(pair[0]);
              sum += pair[1];
              if (sum >= budget) break;
            }
            if (pick.size() <= 1) return java.util.List.of();
            return pick;
          });
    });
  }
}
