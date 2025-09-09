package io.github.panghy.vectorsearch.tasks;

import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.InvalidProtocolBufferException;
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
    return db.runAsync(tr -> tr.get(sk.metaKey()).thenApply(curBytes -> {
          SegmentMeta base;
          try {
            base = curBytes == null ? sm : SegmentMeta.parseFrom(curBytes);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
          SegmentMeta updated = base.toBuilder()
              .setDeletedCount(Math.max(0, base.getDeletedCount() - Math.max(0, removed)))
              .setLastVacuumAtMs(now)
              .build();
          tr.set(sk.metaKey(), updated.toByteArray());
          return updated;
        }))
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
            return db.runAsync(tr -> io.github.panghy.taskqueue.TaskQueues.createTaskQueue(
                        io.github.panghy.taskqueue.TaskQueueConfig.builder(
                                db,
                                indexDirs
                                    .tasksDir()
                                    .createOrOpen(db, java.util.List.of("maint"))
                                    .join(),
                                new ProtoSerializers.StringSerializer(),
                                new ProtoSerializers.MaintenanceTaskSerializer())
                            .estimatedWorkerCount(config.getEstimatedWorkerCount())
                            .defaultTtl(config.getDefaultTtl())
                            .defaultThrottle(config.getDefaultThrottle())
                            .taskNameExtractor(t -> t.hasVacuum()
                                ? ("vacuum-segment:"
                                    + t.getVacuum()
                                        .getSegId())
                                : (t.hasCompact() ? "compact" : "find-candidates"))
                            .build())
                    .join()
                    .enqueueIfNotExists(tr, key, mt))
                .thenApply(x -> null);
          }
          return completedFuture(null);
        });
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
    LOG.info("Compaction skeleton invoked for segIds={}", segIds);
    // In skeleton, simply revert COMPACTING -> SEALED to release the lock.
    Database db = config.getDatabase();
    return db.runAsync(tr -> {
      for (int sid : segIds) {
        byte[] mk = indexDirs.segmentKeys(sid).metaKey();
        byte[] mb = tr.get(mk).join();
        if (mb == null) continue;
        try {
          var sm = SegmentMeta.parseFrom(mb);
          if (sm.getState() == SegmentMeta.State.COMPACTING) {
            var sealed = sm.toBuilder()
                .setState(SegmentMeta.State.SEALED)
                .build();
            tr.set(mk, sealed.toByteArray());
          }
        } catch (InvalidProtocolBufferException e) {
          throw new RuntimeException(e);
        }
      }
      return completedFuture(null);
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
