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
    String segStr = String.format("%06d", segId);
    var sk = indexDirs.segmentKeys(segStr);
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
      Subspace vprefix = new Subspace(indexDirs.segmentsDir().pack(Tuple.from(segStr, "vectors")));
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
            if ((wrote % Math.max(1, checkEvery)) == 0) {
              Long size = tr.getApproximateSize().join();
              if (size != null && size >= softLimit) {
                return completedFuture(new Res(i + 1, removedAcc + removedThis));
              }
            }
          }
          return completedFuture(new Res(i, removedAcc + removedThis));
        })
        .thenCompose(res -> deleteSome(db, sk, kvs, res.next, res.removed, 0, softLimit));
  }

  private CompletableFuture<Void> updateMetaAfterVacuum(
      Database db, FdbDirectories.SegmentKeys sk, SegmentMeta sm, int removed) {
    if (removed <= 0) return completedFuture(null);
    return db.runAsync(tr -> {
      SegmentMeta updated = sm.toBuilder()
          .setDeletedCount(Math.max(0, sm.getDeletedCount() - removed))
          .build();
      tr.set(sk.metaKey(), updated.toByteArray());
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
    return completedFuture(null);
  }
}
