package io.github.panghy.vectorsearch.tasks;

import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories.IndexDirectories;
import io.github.panghy.vectorsearch.fdb.FdbVectorIndex;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.util.Metrics;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Background worker that claims {@link MaintenanceTask} items from the maintenance queue
 * and executes maintenance operations for a single index namespace.
 *
 * <p>Supported task types:
 * <ul>
 *   <li><b>Vacuum</b> — removes tombstoned rows from a segment using
 *       {@link MaintenanceService#vacuumSegment(int, double)}.</li>
 *   <li><b>Compact</b> — placeholder call to
 *       {@link MaintenanceService#compactSegments(java.util.List)} for future merges.</li>
 * </ul>
 */
public final class MaintenanceWorker {
  private final VectorIndexConfig config;
  private final IndexDirectories indexDirs;
  private final TaskQueue<String, MaintenanceTask> queue;

  public MaintenanceWorker(VectorIndexConfig cfg, IndexDirectories dirs, TaskQueue<String, MaintenanceTask> queue) {
    this.config = cfg;
    this.indexDirs = dirs;
    this.queue = queue;
  }

  /**
   * Claims and processes one maintenance task if available.
   *
   * <p>Returns {@code true} if a task was claimed (and completed or failed),
   * or {@code false} if the queue had no visible tasks. Failures are marked via
   * {@code claim.fail()} so the queue can retry.</p>
   */
  public CompletableFuture<Boolean> runOnce() {
    Database db = config.getDatabase();
    return queue.awaitAndClaimTask().thenCompose(claim -> processTask(claim.task(), db)
        .handle((vv, ex) -> ex)
        .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
        .thenApply(v -> true));
  }

  private CompletableFuture<Void> processTask(MaintenanceTask t, Database db) {
    MaintenanceService svc = new MaintenanceService(config, indexDirs);
    if (t.hasVacuum()) {
      var v = t.getVacuum();
      Tracer tracer = Metrics.tracer();
      Span span = tracer.spanBuilder("vectorsearch.vacuum")
          .setSpanKind(SpanKind.INTERNAL)
          .setAttribute("segId", v.getSegId())
          .setAttribute(
              "index.path", io.github.panghy.vectorsearch.util.Metrics.dirPath(indexDirs.indexDir()))
          .startSpan();
      long t0 = System.nanoTime();
      return svc.vacuumSegment(v.getSegId(), v.getMinDeletedRatio()).whenComplete((vv2, ex) -> {
        long durMs = (System.nanoTime() - t0) / 1_000_000;
        Attributes attrs =
            Attributes.builder().put("segId", v.getSegId()).build();
        Metrics.VACUUM_RUN_COUNT.add(1, attrs);
        Metrics.VACUUM_DURATION_MS.record((double) durMs, attrs);
        if (ex != null) {
          span.recordException(ex);
          span.setStatus(io.opentelemetry.api.trace.StatusCode.ERROR);
        }
        span.end();
      });
    }
    if (t.hasCompact()) {
      return svc.compactSegments(t.getCompact().getSegIdsList());
    }
    if (t.hasFindCandidates()) {
      int anchor = t.getFindCandidates().getAnchorSegId();
      return handleFindCandidates(svc, anchor, db);
    }
    return completedFuture(null);
  }

  private CompletableFuture<Void> handleFindCandidates(MaintenanceService svc, int anchor, Database db) {
    if (config.getMaxConcurrentCompactions() <= 0) return completedFuture(null);
    Tracer tracer = Metrics.tracer();
    Span span = tracer.spanBuilder("vectorsearch.compaction")
        .setSpanKind(SpanKind.INTERNAL)
        .setAttribute("anchorSegId", anchor)
        .setAttribute("index.path", io.github.panghy.vectorsearch.util.Metrics.dirPath(indexDirs.indexDir()))
        .startSpan();
    long t0 = System.nanoTime();
    return svc.findCompactionCandidates(anchor).thenCompose(cands -> {
      if (cands.size() <= 1) return completedFuture(null);
      return svc.countInFlightCompactions()
          .thenCompose(inflight -> inflight >= config.getMaxConcurrentCompactions()
              ? completedFuture(null)
              : markCandidatesCompacting(db, cands)
                  .thenCompose(marked -> marked
                      ? FdbVectorIndex.createOrOpen(config)
                          .thenCompose(ix -> ix.requestCompaction(cands)
                              .whenComplete((vv, ex) -> ix.close()))
                      : completedFuture(null)));
    });
  }

  private CompletableFuture<Boolean> markCandidatesCompacting(Database db, List<Integer> cands) {
    return db.runAsync(tr -> {
      List<CompletableFuture<byte[]>> metas = new ArrayList<>();
      for (int sid : cands) metas.add(indexDirs.segmentKeys(tr, sid).thenCompose(sk -> tr.get(sk.metaKey())));
      return allOf(metas.toArray(CompletableFuture[]::new)).thenCompose(v -> {
        for (var f : metas) {
          byte[] mb = f.getNow(null);
          if (mb == null) return completedFuture(false);
          try {
            var sm = SegmentMeta.parseFrom(mb);
            if (sm.getState() != SegmentMeta.State.SEALED) return completedFuture(false);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }
        List<CompletableFuture<Void>> sets = new ArrayList<>();
        for (int sid : cands) {
          sets.add(indexDirs.segmentKeys(tr, sid).thenCompose(sk -> tr.get(sk.metaKey())
              .thenApply(bytes -> {
                try {
                  var sm = SegmentMeta.parseFrom(bytes);
                  var updated = sm.toBuilder()
                      .setState(SegmentMeta.State.COMPACTING)
                      .build();
                  tr.set(sk.metaKey(), updated.toByteArray());
                  return null;
                } catch (InvalidProtocolBufferException e) {
                  throw new RuntimeException(e);
                }
              })));
        }
        return allOf(sets.toArray(CompletableFuture[]::new)).thenApply(x -> true);
      });
    });
  }
}
