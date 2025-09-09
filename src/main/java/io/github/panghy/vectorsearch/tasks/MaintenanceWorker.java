package io.github.panghy.vectorsearch.tasks;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories.IndexDirectories;
import io.github.panghy.vectorsearch.fdb.FdbVectorIndex;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
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
    return completedFuture(indexDirs)
        .thenCompose(d -> queue.awaitAndClaimTask(db).thenCompose(claim -> {
          MaintenanceTask t = claim.task();
          MaintenanceService svc = new MaintenanceService(config, indexDirs);
          CompletableFuture<Void> work;
          if (t.hasVacuum()) {
            var v = t.getVacuum();
            work = svc.vacuumSegment(v.getSegId(), v.getMinDeletedRatio());
          } else if (t.hasCompact()) {
            work = svc.compactSegments(t.getCompact().getSegIdsList());
          } else if (t.hasFindCandidates()) {
            // Heuristic handled by MaintenanceService.findCompactionCandidates(anchor):
            // pick sealed small segments until we approximately fill one target
            int anchor = t.getFindCandidates().getAnchorSegId();
            work = svc.findCompactionCandidates(anchor).thenCompose(cands -> {
              if (cands.size() <= 1) return completedFuture(null);
              // Atomically mark candidates as COMPACTING to avoid overlaps
              return db.runAsync(tr -> {
                    for (int sid : cands) {
                      byte[] mk =
                          indexDirs.segmentKeys(sid).metaKey();
                      byte[] mb = tr.get(mk).join();
                      if (mb == null) return completedFuture(false);
                      try {
                        var sm = io.github.panghy.vectorsearch.proto.SegmentMeta.parseFrom(mb);
                        if (sm.getState()
                            != io.github.panghy.vectorsearch.proto.SegmentMeta.State
                                .SEALED) {
                          return completedFuture(false);
                        }
                      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                        throw new RuntimeException(e);
                      }
                    }
                    for (int sid : cands) {
                      byte[] mk =
                          indexDirs.segmentKeys(sid).metaKey();
                      try {
                        var sm = io.github.panghy.vectorsearch.proto.SegmentMeta.parseFrom(
                            tr.get(mk).join());
                        var updated = sm.toBuilder()
                            .setState(
                                io.github.panghy.vectorsearch.proto.SegmentMeta.State
                                    .COMPACTING)
                            .build();
                        tr.set(mk, updated.toByteArray());
                      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                        throw new RuntimeException(e);
                      }
                    }
                    return completedFuture(true);
                  })
                  .thenCompose(marked -> marked
                      ? FdbVectorIndex.createOrOpen(config)
                          .thenCompose(ix -> ((FdbVectorIndex) ix)
                              .requestCompaction(cands)
                              .whenComplete((vv, ex) -> ix.close()))
                      : completedFuture(null));
            });
          } else {
            work = completedFuture(null);
          }
          return work.handle((vv, ex) -> ex)
              .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
              .thenApply(v -> true);
        }));
  }
}
