package io.github.panghy.vectorsearch.tasks;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories.IndexDirectories;
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
          } else {
            work = completedFuture(null);
          }
          return work.handle((vv, ex) -> ex)
              .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
              .thenApply(v -> true);
        }));
  }
}
