package io.github.panghy.vectorsearch.tasks;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories.IndexDirectories;
import io.github.panghy.vectorsearch.proto.BuildTask;
import java.util.concurrent.CompletableFuture;

/**
 * Background worker that claims {@link io.github.panghy.vectorsearch.proto.BuildTask} from the index's
 * tasks queue and performs segment builds.
 *
 * <p>Current implementation delegates to {@link SegmentBuildService} which writes placeholder PQ/graph
 * artifacts and seals the segment. This will be replaced by a full PQ training + graph builder.</p>
 */
public class SegmentBuildWorker {
  private final VectorIndexConfig config;
  private final IndexDirectories indexDirs;
  private final TaskQueue<String, BuildTask> taskQueue;

  /**
   * Dependency-injected constructor with resolved index directories.
   */
  public SegmentBuildWorker(
      VectorIndexConfig config, IndexDirectories indexDirs, TaskQueue<String, BuildTask> taskQueue) {
    this.config = config;
    this.indexDirs = java.util.Objects.requireNonNull(indexDirs, "indexDirs");
    this.taskQueue = java.util.Objects.requireNonNull(taskQueue, "taskQueue");
  }

  /**
   * Claims and processes a single {@link BuildTask} if available.
   *
   * @return future that completes with true if a task was processed, false if none
   */
  public CompletableFuture<Boolean> runOnce() {
    Database db = config.getDatabase();
    return completedFuture(indexDirs)
        .thenCompose(dirs -> taskQueue.awaitAndClaimTask(db).thenCompose(claim -> {
          BuildTask task = claim.task();
          // Shutdown sentinel: seg_id < 0 means wake and exit loop iteration without building.
          if (task.getSegId() < 0) {
            return claim.complete().thenApply(v -> true);
          }
          // Build artifacts and seal the segment. On failure, mark task failed for immediate retry.
          return new SegmentBuildService(config, indexDirs)
              .build(task.getSegId())
              .handle((v, ex) -> ex)
              .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
              .thenApply(v -> true);
        }));
  }
}
