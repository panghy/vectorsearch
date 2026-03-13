package io.github.panghy.vectorsearch.config;

import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import java.util.Objects;

/**
 * Immutable holder for global (cross-index) task queues.
 *
 * <p>When supplied to {@link VectorIndexConfig}, it signals that the index should enqueue
 * build and maintenance tasks into these shared queues instead of creating per-index local
 * queues and workers.</p>
 *
 * @see VectorIndexConfig#getGlobalTaskQueueConfig()
 * @see VectorIndexConfig#isGlobalTaskQueueEnabled()
 */
public final class GlobalTaskQueueConfig {

  private final TaskQueue<String, GlobalBuildTask> buildQueue;
  private final TaskQueue<String, GlobalMaintenanceTask> maintenanceQueue;

  /**
   * Creates a new {@code GlobalTaskQueueConfig}.
   *
   * @param buildQueue       the shared build task queue (must not be null)
   * @param maintenanceQueue the shared maintenance task queue (must not be null)
   */
  public GlobalTaskQueueConfig(
      TaskQueue<String, GlobalBuildTask> buildQueue, TaskQueue<String, GlobalMaintenanceTask> maintenanceQueue) {
    this.buildQueue = Objects.requireNonNull(buildQueue, "buildQueue must not be null");
    this.maintenanceQueue = Objects.requireNonNull(maintenanceQueue, "maintenanceQueue must not be null");
  }

  /**
   * Returns the global build task queue.
   */
  public TaskQueue<String, GlobalBuildTask> getBuildQueue() {
    return buildQueue;
  }

  /**
   * Returns the global maintenance task queue.
   */
  public TaskQueue<String, GlobalMaintenanceTask> getMaintenanceQueue() {
    return maintenanceQueue;
  }
}
