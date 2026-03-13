package io.github.panghy.vectorsearch.tasks;

import static java.util.Objects.requireNonNull;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import io.github.panghy.taskqueue.TaskClaim;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.proto.TaskKeyMetadata;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/**
 * Adapter that implements {@link TaskQueue TaskQueue&lt;String, MaintenanceTask&gt;} by wrapping
 * each enqueued {@link MaintenanceTask} into a {@link GlobalMaintenanceTask} (with the index
 * directory path) and delegating to the shared global maintenance queue.
 *
 * <p>Only producer methods ({@code enqueue}, {@code enqueueIfNotExists}) are supported. Consumer
 * and inspection methods throw {@link UnsupportedOperationException} because local workers are
 * not started when global mode is enabled.</p>
 */
public final class GlobalMaintenanceQueueAdapter implements TaskQueue<String, MaintenanceTask> {

  private final TaskQueue<String, GlobalMaintenanceTask> globalQueue;
  private final List<String> indexPath;

  /**
   * Creates a new adapter.
   *
   * @param globalQueue the shared global maintenance task queue
   * @param indexPath   the directory path components identifying this index
   */
  public GlobalMaintenanceQueueAdapter(TaskQueue<String, GlobalMaintenanceTask> globalQueue, List<String> indexPath) {
    this.globalQueue = requireNonNull(globalQueue, "globalQueue");
    this.indexPath = List.copyOf(requireNonNull(indexPath, "indexPath"));
  }

  @Override
  public TaskQueueConfig<String, MaintenanceTask> getConfig() {
    throw new UnsupportedOperationException("GlobalMaintenanceQueueAdapter does not have a local config");
  }

  @Override
  public <R> CompletableFuture<R> runAsync(Function<Transaction, CompletableFuture<R>> function) {
    return globalQueue.runAsync(function);
  }

  @Override
  public CompletableFuture<TaskKeyMetadata> enqueue(
      Transaction tx, String key, MaintenanceTask task, Duration ttl, Duration throttle) {
    GlobalMaintenanceTask wrapped = GlobalMaintenanceTask.newBuilder()
        .addAllIndexPath(indexPath)
        .setTask(task)
        .build();
    return globalQueue.enqueue(tx, key, wrapped, ttl, throttle);
  }

  @Override
  public CompletableFuture<TaskKeyMetadata> enqueueIfNotExists(
      Transaction tx, String key, MaintenanceTask task, Duration ttl, Duration throttle, boolean updatePayload) {
    GlobalMaintenanceTask wrapped = GlobalMaintenanceTask.newBuilder()
        .addAllIndexPath(indexPath)
        .setTask(task)
        .build();
    return globalQueue.enqueueIfNotExists(tx, key, wrapped, ttl, throttle, updatePayload);
  }

  @Override
  public CompletableFuture<TaskClaim<String, MaintenanceTask>> awaitAndClaimTask(Database db) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> completeTask(Transaction tx, TaskClaim<String, MaintenanceTask> claim) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> failTask(Transaction tx, TaskClaim<String, MaintenanceTask> claim) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> extendTtl(Transaction tx, TaskClaim<String, MaintenanceTask> claim, Duration ttl) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Boolean> isEmpty(Transaction tx) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Boolean> hasVisibleUnclaimedTasks(Transaction tx) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Boolean> hasClaimedTasks(Transaction tx) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> awaitQueueEmpty(Database db) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }
}
