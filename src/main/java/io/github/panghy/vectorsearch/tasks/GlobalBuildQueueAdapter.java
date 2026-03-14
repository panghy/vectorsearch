package io.github.panghy.vectorsearch.tasks;

import static java.util.Objects.requireNonNull;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import io.github.panghy.taskqueue.TaskClaim;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.proto.TaskKeyMetadata;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/**
 * Adapter that implements {@link TaskQueue TaskQueue&lt;String, BuildTask&gt;} by wrapping each
 * enqueued {@link BuildTask} into a {@link GlobalBuildTask} (with the index directory path) and
 * delegating to the shared global build queue.
 *
 * <p>Only producer methods ({@code enqueue}, {@code enqueueIfNotExists}) and
 * {@link #awaitQueueEmpty(Database)} are supported. Consumer and inspection methods throw
 * {@link UnsupportedOperationException} because local workers are not started when global mode
 * is enabled.</p>
 *
 * <p><b>Note:</b> {@link #awaitQueueEmpty(Database)} delegates to the global queue, so it waits
 * for <em>all</em> global build tasks to drain — not just those belonging to this index.</p>
 */
public final class GlobalBuildQueueAdapter implements TaskQueue<String, BuildTask> {

  private final TaskQueue<String, GlobalBuildTask> globalQueue;
  private final List<String> indexPath;
  private final String keyPrefix;

  /**
   * Creates a new adapter.
   *
   * @param globalQueue the shared global build task queue
   * @param indexPath   the directory path components identifying this index
   */
  public GlobalBuildQueueAdapter(TaskQueue<String, GlobalBuildTask> globalQueue, List<String> indexPath) {
    this.globalQueue = requireNonNull(globalQueue, "globalQueue");
    this.indexPath = List.copyOf(requireNonNull(indexPath, "indexPath"));
    this.keyPrefix = String.join("/", this.indexPath) + ":";
  }

  @Override
  public TaskQueueConfig<String, BuildTask> getConfig() {
    throw new UnsupportedOperationException("GlobalBuildQueueAdapter does not have a local config");
  }

  @Override
  public <R> CompletableFuture<R> runAsync(Function<Transaction, CompletableFuture<R>> function) {
    return globalQueue.runAsync(function);
  }

  @Override
  public CompletableFuture<TaskKeyMetadata> enqueue(
      Transaction tx, String key, BuildTask task, Duration ttl, Duration throttle) {
    GlobalBuildTask wrapped = GlobalBuildTask.newBuilder()
        .addAllIndexPath(indexPath)
        .setTask(task)
        .build();
    return globalQueue.enqueue(tx, keyPrefix + key, wrapped, ttl, throttle);
  }

  @Override
  public CompletableFuture<TaskKeyMetadata> enqueueIfNotExists(Transaction tx, String key, BuildTask task) {
    GlobalBuildTask wrapped = GlobalBuildTask.newBuilder()
        .addAllIndexPath(indexPath)
        .setTask(task)
        .build();
    return globalQueue.enqueueIfNotExists(tx, keyPrefix + key, wrapped);
  }

  @Override
  public CompletableFuture<TaskKeyMetadata> enqueueIfNotExists(
      Transaction tx, String key, BuildTask task, Duration ttl, Duration throttle, boolean updatePayload) {
    GlobalBuildTask wrapped = GlobalBuildTask.newBuilder()
        .addAllIndexPath(indexPath)
        .setTask(task)
        .build();
    return globalQueue.enqueueIfNotExists(tx, keyPrefix + key, wrapped, ttl, throttle, updatePayload);
  }

  @Override
  public CompletableFuture<TaskClaim<String, BuildTask>> awaitAndClaimTask(Database db) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> completeTask(Transaction tx, TaskClaim<String, BuildTask> claim) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> failTask(Transaction tx, TaskClaim<String, BuildTask> claim) {
    throw new UnsupportedOperationException("Local workers are not used when global task queue mode is enabled");
  }

  @Override
  public CompletableFuture<Void> extendTtl(Transaction tx, TaskClaim<String, BuildTask> claim, Duration ttl) {
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

  /**
   * Delegates to the global queue's {@code awaitQueueEmpty}. Note: this waits for <em>all</em>
   * global build tasks to drain, not just those belonging to this index.
   */
  @Override
  public CompletableFuture<Void> awaitQueueEmpty(Database db) {
    return globalQueue.awaitQueueEmpty(db);
  }
}
