package io.github.panghy.vectorsearch.tasks;

import com.apple.foundationdb.async.AsyncUtil;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs a small looped pool of {@link MaintenanceWorker} instances to continuously
 * process maintenance tasks for a given index.
 *
 * <p>Lifecycle:
 * <ul>
 *   <li>{@link #start(int)} creates N loops, each repeatedly calling {@code runOnce()}.</li>
 *   <li>{@link #close()} signals shutdown and enqueues sentinel no-op tasks to wake sleepers.</li>
 * </ul>
 */
public final class MaintenanceWorkerPool implements AutoCloseable {
  private final VectorIndexConfig config;
  private final FdbDirectories.IndexDirectories indexDirs;
  private final TaskQueue<String, MaintenanceTask> queue;
  private final AtomicBoolean running = new AtomicBoolean(false);
  private final List<CompletableFuture<Void>> loops = new ArrayList<>();
  private int threadCount = 0;
  private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(MaintenanceWorkerPool.class);

  public MaintenanceWorkerPool(
      VectorIndexConfig cfg, FdbDirectories.IndexDirectories dirs, TaskQueue<String, MaintenanceTask> queue) {
    this.config = Objects.requireNonNull(cfg);
    this.indexDirs = Objects.requireNonNull(dirs);
    this.queue = Objects.requireNonNull(queue);
  }

  /** Starts {@code n} background worker loops (no-op if {@code n <= 0} or already running). */
  public synchronized void start(int n) {
    if (n <= 0 || running.get()) return;
    running.set(true);
    threadCount = n;
    LOG.debug("MaintenanceWorkerPool starting threads={}", n);
    for (int i = 0; i < n; i++) {
      MaintenanceWorker worker = new MaintenanceWorker(config, indexDirs, queue);
      CompletableFuture<Void> loop = AsyncUtil.whileTrue(() -> {
        if (!running.get()) return CompletableFuture.completedFuture(false);
        return worker.runOnce().handle((ok, ex) -> true).thenApply(x -> running.get());
      });
      loops.add(loop);
    }
  }

  /** Stops all worker loops and best-effort wakes sleepers via sentinel tasks. */
  @Override
  public synchronized void close() {
    running.set(false);
    LOG.debug("MaintenanceWorkerPool stopping; signaling {} sentinel(s)", threadCount);
    // Enqueue a no-op task as sentinel (vacuum with seg_id < 0) to wake workers
    for (int i = 0; i < threadCount; i++) {
      String key = "vacuum-segment:-1:shutdown:" + i + ":" + System.nanoTime();
      MaintenanceTask mt = MaintenanceTask.newBuilder()
          .setVacuum(MaintenanceTask.Vacuum.newBuilder().setSegId(-1).build())
          .build();
      queue.enqueueIfNotExists(key, mt).exceptionally(ex -> null);
    }
    loops.clear();
  }
}
