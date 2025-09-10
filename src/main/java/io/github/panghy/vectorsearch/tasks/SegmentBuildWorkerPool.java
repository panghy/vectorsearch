package io.github.panghy.vectorsearch.tasks;

import com.apple.foundationdb.async.AsyncUtil;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.BuildTask;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs a pool of {@link SegmentBuildWorker} instances that continuously process queued tasks.
 */
public final class SegmentBuildWorkerPool implements AutoCloseable {
  private final VectorIndexConfig config;
  private final FdbDirectories.IndexDirectories indexDirs;
  private final TaskQueue<String, BuildTask> taskQueue;
  private final AtomicBoolean running = new AtomicBoolean(false);
  private final List<CompletableFuture<Void>> loops = new ArrayList<>();
  private int threadCount = 0;
  private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(SegmentBuildWorkerPool.class);

  public SegmentBuildWorkerPool(
      VectorIndexConfig config,
      FdbDirectories.IndexDirectories indexDirs,
      TaskQueue<String, BuildTask> taskQueue) {
    this.config = Objects.requireNonNull(config, "config");
    this.indexDirs = Objects.requireNonNull(indexDirs, "indexDirs");
    this.taskQueue = Objects.requireNonNull(taskQueue, "taskQueue");
  }

  /**
   * Starts {@code n} background threads. No-op if {@code n <= 0} or already started.
   */
  public synchronized void start(int n) {
    if (n <= 0 || running.get()) return;
    running.set(true);
    threadCount = n;
    LOG.debug("SegmentBuildWorkerPool starting threads={}", n);
    for (int i = 0; i < n; i++) {
      SegmentBuildWorker worker = new SegmentBuildWorker(config, indexDirs, taskQueue);
      CompletableFuture<Void> loop = AsyncUtil.whileTrue(() -> {
        if (!running.get()) return CompletableFuture.completedFuture(false);
        // runOnce returns a future that completes when a task is processed (or sentinel)
        return worker.runOnce()
            .handle((ok, ex) -> true) // swallow errors, keep looping
            .thenApply(x -> running.get());
      });
      loops.add(loop);
    }
  }

  /**
   * Stops all threads and waits briefly for termination.
   */
  @Override
  public synchronized void close() {
    running.set(false);
    LOG.debug("SegmentBuildWorkerPool stopping; signaling {} sentinel(s)", threadCount);
    // Enqueue sentinel tasks to wake up blocked workers so they can exit.
    // Best-effort async sentinel enqueue; no blocking join
    for (int i = 0; i < threadCount; i++) {
      String key = "build-segment:-1:shutdown:" + i + ":" + System.nanoTime();
      var task = BuildTask.newBuilder().setSegId(-1).build();
      taskQueue.enqueueIfNotExists(key, task).exceptionally(ex -> null);
    }
    // let whileTrue loops observe running=false and exit after sentinel wakes them
    loops.clear();
  }
}
