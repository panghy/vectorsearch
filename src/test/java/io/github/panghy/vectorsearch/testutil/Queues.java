package io.github.panghy.vectorsearch.testutil;

import com.apple.foundationdb.Database;
import io.github.panghy.taskqueue.TaskQueue;
import java.lang.reflect.Method;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/** Test utilities for waiting on TaskQueue state in a version-tolerant way. */
public final class Queues {
  private Queues() {}

  /**
   * Waits until the queue is completely empty (no visible or claimed tasks).
   * If running against TaskQueue >= 0.10.0, calls awaitQueueEmpty; otherwise polls.
   */
  public static CompletableFuture<Void> awaitQueueEmpty(TaskQueue<?, ?> q, Database db) {
    try {
      Method m = q.getClass().getMethod("awaitQueueEmpty", Database.class);
      Object f = m.invoke(q, db);
      @SuppressWarnings("unchecked")
      CompletableFuture<?> cf = (CompletableFuture<?>) f;
      return cf.thenApply(v -> null);
    } catch (NoSuchMethodException nsme) {
      final long pollDelayMs = 50L;
      return com.apple.foundationdb.async.AsyncUtil.whileTrue(() -> db.runAsync(
                  tr -> q.hasVisibleUnclaimedTasks(tr)
                      .thenCombine(
                          q.hasClaimedTasks(tr),
                          (vis, clm) -> Boolean.TRUE.equals(vis) || Boolean.TRUE.equals(clm)))
              .thenCompose(keep -> keep
                  ? CompletableFuture.supplyAsync(
                      () -> true,
                      CompletableFuture.delayedExecutor(pollDelayMs, TimeUnit.MILLISECONDS))
                  : CompletableFuture.completedFuture(false)))
          .thenApply(v -> null);
    } catch (Exception e) {
      CompletableFuture<Void> f = new CompletableFuture<>();
      f.completeExceptionally(e);
      return f;
    }
  }
}
