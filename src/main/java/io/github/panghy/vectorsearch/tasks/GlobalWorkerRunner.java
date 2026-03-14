package io.github.panghy.vectorsearch.tasks;

import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.async.AsyncUtil;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.config.GlobalTaskQueueConfig;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.config.WorkerConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.fdb.FdbDirectories.IndexDirectories;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import io.github.panghy.vectorsearch.proto.IndexMeta;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Standalone worker that claims tasks from global (cross-index) build and maintenance queues
 * and dispatches them to the appropriate per-index services.
 *
 * <p>For each claimed task the runner:
 * <ol>
 *   <li>Extracts the {@code index_path} from the wrapper proto.</li>
 *   <li>Resolves (and caches) the {@link IndexDirectories} for that path.</li>
 *   <li>Reads the persisted {@link IndexMeta} to reconstruct a {@link VectorIndexConfig}
 *       with data params from the index and operational settings from the template.</li>
 *   <li>Delegates to {@link SegmentBuildService} (build) or {@link MaintenanceWorker}-style
 *       processing (maintenance).</li>
 * </ol>
 *
 * <p>Uses {@link AsyncUtil#whileTrue} for worker loops, matching the pattern of existing
 * pools like {@link SegmentBuildWorkerPool} and {@link MaintenanceWorkerPool}.</p>
 *
 * @see GlobalTaskQueueConfig
 */
public final class GlobalWorkerRunner implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(GlobalWorkerRunner.class);

  private final Database db;
  private final WorkerConfig workerConfig;
  private final TaskQueue<String, GlobalBuildTask> buildQueue;
  private final TaskQueue<String, GlobalMaintenanceTask> maintenanceQueue;
  private final AtomicBoolean running = new AtomicBoolean(false);
  private final List<CompletableFuture<Void>> loops = new ArrayList<>();
  private int buildThreadCount = 0;
  private int maintThreadCount = 0;

  // Cache IndexDirectories per index path to avoid repeated DirectoryLayer opens.
  private final ConcurrentHashMap<List<String>, CompletableFuture<IndexDirectories>> indexDirsCache =
      new ConcurrentHashMap<>();

  /**
   * Creates a new {@code GlobalWorkerRunner}.
   *
   * @param db           the FoundationDB database handle
   * @param workerConfig operational settings applied to every index processed by this runner
   * @param globalConfig the global task queue configuration holding both queues
   */
  public GlobalWorkerRunner(Database db, WorkerConfig workerConfig, GlobalTaskQueueConfig globalConfig) {
    this.db = requireNonNull(db, "db");
    this.workerConfig = requireNonNull(workerConfig, "workerConfig");
    requireNonNull(globalConfig, "globalConfig");
    this.buildQueue = globalConfig.getBuildQueue();
    this.maintenanceQueue = globalConfig.getMaintenanceQueue();
  }

  /**
   * Starts {@code buildWorkers} build loops and {@code maintWorkers} maintenance loops.
   *
   * <p>No-op if already started or both counts are zero.</p>
   *
   * @param buildWorkers number of build worker loops
   * @param maintWorkers number of maintenance worker loops
   */
  public synchronized void start(int buildWorkers, int maintWorkers) {
    if (running.get()) return;
    if (buildWorkers <= 0 && maintWorkers <= 0) return;
    running.set(true);
    buildThreadCount = Math.max(0, buildWorkers);
    maintThreadCount = Math.max(0, maintWorkers);
    LOG.debug("GlobalWorkerRunner starting build={} maint={}", buildThreadCount, maintThreadCount);

    for (int i = 0; i < buildThreadCount; i++) {
      CompletableFuture<Void> loop = AsyncUtil.whileTrue(() -> {
        if (!running.get()) return completedFuture(false);
        return runOnceBuild()
            .handle((ok, ex) -> {
              if (ex != null) LOG.debug("Build loop error (will retry)", ex);
              return true;
            })
            .thenApply(x -> running.get());
      });
      loops.add(loop);
    }

    for (int i = 0; i < maintThreadCount; i++) {
      CompletableFuture<Void> loop = AsyncUtil.whileTrue(() -> {
        if (!running.get()) return completedFuture(false);
        return runOnceMaint()
            .handle((ok, ex) -> {
              if (ex != null) LOG.debug("Maintenance loop error (will retry)", ex);
              return true;
            })
            .thenApply(x -> running.get());
      });
      loops.add(loop);
    }
  }

  /** Stops all worker loops and enqueues sentinel tasks to wake blocked workers. */
  @Override
  public synchronized void close() {
    running.set(false);
    LOG.debug(
        "GlobalWorkerRunner stopping; signaling sentinels build={} maint={}",
        buildThreadCount,
        maintThreadCount);
    // Build sentinels
    for (int i = 0; i < buildThreadCount; i++) {
      String key = "global-build:-1:shutdown:" + i + ":" + System.nanoTime();
      GlobalBuildTask sentinel = GlobalBuildTask.newBuilder()
          .setTask(BuildTask.newBuilder().setSegId(-1).build())
          .build();
      buildQueue.enqueueIfNotExists(key, sentinel).exceptionally(ex -> null);
    }
    // Maintenance sentinels
    for (int i = 0; i < maintThreadCount; i++) {
      String key = "global-maint:-1:shutdown:" + i + ":" + System.nanoTime();
      GlobalMaintenanceTask sentinel = GlobalMaintenanceTask.newBuilder()
          .setTask(MaintenanceTask.newBuilder()
              .setVacuum(MaintenanceTask.Vacuum.newBuilder()
                  .setSegId(-1)
                  .build())
              .build())
          .build();
      maintenanceQueue.enqueueIfNotExists(key, sentinel).exceptionally(ex -> null);
    }
    loops.clear();
  }

  /**
   * Claims and processes a single global build task.
   *
   * @return future completing with {@code true} if a task was processed
   */
  CompletableFuture<Boolean> runOnceBuild() {
    return buildQueue.awaitAndClaimTask().thenCompose(claim -> {
      GlobalBuildTask gbt = claim.task();
      BuildTask bt = gbt.getTask();
      // Sentinel: seg_id < 0 means shutdown signal
      if (bt.getSegId() < 0) {
        return claim.complete().thenApply(v -> true);
      }
      List<String> indexPath = gbt.getIndexPathList();
      return resolveIndexDirs(indexPath)
          .thenCompose(dirs -> buildConfigForIndex(dirs, indexPath).thenCompose(cfg -> {
            SegmentBuildService svc = new SegmentBuildService(cfg, dirs);
            return svc.build(bt.getSegId());
          }))
          .handle((v, ex) -> ex)
          .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
          .thenApply(v -> true);
    });
  }

  /**
   * Claims and processes a single global maintenance task.
   *
   * @return future completing with {@code true} if a task was processed
   */
  CompletableFuture<Boolean> runOnceMaint() {
    return maintenanceQueue.awaitAndClaimTask().thenCompose(claim -> {
      GlobalMaintenanceTask gmt = claim.task();
      MaintenanceTask mt = gmt.getTask();
      // Sentinel: vacuum with seg_id < 0 means shutdown signal
      if (mt.hasVacuum() && mt.getVacuum().getSegId() < 0) {
        return claim.complete().thenApply(v -> true);
      }
      List<String> indexPath = gmt.getIndexPathList();
      return resolveIndexDirs(indexPath)
          .thenCompose(dirs -> buildConfigForIndex(dirs, indexPath)
              .thenCompose(cfg -> processMaintenanceTask(mt, cfg, dirs)))
          .handle((v, ex) -> ex)
          .thenCompose(ex -> (ex == null) ? claim.complete() : claim.fail())
          .thenApply(v -> true);
    });
  }

  private CompletableFuture<Void> processMaintenanceTask(
      MaintenanceTask t, VectorIndexConfig cfg, IndexDirectories dirs) {
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    if (t.hasVacuum()) {
      var v = t.getVacuum();
      return svc.vacuumSegment(v.getSegId(), v.getMinDeletedRatio());
    }
    if (t.hasCompact()) {
      return svc.compactSegments(t.getCompact().getSegIdsList());
    }
    if (t.hasFindCandidates()) {
      return svc.findCompactionCandidates(t.getFindCandidates().getAnchorSegId())
          .thenApply(v -> null);
    }
    return completedFuture(null);
  }

  /**
   * Resolves and caches {@link IndexDirectories} for the given index path.
   */
  private CompletableFuture<IndexDirectories> resolveIndexDirs(List<String> indexPath) {
    return indexDirsCache.computeIfAbsent(
        List.copyOf(indexPath),
        path -> DirectoryLayer.getDefault()
            .open(db, path)
            .thenCompose(indexDir -> FdbDirectories.openIndex(indexDir, db)));
  }

  /**
   * Reads persisted {@link IndexMeta} from the index and reconstructs a {@link VectorIndexConfig}
   * using data params from the meta and operational settings from the {@link WorkerConfig}.
   *
   * <p>Package-private for testing.</p>
   */
  CompletableFuture<VectorIndexConfig> buildConfigForIndex(IndexDirectories dirs, List<String> indexPath) {
    return db.readAsync(tr -> tr.get(dirs.metaKey())).thenCompose(metaBytes -> {
      if (metaBytes == null) {
        return CompletableFuture.failedFuture(
            new IllegalStateException("No IndexMeta found for path: " + indexPath));
      }
      IndexMeta meta;
      try {
        meta = IndexMeta.parseFrom(metaBytes);
      } catch (InvalidProtocolBufferException e) {
        return CompletableFuture.failedFuture(
            new IllegalStateException("Corrupted IndexMeta at path: " + indexPath, e));
      }
      VectorIndexConfig.Metric metric = (meta.getMetric() == IndexMeta.Metric.METRIC_COSINE)
          ? VectorIndexConfig.Metric.COSINE
          : VectorIndexConfig.Metric.L2;
      WorkerConfig wc = workerConfig;
      // Build config with data params from IndexMeta + ALL operational settings from WorkerConfig
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, dirs.indexDir())
          .dimension(meta.getDimension())
          .metric(metric)
          .maxSegmentSize(
              meta.getMaxSegmentSize() > 0 ? meta.getMaxSegmentSize() : wc.getDefaultMaxSegmentSize())
          .pqM(meta.getPqM() > 0 ? meta.getPqM() : wc.getDefaultPqM())
          .pqK(meta.getPqK() > 1 ? meta.getPqK() : wc.getDefaultPqK())
          .graphDegree(meta.getGraphDegree() > 0 ? meta.getGraphDegree() : wc.getDefaultGraphDegree())
          .oversample(meta.getOversample() > 0 ? meta.getOversample() : wc.getDefaultOversample())
          .graphBuildBreadth(
              meta.getGraphBuildBreadth() > 0
                  ? meta.getGraphBuildBreadth()
                  : wc.getDefaultGraphBuildBreadth())
          .graphAlpha(meta.getGraphAlpha() > 0 ? meta.getGraphAlpha() : wc.getDefaultGraphAlpha())
          // No local workers for global runner
          .localWorkerThreads(0)
          .localMaintenanceWorkerThreads(0)
          // ALL operational settings from WorkerConfig
          .estimatedWorkerCount(wc.getEstimatedWorkerCount())
          .defaultTtl(wc.getDefaultTtl())
          .defaultThrottle(wc.getDefaultThrottle())
          .maxConcurrentCompactions(wc.getMaxConcurrentCompactions())
          .buildTxnLimitBytes(wc.getBuildTxnLimitBytes())
          .buildTxnSoftLimitRatio(wc.getBuildTxnSoftLimitRatio())
          .buildSizeCheckEvery(wc.getBuildSizeCheckEvery())
          .vacuumCooldown(wc.getVacuumCooldown())
          .vacuumMinDeletedRatio(wc.getVacuumMinDeletedRatio())
          .autoFindCompactionCandidates(wc.isAutoFindCompactionCandidates())
          .compactionMinSegments(wc.getCompactionMinSegments())
          .compactionMaxSegments(wc.getCompactionMaxSegments())
          .compactionMinFragmentation(wc.getCompactionMinFragmentation())
          .compactionAgeBiasWeight(wc.getCompactionAgeBiasWeight())
          .compactionSizeBiasWeight(wc.getCompactionSizeBiasWeight())
          .compactionFragBiasWeight(wc.getCompactionFragBiasWeight())
          .codebookBatchLoadSize(wc.getCodebookBatchLoadSize())
          .adjacencyBatchLoadSize(wc.getAdjacencyBatchLoadSize())
          .prefetchCodebooksEnabled(wc.isPrefetchCodebooksEnabled())
          .prefetchCodebooksSync(wc.isPrefetchCodebooksSync())
          .instantSource(wc.getInstantSource())
          .metricAttributes(wc.getMetricAttributes())
          .build();
      return completedFuture(cfg);
    });
  }

  /** Returns {@code true} if the runner is currently active. Visible for testing. */
  boolean isRunning() {
    return running.get();
  }
}
