package io.github.panghy.vectorsearch.workers;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import io.github.panghy.taskqueue.TaskClaim;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.proto.LinkTask;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.metrics.LongCounter;
import io.opentelemetry.api.metrics.LongHistogram;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Worker that processes link tasks to build the proximity graph.
 *
 * <p>This worker:
 * <ul>
 *   <li>Claims multiple tasks in batches for efficiency</li>
 *   <li>Persists PQ codes to storage blocks</li>
 *   <li>Discovers neighbors using beam search</li>
 *   <li>Updates adjacency lists with bidirectional links</li>
 *   <li>Adaptively adjusts batch size based on transaction timing</li>
 * </ul>
 */
public class LinkWorker implements Runnable {

  private static final Logger LOGGER = Logger.getLogger(LinkWorker.class.getName());

  // OpenTelemetry instrumentation
  private static final Tracer TRACER = GlobalOpenTelemetry.getTracer("io.github.panghy.vectorsearch", "0.1.0");
  private static final Meter METER = GlobalOpenTelemetry.getMeter("io.github.panghy.vectorsearch");

  // Metrics
  private static final LongCounter TASKS_CLAIMED = METER.counterBuilder("vectorsearch.linkworker.tasks.claimed")
      .setDescription("Number of link tasks claimed")
      .setUnit("tasks")
      .build();

  private static final LongCounter TASKS_COMPLETED = METER.counterBuilder("vectorsearch.linkworker.tasks.completed")
      .setDescription("Number of link tasks completed")
      .setUnit("tasks")
      .build();

  private static final LongCounter TASKS_FAILED = METER.counterBuilder("vectorsearch.linkworker.tasks.failed")
      .setDescription("Number of link tasks failed")
      .setUnit("tasks")
      .build();

  private static final LongHistogram BATCH_SIZE = METER.histogramBuilder("vectorsearch.linkworker.batch.size")
      .setDescription("Size of task batches processed")
      .setUnit("tasks")
      .ofLongs()
      .build();

  private static final LongHistogram TRANSACTION_DURATION = METER.histogramBuilder(
          "vectorsearch.linkworker.transaction.duration")
      .setDescription("Duration of link transactions")
      .setUnit("ms")
      .ofLongs()
      .build();

  // Configuration
  private static final Duration CLAIM_TIMEOUT = Duration.ofMillis(100);
  private static final Duration CLAIM_BUDGET = Duration.ofMillis(500);
  private static final Duration TRANSACTION_BUDGET = Duration.ofSeconds(4);
  private static final int MIN_BATCH_SIZE = 1;
  private static final int MAX_BATCH_SIZE = 100;
  private static final int INITIAL_BATCH_SIZE = 10;

  // Dependencies
  private final Database database;
  private final TaskQueue<Long, LinkTask> taskQueue;
  private final VectorIndexKeys keys;
  private final PqBlockStorage pqBlockStorage;
  private final NodeAdjacencyStorage nodeAdjacencyStorage;
  private final EntryPointStorage entryPointStorage;
  private final CodebookStorage codebookStorage;
  private final BeamSearchEngine searchEngine;
  private final InstantSource instantSource;

  // Runtime state
  private final AtomicBoolean running = new AtomicBoolean(true);
  private final AtomicInteger currentBatchSize = new AtomicInteger(INITIAL_BATCH_SIZE);
  private final AtomicLong totalProcessed = new AtomicLong(0);
  private final AtomicLong totalFailed = new AtomicLong(0);

  // Worker configuration
  private final int graphDegree;
  private final int maxSearchVisits;
  private final double pruningAlpha;

  /**
   * Creates a new link worker.
   *
   * @param database              FDB database instance
   * @param taskQueue            task queue for link operations
   * @param keys                 vector index keys helper
   * @param pqBlockStorage       storage for PQ blocks
   * @param nodeAdjacencyStorage storage for adjacency lists
   * @param entryPointStorage    storage for entry points
   * @param codebookStorage      storage for codebooks
   * @param searchEngine         beam search engine for neighbor discovery
   * @param graphDegree         maximum neighbors per node
   * @param maxSearchVisits     maximum nodes to visit during search
   * @param pruningAlpha        alpha parameter for robust pruning
   * @param instantSource       time source
   */
  public LinkWorker(
      Database database,
      TaskQueue<Long, LinkTask> taskQueue,
      VectorIndexKeys keys,
      PqBlockStorage pqBlockStorage,
      NodeAdjacencyStorage nodeAdjacencyStorage,
      EntryPointStorage entryPointStorage,
      CodebookStorage codebookStorage,
      BeamSearchEngine searchEngine,
      int graphDegree,
      int maxSearchVisits,
      double pruningAlpha,
      InstantSource instantSource) {
    this.database = database;
    this.taskQueue = taskQueue;
    this.keys = keys;
    this.pqBlockStorage = pqBlockStorage;
    this.nodeAdjacencyStorage = nodeAdjacencyStorage;
    this.entryPointStorage = entryPointStorage;
    this.codebookStorage = codebookStorage;
    this.searchEngine = searchEngine;
    this.graphDegree = graphDegree;
    this.maxSearchVisits = maxSearchVisits;
    this.pruningAlpha = pruningAlpha;
    this.instantSource = instantSource;
  }

  @Override
  public void run() {
    LOGGER.info("Link worker started");

    while (running.get()) {
      try {
        processBatch();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        LOGGER.info("Link worker interrupted");
        break;
      } catch (Exception e) {
        LOGGER.log(Level.SEVERE, "Error processing batch", e);
        // Continue processing after error
        try {
          Thread.sleep(1000); // Brief pause after error
        } catch (InterruptedException ie) {
          Thread.currentThread().interrupt();
          break;
        }
      }
    }

    LOGGER.info(String.format(
        "Link worker stopped. Processed: %d, Failed: %d", totalProcessed.get(), totalFailed.get()));
  }

  /**
   * Processes a batch of link tasks.
   */
  private void processBatch() throws InterruptedException {
    Span span = TRACER.spanBuilder("LinkWorker.processBatch")
        .setSpanKind(SpanKind.INTERNAL)
        .startSpan();

    try {
      // Phase 1: Claim tasks
      List<TaskClaim<Long, LinkTask>> claims = claimTasks();
      if (claims.isEmpty()) {
        // No tasks available, wait a bit
        Thread.sleep(100);
        return;
      }

      BATCH_SIZE.record(claims.size());
      span.setAttribute("batch.size", claims.size());

      // Phase 2: Process tasks in a single transaction
      Instant startTime = instantSource.instant();
      boolean success = processTasksInTransaction(claims);
      Duration transactionDuration = Duration.between(startTime, instantSource.instant());

      TRANSACTION_DURATION.record(transactionDuration.toMillis());
      span.setAttribute("transaction.duration_ms", transactionDuration.toMillis());

      // Phase 3: Complete or fail tasks based on result
      if (success) {
        completeTasks(claims);
        totalProcessed.addAndGet(claims.size());
        adjustBatchSize(transactionDuration, true);
        span.setStatus(StatusCode.OK);
      } else {
        failTasks(claims);
        totalFailed.addAndGet(claims.size());
        adjustBatchSize(transactionDuration, false);
        span.setStatus(StatusCode.ERROR, "Transaction failed");
      }

    } finally {
      span.end();
    }
  }

  /**
   * Claims multiple tasks up to the current batch size.
   */
  private List<TaskClaim<Long, LinkTask>> claimTasks() throws InterruptedException {
    List<TaskClaim<Long, LinkTask>> claims = new ArrayList<>();
    Instant claimDeadline = instantSource.instant().plus(CLAIM_BUDGET);
    int targetSize = currentBatchSize.get();

    for (int i = 0; i < targetSize; i++) {
      if (instantSource.instant().isAfter(claimDeadline)) {
        LOGGER.fine("Claim budget exhausted after " + i + " claims");
        break;
      }

      try {
        CompletableFuture<TaskClaim<Long, LinkTask>> claimFuture = taskQueue.awaitAndClaimTask(database);

        TaskClaim<Long, LinkTask> claim = claimFuture.get(CLAIM_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS);

        claims.add(claim);
        TASKS_CLAIMED.add(1);

      } catch (TimeoutException e) {
        // No more tasks immediately available
        LOGGER.fine("No task available within timeout");
        break;
      } catch (Exception e) {
        LOGGER.log(Level.WARNING, "Failed to claim task", e);
        break;
      }
    }

    LOGGER.fine("Claimed " + claims.size() + " tasks");
    return claims;
  }

  /**
   * Processes all claimed tasks in a single transaction.
   */
  private boolean processTasksInTransaction(List<TaskClaim<Long, LinkTask>> claims) {
    try {
      return database.runAsync(tr -> {
            Instant deadline = instantSource.instant().plus(TRANSACTION_BUDGET);

            // Group tasks by PQ block for efficient updates
            Map<Long, List<TaskClaim<Long, LinkTask>>> tasksByBlock = groupTasksByBlock(claims);

            // Process each group
            List<CompletableFuture<Void>> futures = new ArrayList<>();

            for (Map.Entry<Long, List<TaskClaim<Long, LinkTask>>> entry : tasksByBlock.entrySet()) {
              futures.add(processBlockGroup(tr, entry.getKey(), entry.getValue(), deadline));
            }

            // Wait for all operations to complete
            return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> {
                  // Check if we're still within time budget
                  if (instantSource.instant().isAfter(deadline)) {
                    throw new RuntimeException("Transaction exceeded time budget");
                  }
                  return true;
                });
          })
          .get(TRANSACTION_BUDGET.toSeconds() + 1, TimeUnit.SECONDS);

    } catch (Exception e) {
      LOGGER.log(Level.WARNING, "Transaction failed", e);
      return false;
    }
  }

  /**
   * Groups tasks by their PQ block number for efficient batch updates.
   */
  private Map<Long, List<TaskClaim<Long, LinkTask>>> groupTasksByBlock(List<TaskClaim<Long, LinkTask>> claims) {
    Map<Long, List<TaskClaim<Long, LinkTask>>> groups = new HashMap<>();

    for (TaskClaim<Long, LinkTask> claim : claims) {
      long nodeId = claim.taskKey();
      long blockNumber = pqBlockStorage.getBlockNumber(nodeId);
      groups.computeIfAbsent(blockNumber, k -> new ArrayList<>()).add(claim);
    }

    return groups;
  }

  /**
   * Processes all tasks in a single PQ block group.
   */
  private CompletableFuture<Void> processBlockGroup(
      Transaction tr, long blockNumber, List<TaskClaim<Long, LinkTask>> claims, Instant deadline) {

    // Step 1: Persist PQ codes for all nodes in this block
    List<CompletableFuture<Void>> persistFutures = new ArrayList<>();
    for (TaskClaim<Long, LinkTask> claim : claims) {
      LinkTask task = claim.task();
      persistFutures.add(pqBlockStorage.storePqCode(
          tr, claim.taskKey(), task.getPqCode().toByteArray(), (int) task.getCodebookVersion()));
    }

    return CompletableFuture.allOf(persistFutures.toArray(new CompletableFuture[0]))
        .thenCompose(v -> {
          // Step 2: Discover neighbors for all nodes
          List<CompletableFuture<List<Long>>> neighborFutures = new ArrayList<>();

          for (TaskClaim<Long, LinkTask> claim : claims) {
            neighborFutures.add(discoverNeighbors(tr, claim.taskKey(), claim.task()));
          }

          return CompletableFuture.allOf(neighborFutures.toArray(new CompletableFuture[0]))
              .thenCompose($ -> {
                // Step 3: Update adjacency lists with back-links
                List<CompletableFuture<Void>> updateFutures = new ArrayList<>();

                for (int i = 0; i < claims.size(); i++) {
                  TaskClaim<Long, LinkTask> claim = claims.get(i);
                  List<Long> neighbors =
                      neighborFutures.get(i).join();

                  updateFutures.add(updateAdjacencyWithBacklinks(tr, claim.taskKey(), neighbors));
                }

                return CompletableFuture.allOf(updateFutures.toArray(new CompletableFuture[0]));
              });
        });
  }

  /**
   * Discovers neighbors for a node using beam search.
   */
  private CompletableFuture<List<Long>> discoverNeighbors(Transaction tr, long nodeId, LinkTask task) {

    // Load entry points
    return entryPointStorage.getEntryPoints(tr).thenCompose(entryPoints -> {
      if (entryPoints.isEmpty()) {
        // No entry points yet, this might be one of the first nodes
        return completedFuture(new ArrayList<>());
      }

      // Use beam search to find neighbors
      // Note: This is a simplified version. The actual implementation
      // would need to reconstruct the vector from PQ codes or use
      // the vector_prefix if available
      return searchEngine
          .searchForNeighbors(
              tr,
              nodeId,
              task.getPqCode().toByteArray(),
              (int) task.getCodebookVersion(),
              entryPoints,
              graphDegree * 2, // Search for more candidates
              maxSearchVisits)
          .thenApply(candidates -> {
            // Apply robust pruning to select final neighbors
            // Note: candidates is already a List<Long> from searchForNeighbors
            return nodeAdjacencyStorage.robustPruneSimple(candidates, graphDegree, pruningAlpha);
          });
    });
  }

  /**
   * Updates adjacency list for a node and adds back-links.
   */
  private CompletableFuture<Void> updateAdjacencyWithBacklinks(Transaction tr, long nodeId, List<Long> neighbors) {

    // Update this node's adjacency list
    return nodeAdjacencyStorage.getNodeAdjacency(tr, nodeId).thenCompose(existing -> {
      Set<Long> currentNeighbors =
          existing != null ? new HashSet<>(existing.getNeighborsList()) : new HashSet<>();

      // Merge with new neighbors
      currentNeighbors.addAll(neighbors);

      // Prune to degree limit if needed
      final List<Long> prunedNeighbors;
      if (currentNeighbors.size() > graphDegree) {
        prunedNeighbors = nodeAdjacencyStorage.robustPruneSimple(
            new ArrayList<>(currentNeighbors), graphDegree, pruningAlpha);
      } else {
        prunedNeighbors = new ArrayList<>(currentNeighbors);
      }

      // Update this node
      NodeAdjacency updated = NodeAdjacency.newBuilder()
          .setNodeId(nodeId)
          .addAllNeighbors(prunedNeighbors)
          .setVersion(existing != null ? existing.getVersion() + 1 : 1)
          .setState(NodeAdjacency.State.ACTIVE)
          .build();

      return nodeAdjacencyStorage.storeNodeAdjacency(tr, nodeId, updated).thenCompose(v -> {
        // Add back-links to all neighbors
        List<CompletableFuture<Void>> backLinkFutures = new ArrayList<>();

        for (Long neighborId : prunedNeighbors) {
          backLinkFutures.add(addBackLink(tr, neighborId, nodeId));
        }

        return CompletableFuture.allOf(backLinkFutures.toArray(new CompletableFuture[0]));
      });
    });
  }

  /**
   * Adds a back-link from neighbor to node.
   */
  private CompletableFuture<Void> addBackLink(Transaction tr, long neighborId, long nodeId) {
    return nodeAdjacencyStorage.getNodeAdjacency(tr, neighborId).thenCompose(existing -> {
      if (existing == null) {
        // Neighbor doesn't exist yet, create empty adjacency
        existing = NodeAdjacency.newBuilder()
            .setNodeId(neighborId)
            .setState(NodeAdjacency.State.ACTIVE)
            .build();
      }

      Set<Long> neighbors = new HashSet<>(existing.getNeighborsList());
      if (neighbors.contains(nodeId)) {
        // Back-link already exists
        return completedFuture(null);
      }

      neighbors.add(nodeId);

      // Prune if over degree limit
      final List<Long> prunedNeighbors;
      if (neighbors.size() > graphDegree) {
        prunedNeighbors =
            nodeAdjacencyStorage.robustPruneSimple(new ArrayList<>(neighbors), graphDegree, pruningAlpha);
      } else {
        prunedNeighbors = new ArrayList<>(neighbors);
      }

      NodeAdjacency updated = existing.toBuilder()
          .clearNeighbors()
          .addAllNeighbors(prunedNeighbors)
          .setVersion(existing.getVersion() + 1)
          .build();

      return nodeAdjacencyStorage.storeNodeAdjacency(tr, neighborId, updated);
    });
  }

  /**
   * Marks all tasks as completed.
   */
  private void completeTasks(List<TaskClaim<Long, LinkTask>> claims) {
    for (TaskClaim<Long, LinkTask> claim : claims) {
      try {
        claim.complete().get(5, TimeUnit.SECONDS);
        TASKS_COMPLETED.add(1);
      } catch (Exception e) {
        LOGGER.log(Level.WARNING, "Failed to complete task " + claim.taskKey(), e);
        TASKS_FAILED.add(1);
      }
    }
  }

  /**
   * Marks all tasks as failed for retry.
   */
  private void failTasks(List<TaskClaim<Long, LinkTask>> claims) {
    for (TaskClaim<Long, LinkTask> claim : claims) {
      try {
        claim.fail().get(5, TimeUnit.SECONDS);
        TASKS_FAILED.add(1);
      } catch (Exception e) {
        LOGGER.log(Level.WARNING, "Failed to fail task " + claim.taskKey(), e);
      }
    }
  }

  /**
   * Adjusts batch size based on transaction performance.
   */
  private void adjustBatchSize(Duration transactionDuration, boolean success) {
    int currentSize = currentBatchSize.get();
    int newSize;

    if (!success) {
      // Transaction failed, reduce batch size significantly
      newSize = Math.max(MIN_BATCH_SIZE, currentSize / 2);
    } else if (transactionDuration.toMillis() < 3000) {
      // Transaction fast, increase batch size
      newSize = Math.min(MAX_BATCH_SIZE, (int) (currentSize * 1.2));
    } else if (transactionDuration.toMillis() > 4000) {
      // Transaction slow, decrease batch size
      newSize = Math.max(MIN_BATCH_SIZE, (int) (currentSize * 0.8));
    } else {
      // Transaction time good, keep current size
      newSize = currentSize;
    }

    if (newSize != currentSize) {
      currentBatchSize.set(newSize);
      LOGGER.info("Adjusted batch size from " + currentSize + " to " + newSize);
    }
  }

  /**
   * Stops the worker gracefully.
   */
  public void stop() {
    LOGGER.info("Stopping link worker");
    running.set(false);
  }

  /**
   * Gets the current batch size.
   */
  public int getCurrentBatchSize() {
    return currentBatchSize.get();
  }

  /**
   * Gets total tasks processed.
   */
  public long getTotalProcessed() {
    return totalProcessed.get();
  }

  /**
   * Gets total tasks failed.
   */
  public long getTotalFailed() {
    return totalFailed.get();
  }
}
