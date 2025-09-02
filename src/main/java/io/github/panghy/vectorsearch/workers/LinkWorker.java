package io.github.panghy.vectorsearch.workers;

import static java.util.Objects.requireNonNullElseGet;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.async.AsyncUtil;
import io.github.panghy.taskqueue.TaskClaim;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.vectorsearch.graph.RobustPruning;
import io.github.panghy.vectorsearch.graph.RobustPruning.Candidate;
import io.github.panghy.vectorsearch.graph.RobustPruning.PruningConfig;
import io.github.panghy.vectorsearch.proto.LinkTask;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.proto.PqEncodedVector;
import io.github.panghy.vectorsearch.proto.RawVector;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.search.SearchResult;
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
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

  private static final Logger LOGGER = LoggerFactory.getLogger(LinkWorker.class);

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
  private static final int MAX_BATCH_SIZE = 10000;
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
   * @param database             FDB database instance
   * @param taskQueue            task queue for link operations
   * @param keys                 vector index keys helper
   * @param pqBlockStorage       storage for PQ blocks
   * @param nodeAdjacencyStorage storage for adjacency lists
   * @param entryPointStorage    storage for entry points
   * @param codebookStorage      storage for codebooks
   * @param searchEngine         beam search engine for neighbor discovery
   * @param graphDegree          maximum neighbors per node
   * @param maxSearchVisits      maximum nodes to visit during search
   * @param pruningAlpha         alpha parameter for robust pruning
   * @param instantSource        time source
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

    // Use AsyncUtil.whileTrue for the main loop
    AsyncUtil.whileTrue(() -> {
          if (!running.get()) {
            return completedFuture(false);
          }

          return processBatchAsync().thenApply(v -> running.get()).exceptionally(e -> {
            if (e.getCause() instanceof InterruptedException) {
              Thread.currentThread().interrupt();
              LOGGER.info("Link worker interrupted");
              return false;
            }
            LOGGER.error("Error processing batch", e);
            return running.get();
          });
        })
        .join();

    LOGGER.info("Link worker stopped. Processed: {}, Failed: {}", totalProcessed.get(), totalFailed.get());
  }

  /**
   * Processes a batch of link tasks asynchronously.
   */
  CompletableFuture<Void> processBatchAsync() {
    Span span = TRACER.spanBuilder("LinkWorker.processBatch")
        .setSpanKind(SpanKind.INTERNAL)
        .startSpan();

    // Phase 1: Claim tasks
    return claimTasksAsync()
        .thenCompose(claims -> {
          if (claims.isEmpty()) {
            // No tasks available, wait a bit
            CompletableFuture<Void> delay = new CompletableFuture<>();
            return delay.completeOnTimeout(null, 100, TimeUnit.MILLISECONDS);
          }

          BATCH_SIZE.record(claims.size());
          span.setAttribute("batch.size", claims.size());

          // Phase 2: Process tasks using common logic
          Instant startTime = instantSource.instant();
          return processClaimedTasksAsync(claims).thenAccept(success -> {
            Duration transactionDuration = Duration.between(startTime, instantSource.instant());
            TRANSACTION_DURATION.record(transactionDuration.toMillis());
            span.setAttribute("transaction.duration_ms", transactionDuration.toMillis());

            // Phase 3: Update metrics and adjust batch size
            if (success) {
              totalProcessed.addAndGet(claims.size());
              adjustBatchSize(transactionDuration, true);
              span.setStatus(StatusCode.OK);
            } else {
              totalFailed.addAndGet(claims.size());
              adjustBatchSize(transactionDuration, false);
              span.setStatus(StatusCode.ERROR, "Transaction failed");
            }
          });
        })
        .whenComplete((v, e) -> span.end());
  }

  /**
   * Claims multiple tasks up to the current batch size asynchronously.
   */
  CompletableFuture<List<TaskClaim<Long, LinkTask>>> claimTasksAsync() {
    List<TaskClaim<Long, LinkTask>> claims = new ArrayList<>();
    Instant claimDeadline = instantSource.instant().plus(CLAIM_BUDGET);
    int targetSize = currentBatchSize.get();

    return claimTasksRecursive(claims, targetSize, claimDeadline, 0);
  }

  /**
   * Recursively claims tasks until batch size is reached or deadline expires.
   */
  private CompletableFuture<List<TaskClaim<Long, LinkTask>>> claimTasksRecursive(
      List<TaskClaim<Long, LinkTask>> claims, int targetSize, Instant claimDeadline, int index) {

    if (index >= targetSize || instantSource.instant().isAfter(claimDeadline)) {
      if (index < targetSize) {
        LOGGER.debug("Claim budget exhausted after {} claims", index);
      }
      LOGGER.debug("Claimed {} tasks", claims.size());
      return completedFuture(claims);
    }

    CompletableFuture<TaskClaim<Long, LinkTask>> claimFuture = taskQueue.awaitAndClaimTask(database);

    // Add timeout to the claim future
    CompletableFuture<TaskClaim<Long, LinkTask>> timeoutFuture = new CompletableFuture<>();
    claimFuture.orTimeout(CLAIM_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS).whenComplete((claim, error) -> {
      if (error != null) {
        if (error instanceof TimeoutException) {
          LOGGER.debug("No task available within timeout");
        } else {
          LOGGER.warn("Failed to claim task", error);
        }
        timeoutFuture.complete(null);
      } else {
        timeoutFuture.complete(claim);
      }
    });

    return timeoutFuture.thenCompose(claim -> {
      if (claim == null) {
        // No more tasks or error occurred, return what we have
        LOGGER.info("Claimed {} tasks", claims.size());
        return completedFuture(claims);
      }

      claims.add(claim);
      TASKS_CLAIMED.add(1);
      return claimTasksRecursive(claims, targetSize, claimDeadline, index + 1);
    });
  }

  /**
   * Common method to process claimed tasks asynchronously.
   * Handles the transaction, completion/failure, and returns the success status.
   */
  private CompletableFuture<Boolean> processClaimedTasksAsync(List<TaskClaim<Long, LinkTask>> claims) {
    return processTasksInTransactionAsync(claims).thenCompose(success -> {
      if (success) {
        LOGGER.info("Processed {} tasks", claims.size());
        return completeTasksAsync(claims).thenApply(v -> true);
      } else {
        LOGGER.warn("Failed to process {} tasks", claims.size());
        return failTasksAsync(claims).thenApply(v -> false);
      }
    });
  }

  /**
   * Processes all claimed tasks in a single transaction asynchronously.
   */
  private CompletableFuture<Boolean> processTasksInTransactionAsync(List<TaskClaim<Long, LinkTask>> claims) {
    LOGGER.info("Claimed {} tasks", claims.size());
    return database.runAsync(tr -> {
          // Group tasks by PQ block for efficient updates
          Map<Long, List<TaskClaim<Long, LinkTask>>> tasksByBlock = groupTasksByBlock(claims);

          // Process each group
          List<CompletableFuture<Void>> futures = new ArrayList<>();

          for (Map.Entry<Long, List<TaskClaim<Long, LinkTask>>> entry : tasksByBlock.entrySet()) {
            futures.add(processBlockGroup(tr, entry.getValue()));
          }

          // Wait for all operations to complete
          return allOf(futures.toArray(CompletableFuture[]::new)).thenApply($ -> true);
        })
        .orTimeout(TRANSACTION_BUDGET.toSeconds() + 1, TimeUnit.SECONDS)
        .handle((success, error) -> {
          if (error != null) {
            LOGGER.warn("Transaction failed", error);
            return false;
          }
          return success;
        });
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
  private CompletableFuture<Void> processBlockGroup(Transaction tr, List<TaskClaim<Long, LinkTask>> claims) {

    // First, process any raw vectors to encode them
    List<CompletableFuture<PqEncodedData>> encodingFutures = new ArrayList<>();
    for (TaskClaim<Long, LinkTask> claim : claims) {
      LinkTask task = claim.task();
      encodingFutures.add(getOrEncodeVector(task));
    }

    return allOf(encodingFutures.toArray(CompletableFuture[]::new)).thenCompose(v -> {
      // Now all tasks have PQ-encoded data
      List<PqEncodedData> encodedData =
          encodingFutures.stream().map(CompletableFuture::join).collect(Collectors.toList());

      // Get the codebook version (all should be the same after encoding)
      final int codebookVersion = encodedData.isEmpty() ? 1 : encodedData.get(0).codebookVersion;

      // Validate that the codebook version exists
      return codebookStorage.loadCodebooks(codebookVersion).thenCompose(codebooks -> {
        if (codebooks == null) {
          throw new IllegalArgumentException("Codebook version " + codebookVersion + " does not exist");
        }

        // Step 1: Persist PQ codes for all nodes in this block
        List<CompletableFuture<Void>> persistFutures = new ArrayList<>();
        for (int i = 0; i < claims.size(); i++) {
          TaskClaim<Long, LinkTask> claim = claims.get(i);
          PqEncodedData encoded = encodedData.get(i);
          if (encoded.pqCode != null) {
            persistFutures.add(pqBlockStorage.storePqCode(
                tr, claim.taskKey(), encoded.pqCode, encoded.codebookVersion));
          }
        }

        return allOf(persistFutures.toArray(CompletableFuture[]::new)).thenCompose($ -> {
          // Step 2: Discover neighbors for all nodes
          List<CompletableFuture<List<Long>>> neighborFutures = new ArrayList<>();

          for (int i = 0; i < claims.size(); i++) {
            TaskClaim<Long, LinkTask> claim = claims.get(i);
            PqEncodedData encoded = encodedData.get(i);
            neighborFutures.add(discoverNeighborsWithEncoded(
                tr, claim.taskKey(), encoded.pqCode, encoded.codebookVersion));
          }

          return allOf(neighborFutures.toArray(CompletableFuture[]::new))
              .thenCompose($$ -> {
                // Step 3: Update adjacency lists with back-links
                List<CompletableFuture<Void>> updateFutures = new ArrayList<>();

                for (int i = 0; i < claims.size(); i++) {
                  TaskClaim<Long, LinkTask> claim = claims.get(i);
                  List<Long> neighbors =
                      neighborFutures.get(i).join();

                  updateFutures.add(updateAdjacencyWithBacklinks(
                      tr, claim.taskKey(), neighbors, codebookVersion));
                }
                return allOf(updateFutures.toArray(CompletableFuture[]::new));
              });
        });
      });
    });
  }

  /**
   * Helper class to hold PQ-encoded data.
   */
  private static class PqEncodedData {
    final byte[] pqCode;
    final int codebookVersion;

    PqEncodedData(byte[] pqCode, int codebookVersion) {
      this.pqCode = pqCode;
      this.codebookVersion = codebookVersion;
    }
  }

  /**
   * Gets PQ-encoded data from task or encodes raw vector.
   */
  private CompletableFuture<PqEncodedData> getOrEncodeVector(LinkTask task) {
    switch (task.getVectorDataCase()) {
      case PQ_ENCODED:
        // Already encoded
        PqEncodedVector pqEncoded = task.getPqEncoded();
        return completedFuture(
            new PqEncodedData(pqEncoded.getPqCode().toByteArray(), (int) pqEncoded.getCodebookVersion()));

      case RAW_VECTOR:
        // Need to encode with current codebooks
        RawVector rawVector = task.getRawVector();
        float[] vector = new float[rawVector.getValuesCount()];
        for (int i = 0; i < vector.length; i++) {
          vector[i] = rawVector.getValues(i);
        }

        // Get the active codebook version and ProductQuantizer from storage
        return codebookStorage.getActiveVersion().thenCompose(cbv -> {
          if (cbv < 0) {
            return CompletableFuture.failedFuture(
                new IllegalStateException("No active codebook version available"));
          }
          return codebookStorage.getProductQuantizer(cbv).thenApply(pq -> {
            if (pq == null) {
              throw new IllegalStateException(
                  "Cannot encode vector - ProductQuantizer not available for version " + cbv);
            }
            // Encode the vector
            byte[] pqCode = pq.encode(vector);
            return new PqEncodedData(pqCode, cbv);
          });
        });

      case VECTORDATA_NOT_SET:
      default:
        // No vector data, can't process
        return CompletableFuture.failedFuture(new IllegalArgumentException("LinkTask has no vector data"));
    }
  }

  /**
   * Discovers neighbors for a node using beam search with encoded data.
   */
  private CompletableFuture<List<Long>> discoverNeighborsWithEncoded(
      Transaction tr, long nodeId, byte[] pqCode, int codebookVersion) {
    // If no PQ code, can't search for neighbors
    if (pqCode == null) {
      return completedFuture(new ArrayList<>());
    }

    // Load entry points
    return entryPointStorage.getEntryPoints(tr).thenCompose(entryPoints -> {
      if (entryPoints.isEmpty()) {
        // No entry points yet, this might be one of the first nodes
        return completedFuture(new ArrayList<>());
      }

      // Use beam search to find neighbors
      return searchEngine
          .searchForNeighbors(
              tr,
              nodeId,
              pqCode,
              codebookVersion,
              entryPoints,
              graphDegree * 2, // Search for more candidates
              maxSearchVisits)
          .thenApply(searchResults -> {
            // Convert SearchResults to RobustPruning.Candidates
            List<Candidate> candidates = new ArrayList<>();
            for (SearchResult result : searchResults) {
              candidates.add(Candidate.builder()
                  .nodeId(result.getNodeId())
                  .distanceToQuery(result.getDistance())
                  .build());
            }

            // Apply proper robust pruning with distances
            PruningConfig config = PruningConfig.builder()
                .maxDegree(graphDegree)
                .alpha(pruningAlpha)
                .build();

            return RobustPruning.prune(candidates, config);
          });
    });
  }

  /**
   * Updates adjacency list for a node and adds back-links.
   */
  private CompletableFuture<Void> updateAdjacencyWithBacklinks(
      Transaction tr, long nodeId, List<Long> neighbors, int codebookVersion) {

    // Update this node's adjacency list
    return nodeAdjacencyStorage.getNodeAdjacency(tr, nodeId).thenCompose(existing -> {
      Set<Long> currentNeighbors =
          existing != null ? new HashSet<>(existing.getNeighborsList()) : new HashSet<>();

      // Merge with new neighbors
      currentNeighbors.addAll(neighbors);

      // Prune to degree limit if needed
      final CompletableFuture<List<Long>> prunedNeighborsFuture;
      if (currentNeighbors.size() > graphDegree) {
        // Compute distances for all neighbors to apply proper robust pruning
        List<CompletableFuture<Candidate>> candidateFutures = new ArrayList<>();
        for (Long neighborId : currentNeighbors) {
          // Load PQ codes and compute distance
          CompletableFuture<Candidate> candidateFuture = pqBlockStorage
              .loadPqCode(neighborId, codebookVersion)
              .thenApply(pqCode -> {
                if (pqCode == null) {
                  // Node doesn't have PQ code yet, give it max distance
                  return Candidate.builder()
                      .nodeId(neighborId)
                      .distanceToQuery(Float.MAX_VALUE)
                      .build();
                }
                // For now, use a simplified distance (could be improved with actual PQ distance)
                // In a full implementation, we'd compute the actual PQ distance between nodes
                float distance = neighborId.equals(nodeId) ? 0.0f : 1.0f;
                return Candidate.builder()
                    .nodeId(neighborId)
                    .distanceToQuery(distance)
                    .build();
              });
          candidateFutures.add(candidateFuture);
        }

        // Wait for all distances and apply robust pruning
        prunedNeighborsFuture = allOf(candidateFutures.toArray(CompletableFuture[]::new))
            .thenApply(v -> {
              List<Candidate> candidates = candidateFutures.stream()
                  .map(CompletableFuture::join)
                  .collect(Collectors.toList());

              PruningConfig config = PruningConfig.builder()
                  .maxDegree(graphDegree)
                  .alpha(pruningAlpha)
                  .build();

              return RobustPruning.prune(candidates, config);
            });
      } else {
        prunedNeighborsFuture = completedFuture(new ArrayList<>(currentNeighbors));
      }

      return prunedNeighborsFuture.thenCompose(prunedNeighbors -> {

        // Update this node
        NodeAdjacency updated = NodeAdjacency.newBuilder()
            .setNodeId(nodeId)
            .addAllNeighbors(prunedNeighbors)
            .setVersion(existing != null ? existing.getVersion() + 1 : 1)
            .setState(NodeAdjacency.State.ACTIVE)
            .build();

        return nodeAdjacencyStorage
            .storeNodeAdjacency(tr, nodeId, updated)
            .thenCompose(v -> {
              // Add back-links to all neighbors
              List<CompletableFuture<Void>> backLinkFutures = new ArrayList<>();

              for (Long neighborId : prunedNeighbors) {
                backLinkFutures.add(addBackLink(tr, neighborId, nodeId, codebookVersion));
              }

              return allOf(backLinkFutures.toArray(CompletableFuture[]::new));
            });
      });
    });
  }

  /**
   * Adds a back-link from neighbor to node.
   */
  private CompletableFuture<Void> addBackLink(Transaction tr, long neighborId, long nodeId, int codebookVersion) {
    return nodeAdjacencyStorage.getNodeAdjacency(tr, neighborId).thenCompose(existingNode -> {
      final NodeAdjacency existing;
      // Neighbor doesn't exist yet, create empty adjacency
      existing = requireNonNullElseGet(existingNode, () -> NodeAdjacency.newBuilder()
          .setNodeId(neighborId)
          .setState(NodeAdjacency.State.ACTIVE)
          .build());

      Set<Long> neighbors = new HashSet<>(existing.getNeighborsList());
      if (neighbors.contains(nodeId)) {
        // Back-link already exists
        return completedFuture(null);
      }

      neighbors.add(nodeId);

      // Prune if over degree limit
      final CompletableFuture<List<Long>> prunedNeighborsFuture;
      if (neighbors.size() > graphDegree) {
        // Compute distances for robust pruning of back-links
        List<CompletableFuture<Candidate>> candidateFutures = new ArrayList<>();
        for (Long candidateId : neighbors) {
          CompletableFuture<Candidate> candidateFuture = pqBlockStorage
              .loadPqCode(candidateId, codebookVersion)
              .thenApply(pqCode -> {
                if (pqCode == null) {
                  return Candidate.builder()
                      .nodeId(candidateId)
                      .distanceToQuery(Float.MAX_VALUE)
                      .build();
                }
                // For now, use a simplified distance (could be improved with actual PQ distance)
                float distance = candidateId.equals(neighborId) ? 0.0f : 1.0f;
                return Candidate.builder()
                    .nodeId(candidateId)
                    .distanceToQuery(distance)
                    .build();
              });
          candidateFutures.add(candidateFuture);
        }

        prunedNeighborsFuture = allOf(candidateFutures.toArray(new CompletableFuture[0]))
            .thenApply(v -> {
              List<Candidate> candidates = candidateFutures.stream()
                  .map(CompletableFuture::join)
                  .collect(Collectors.toList());

              PruningConfig config = PruningConfig.builder()
                  .maxDegree(graphDegree)
                  .alpha(pruningAlpha)
                  .build();

              return RobustPruning.prune(candidates, config);
            });
      } else {
        prunedNeighborsFuture = completedFuture(new ArrayList<>(neighbors));
      }

      return prunedNeighborsFuture.thenApply(prunedNeighbors -> {
        NodeAdjacency updated = existing.toBuilder()
            .clearNeighbors()
            .addAllNeighbors(prunedNeighbors)
            .setVersion(existing.getVersion() + 1)
            .build();

        nodeAdjacencyStorage.storeNodeAdjacency(tr, neighborId, updated);
        return null;
      });
    });
  }

  /**
   * Marks all tasks as completed asynchronously.
   */
  private CompletableFuture<Void> completeTasksAsync(List<TaskClaim<Long, LinkTask>> claims) {
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (TaskClaim<Long, LinkTask> claim : claims) {
      CompletableFuture<Void> completeFuture = claim.complete()
          .thenAccept(v -> TASKS_COMPLETED.add(1))
          .exceptionally(e -> {
            LOGGER.warn("Failed to complete task {}", claim.taskKey(), e);
            TASKS_FAILED.add(1);
            return null;
          });
      futures.add(completeFuture);
    }

    return allOf(futures.toArray(CompletableFuture[]::new));
  }

  /**
   * Marks all tasks as failed for retry asynchronously.
   */
  private CompletableFuture<Void> failTasksAsync(List<TaskClaim<Long, LinkTask>> claims) {
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (TaskClaim<Long, LinkTask> claim : claims) {
      CompletableFuture<Void> failFuture = claim.fail()
          .thenAccept(v -> TASKS_FAILED.add(1))
          .exceptionally(e -> {
            LOGGER.warn("Failed to fail task {}", claim.taskKey(), e);
            return null;
          });
      futures.add(failFuture);
    }

    return allOf(futures.toArray(CompletableFuture[]::new));
  }

  /**
   * Adjusts batch size based on transaction performance.
   */
  void adjustBatchSize(Duration transactionDuration, boolean success) {
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
      LOGGER.info("Adjusted batch size from {} to {}", currentSize, newSize);
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

  /**
   * Processes a single task and returns when complete.
   * This is primarily for testing purposes.
   *
   * @return CompletableFuture that completes when the task is processed
   */
  public CompletableFuture<Boolean> processOneTask() {
    // Use claimTask with a timeout to avoid indefinite waiting
    CompletableFuture<TaskClaim<Long, LinkTask>> claimFuture = taskQueue.awaitAndClaimTask(database);

    return claimFuture
        .orTimeout(CLAIM_TIMEOUT.toMillis(), TimeUnit.MILLISECONDS)
        .thenCompose(claim -> {
          TASKS_CLAIMED.add(1);
          List<TaskClaim<Long, LinkTask>> claims = List.of(claim);

          // Use common processing logic
          return processClaimedTasksAsync(claims).thenApply(success -> {
            if (success) {
              totalProcessed.incrementAndGet();
            } else {
              totalFailed.incrementAndGet();
            }
            return success;
          });
        })
        .exceptionally(e -> {
          if (e instanceof TimeoutException) {
            // No task available within timeout
            return false;
          }
          LOGGER.warn("Failed to process single task", e);
          return false;
        });
  }

  /**
   * Processes all available tasks until the queue is empty.
   * This is primarily for testing purposes.
   *
   * @param maxTasks maximum number of tasks to process (0 for unlimited)
   * @return CompletableFuture with the number of tasks processed
   */
  public CompletableFuture<Integer> processAllAvailableTasks(int maxTasks) {
    return processTasksRecursively(new AtomicInteger(0), maxTasks);
  }

  /**
   * Helper method for iterative task processing to avoid stack overflow.
   */
  private CompletableFuture<Integer> processTasksRecursively(AtomicInteger processedCount, int maxTasks) {
    CompletableFuture<Integer> result = new CompletableFuture<>();

    class TaskLoop {
      void next() {
        int count = processedCount.get();
        if (maxTasks > 0 && count >= maxTasks) {
          result.complete(count);
          return;
        }
        processOneTask()
            .thenAccept(processed -> {
              if (!processed) {
                // No more tasks available
                result.complete(processedCount.get());
              } else {
                processedCount.incrementAndGet();
                next();
              }
            })
            .exceptionally(ex -> {
              result.completeExceptionally(ex);
              return null;
            });
      }
    }
    new TaskLoop().next();
    return result;
  }
}
