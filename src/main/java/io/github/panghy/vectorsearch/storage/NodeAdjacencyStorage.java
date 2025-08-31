package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.graph.RobustPruning;
import io.github.panghy.vectorsearch.graph.RobustPruning.Candidate;
import io.github.panghy.vectorsearch.graph.RobustPruning.PruningConfig;
import io.github.panghy.vectorsearch.proto.EntryList;
import io.github.panghy.vectorsearch.proto.GraphMeta;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Storage handler for graph adjacency lists in FoundationDB.
 * Manages per-node neighbor lists with degree bounds and bidirectional consistency.
 * Implements DiskANN-style robust pruning for diverse neighbor selection.
 * <p>
 * All methods that interact with the database return CompletableFuture to support
 * non-blocking async operations. Callers should compose these futures within
 * FDB transactions.
 */
public class NodeAdjacencyStorage {
  private static final Logger LOGGER = LoggerFactory.getLogger(NodeAdjacencyStorage.class);

  private final Database db;
  private final VectorIndexKeys keys;
  private final int graphDegree;
  private final InstantSource instantSource;

  // Cache for frequently accessed adjacency lists
  private final AsyncLoadingCache<Long, NodeAdjacency> adjacencyCache;

  // Default pruning alpha for diversity vs proximity trade-off
  private static final double DEFAULT_PRUNE_ALPHA = 1.2;

  public NodeAdjacencyStorage(
      Database db,
      VectorIndexKeys keys,
      int graphDegree,
      InstantSource instantSource,
      int maxCacheSize,
      Duration cacheTtl) {
    this.db = db;
    this.keys = keys;
    this.graphDegree = graphDegree;
    this.instantSource = instantSource;
    this.adjacencyCache = Caffeine.newBuilder()
        .maximumSize(maxCacheSize)
        .expireAfterWrite(cacheTtl)
        .buildAsync((nodeId, executor) -> loadAdjacencyFromDb(nodeId));
  }

  /**
   * Stores or updates the adjacency list for a node.
   *
   * @param tx        the transaction to use
   * @param nodeId    the node ID
   * @param neighbors list of neighbor node IDs (will be sorted and deduplicated)
   * @return future completing when stored
   */
  public CompletableFuture<Void> storeAdjacency(Transaction tx, long nodeId, List<Long> neighbors) {
    // Sort and deduplicate neighbors
    List<Long> sortedNeighbors =
        neighbors.stream().distinct().sorted().limit(graphDegree).collect(Collectors.toList());

    byte[] key = keys.nodeAdjacencyKey(nodeId);

    // Read existing for version increment
    return readProto(tx, key, NodeAdjacency.parser()).thenApply(existing -> {
      NodeAdjacency.Builder builder = NodeAdjacency.newBuilder()
          .setNodeId(nodeId)
          .addAllNeighbors(sortedNeighbors)
          .setVersion(existing != null ? existing.getVersion() + 1 : 1)
          .setUpdatedAt(currentTimestamp());

      NodeAdjacency updated = builder.build();
      tx.set(key, updated.toByteArray());
      return null;
    });
  }

  /**
   * Stores a complete NodeAdjacency message.
   * This is used by LinkWorker for direct adjacency updates.
   *
   * @param tx        the transaction to use
   * @param nodeId    the node ID
   * @param adjacency the complete adjacency message
   * @return future completing when stored
   */
  public CompletableFuture<Void> storeNodeAdjacency(Transaction tx, long nodeId, NodeAdjacency adjacency) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    tx.set(key, adjacency.toByteArray());
    return completedFuture(null);
  }

  /**
   * Gets the node adjacency for a node.
   * Alias for loadAdjacency for LinkWorker compatibility.
   *
   * @param tx     the transaction to use
   * @param nodeId the node ID
   * @return future with adjacency or null if not found
   */
  public CompletableFuture<NodeAdjacency> getNodeAdjacency(Transaction tx, long nodeId) {
    return loadAdjacency(tx, nodeId);
  }

  /**
   * Loads the adjacency list for a node from the transaction.
   *
   * @param tx     the transaction to use
   * @param nodeId the node ID
   * @return future with adjacency or null if not found
   */
  public CompletableFuture<NodeAdjacency> loadAdjacency(Transaction tx, long nodeId) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    return readProto(tx, key, NodeAdjacency.parser()).thenApply(adjacency -> {
      // Update cache if found
      if (adjacency != null) {
        adjacencyCache.put(nodeId, completedFuture(adjacency));
      }
      return adjacency;
    });
  }

  /**
   * Loads the adjacency list for a node from cache or database.
   * This method doesn't require a transaction and uses the cache.
   *
   * @param nodeId the node ID
   * @return future with adjacency or null if not found
   */
  public CompletableFuture<NodeAdjacency> loadAdjacencyAsync(long nodeId) {
    return adjacencyCache.get(nodeId);
  }

  /**
   * Batch loads adjacency lists for multiple nodes.
   *
   * @param tx      the transaction to use
   * @param nodeIds list of node IDs
   * @return future with map of nodeId to adjacency
   */
  public CompletableFuture<Map<Long, NodeAdjacency>> batchLoadAdjacency(Transaction tx, List<Long> nodeIds) {
    if (nodeIds.isEmpty()) {
      return completedFuture(Collections.emptyMap());
    }
    CompletableFuture<Map<Long, NodeAdjacency>> resultFuture = completedFuture(new HashMap<>(nodeIds.size()));

    for (long nodeId : nodeIds) {
      CompletableFuture<NodeAdjacency> future = loadAdjacency(tx, nodeId);
      resultFuture = resultFuture.thenCombine(future, (result, adj) -> {
        if (adj != null) {
          result.put(nodeId, adj);
        }
        return result;
      });
    }
    return resultFuture;
  }

  /**
   * Adds a neighbor to a node's adjacency list.
   * Maintains sorted order and degree bound.
   *
   * @param tx         the transaction to use
   * @param nodeId     the node ID
   * @param neighborId the neighbor to add
   * @return future completing when added
   */
  public CompletableFuture<Void> addNeighbor(Transaction tx, long nodeId, long neighborId) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    return readProto(tx, key, NodeAdjacency.parser()).thenAccept(existing -> {
      List<Long> neighbors;
      if (existing == null) {
        neighbors = new ArrayList<>();
      } else {
        neighbors = new ArrayList<>(existing.getNeighborsList());
      }

      // Add if not present and under degree limit
      if (!neighbors.contains(neighborId) && neighbors.size() < graphDegree) {
        neighbors.add(neighborId);
        Collections.sort(neighbors);
      }

      NodeAdjacency updated = NodeAdjacency.newBuilder()
          .setNodeId(nodeId)
          .addAllNeighbors(neighbors)
          .setVersion(existing != null ? existing.getVersion() + 1 : 1)
          .setUpdatedAt(currentTimestamp())
          .build();

      tx.set(key, updated.toByteArray());
    });
  }

  /**
   * Removes a neighbor from a node's adjacency list.
   *
   * @param tx         the transaction to use
   * @param nodeId     the node ID
   * @param neighborId the neighbor to remove
   * @return future completing when removed
   */
  public CompletableFuture<Void> removeNeighbor(Transaction tx, long nodeId, long neighborId) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    return readProto(tx, key, NodeAdjacency.parser()).thenAccept(existing -> {
      if (existing == null) {
        return;
      }

      List<Long> neighbors = existing.getNeighborsList().stream()
          .filter(n -> n != neighborId)
          .collect(Collectors.toList());

      NodeAdjacency updated = NodeAdjacency.newBuilder()
          .setNodeId(nodeId)
          .addAllNeighbors(neighbors)
          .setVersion(existing.getVersion() + 1)
          .setUpdatedAt(currentTimestamp())
          .build();

      tx.set(key, updated.toByteArray());
    });
  }

  /**
   * Adds bidirectional edges between a node and its neighbors.
   * Uses simple pruning when degree limit is exceeded.
   * All updates happen in the provided transaction.
   *
   * @param tx        the transaction to use
   * @param nodeId    the central node
   * @param neighbors the neighbors to link bidirectionally
   * @return future completing when all links are added
   */
  public CompletableFuture<Void> addBackLinks(Transaction tx, long nodeId, List<Long> neighbors) {
    return addBackLinksWithDistanceFunction(tx, nodeId, neighbors, null);
  }

  /**
   * Adds bidirectional edges between a node and its neighbors.
   * Uses robust pruning with distances when degree limit is exceeded.
   * All updates happen in the provided transaction.
   *
   * @param tx               the transaction to use
   * @param nodeId           the central node
   * @param neighbors        the neighbors to link bidirectionally
   * @param distanceFunction optional function to compute distances for pruning (null for simple pruning)
   * @return future completing when all links are added
   */
  public CompletableFuture<Void> addBackLinksWithDistanceFunction(
      Transaction tx,
      long nodeId,
      List<Long> neighbors,
      BiFunction<Long, Long, CompletableFuture<Float>> distanceFunction) {
    if (neighbors.isEmpty()) {
      return completedFuture(null);
    }
    CompletableFuture<Void> resultFuture = completedFuture(null);

    for (long neighborId : neighbors) {
      byte[] key = keys.nodeAdjacencyKey(neighborId);

      CompletableFuture<Void> future;
      if (distanceFunction != null) {
        // Use distance-based pruning
        future = readProto(tx, key, NodeAdjacency.parser()).thenCompose(existing -> {
          List<Long> neighborList;
          if (existing == null) {
            neighborList = new ArrayList<>();
          } else {
            neighborList = new ArrayList<>(existing.getNeighborsList());
          }

          // Add nodeId as back-link if not present
          if (!neighborList.contains(nodeId)) {
            neighborList.add(nodeId);

            // If we exceed the degree limit, use robust pruning with distances
            if (neighborList.size() > graphDegree) {
              // Compute distances for all neighbors from the perspective of neighborId
              CompletableFuture<List<Candidate>> candidateFutures =
                  completedFuture(new ArrayList<>(neighborList.size()));
              for (Long neighbor : neighborList) {
                CompletableFuture<Candidate> candidateFuture = distanceFunction
                    .apply(neighborId, neighbor)
                    .thenApply(distance -> Candidate.builder()
                        .nodeId(neighbor)
                        .distanceToQuery(distance)
                        .build());
                candidateFutures = candidateFutures.thenCombine(candidateFuture, (list, candidate) -> {
                  list.add(candidate);
                  return list;
                });
              }

              // Wait for all distances and apply robust pruning
              return candidateFutures
                  .thenApply(candidates -> {
                    PruningConfig config = PruningConfig.builder()
                        .maxDegree(graphDegree)
                        .alpha(DEFAULT_PRUNE_ALPHA)
                        .build();

                    return RobustPruning.prune(candidates, config);
                  })
                  .thenAccept(prunedNeighbors -> {
                    NodeAdjacency updated = NodeAdjacency.newBuilder()
                        .setNodeId(neighborId)
                        .addAllNeighbors(prunedNeighbors)
                        .setVersion(existing != null ? existing.getVersion() + 1 : 1)
                        .setUpdatedAt(currentTimestamp())
                        .build();

                    tx.set(key, updated.toByteArray());
                  });
            } else {
              Collections.sort(neighborList);
              NodeAdjacency updated = NodeAdjacency.newBuilder()
                  .setNodeId(neighborId)
                  .addAllNeighbors(neighborList)
                  .setVersion(existing != null ? existing.getVersion() + 1 : 1)
                  .setUpdatedAt(currentTimestamp())
                  .build();

              tx.set(key, updated.toByteArray());
              return completedFuture(null);
            }
          }
          return completedFuture(null);
        });
      } else {
        // Fallback to simple pruning without distances
        future = readProto(tx, key, NodeAdjacency.parser()).thenApply(existing -> {
          List<Long> neighborList;
          if (existing == null) {
            neighborList = new ArrayList<>();
          } else {
            neighborList = new ArrayList<>(existing.getNeighborsList());
          }

          // Add nodeId as back-link if not present
          if (!neighborList.contains(nodeId)) {
            neighborList.add(nodeId);

            // If we exceed the degree limit, use simple pruning
            if (neighborList.size() > graphDegree) {
              Collections.sort(neighborList);
              neighborList = neighborList.subList(0, graphDegree);
            } else {
              Collections.sort(neighborList);
            }

            NodeAdjacency updated = NodeAdjacency.newBuilder()
                .setNodeId(neighborId)
                .addAllNeighbors(neighborList)
                .setVersion(existing != null ? existing.getVersion() + 1 : 1)
                .setUpdatedAt(currentTimestamp())
                .build();

            tx.set(key, updated.toByteArray());
          }

          return null;
        });
      }

      resultFuture = resultFuture.thenCompose($ -> future);
    }
    return resultFuture;
  }

  /**
   * Removes bidirectional edges between a node and its neighbors.
   * All updates happen in the provided transaction.
   *
   * @param tx        the transaction to use
   * @param nodeId    the central node
   * @param neighbors the neighbors to unlink
   * @return future completing when all links are removed
   */
  public CompletableFuture<Void> removeBackLinks(Transaction tx, long nodeId, List<Long> neighbors) {
    if (neighbors.isEmpty()) {
      return completedFuture(null);
    }
    CompletableFuture<Void> resultFuture = completedFuture(null);

    for (long neighborId : neighbors) {
      byte[] key = keys.nodeAdjacencyKey(neighborId);
      CompletableFuture<Void> future = readProto(tx, key, NodeAdjacency.parser())
          .thenAccept(existing -> {
            if (existing == null) {
              return;
            }

            List<Long> neighborList = existing.getNeighborsList().stream()
                .filter(n -> n != nodeId)
                .collect(Collectors.toList());

            NodeAdjacency updated = NodeAdjacency.newBuilder()
                .setNodeId(neighborId)
                .addAllNeighbors(neighborList)
                .setVersion(existing.getVersion() + 1)
                .setUpdatedAt(currentTimestamp())
                .build();

            tx.set(key, updated.toByteArray());
          });
      resultFuture = resultFuture.thenCompose($ -> future);
    }
    return resultFuture;
  }

  /**
   * Robust prune algorithm for diverse neighbor selection (DiskANN-style).
   * Keeps neighbors that are both close and diverse.
   *
   * @param candidates sorted list of candidates with distances
   * @param maxDegree  maximum neighbors to keep
   * @param alpha      diversity vs proximity trade-off (lower = more diverse)
   * @return pruned list of node IDs
   */
  public List<Long> robustPrune(List<ScoredNode> candidates, int maxDegree, double alpha) {
    // Convert ScoredNode to RobustPruning.Candidate format
    List<Candidate> pruningCandidates = candidates.stream()
        .map(node -> Candidate.builder()
            .nodeId(node.nodeId)
            .distanceToQuery((float) node.distance)
            .build())
        .collect(Collectors.toList());

    // Create pruning configuration
    PruningConfig config =
        PruningConfig.builder().maxDegree(maxDegree).alpha(alpha).build();

    // Use the RobustPruning implementation
    return RobustPruning.prune(pruningCandidates, config);
  }

  /**
   * Simple robust prune for nodes without distance information.
   * Falls back to keeping the first maxDegree nodes.
   *
   * @param candidateIds list of node IDs
   * @param maxDegree    maximum neighbors to keep
   * @param alpha        diversity vs proximity trade-off (unused without distances)
   * @return pruned list of node IDs
   */
  public List<Long> robustPruneSimple(List<Long> candidateIds, int maxDegree, double alpha) {
    if (candidateIds.size() <= maxDegree) {
      return new ArrayList<>(candidateIds);
    }
    // Without distance information, just keep first maxDegree
    // Note: alpha parameter is kept for API consistency with full robust pruning
    // which would use it for diversity control if distance information was available
    return candidateIds.subList(0, maxDegree);
  }

  /**
   * Merges new neighbors with existing ones and prunes to degree limit.
   * Simple version without distance information.
   *
   * @param current   current neighbors
   * @param additions new neighbors to add (without distances)
   * @param maxDegree maximum total neighbors
   * @return merged and pruned list
   */
  public List<Long> mergePrune(List<Long> current, List<Long> additions, int maxDegree) {
    Set<Long> merged = new LinkedHashSet<>(current);
    merged.addAll(additions);

    List<Long> result = new ArrayList<>(merged);
    Collections.sort(result);

    if (result.size() > maxDegree) {
      // Simple strategy: keep first maxDegree after sorting
      return result.subList(0, maxDegree);
    }

    return result;
  }

  /**
   * Merges new neighbors with existing ones and prunes to degree limit using robust pruning.
   *
   * @param current   current neighbors
   * @param additions new neighbors to add with distances
   * @param maxDegree maximum total neighbors
   * @return merged and pruned list
   */
  public List<Long> mergePruneWithDistances(List<Long> current, List<ScoredNode> additions, int maxDegree) {
    // Note: This requires a distance function for existing neighbors
    // For now, use a simple merge without distance-based pruning for existing nodes
    // This would be improved by having distances for all nodes
    Set<Long> merged = new LinkedHashSet<>(current);
    for (ScoredNode addition : additions) {
      merged.add(addition.nodeId);
    }

    if (merged.size() <= maxDegree) {
      return new ArrayList<>(merged);
    }

    // If we exceed the degree, use robust pruning on the additions
    // and keep the most important current neighbors
    List<Long> prunedAdditions =
        robustPrune(additions, maxDegree - Math.min(current.size(), maxDegree / 2), DEFAULT_PRUNE_ALPHA);
    Set<Long> result = new LinkedHashSet<>();
    // Keep some existing neighbors
    int keepExisting = Math.min(current.size(), maxDegree / 2);
    for (int i = 0; i < keepExisting && i < current.size(); i++) {
      result.add(current.get(i));
    }
    result.addAll(prunedAdditions);

    return new ArrayList<>(result);
  }

  /**
   * Stores the entry list for search initialization.
   *
   * @param tx        the transaction to use
   * @param entryList the entry points
   */
  public void storeEntryList(Transaction tx, EntryList entryList) {
    byte[] key = keys.entryListKey();
    tx.set(key, entryList.toByteArray());
  }

  /**
   * Loads the entry list for search initialization.
   *
   * @param tx the transaction to use
   * @return future with entry list or null if not found
   */
  public CompletableFuture<EntryList> loadEntryList(Transaction tx) {
    return readProto(tx, keys.entryListKey(), EntryList.parser());
  }

  /**
   * Stores graph metadata and statistics.
   *
   * @param tx        the transaction to use
   * @param graphMeta the metadata to store
   */
  public void storeGraphMeta(Transaction tx, GraphMeta graphMeta) {
    byte[] key = keys.graphConnectivityKey();
    tx.set(key, graphMeta.toByteArray());
  }

  /**
   * Loads graph metadata and statistics.
   *
   * @param tx the transaction to use
   * @return future with graph metadata or null if not found
   */
  public CompletableFuture<GraphMeta> loadGraphMeta(Transaction tx) {
    return readProto(tx, keys.graphConnectivityKey(), GraphMeta.parser());
  }

  /**
   * Deletes a node's adjacency list.
   *
   * @param tx     the transaction to use
   * @param nodeId the node to delete
   */
  public void deleteNode(Transaction tx, long nodeId) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    tx.clear(key);
    adjacencyCache.synchronous().invalidate(nodeId);
  }

  /**
   * Clears the adjacency cache.
   */
  public void clearCache() {
    adjacencyCache.synchronous().invalidateAll();
    LOGGER.debug("Cleared adjacency cache");
  }

  // Helper methods

  private CompletableFuture<NodeAdjacency> loadAdjacencyFromDb(Long nodeId) {
    byte[] key = keys.nodeAdjacencyKey(nodeId);
    return db.runAsync(tx -> readProto(tx, key, NodeAdjacency.parser()));
  }

  private Timestamp currentTimestamp() {
    Instant now = instantSource.instant();
    return Timestamp.newBuilder()
        .setSeconds(now.getEpochSecond())
        .setNanos(now.getNano())
        .build();
  }

  /**
   * Node with distance score for pruning operations.
   */
  public record ScoredNode(long nodeId, double distance) {}
}
