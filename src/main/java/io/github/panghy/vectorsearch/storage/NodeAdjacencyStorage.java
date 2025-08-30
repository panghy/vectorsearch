package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.github.benmanes.caffeine.cache.AsyncLoadingCache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.EntryList;
import io.github.panghy.vectorsearch.proto.GraphMeta;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Storage handler for graph adjacency lists in FoundationDB.
 * Manages per-node neighbor lists with degree bounds and bidirectional consistency.
 * Implements DiskANN-style robust pruning for diverse neighbor selection.
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
   * @param nodeId the node ID
   * @param neighbors list of neighbor node IDs (will be sorted and deduplicated)
   * @return future completing when stored
   */
  public CompletableFuture<Void> storeAdjacency(long nodeId, List<Long> neighbors) {
    // Sort and deduplicate neighbors
    List<Long> sortedNeighbors =
        neighbors.stream().distinct().sorted().limit(graphDegree).collect(Collectors.toList());

    return db.runAsync(tx -> {
      byte[] key = keys.nodeAdjacencyKey(nodeId);

      // Read existing for version increment
      return readProto(tx, key, NodeAdjacency.parser()).thenApply(existing -> {
        NodeAdjacency.Builder builder = NodeAdjacency.newBuilder()
            .setNodeId(nodeId)
            .addAllNeighbors(sortedNeighbors)
            .setVersion(existing != null ? existing.getVersion() + 1 : 1)
            .setUpdatedAt(currentTimestamp());

        // Preserve last accessed time if it exists (future enhancement)

        NodeAdjacency updated = builder.build();
        tx.set(key, updated.toByteArray());

        // Update cache
        adjacencyCache.put(nodeId, completedFuture(updated));

        return null;
      });
    });
  }

  /**
   * Loads the adjacency list for a node.
   *
   * @param nodeId the node ID
   * @return future with adjacency or null if not found
   */
  public CompletableFuture<NodeAdjacency> loadAdjacency(long nodeId) {
    return adjacencyCache.get(nodeId);
  }

  /**
   * Batch loads adjacency lists for multiple nodes.
   *
   * @param nodeIds list of node IDs
   * @return future with map of nodeId to adjacency
   */
  public CompletableFuture<Map<Long, NodeAdjacency>> batchLoadAdjacency(List<Long> nodeIds) {
    Map<Long, CompletableFuture<NodeAdjacency>> futures = new HashMap<>();
    for (long nodeId : nodeIds) {
      futures.put(nodeId, adjacencyCache.get(nodeId));
    }

    return CompletableFuture.allOf(futures.values().toArray(new CompletableFuture[0]))
        .thenApply(v -> {
          Map<Long, NodeAdjacency> result = new HashMap<>();
          futures.forEach((nodeId, future) -> {
            NodeAdjacency adj = future.join();
            if (adj != null) {
              result.put(nodeId, adj);
            }
          });
          return result;
        });
  }

  /**
   * Adds a neighbor to a node's adjacency list.
   * Maintains sorted order and degree bound.
   *
   * @param nodeId the node ID
   * @param neighborId the neighbor to add
   * @return future completing when added
   */
  public CompletableFuture<Void> addNeighbor(long nodeId, long neighborId) {
    return db.runAsync(tx -> {
      byte[] key = keys.nodeAdjacencyKey(nodeId);

      return readProto(tx, key, NodeAdjacency.parser()).thenApply(existing -> {
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
        adjacencyCache.put(nodeId, completedFuture(updated));

        return null;
      });
    });
  }

  /**
   * Removes a neighbor from a node's adjacency list.
   *
   * @param nodeId the node ID
   * @param neighborId the neighbor to remove
   * @return future completing when removed
   */
  public CompletableFuture<Void> removeNeighbor(long nodeId, long neighborId) {
    return db.runAsync(tx -> {
      byte[] key = keys.nodeAdjacencyKey(nodeId);

      return readProto(tx, key, NodeAdjacency.parser()).thenApply(existing -> {
        if (existing == null) {
          return null;
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
        adjacencyCache.put(nodeId, completedFuture(updated));

        return null;
      });
    });
  }

  /**
   * Adds bidirectional edges between a node and its neighbors.
   * Updates are done in small batches to avoid large transactions.
   *
   * @param nodeId the central node
   * @param neighbors the neighbors to link bidirectionally
   * @param batchSize number of neighbors to update per transaction
   * @return future completing when all links are added
   */
  public CompletableFuture<Void> batchAddBackLinks(long nodeId, List<Long> neighbors, int batchSize) {
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    // Process neighbors in batches
    for (int i = 0; i < neighbors.size(); i += batchSize) {
      List<Long> batch = neighbors.subList(i, Math.min(i + batchSize, neighbors.size()));

      CompletableFuture<Void> batchFuture = db.runAsync(tx -> {
        List<CompletableFuture<Void>> updates = new ArrayList<>();

        for (long neighborId : batch) {
          byte[] key = keys.nodeAdjacencyKey(neighborId);

          CompletableFuture<Void> update = readProto(tx, key, NodeAdjacency.parser())
              .thenApply(existing -> {
                List<Long> neighborList;
                if (existing == null) {
                  neighborList = new ArrayList<>();
                } else {
                  neighborList = new ArrayList<>(existing.getNeighborsList());
                }

                // Add nodeId as back-link if not present and under degree
                if (!neighborList.contains(nodeId) && neighborList.size() < graphDegree) {
                  neighborList.add(nodeId);
                  Collections.sort(neighborList);

                  NodeAdjacency updated = NodeAdjacency.newBuilder()
                      .setNodeId(neighborId)
                      .addAllNeighbors(neighborList)
                      .setVersion(existing != null ? existing.getVersion() + 1 : 1)
                      .setUpdatedAt(currentTimestamp())
                      .build();

                  tx.set(key, updated.toByteArray());
                  adjacencyCache.put(neighborId, completedFuture(updated));
                }

                return null;
              });

          updates.add(update);
        }

        return CompletableFuture.allOf(updates.toArray(new CompletableFuture[0]));
      });

      futures.add(batchFuture);
    }

    return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
  }

  /**
   * Removes bidirectional edges between a node and its neighbors.
   *
   * @param nodeId the central node
   * @param neighbors the neighbors to unlink
   * @param batchSize number of neighbors to update per transaction
   * @return future completing when all links are removed
   */
  public CompletableFuture<Void> batchRemoveBackLinks(long nodeId, List<Long> neighbors, int batchSize) {
    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (int i = 0; i < neighbors.size(); i += batchSize) {
      List<Long> batch = neighbors.subList(i, Math.min(i + batchSize, neighbors.size()));

      CompletableFuture<Void> batchFuture = db.runAsync(tx -> {
        List<CompletableFuture<Void>> updates = new ArrayList<>();

        for (long neighborId : batch) {
          byte[] key = keys.nodeAdjacencyKey(neighborId);

          CompletableFuture<Void> update = readProto(tx, key, NodeAdjacency.parser())
              .thenApply(existing -> {
                if (existing == null) {
                  return null;
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
                adjacencyCache.put(neighborId, completedFuture(updated));

                return null;
              });

          updates.add(update);
        }

        return CompletableFuture.allOf(updates.toArray(new CompletableFuture[0]));
      });

      futures.add(batchFuture);
    }

    return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
  }

  /**
   * Robust prune algorithm for diverse neighbor selection (DiskANN-style).
   * Keeps neighbors that are both close and diverse.
   *
   * @param candidates sorted list of candidates with distances
   * @param maxDegree maximum neighbors to keep
   * @param alpha diversity vs proximity trade-off (higher = more diverse)
   * @return pruned list of node IDs
   */
  public List<Long> robustPrune(List<ScoredNode> candidates, int maxDegree, double alpha) {
    if (candidates.isEmpty()) {
      return new ArrayList<>();
    }

    List<Long> pruned = new ArrayList<>();
    Set<Long> prunedSet = new HashSet<>();

    for (ScoredNode candidate : candidates) {
      if (pruned.size() >= maxDegree) {
        break;
      }

      if (prunedSet.contains(candidate.nodeId)) {
        continue;
      }

      // First node is always added
      if (pruned.isEmpty()) {
        pruned.add(candidate.nodeId);
        prunedSet.add(candidate.nodeId);
        continue;
      }

      // Check if candidate is dominated by any already selected neighbor
      // A node is dominated if a selected node can serve as its representative
      // This happens when candidate.distance <= alpha * selected.distance
      // (i.e., reaching candidate through selected is not much worse)
      boolean dominated = false;
      for (Long selected : pruned) {
        // Find distance of selected node
        double selectedDist = candidates.stream()
            .filter(c -> c.nodeId == selected)
            .findFirst()
            .map(c -> c.distance)
            .orElse(Double.MAX_VALUE);

        // Candidate is dominated if it's not significantly better than an existing node
        // i.e., if candidate is farther and within alpha factor of a closer node
        if (selectedDist <= candidate.distance && candidate.distance <= alpha * selectedDist) {
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        pruned.add(candidate.nodeId);
        prunedSet.add(candidate.nodeId);
      }
    }

    return pruned;
  }

  /**
   * Merges new neighbors with existing ones and prunes to degree limit.
   *
   * @param current current neighbors
   * @param additions new neighbors to add
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
      // In practice, might want to use distance-based pruning
      return result.subList(0, maxDegree);
    }

    return result;
  }

  /**
   * Stores the entry list for search initialization.
   *
   * @param entryList the entry points
   * @return future completing when stored
   */
  public CompletableFuture<Void> storeEntryList(EntryList entryList) {
    return db.runAsync(tx -> {
      byte[] key = keys.entryListKey();
      tx.set(key, entryList.toByteArray());
      return completedFuture(null);
    });
  }

  /**
   * Loads the entry list for search initialization.
   *
   * @return future with entry list or null if not found
   */
  public CompletableFuture<EntryList> loadEntryList() {
    return db.runAsync(tx -> readProto(tx, keys.entryListKey(), EntryList.parser()));
  }

  /**
   * Stores graph metadata and statistics.
   *
   * @param graphMeta the metadata to store
   * @return future completing when stored
   */
  public CompletableFuture<Void> storeGraphMeta(GraphMeta graphMeta) {
    return db.runAsync(tx -> {
      byte[] key = keys.graphConnectivityKey();
      tx.set(key, graphMeta.toByteArray());
      return completedFuture(null);
    });
  }

  /**
   * Loads graph metadata and statistics.
   *
   * @return future with graph metadata or null if not found
   */
  public CompletableFuture<GraphMeta> loadGraphMeta() {
    return db.runAsync(tx -> readProto(tx, keys.graphConnectivityKey(), GraphMeta.parser()));
  }

  /**
   * Deletes a node's adjacency list.
   *
   * @param nodeId the node to delete
   * @return future completing when deleted
   */
  public CompletableFuture<Void> deleteNode(long nodeId) {
    return db.runAsync(tx -> {
      byte[] key = keys.nodeAdjacencyKey(nodeId);
      tx.clear(key);
      adjacencyCache.synchronous().invalidate(nodeId);
      return completedFuture(null);
    });
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
  public static class ScoredNode {
    public final long nodeId;
    public final double distance;

    public ScoredNode(long nodeId, double distance) {
      this.nodeId = nodeId;
      this.distance = distance;
    }
  }
}
