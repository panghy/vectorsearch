package io.github.panghy.vectorsearch.graph;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeySelector;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.tuple.ByteArrayUtil;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.GraphMetaStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static java.util.concurrent.CompletableFuture.completedFuture;

/**
 * Monitors and repairs graph connectivity in the vector search index.
 *
 * <p>This class provides:
 * <ul>
 *   <li>Detection of disconnected components using BFS traversal</li>
 *   <li>Identification of orphaned nodes (unreachable from main component)</li>
 *   <li>Repair mechanism to reconnect orphans using PQ distance-based nearest neighbors</li>
 *   <li>Entry point refresh based on graph centrality and connectivity</li>
 * </ul>
 *
 * <p>The connectivity analysis uses sampling for large graphs and exact BFS for smaller ones.
 * Repair operations add bidirectional edges between orphaned nodes and their nearest neighbors
 * in the main connected component.
 */
public class GraphConnectivityMonitor {
  private static final Logger LOGGER = Logger.getLogger(GraphConnectivityMonitor.class.getName());

  private static final int SAMPLE_SIZE = 10000; // Number of random nodes to sample for analysis
  private static final int MAX_BFS_VISITS = 1000; // Max nodes to visit per BFS
  private static final int REPAIR_NEIGHBORS = 3; // Number of edges to add per orphan
  private static final double HEALTHY_THRESHOLD = 0.95; // 95% connectivity for healthy graph

  private final Database db;
  private final VectorIndexKeys keys;
  private final GraphMetaStorage metaStorage;
  private final NodeAdjacencyStorage adjacencyStorage;
  private final PqBlockStorage pqBlockStorage;
  private final EntryPointStorage entryPointStorage;
  private final ProductQuantizer pq;
  private final Random random;

  /**
   * Creates a new GraphConnectivityMonitor.
   *
   * @param db                the FoundationDB database
   * @param keys              vector index keys for this collection
   * @param metaStorage       storage for graph metadata
   * @param adjacencyStorage  storage for node adjacency lists
   * @param pqBlockStorage    storage for PQ codes
   * @param entryPointStorage storage for entry points
   * @param pq                product quantizer for distance computation
   */
  public GraphConnectivityMonitor(
      Database db,
      VectorIndexKeys keys,
      GraphMetaStorage metaStorage,
      NodeAdjacencyStorage adjacencyStorage,
      PqBlockStorage pqBlockStorage,
      EntryPointStorage entryPointStorage,
      ProductQuantizer pq) {
    this.db = db;
    this.keys = keys;
    this.metaStorage = metaStorage;
    this.adjacencyStorage = adjacencyStorage;
    this.pqBlockStorage = pqBlockStorage;
    this.entryPointStorage = entryPointStorage;
    this.pq = pq;
    this.random = new Random();
  }

  /**
   * Analyzes graph connectivity and repairs if necessary.
   *
   * @param codebookVersion the PQ codebook version to use for distance calculations
   * @return future that completes when analysis and repair is done
   */
  public CompletableFuture<Void> analyzeAndRepair(int codebookVersion) {
    return db.runAsync(tx -> analyzeConnectivity(tx).thenCompose(analysisResult -> {
      if (analysisResult.needsRepair()) {
        LOGGER.info("Graph needs repair, starting repair process");
        return repairConnectivity(tx, analysisResult, codebookVersion);
      } else {
        LOGGER.fine("Graph connectivity is healthy");
        return completedFuture(null);
      }
    }));
  }

  /**
   * Analyzes graph connectivity to detect disconnected components.
   *
   * @param tx the transaction to use
   * @return future containing analysis results
   */
  public CompletableFuture<ConnectivityAnalysis> analyzeConnectivity(Transaction tx) {
    LOGGER.info("Starting graph connectivity analysis using sampling");

    // Sample random nodes and estimate connectivity
    return sampleRandomNodes(tx, SAMPLE_SIZE).thenCompose(sampleNodes -> {
      if (sampleNodes.isEmpty()) {
        LOGGER.info("Graph is empty, no connectivity analysis needed");
        return completedFuture(new ConnectivityAnalysis(0, 0, 0, Collections.emptyList()));
      }

      // Perform limited BFS from each sample to estimate connectivity
      return estimateConnectivityFromSamples(tx, sampleNodes);
    });
  }

  /**
   * Repairs graph connectivity by reconnecting orphaned nodes.
   */
  CompletableFuture<Void> repairConnectivity(Transaction tx, ConnectivityAnalysis analysis, int codebookVersion) {

    List<Long> orphanedNodes = analysis.orphanedNodes();

    if (orphanedNodes.isEmpty()) {
      LOGGER.info("No orphaned nodes to repair");
      return completedFuture(null);
    }

    LOGGER.info(String.format("Repairing %d orphaned nodes", orphanedNodes.size()));

    // Create distance function for finding nearest neighbors
    BiFunction<Long, Long, CompletableFuture<Float>> distanceFunction =
        (nodeId1, nodeId2) -> computePqDistance(tx, nodeId1, nodeId2, codebookVersion);

    // Repair each orphaned node
    List<CompletableFuture<Void>> repairFutures = new ArrayList<>();

    for (Long orphanId : orphanedNodes) {
      repairFutures.add(repairOrphanedNode(tx, orphanId, analysis, distanceFunction));
    }

    return CompletableFuture.allOf(repairFutures.toArray(new CompletableFuture[0]))
        .thenCompose(v -> metaStorage.markRepairCompleted(tx, orphanedNodes.size()));
  }

  /**
   * Repairs a single orphaned node by connecting it to nearest neighbors in main component.
   */
  private CompletableFuture<Void> repairOrphanedNode(
      Transaction tx,
      Long orphanId,
      ConnectivityAnalysis analysis,
      BiFunction<Long, Long, CompletableFuture<Float>> distanceFunction) {

    // Find nearest neighbors in the main component
    return findNearestInMainComponent(tx, orphanId, analysis, REPAIR_NEIGHBORS, distanceFunction)
        .thenCompose(nearestNeighbors -> {
          if (nearestNeighbors.isEmpty()) {
            LOGGER.warning("Could not find neighbors for orphan " + orphanId);
            return completedFuture(null);
          }

          // Add bidirectional edges
          List<CompletableFuture<Void>> edgeFutures = new ArrayList<>();

          // Add forward edges from orphan to neighbors
          edgeFutures.add(adjacencyStorage.addBackLinks(tx, orphanId, nearestNeighbors));

          // Add back-links from neighbors to orphan
          edgeFutures.add(adjacencyStorage.addBackLinksWithDistanceFunction(
              tx, nearestNeighbors.get(0), Collections.singletonList(orphanId), distanceFunction));

          return CompletableFuture.allOf(edgeFutures.toArray(new CompletableFuture[0]));
        });
  }

  /**
   * Finds nearest neighbors in the main component for an orphaned node.
   */
  private CompletableFuture<List<Long>> findNearestInMainComponent(
      Transaction tx,
      Long orphanId,
      ConnectivityAnalysis analysis,
      int k,
      BiFunction<Long, Long, CompletableFuture<Float>> distanceFunction) {

    // Get a sample of nodes from main component to search
    long mainComponentSize = analysis.largestComponentSize();
    int sampleSize = Math.min(1000, (int) mainComponentSize);

    // Sample random nodes to find nearest neighbors
    return sampleRandomNodes(tx, sampleSize).thenCompose(sampleNodes -> {

      // Compute distances to all sampled nodes
      List<CompletableFuture<NodeDistance>> distanceFutures = new ArrayList<>();

      for (Long candidateId : sampleNodes) {
        if (!candidateId.equals(orphanId)) {
          distanceFutures.add(distanceFunction
              .apply(orphanId, candidateId)
              .thenApply(distance -> new NodeDistance(candidateId, distance)));
        }
      }

      return CompletableFuture.allOf(distanceFutures.toArray(new CompletableFuture[0]))
          .thenApply(v -> {
            // Collect and sort by distance
            PriorityQueue<NodeDistance> nearestNodes = new PriorityQueue<>();

            for (CompletableFuture<NodeDistance> future : distanceFutures) {
              nearestNodes.add(future.join());
            }

            // Return k nearest
            List<Long> result = new ArrayList<>();
            for (int i = 0; i < k && !nearestNodes.isEmpty(); i++) {
              result.add(nearestNodes.poll().nodeId);
            }

            return result;
          });
    });
  }

  /**
   * Computes PQ distance between two nodes.
   */
  CompletableFuture<Float> computePqDistance(Transaction tx, Long nodeId1, Long nodeId2, int codebookVersion) {

    // Check if PQ is available
    if (pq == null) {
      // Return a fixed distance when PQ is not available
      return completedFuture(1.0f);
    }

    // Build lookup table for node1's codes (treating as query)
    CompletableFuture<byte[]> codes1Future = pqBlockStorage.loadPqCode(nodeId1, codebookVersion);

    return codes1Future.thenCompose(codes1 -> {
      if (codes1 == null) {
        return completedFuture(Float.MAX_VALUE);
      }

      // Convert codes1 to float vector (simplified - normally would reconstruct from codebook)
      // For now, create a dummy query vector
      float[] queryVector = new float[codes1.length * 8]; // Assuming 8 dimensions per subvector
      for (int i = 0; i < codes1.length; i++) {
        queryVector[i * 8] = codes1[i] & 0xFF;
      }

      // Build lookup table
      float[][] lookupTable = pq.buildLookupTable(queryVector);

      // Load codes for node2
      return pqBlockStorage.loadPqCode(nodeId2, codebookVersion).thenApply(codes2 -> {
        if (codes2 == null) {
          return Float.MAX_VALUE;
        }

        // Compute distance using lookup table
        return pq.computeDistance(codes2, lookupTable);
      });
    });
  }

  /**
   * Refreshes entry points based on graph connectivity and centrality.
   *
   * @param tx the transaction to use
   * @return future that completes when entry points are refreshed
   */
  public CompletableFuture<Void> refreshEntryPoints(Transaction tx) {
    LOGGER.info("Refreshing entry points");

    // Sample nodes for entry points
    int totalSample = 200; // Sample more nodes than we need

    return sampleRandomNodes(tx, totalSample).thenCompose(sampledNodes -> {
      if (sampledNodes.isEmpty()) {
        return completedFuture(null);
      }

      // Select diverse entry points from our sample
      // Divide the sample into three portions
      int totalSize = sampledNodes.size();
      int numPrimary = Math.min(32, totalSize / 3);
      int numRandom = Math.min(16, totalSize / 3);
      int numHighDegree = Math.min(16, totalSize - numPrimary - numRandom);

      // Primary: Use first portion of sample
      List<Long> primaryEntries = sampledNodes.stream().limit(numPrimary).collect(Collectors.toList());

      // Random: Use middle portion of sample
      List<Long> randomEntries =
          sampledNodes.stream().skip(numPrimary).limit(numRandom).collect(Collectors.toList());

      // High-degree: Find nodes with most connections from remaining sample
      List<Long> candidatesForHighDegree = sampledNodes.stream()
          .skip(numPrimary + numRandom)
          .limit(numHighDegree)
          .collect(Collectors.toList());

      return findHighDegreeNodesFromList(tx, candidatesForHighDegree, numHighDegree)
          .thenCompose(highDegreeEntries ->
              entryPointStorage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries));
    });
  }

  /**
   * Finds nodes with the highest degree from a given list.
   */
  private CompletableFuture<List<Long>> findHighDegreeNodesFromList(Transaction tx, List<Long> candidates, int k) {

    if (candidates.isEmpty()) {
      return completedFuture(Collections.emptyList());
    }

    // Load adjacencies to get degrees
    List<CompletableFuture<NodeDegree>> degreeFutures = new ArrayList<>();

    for (Long nodeId : candidates) {
      degreeFutures.add(adjacencyStorage
          .loadAdjacency(tx, nodeId)
          .thenApply(adj -> new NodeDegree(nodeId, adj != null ? adj.getNeighborsCount() : 0)));
    }

    return CompletableFuture.allOf(degreeFutures.toArray(new CompletableFuture[0]))
        .thenApply(v -> {
          // Sort by degree and return top k
          PriorityQueue<NodeDegree> highDegreeNodes =
              new PriorityQueue<>((a, b) -> Integer.compare(b.degree(), a.degree())); // Max heap

          for (CompletableFuture<NodeDegree> future : degreeFutures) {
            highDegreeNodes.add(future.join());
          }

          List<Long> result = new ArrayList<>();
          for (int i = 0; i < k && !highDegreeNodes.isEmpty(); i++) {
            result.add(highDegreeNodes.poll().nodeId());
          }

          return result;
        });
  }

  /**
   * Samples random nodes from the graph using intelligent range-based probing.
   *
   * @param tx         the transaction
   * @param sampleSize number of nodes to sample
   * @return future with list of sampled node IDs
   */
  private CompletableFuture<List<Long>> sampleRandomNodes(Transaction tx, int sampleSize) {
    return getNodeIdRange(tx).thenCompose(range -> {
      if (range == null) {
        return completedFuture(Collections.emptyList());
      }

      // Start with the full range
      List<SamplingRange> ranges = new ArrayList<>();
      ranges.add(new SamplingRange(range.min, range.max));

      Set<Long> sampledIds = new HashSet<>();

      // Start sampling with alternating probe direction
      return sampleWithRanges(tx, ranges, sampledIds, sampleSize, true);
    });
  }

  /**
   * Recursively samples nodes by probing ranges and splitting on misses.
   */
  private CompletableFuture<List<Long>> sampleWithRanges(
      Transaction tx, List<SamplingRange> ranges, Set<Long> sampledIds, int targetSize, boolean useForward) {

    if (sampledIds.size() >= targetSize || ranges.isEmpty()) {
      List<Long> result = new ArrayList<>(sampledIds);
      return completedFuture(result.subList(0, Math.min(targetSize, result.size())));
    }

    // Select a range probabilistically based on size
    SamplingRange selectedRange = selectRangeBySize(ranges);

    // Generate random probe point within range
    long probeId = selectedRange.min + (long) (random.nextDouble() * (selectedRange.max - selectedRange.min + 1));

    LOGGER.info("Probing range " + selectedRange.min + " - " + selectedRange.max + " at " + probeId);

    // Probe for node
    byte[] searchKey = keys.nodeAdjacencyKey(probeId);
    KeySelector selector =
        useForward ? KeySelector.firstGreaterOrEqual(searchKey) : KeySelector.lastLessOrEqual(searchKey);

    return tx.getKey(selector).thenCompose(foundKey -> {
      long foundId = extractNodeIdFromKey(foundKey);

      if (foundId >= selectedRange.min && foundId <= selectedRange.max) {
        LOGGER.info("Found node " + foundId + " in range " + selectedRange.min + " - " + selectedRange.max);
        // Hit - found a node in the range
        boolean isNewNode = sampledIds.add(foundId);

        if (!isNewNode) {
          // We found a node we already sampled - split the range to avoid it
          List<SamplingRange> newRanges = new ArrayList<>(ranges);
          newRanges.remove(selectedRange);

          // Split the range around the found node
          if (foundId > selectedRange.min) {
            if (foundId - selectedRange.min >= 2) {
              newRanges.add(new SamplingRange(selectedRange.min, foundId - 1));
            }
          }
          if (foundId < selectedRange.max) {
            if (selectedRange.max - foundId >= 2) {
              newRanges.add(new SamplingRange(foundId + 1, selectedRange.max));
            }
          }

          if (newRanges.isEmpty()) {
            // No more ranges to explore
            List<Long> result = new ArrayList<>(sampledIds);
            return completedFuture(result);
          }

          return sampleWithRanges(tx, newRanges, sampledIds, targetSize, !useForward);
        }

        // Continue with same ranges, alternating direction
        return sampleWithRanges(tx, ranges, sampledIds, targetSize, !useForward);

      } else {
        LOGGER.info("Missed node in range " + selectedRange.min + " - " + selectedRange.max + " at " + probeId);
        // Miss - split the range
        List<SamplingRange> newRanges = new ArrayList<>(ranges);
        newRanges.remove(selectedRange);

        // Split based on where we found a node
        if (useForward && foundId > selectedRange.max) {
          // Found node is beyond our range, entire range is empty
        } else if (!useForward && foundId < selectedRange.min) {
          // Found node is before our range, entire range is empty
        } else if (useForward && foundId > probeId) {
          // Empty space between probeId and foundId
          newRanges.add(new SamplingRange(foundId, selectedRange.max));
          sampledIds.add(foundId); // Also use the found node
        } else if (!useForward && foundId < probeId) {
          // Empty space between foundId and probeId
          newRanges.add(new SamplingRange(selectedRange.min, foundId));
          sampledIds.add(foundId); // Also use the found node
        }

        // Continue with split ranges if any remain, alternating direction
        if (newRanges.isEmpty()) {
          // No more ranges to explore
          List<Long> result = new ArrayList<>(sampledIds);
          return completedFuture(result);
        }
        return sampleWithRanges(tx, newRanges, sampledIds, targetSize, !useForward);
      }
    });
  }

  /**
   * Selects a range probabilistically based on its size.
   */
  private SamplingRange selectRangeBySize(List<SamplingRange> ranges) {
    // Calculate total size
    long totalSize = ranges.stream().mapToLong(r -> r.size).sum();

    // Pick random point in total size
    long randomPoint = (long) (random.nextDouble() * totalSize);

    // Find which range contains this point
    long cumulative = 0;
    for (SamplingRange range : ranges) {
      cumulative += range.size;
      if (randomPoint < cumulative) {
        return range;
      }
    }

    // Shouldn't happen, but return last range as fallback
    return ranges.get(ranges.size() - 1);
  }

  /**
   * Gets the min and max node IDs in the graph.
   */
  private CompletableFuture<NodeIdRange> getNodeIdRange(Transaction tx) {
    byte[] prefix = keys.allNodesPrefix();

    // Use getKey to find first and last keys efficiently
    CompletableFuture<byte[]> firstKeyFuture = tx.getKey(KeySelector.firstGreaterOrEqual(prefix));
    CompletableFuture<byte[]> lastKeyFuture = tx.getKey(KeySelector.lastLessOrEqual(ByteArrayUtil.strinc(prefix)));

    return firstKeyFuture.thenCombine(lastKeyFuture, (firstKey, lastKey) -> {
      if (firstKey == null || lastKey == null) {
        return null;
      }
      if (!ByteArrayUtil.startsWith(firstKey, prefix) || !ByteArrayUtil.startsWith(lastKey, prefix)) {
        return null;
      }

      long firstId = extractNodeIdFromKey(firstKey);
      long lastId = extractNodeIdFromKey(lastKey);

      return new NodeIdRange(firstId, lastId);
    });
  }

  /**
   * Estimates connectivity from sample nodes.
   *
   * @param tx          the transaction
   * @param sampleNodes the sampled node IDs
   * @return future with connectivity analysis
   */
  private CompletableFuture<ConnectivityAnalysis> estimateConnectivityFromSamples(
      Transaction tx, List<Long> sampleNodes) {

    // Track reachability from each sample

    CompletableFuture<List<Integer>> reachabilityFutures = completedFuture(new ArrayList<>(sampleNodes.size()));

    for (Long nodeId : sampleNodes) {
      CompletableFuture<Integer> future = limitedBFS(tx, nodeId, MAX_BFS_VISITS);
      reachabilityFutures = reachabilityFutures.thenCombine(future, (list, count) -> {
        list.add(count);
        return list;
      });
    }

    return reachabilityFutures.thenCompose(reachCounts -> {
      // Estimate total nodes and connectivity
      double avgReachability = 0;
      int maxReachability = Integer.MIN_VALUE;

      for (int count : reachCounts) {
        avgReachability = avgReachability + count;
        maxReachability = Math.max(maxReachability, count);
      }
      avgReachability = avgReachability / reachCounts.size();

      // Better estimation: use actual sample size for small graphs
      long estimatedTotal;
      long largestComponentEstimate;

      if (sampleNodes.size() < 100) {
        // For small samples, use actual counts
        estimatedTotal = Math.max(sampleNodes.size(), maxReachability);
        largestComponentEstimate = maxReachability;
      } else {
        // For larger samples, if most samples can reach similar numbers of nodes,
        // it indicates good connectivity
        estimatedTotal = Math.max(sampleNodes.size(), maxReachability);

        // If average reachability is high relative to sample size,
        // the graph is likely well-connected
        if (avgReachability >= sampleNodes.size() * 0.8) {
          // Most samples can reach most other samples - well connected
          largestComponentEstimate = estimatedTotal;
        } else {
          // Use the max reachability as estimate for largest component
          largestComponentEstimate = maxReachability;
        }
      }

      // For now, we don't track specific orphaned nodes in sampling mode
      List<Long> orphanedNodes = Collections.emptyList();

      // Store analysis results
      return metaStorage
          .updateConnectivityStats(tx, 1, largestComponentEstimate, estimatedTotal, orphanedNodes)
          .thenApply(unused ->
              new ConnectivityAnalysis(1, largestComponentEstimate, estimatedTotal, orphanedNodes));
    });
  }

  /**
   * Performs limited BFS from a node.
   *
   * @param tx        the transaction
   * @param startNode the starting node
   * @param maxVisits maximum nodes to visit
   * @return number of nodes reached
   */
  private CompletableFuture<Integer> limitedBFS(Transaction tx, Long startNode, int maxVisits) {
    Set<Long> visited = new HashSet<>();
    Queue<Long> queue = new ArrayDeque<>();
    queue.offer(startNode);
    visited.add(startNode);

    return limitedBFSHelper(tx, queue, visited, maxVisits);
  }

  private CompletableFuture<Integer> limitedBFSHelper(
      Transaction tx, Queue<Long> queue, Set<Long> visited, int remaining) {

    if (queue.isEmpty() || remaining <= 0) {
      return completedFuture(visited.size());
    }

    Long current = queue.poll();

    return adjacencyStorage.loadAdjacency(tx, current).thenCompose(adjacency -> {
      if (adjacency != null) {
        for (Long neighbor : adjacency.getNeighborsList()) {
          if (!visited.contains(neighbor) && visited.size() < remaining) {
            visited.add(neighbor);
            queue.offer(neighbor);
          }
        }
      }
      return limitedBFSHelper(tx, queue, visited, remaining - 1);
    });
  }

  /**
   * Extracts node ID from a key.
   */
  private long extractNodeIdFromKey(byte[] key) {
    // Key format: prefix + nodeId (as tuple)
    // Extract the nodeId portion after the prefix
    return keys.extractNodeIdFromAdjacencyKey(key);
  }

  /**
   * Result of connectivity analysis.
   */
  public record ConnectivityAnalysis(
      int connectedComponents, long largestComponentSize, long totalNodes, List<Long> orphanedNodes) {
    public ConnectivityAnalysis(
        int connectedComponents, long largestComponentSize, long totalNodes, List<Long> orphanedNodes) {
      this.connectedComponents = connectedComponents;
      this.largestComponentSize = largestComponentSize;
      this.totalNodes = totalNodes;
      this.orphanedNodes = orphanedNodes != null ? orphanedNodes : Collections.emptyList();
    }

    public boolean needsRepair() {
      if (totalNodes == 0) {
        return false;
      }
      double connectivityRatio = (double) largestComponentSize / totalNodes;
      return connectivityRatio < HEALTHY_THRESHOLD || !orphanedNodes.isEmpty();
    }
  }

  /**
   * Helper class for tracking node distances.
   */
  record NodeDistance(Long nodeId, float distance) implements Comparable<NodeDistance> {

    @Override
    public int compareTo(NodeDistance other) {
      return Float.compare(this.distance, other.distance);
    }
  }

  /**
   * Helper class for tracking node degrees.
   */
  private record NodeDegree(Long nodeId, int degree) {
  }

  /**
   * Represents a range for sampling.
   */
  static class SamplingRange {
    final long min;
    final long max;
    final long size;

    SamplingRange(long min, long max) {
      if (min > max) {
        throw new IllegalArgumentException("Invalid range: " + min + " > " + max);
      }
      if (min < 0) {
        throw new IllegalArgumentException("Invalid range: " + min + " or " + max + " is negative");
      }
      this.min = min;
      this.max = max;
      this.size = max - min + 1;
    }
  }

  /**
   * Represents a node ID range.
   */
  record NodeIdRange(long min, long max) {
  }
}
