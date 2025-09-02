package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;

import com.apple.foundationdb.Transaction;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.GraphMeta;
import java.time.Instant;
import java.time.InstantSource;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Storage layer for graph connectivity metadata.
 * Tracks connectivity analysis results, orphaned nodes, and graph statistics.
 *
 * <p>This class manages the GraphMeta protobuf which contains:
 * <ul>
 *   <li>Connectivity analysis timestamp and results</li>
 *   <li>Number of connected components</li>
 *   <li>Size of the largest component</li>
 *   <li>List of orphaned nodes needing repair</li>
 *   <li>Graph statistics (degree distribution, etc.)</li>
 * </ul>
 *
 * <p>All operations use transaction parameters to allow callers to control
 * transaction boundaries and batch operations efficiently.
 */
public class GraphMetaStorage {
  private static final Logger LOGGER = LoggerFactory.getLogger(GraphMetaStorage.class);

  private final VectorIndexKeys keys;
  private final InstantSource instantSource;

  /**
   * Creates a new GraphMetaStorage instance.
   *
   * @param keys          vector index keys for this collection
   * @param instantSource source for timestamps
   */
  public GraphMetaStorage(VectorIndexKeys keys, InstantSource instantSource) {
    this.keys = keys;
    this.instantSource = instantSource;
  }

  /**
   * Stores or updates the graph metadata.
   *
   * @param tx         the transaction to use
   * @param graphMeta  the graph metadata to store
   * @return future that completes when the metadata is stored
   */
  public CompletableFuture<Void> storeGraphMeta(Transaction tx, GraphMeta graphMeta) {
    byte[] key = keys.graphMetaKey();
    tx.set(key, graphMeta.toByteArray());

    LOGGER.debug(
        "Stored graph metadata: components={}, largest={}, total={}, orphans={}",
        graphMeta.getConnectedComponents(),
        graphMeta.getLargestComponentSize(),
        graphMeta.getTotalNodes(),
        graphMeta.getOrphanedNodesCount());

    return CompletableFuture.completedFuture(null);
  }

  /**
   * Loads the graph metadata.
   *
   * @param tx the transaction to use
   * @return future containing the graph metadata, or null if not found
   */
  public CompletableFuture<GraphMeta> loadGraphMeta(Transaction tx) {
    byte[] key = keys.graphMetaKey();
    return readProto(tx, key, GraphMeta.parser());
  }

  /**
   * Updates connectivity analysis results.
   *
   * @param tx                    the transaction to use
   * @param connectedComponents   number of disconnected subgraphs
   * @param largestComponentSize  nodes in the main component
   * @param totalNodes            total active nodes
   * @param orphanedNodes         list of orphaned node IDs
   * @return future that completes when the metadata is updated
   */
  public CompletableFuture<Void> updateConnectivityStats(
      Transaction tx,
      int connectedComponents,
      long largestComponentSize,
      long totalNodes,
      List<Long> orphanedNodes) {

    byte[] key = keys.graphMetaKey();

    return readProto(tx, key, GraphMeta.parser()).thenApply(existing -> {
      GraphMeta.Builder builder = existing != null ? existing.toBuilder() : GraphMeta.newBuilder();

      // Update connectivity stats
      builder.setLastAnalysisTimestamp(currentTimestamp());
      builder.setConnectedComponents(connectedComponents);
      builder.setLargestComponentSize(largestComponentSize);
      builder.setTotalNodes(totalNodes);

      // Update orphaned nodes list
      builder.clearOrphanedNodes();
      if (orphanedNodes != null && !orphanedNodes.isEmpty()) {
        builder.addAllOrphanedNodes(orphanedNodes);
      }

      // Calculate connectivity percentage
      double connectivityPercentage = totalNodes > 0 ? (double) largestComponentSize / totalNodes * 100.0 : 0.0;

      LOGGER.info(
          "Connectivity analysis: {} components, main={:.1f}% ({}/{} nodes), {} orphans",
          connectedComponents,
          connectivityPercentage,
          largestComponentSize,
          totalNodes,
          orphanedNodes != null ? orphanedNodes.size() : 0);

      tx.set(key, builder.build().toByteArray());
      return null;
    });
  }

  /**
   * Updates graph statistics (degree distribution).
   *
   * @param tx               the transaction to use
   * @param avgDegree        average node degree
   * @param maxDegree        maximum node degree
   * @param minDegree        minimum node degree
   * @param isolatedNodes    number of nodes with no neighbors
   * @return future that completes when the statistics are updated
   */
  public CompletableFuture<Void> updateGraphStatistics(
      Transaction tx, double avgDegree, int maxDegree, int minDegree, int isolatedNodes) {

    byte[] key = keys.graphMetaKey();

    return readProto(tx, key, GraphMeta.parser()).thenApply(existing -> {
      GraphMeta.Builder builder = existing != null ? existing.toBuilder() : GraphMeta.newBuilder();

      builder.setAverageDegree(avgDegree);
      // Max/min degree and isolated nodes are not in the proto currently
      // We could add them to total_edges or clustering_coefficient if needed

      tx.set(key, builder.build().toByteArray());

      LOGGER.debug(
          "Updated graph statistics: avg_degree={:.2f}, max={}, min={}, isolated={}",
          avgDegree,
          maxDegree,
          minDegree,
          isolatedNodes);

      return null;
    });
  }

  /**
   * Marks that a repair operation has completed.
   *
   * @param tx            the transaction to use
   * @param repairedNodes number of nodes that were repaired
   * @return future that completes when the repair is marked
   */
  public CompletableFuture<Void> markRepairCompleted(Transaction tx, int repairedNodes) {
    byte[] key = keys.graphMetaKey();

    return readProto(tx, key, GraphMeta.parser()).thenApply(existing -> {
      if (existing == null) {
        LOGGER.warn("Cannot mark repair completed: no graph metadata found");
        return null;
      }

      GraphMeta.Builder builder = existing.toBuilder();
      // Repair timestamp is not in the current proto
      // We can track it in last_analysis_timestamp if needed
      builder.setLastAnalysisTimestamp(currentTimestamp());
      builder.clearOrphanedNodes(); // Clear orphans after repair

      tx.set(key, builder.build().toByteArray());

      LOGGER.info("Graph repair completed: {} nodes reconnected", repairedNodes);

      return null;
    });
  }

  /**
   * Checks if connectivity analysis is needed based on time since last analysis.
   *
   * @param tx             the transaction to use
   * @param maxAgeSeconds  maximum age in seconds before reanalysis is needed
   * @return future containing true if analysis is needed
   */
  public CompletableFuture<Boolean> isAnalysisNeeded(Transaction tx, long maxAgeSeconds) {
    return loadGraphMeta(tx).thenApply(meta -> {
      if (meta == null || !meta.hasLastAnalysisTimestamp()) {
        return true; // No previous analysis
      }

      Instant lastAnalysis = Instant.ofEpochSecond(
          meta.getLastAnalysisTimestamp().getSeconds(),
          meta.getLastAnalysisTimestamp().getNanos());

      Instant now = instantSource.instant();
      long ageSeconds = now.getEpochSecond() - lastAnalysis.getEpochSecond();

      return ageSeconds >= maxAgeSeconds;
    });
  }

  /**
   * Gets the connectivity health status.
   *
   * @param tx                  the transaction to use
   * @param healthyThreshold    percentage threshold for healthy connectivity (e.g., 0.95 for 95%)
   * @return future containing true if graph is healthy
   */
  public CompletableFuture<Boolean> isGraphHealthy(Transaction tx, double healthyThreshold) {
    return loadGraphMeta(tx).thenApply(meta -> {
      if (meta == null || meta.getTotalNodes() == 0) {
        return true; // Empty or uninitialized graph is considered healthy
      }

      double connectivityRatio = (double) meta.getLargestComponentSize() / meta.getTotalNodes();
      boolean isHealthy = connectivityRatio >= healthyThreshold;

      if (!isHealthy) {
        LOGGER.warn(
            "Graph connectivity below threshold: {:.1f}% < {:.1f}%",
            connectivityRatio * 100, healthyThreshold * 100);
      }

      return isHealthy;
    });
  }

  /**
   * Clears the graph metadata.
   *
   * @param tx the transaction to use
   * @return future that completes when the metadata is cleared
   */
  public CompletableFuture<Void> clearGraphMeta(Transaction tx) {
    byte[] key = keys.graphMetaKey();
    tx.clear(key);
    return CompletableFuture.completedFuture(null);
  }

  private Timestamp currentTimestamp() {
    Instant now = instantSource.instant();
    return Timestamp.newBuilder()
        .setSeconds(now.getEpochSecond())
        .setNanos(now.getNano())
        .build();
  }
}
