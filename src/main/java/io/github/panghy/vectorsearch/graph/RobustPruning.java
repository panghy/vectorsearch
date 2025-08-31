package io.github.panghy.vectorsearch.graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.logging.Logger;
import lombok.Builder;
import lombok.Getter;

/**
 * Implements DiskANN-style robust pruning algorithm for neighbor selection in graph construction.
 *
 * <p>The algorithm maintains diversity in the neighbor list by pruning candidates that are
 * too similar to already selected neighbors. This helps create a well-connected graph with
 * good search properties.
 *
 * <p>Based on the Vamana algorithm from DiskANN paper:
 * "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
 *
 * <p>Key features:
 * <ul>
 *   <li>Preserves closest neighbors while maintaining diversity</li>
 *   <li>Uses alpha parameter to control diversity vs proximity trade-off</li>
 *   <li>Supports both distance-to-query and pairwise distance pruning</li>
 *   <li>Handles bidirectional edge maintenance</li>
 * </ul>
 */
public class RobustPruning {
  private static final Logger LOGGER = Logger.getLogger(RobustPruning.class.getName());

  /**
   * Default alpha parameter for diversity control.
   * Values typically range from 0.95 to 1.2:
   * - Lower values (0.95): More aggressive pruning, higher diversity
   * - Higher values (1.2): Less aggressive pruning, more proximity-focused
   */
  public static final double DEFAULT_ALPHA = 1.2;

  /**
   * Represents a candidate neighbor for pruning consideration.
   */
  @Getter
  @Builder
  public static class Candidate {
    private final long nodeId;
    private final float distanceToQuery;

    /**
     * Natural ordering by distance (ascending).
     */
    public int compareTo(Candidate other) {
      return Float.compare(this.distanceToQuery, other.distanceToQuery);
    }
  }

  /**
   * Configuration for the robust pruning algorithm.
   */
  @Getter
  @Builder
  public static class PruningConfig {
    @Builder.Default
    private final int maxDegree = 64;

    @Builder.Default
    private final double alpha = DEFAULT_ALPHA;

    @Builder.Default
    private final boolean usePairwiseDistance = false;

    @Builder.Default
    private final boolean maintainConnectivity = true;
  }

  /**
   * Performs robust pruning on a list of candidates sorted by distance to query.
   *
   * <p>The algorithm iterates through candidates in order of proximity and keeps
   * those that are not dominated by already selected neighbors. A candidate is
   * dominated if its distance to the query is within alpha * distance of an
   * already selected neighbor.
   *
   * @param candidates List of candidates sorted by distance to query (ascending)
   * @param config Pruning configuration
   * @return List of selected neighbor IDs (up to maxDegree)
   */
  public static List<Long> prune(List<Candidate> candidates, PruningConfig config) {
    if (candidates.isEmpty()) {
      return Collections.emptyList();
    }

    // Ensure candidates are sorted
    List<Candidate> sortedCandidates = new ArrayList<>(candidates);
    sortedCandidates.sort(Candidate::compareTo);

    List<Long> selected = new ArrayList<>();
    Set<Long> selectedSet = new HashSet<>();

    for (Candidate candidate : sortedCandidates) {
      if (selected.size() >= config.getMaxDegree()) {
        break;
      }

      // Check if candidate is dominated by any selected neighbor
      boolean dominated = false;
      for (int i = 0; i < selected.size(); i++) {
        final Long selectedId = selected.get(i);
        Candidate selectedCandidate = sortedCandidates.stream()
            .filter(c -> c.getNodeId() == selectedId)
            .findFirst()
            .orElse(null);

        if (selectedCandidate != null && isDominated(candidate, selectedCandidate, config)) {
          dominated = true;
          break;
        }
      }

      if (!dominated && !selectedSet.contains(candidate.getNodeId())) {
        selected.add(candidate.getNodeId());
        selectedSet.add(candidate.getNodeId());

        LOGGER.fine(String.format(
            "Selected node %d with distance %.4f (count: %d/%d)",
            candidate.getNodeId(), candidate.getDistanceToQuery(), selected.size(), config.getMaxDegree()));
      }
    }

    return selected;
  }

  /**
   * Performs robust pruning with pairwise distance computation.
   *
   * <p>This variant uses actual pairwise distances between candidates for more
   * accurate pruning decisions, at the cost of additional distance computations.
   *
   * @param candidates List of candidates sorted by distance to query
   * @param config Pruning configuration
   * @param pairwiseDistance Function to compute distance between two nodes
   * @return List of selected neighbor IDs
   */
  public static List<Long> pruneWithPairwiseDistance(
      List<Candidate> candidates, PruningConfig config, BiFunction<Long, Long, Float> pairwiseDistance) {

    if (candidates.isEmpty()) {
      return Collections.emptyList();
    }

    // Ensure candidates are sorted
    List<Candidate> sortedCandidates = new ArrayList<>(candidates);
    sortedCandidates.sort(Candidate::compareTo);

    List<Long> selected = new ArrayList<>();
    Set<Long> selectedSet = new HashSet<>();

    for (Candidate candidate : sortedCandidates) {
      if (selected.size() >= config.getMaxDegree()) {
        break;
      }

      // Check if candidate is dominated using pairwise distances
      boolean dominated = false;
      for (Long selectedId : selected) {
        float pairDist = pairwiseDistance.apply(candidate.getNodeId(), selectedId);

        // If candidate is very close to an already selected node, it's dominated
        if (pairDist < config.getAlpha() * candidate.getDistanceToQuery()) {
          dominated = true;
          LOGGER.fine(String.format(
              "Node %d dominated by %d (pairwise dist: %.4f < %.4f)",
              candidate.getNodeId(),
              selectedId,
              pairDist,
              config.getAlpha() * candidate.getDistanceToQuery()));
          break;
        }
      }

      if (!dominated && !selectedSet.contains(candidate.getNodeId())) {
        selected.add(candidate.getNodeId());
        selectedSet.add(candidate.getNodeId());

        LOGGER.fine(String.format(
            "Selected node %d with distance %.4f (count: %d/%d)",
            candidate.getNodeId(), candidate.getDistanceToQuery(), selected.size(), config.getMaxDegree()));
      }
    }

    return selected;
  }

  /**
   * Merges existing neighbors with new candidates and prunes to maintain degree bound.
   *
   * <p>This is used when updating an existing node's adjacency list with new candidates.
   * The algorithm combines current neighbors with new candidates, removes duplicates,
   * and prunes to the maximum degree if necessary.
   *
   * @param currentNeighbors Existing neighbor list
   * @param newCandidates New candidates to consider
   * @param config Pruning configuration
   * @param distanceFunction Function to compute distance to query for any node
   * @return Merged and pruned neighbor list
   */
  public static List<Long> mergeAndPrune(
      List<Long> currentNeighbors,
      List<Candidate> newCandidates,
      PruningConfig config,
      Function<Long, Float> distanceFunction) {

    // Combine current neighbors and new candidates
    Set<Long> allNodes = new HashSet<>(currentNeighbors);
    List<Candidate> allCandidates = new ArrayList<>();

    // Add existing neighbors as candidates
    for (Long nodeId : currentNeighbors) {
      if (!allNodes.contains(nodeId)) {
        continue;
      }
      float distance = distanceFunction.apply(nodeId);
      allCandidates.add(
          Candidate.builder().nodeId(nodeId).distanceToQuery(distance).build());
    }

    // Add new candidates
    for (Candidate candidate : newCandidates) {
      if (!allNodes.contains(candidate.getNodeId())) {
        allCandidates.add(candidate);
        allNodes.add(candidate.getNodeId());
      }
    }

    // Sort by distance
    allCandidates.sort(Candidate::compareTo);

    // If within degree limit, keep all
    if (allCandidates.size() <= config.getMaxDegree()) {
      return allCandidates.stream().map(Candidate::getNodeId).toList();
    }

    // Otherwise, apply robust pruning
    return prune(allCandidates, config);
  }

  /**
   * Checks if a candidate is dominated by a selected neighbor.
   *
   * <p>A candidate is dominated if it's too similar to an already selected neighbor,
   * as determined by the alpha parameter. This promotes diversity in the neighbor set.
   *
   * @param candidate The candidate to check
   * @param selected An already selected neighbor
   * @param config Pruning configuration
   * @return true if candidate is dominated, false otherwise
   */
  private static boolean isDominated(Candidate candidate, Candidate selected, PruningConfig config) {
    // The dominance test from the original ideation's RobustPrune algorithm:
    // A candidate c is dominated by a selected neighbor r if:
    //   dist_q(c) <= tau * dist_q(r)
    // where tau is the alpha parameter.
    //
    // This means we skip candidates that are too close to already selected nodes,
    // promoting diversity in the neighbor set.
    return candidate.getDistanceToQuery() <= config.getAlpha() * selected.getDistanceToQuery();
  }

  /**
   * Prunes back-links when a node's neighbor list changes.
   *
   * <p>This ensures bidirectional consistency in the graph. When node A removes
   * node B from its neighbor list, node B should also remove A from its list.
   *
   * @param nodeId The node whose neighbors changed
   * @param oldNeighbors Previous neighbor list
   * @param newNeighbors New neighbor list after pruning
   * @return List of nodes that need their back-links updated
   */
  public static List<Long> getBackLinksToRemove(long nodeId, List<Long> oldNeighbors, List<Long> newNeighbors) {

    Set<Long> oldSet = new HashSet<>(oldNeighbors);
    Set<Long> newSet = new HashSet<>(newNeighbors);

    // Find neighbors that were removed
    List<Long> removed = new ArrayList<>();
    for (Long neighbor : oldSet) {
      if (!newSet.contains(neighbor)) {
        removed.add(neighbor);
      }
    }

    return removed;
  }

  /**
   * Gets the list of new back-links to add.
   *
   * <p>When node A adds node B as a neighbor, node B should add A as well
   * (subject to its own degree constraints).
   *
   * @param nodeId The node whose neighbors changed
   * @param oldNeighbors Previous neighbor list
   * @param newNeighbors New neighbor list after pruning
   * @return List of nodes that need new back-links added
   */
  public static List<Long> getBackLinksToAdd(long nodeId, List<Long> oldNeighbors, List<Long> newNeighbors) {

    Set<Long> oldSet = new HashSet<>(oldNeighbors);
    Set<Long> newSet = new HashSet<>(newNeighbors);

    // Find neighbors that were added
    List<Long> added = new ArrayList<>();
    for (Long neighbor : newSet) {
      if (!oldSet.contains(neighbor)) {
        added.add(neighbor);
      }
    }

    return added;
  }
}
