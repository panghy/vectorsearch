package io.github.panghy.vectorsearch.search;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Internal data structure for tracking candidates during beam search.
 * Maintains both the node information and search state (visited status).
 */
@Data
@AllArgsConstructor
public class SearchCandidate implements Comparable<SearchCandidate> {
  /** The ID of the candidate node. */
  private final long nodeId;

  /** The distance from the query vector. */
  private final float distance;

  /** Whether this node has been visited (expanded) during search. */
  private boolean visited;

  /**
   * Creates a new unvisited search candidate.
   *
   * @param nodeId the node ID
   * @param distance the distance from query
   * @return a new unvisited candidate
   */
  public static SearchCandidate unvisited(long nodeId, float distance) {
    return new SearchCandidate(nodeId, distance, false);
  }

  /**
   * Marks this candidate as visited.
   */
  public void markVisited() {
    this.visited = true;
  }

  /**
   * Compares candidates by distance for priority queue ordering.
   * Natural ordering is by increasing distance (closer candidates first).
   *
   * @param other the other candidate to compare to
   * @return negative if this candidate is closer, positive if farther, 0 if equal
   */
  @Override
  public int compareTo(SearchCandidate other) {
    return Float.compare(this.distance, other.distance);
  }

  /**
   * Converts this candidate to a search result.
   *
   * @return a SearchResult with the same nodeId and distance
   */
  public SearchResult toResult() {
    return new SearchResult(nodeId, distance);
  }

  @Override
  public String toString() {
    return String.format("SearchCandidate(nodeId=%d, distance=%.4f, visited=%s)", nodeId, distance, visited);
  }
}
