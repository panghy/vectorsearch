package io.github.panghy.vectorsearch.search;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Represents a search result containing a node ID and its distance from the query vector.
 * Used as the output of approximate nearest neighbor search operations.
 */
@Data
@AllArgsConstructor
public class SearchResult implements Comparable<SearchResult> {
  /** The ID of the node in the vector index. */
  private final long nodeId;

  /** The distance from the query vector (lower is more similar). */
  private final float distance;

  /**
   * Compares results by distance for sorting.
   * Natural ordering is by increasing distance (closer vectors first).
   *
   * @param other the other search result to compare to
   * @return negative if this result is closer, positive if farther, 0 if equal
   */
  @Override
  public int compareTo(SearchResult other) {
    return Float.compare(this.distance, other.distance);
  }

  @Override
  public String toString() {
    return String.format("SearchResult(nodeId=%d, distance=%.4f)", nodeId, distance);
  }
}
