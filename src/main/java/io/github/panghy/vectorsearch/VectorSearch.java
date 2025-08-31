package io.github.panghy.vectorsearch;

import io.github.panghy.vectorsearch.search.SearchResult;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * Interface for vector search operations on a distributed index.
 *
 * <p>This interface provides methods for inserting, searching, and deleting vectors
 * in a high-performance approximate nearest neighbor (ANN) index. Implementations
 * typically use techniques like Product Quantization (PQ) for compression and
 * graph-based algorithms like DiskANN for efficient search.
 *
 * <p>All operations are asynchronous and return {@link CompletableFuture} to support
 * non-blocking I/O and high throughput. The index maintains ACID properties through
 * the underlying storage system (e.g., FoundationDB).
 *
 * <p>Key features:
 * <ul>
 *   <li>Automatic ID assignment for new vectors</li>
 *   <li>Batch operations for efficiency</li>
 *   <li>Configurable search parameters for recall/latency trade-offs</li>
 *   <li>Online index updates without downtime</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * VectorSearch index = FdbVectorSearch.createOrOpen(config, database).join();
 *
 * // Insert vectors with automatic ID assignment
 * float[][] vectors = loadVectors();
 * List<Long> assignedIds = index.insert(Arrays.asList(vectors)).join();
 *
 * // Search for similar vectors
 * float[] query = getQueryVector();
 * List<SearchResult> results = index.search(query, 10).join();
 *
 * // Delete vectors by ID
 * index.delete(List.of(id1, id2, id3)).join();
 * }</pre>
 */
public interface VectorSearch {

  /**
   * Inserts new vectors into the index with automatically assigned IDs.
   *
   * <p>This method assigns unique IDs to each vector and enqueues them for
   * asynchronous graph construction. Vectors become searchable after the
   * link workers process them (typically within seconds).
   *
   * <p>The returned list contains the assigned IDs in the same order as the
   * input vectors. IDs are guaranteed to be unique within the collection.
   *
   * @param vectors the vectors to insert, each must have the configured dimension
   * @return a future containing a list of assigned IDs corresponding to each vector
   * @throws IllegalArgumentException if any vector has incorrect dimension
   * @throws IllegalStateException if the index is not initialized
   */
  CompletableFuture<List<Long>> insert(List<float[]> vectors);

  /**
   * Inserts new vectors into the index with automatically assigned IDs.
   *
   * <p>Convenience method for array input. See {@link #insert(List)}.
   *
   * @param vectors the vectors to insert
   * @return a future containing a list of assigned IDs corresponding to each vector
   */
  default CompletableFuture<List<Long>> insert(float[]... vectors) {
    return insert(List.of(vectors));
  }

  /**
   * Updates existing vectors or inserts new ones with specified IDs.
   *
   * <p>This method allows explicit ID assignment, useful for:
   * <ul>
   *   <li>Updating existing vectors with new values</li>
   *   <li>Maintaining external ID mappings</li>
   *   <li>Bulk loading with predetermined IDs</li>
   * </ul>
   *
   * <p>If a vector with the given ID already exists, it will be updated.
   * The update process re-links the node in the graph with new neighbors
   * based on the updated vector value.
   *
   * @param vectors map of IDs to vectors
   * @return a future that completes when all vectors are enqueued
   * @throws IllegalArgumentException if any vector has incorrect dimension
   * @throws IllegalStateException if the index is not initialized
   */
  CompletableFuture<Void> upsert(Map<Long, float[]> vectors);

  /**
   * Searches for the k nearest neighbors of a query vector.
   *
   * <p>This method performs approximate nearest neighbor search using the
   * configured distance metric (L2, cosine, or inner product). The search
   * traverses the graph structure starting from entry points and expands
   * the search frontier using a beam search algorithm.
   *
   * <p>Search quality can be tuned via configuration parameters:
   * <ul>
   *   <li>searchList: controls beam width (default: max(16, k))</li>
   *   <li>maxVisits: limits node expansions for latency bounds</li>
   * </ul>
   *
   * @param queryVector the vector to search for
   * @param k the number of nearest neighbors to return
   * @return a future containing the search results sorted by distance
   * @throws IllegalArgumentException if queryVector has incorrect dimension
   * @throws IllegalStateException if the index is not initialized
   */
  CompletableFuture<List<SearchResult>> search(float[] queryVector, int k);

  /**
   * Searches for nearest neighbors with custom search parameters.
   *
   * <p>This method allows fine-tuning search parameters per query for
   * optimal recall/latency trade-offs.
   *
   * @param queryVector the vector to search for
   * @param k the number of nearest neighbors to return
   * @param searchList the beam width for search (larger = better recall, higher latency)
   * @param maxVisits maximum nodes to visit (caps latency)
   * @return a future containing the search results sorted by distance
   */
  CompletableFuture<List<SearchResult>> search(float[] queryVector, int k, int searchList, int maxVisits);

  /**
   * Deletes vectors from the index by their IDs.
   *
   * <p>This method enqueues deletion tasks that:
   * <ul>
   *   <li>Remove the node from the graph</li>
   *   <li>Clean up back-links from neighboring nodes</li>
   *   <li>Maintain graph connectivity</li>
   * </ul>
   *
   * <p>Deletion is idempotent - deleting non-existent IDs is a no-op.
   *
   * @param ids the IDs of vectors to delete
   * @return a future that completes when deletions are enqueued
   */
  CompletableFuture<Void> delete(List<Long> ids);

  /**
   * Deletes vectors from the index by their IDs.
   *
   * <p>Convenience method for varargs input. See {@link #delete(List)}.
   *
   * @param ids the IDs of vectors to delete
   * @return a future that completes when deletions are enqueued
   */
  default CompletableFuture<Void> delete(Long... ids) {
    return delete(List.of(ids));
  }

  /**
   * Gets statistics about the index.
   *
   * <p>Statistics may include:
   * <ul>
   *   <li>Total vectors indexed</li>
   *   <li>Graph connectivity metrics</li>
   *   <li>Cache hit rates</li>
   *   <li>Worker queue depths</li>
   * </ul>
   *
   * @return a future containing index statistics
   */
  CompletableFuture<IndexStats> getStats();

  /**
   * Checks if the index is healthy and ready for operations.
   *
   * <p>Health checks include:
   * <ul>
   *   <li>Storage connectivity</li>
   *   <li>Worker availability</li>
   *   <li>Graph connectivity above threshold</li>
   * </ul>
   *
   * @return a future containing true if healthy, false otherwise
   */
  CompletableFuture<Boolean> isHealthy();

  /**
   * Forces a refresh of search entry points.
   *
   * <p>Entry points are normally refreshed periodically in the background.
   * This method triggers an immediate refresh, useful after bulk insertions
   * or deletions.
   *
   * @return a future that completes when refresh is done
   */
  CompletableFuture<Void> refreshEntryPoints();

  /**
   * Container for index statistics.
   */
  class IndexStats {
    private final long vectorCount;
    private final long graphEdgeCount;
    private final double graphConnectivity;
    private final long queueDepth;
    private final double cacheHitRate;

    public IndexStats(
        long vectorCount, long graphEdgeCount, double graphConnectivity, long queueDepth, double cacheHitRate) {
      this.vectorCount = vectorCount;
      this.graphEdgeCount = graphEdgeCount;
      this.graphConnectivity = graphConnectivity;
      this.queueDepth = queueDepth;
      this.cacheHitRate = cacheHitRate;
    }

    public long getVectorCount() {
      return vectorCount;
    }

    public long getGraphEdgeCount() {
      return graphEdgeCount;
    }

    public double getGraphConnectivity() {
      return graphConnectivity;
    }

    public long getQueueDepth() {
      return queueDepth;
    }

    public double getCacheHitRate() {
      return cacheHitRate;
    }
  }
}
