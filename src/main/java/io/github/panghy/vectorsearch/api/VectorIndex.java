package io.github.panghy.vectorsearch.api;

import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbVectorIndex;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Primary entry point to the vector search index backed by FoundationDB.
 *
 * <p>This interface exposes an asynchronous API for lifecycle management, ingestion, query, and
 * maintenance. Implementations maintain a segmented index with strict-cap rotation and background
 * sealing:
 *
 * <ul>
 *   <li>ACTIVE → accepts writes until {@code maxSegmentSize} is reached</li>
 *   <li>PENDING → write-protected; background builders produce PQ codebooks/codes and a graph</li>
 *   <li>SEALED → immutable and optimized for search (PQ/graph traversal + exact re-rank)</li>
 * </ul>
 *
 * <p>Concurrency and performance:
 * <ul>
 *   <li>All methods return {@link CompletableFuture}; storage paths are non-blocking and batch I/O
 *       within FDB transaction limits.</li>
 *   <li>Queries scan multiple segments; SEALED use PQ/graph, ACTIVE/PENDING use brute-force and are
 *       re-ranked exactly. Returned {@link SearchResult#score()} is higher-is-better. For L2 it is
 *       {@code -distance}; {@link SearchResult#distance()} carries the positive distance.</li>
 *   <li>Implementations may optionally prefetch PQ codebooks for SEALED segments to warm caches.</li>
 * </ul>
 */
public interface VectorIndex extends AutoCloseable {
  /**
   * Creates or opens an index under the provided DirectoryLayer root.
   *
   * <p>Behavior and invariants:
   * <ul>
   *   <li>Creates index metadata and the initial ACTIVE segment if missing; otherwise validates the
   *       existing metadata matches the provided {@link VectorIndexConfig} (dimension, metric,
   *       PQ/graph parameters, oversample, maxSegmentSize).</li>
   *   <li>Initializes internal queues and a compact segment registry (segmentsIndex) used for
   *       scalable listing; no legacy per-meta scan is used.</li>
   *   <li>May auto-start local workers based on config:
   *       {@code localWorkerThreads} for segment build, and
   *       {@code localMaintenanceWorkerThreads} for maintenance.</li>
   * </ul>
   *
   * <p>Call {@link #close()} to stop any auto-started workers and release resources.
   */
  static CompletableFuture<VectorIndex> createOrOpen(VectorIndexConfig config) {
    return FdbVectorIndex.createOrOpen(config).thenApply(ix -> (VectorIndex) ix);
  }

  /**
   * Inserts one vector into the current ACTIVE segment.
   *
   * <p>Semantics:
   * <ul>
   *   <li>Returns the assigned {@code [segmentId, vectorId]}.</li>
   *   <li>If the ACTIVE segment reaches capacity, the implementation atomically rotates:
   *       current → PENDING (sealed later by a builder) and opens the next ACTIVE; a build task is
   *       enqueued for the PENDING segment.</li>
   *   <li>{@code embedding.length} must equal {@link VectorIndexConfig#getDimension()}.</li>
   *   <li>{@code payload} is optional and stored verbatim with the vector.</li>
   * </ul>
   */
  CompletableFuture<int[]> add(float[] embedding, byte[] payload);

  /**
   * Inserts many vectors efficiently.
   *
   * <p>Semantics and guarantees:
   * <ul>
   *   <li>Preserves input order in the returned IDs list.</li>
   *   <li>Batches writes per transaction and across rotations to minimize contention.</li>
   *   <li>Enqueues exactly one build task for each segment that transitions ACTIVE→PENDING.</li>
   * </ul>
   */
  CompletableFuture<List<int[]>> addAll(float[][] embeddings, byte[][] payloads);

  /**
   * kNN query with default traversal parameters.
   *
   * <p>SEALED segments use PQ/graph traversal followed by exact re-rank; ACTIVE/PENDING segments are
   * scanned with brute-force and re-ranked exactly. Results are globally merged by score and
   * truncated to {@code k}.
   */
  CompletableFuture<List<SearchResult>> query(float[] q, int k);

  /**
   * kNN query with explicit traversal parameters.
   *
   * <p>Notes:
   * <ul>
   *   <li>{@link SearchParams.Mode#BEST_FIRST} is the default. {@link SearchParams.Mode#BEAM} is
   *       supported but considered legacy; implementations may log a one-time warning.</li>
   *   <li>For COSINE, if {@code normalizeOnRead=true} the implementation normalizes at exact
   *       re-rank time; otherwise raw cosine similarity is used.</li>
   *   <li>Implementations may prefetch PQ codebooks for SEALED segments. If the configuration enables
   *       a test-only synchronous prefetch flag, query waits for prefetch to complete before search.
   *   </li>
   * </ul>
   */
  CompletableFuture<List<SearchResult>> query(float[] q, int k, SearchParams params);

  /**
   * Marks a single vector as deleted (tombstone) and updates per-segment counters.
   *
   * <p>If the segment's deleted ratio meets {@link VectorIndexConfig#getVacuumMinDeletedRatio()}, a
   * threshold-aware maintenance task may be enqueued (subject to
   * {@link VectorIndexConfig#getVacuumCooldown()}). Vacuum physically removes vector, PQ code, and
   * adjacency entries and decrements {@code deleted_count}.</p>
   */
  CompletableFuture<Void> delete(int segId, int vecId);

  /**
   * Batch delete convenience for many {@code [segmentId, vectorId]} pairs.
   *
   * <p>Enqueues at most one vacuum task per affected segment when the deleted ratio threshold is
   * satisfied, observing cooldown semantics.
   */
  CompletableFuture<Void> deleteAll(int[][] ids);

  /**
   * Waits until background indexing is drained.
   *
   * <p>Implementation typically polls the build queue using
   * {@code hasVisibleUnclaimedTasks OR hasClaimedTasks} with a short delay and completes when both
   * are false. This method is intended for tests and maintenance flows; production queries should
   * not rely on it.</p>
   */
  CompletableFuture<Void> awaitIndexingComplete();

  /** Approximate number of decoded PQ codebooks currently resident in cache. */
  long getCodebookCacheSize();

  /** Approximate number of adjacency lists currently resident in cache. */
  long getAdjacencyCacheSize();

  /**
   * Closes the index and stops any auto-started worker pools.
   *
   * <p>Does not drop data; it only releases resources tied to this instance.
   */
  @Override
  void close();
}
