package io.github.panghy.vectorsearch.api;

import com.apple.foundationdb.Transaction;
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
  /**
   * Opens an existing index or creates a new one using the provided configuration.
   *
   * @param config index configuration and resources (FDB {@code Database}, root directory,
   *               dimension, metric, PQ/graph parameters, etc.)
   * @return a future that resolves to a ready {@link VectorIndex}
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
  /**
   * Inserts a single vector into the current ACTIVE segment.
   *
   * @param embedding vector values; length must equal {@link VectorIndexConfig#getDimension()}.
   *                  Implementations may throw {@link IllegalArgumentException} if the length
   *                  does not match.
   * @param payload   optional payload bytes to store verbatim with the vector; may be {@code null}.
   * @return a future with the assigned identifier {@code [segmentId, vectorId]}.
   */
  CompletableFuture<int[]> add(float[] embedding, byte[] payload);

  /**
   * Inserts one vector using the provided FoundationDB {@link Transaction}.
   *
   * <p>Nuance: this overload attempts to perform the entire write in the caller-supplied transaction
   * without internal chunking. If the write set would exceed FoundationDB limits (≈10MB or 5s),
   * the caller is expected to control batch sizes or split work across transactions. If rotation is
   * necessary, the implementation performs the ACTIVE→PENDING transition and opens the next ACTIVE
   * within the same transaction before continuing writes.
   */
  /**
   * Inserts one vector using a caller-supplied FoundationDB {@link Transaction}.
   *
   * <p>Single-transaction nuance: this overload attempts the entire write in {@code tx}
   * without internal chunking. If the write set would exceed FDB limits (~10MB of mutations or
   * ~5 seconds), callers must dial back batch sizes or split the work themselves.</p>
   *
   * <p>If rotation is needed, the ACTIVE→PENDING transition and creation of the next ACTIVE are
   * performed inside {@code tx} before continuing writes.</p>
   *
   * @param tx        existing FDB transaction to use
   * @param embedding vector values; length must equal {@link VectorIndexConfig#getDimension()}.
   * @param payload   optional payload bytes, may be {@code null}
   * @return a future with the assigned identifier {@code [segmentId, vectorId]}.
   */
  CompletableFuture<int[]> add(Transaction tx, float[] embedding, byte[] payload);

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
  /**
   * Inserts multiple vectors efficiently across one or more transactions.
   *
   * <p>Behavior when {@code payloads} length differs from {@code embeddings.length}:</p>
   * <ul>
   *   <li>If {@code payloads == null} or {@code payloads.length < embeddings.length}, missing
   *       entries are treated as {@code null} payloads.</li>
   *   <li>If {@code payloads.length > embeddings.length}, extra payloads are ignored.</li>
   * </ul>
   *
   * @param embeddings array of vectors; each vector's length must equal the configured dimension
   * @param payloads   optional per-vector payloads; may be {@code null} or shorter than embeddings
   * @return a future with assigned IDs per vector (same order as input)
   */
  CompletableFuture<List<int[]>> addAll(float[][] embeddings, byte[][] payloads);

  /**
   * Inserts many vectors using a single caller-managed FDB {@link Transaction}.
   *
   * <p>Unlike the default overload which may span multiple transactions to respect size/time
   * limits, this method writes all vectors in the provided transaction and may rotate segments
   * (mark PENDING and open next ACTIVE) inline as capacity is reached. Callers should dial back the
   * batch size to remain within FDB limits.
   */
  /**
   * Inserts multiple vectors using a single caller-managed {@link Transaction}.
   *
   * <p>Unlike {@link #addAll(float[][], byte[][])} which may span multiple transactions to respect
   * size/time heuristics, this method performs all mutations inside {@code tx}. If the write set is
   * too large, callers should reduce batch sizes accordingly.</p>
   *
   * <p>Rotation (ACTIVE→PENDING and opening next ACTIVE) occurs inside {@code tx} as needed.</p>
   *
   * @param tx         existing FDB transaction to use
   * @param embeddings vectors to insert; each vector length must equal the configured dimension
   * @param payloads   optional per-vector payloads (see {@link #addAll(float[][], byte[][])} rules)
   * @return a future with assigned IDs per vector (same order as input)
   */
  CompletableFuture<List<int[]>> addAll(Transaction tx, float[][] embeddings, byte[][] payloads);

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
  /**
   * Marks one vector as deleted (tombstone) and updates per-segment counters.
   *
   * @param segId segment identifier
   * @param vecId vector identifier within the segment
   * @return a future that completes when the mutation is persisted
   */
  CompletableFuture<Void> delete(int segId, int vecId);

  /**
   * Marks one vector deleted within the provided caller-managed {@link Transaction}.
   *
   * <p>Behavior matches {@link #delete(int, int)} but performs all reads/writes in the supplied
   * transaction (no internal chunking). Threshold-aware maintenance enqueue (vacuum) will also use
   * the same transaction if configured.
   */
  /**
   * Single-transaction delete using a caller-provided {@link Transaction}.
   *
   * @param tx    existing FDB transaction to use
   * @param segId segment identifier
   * @param vecId vector identifier within the segment
   * @return a future that completes when the mutation is persisted
   */
  CompletableFuture<Void> delete(Transaction tx, int segId, int vecId);

  /**
   * Batch delete convenience for many {@code [segmentId, vectorId]} pairs.
   *
   * <p>Enqueues at most one vacuum task per affected segment when the deleted ratio threshold is
   * satisfied, observing cooldown semantics.
   */
  /**
   * Batch delete convenience for many {@code [segmentId, vectorId]} pairs.
   *
   * @param ids array of {@code [segId, vecId]} pairs; {@code null} or empty is a no-op
   * @return a future that completes when all mutations are persisted
   */
  CompletableFuture<Void> deleteAll(int[][] ids);

  /**
   * Batch delete using a single caller-managed {@link Transaction}. See {@link #delete(Transaction,
   * int, int)} for details and caveats.
   */
  /**
   * Batch delete within a single caller-provided {@link Transaction}.
   *
   * @param tx  existing FDB transaction to use
   * @param ids array of {@code [segId, vecId]} pairs; {@code null} or empty is a no-op
   * @return a future that completes when mutations are persisted
   */
  CompletableFuture<Void> deleteAll(Transaction tx, int[][] ids);

  /**
   * Waits until background indexing is drained.
   *
   * <p>Implementation typically polls the build queue using
   * {@code hasVisibleUnclaimedTasks OR hasClaimedTasks} with a short delay and completes when both
   * are false. This method is intended for tests and maintenance flows; production queries should
   * not rely on it.</p>
   */
  /**
   * Waits until the build queue is empty (no visible or claimed tasks remain).
   *
   * <p>Intended primarily for tests and maintenance flows; production traffic should not block on
   * index background work.</p>
   *
   * @return a future that completes when the build queue drains
   */
  CompletableFuture<Void> awaitIndexingComplete();

  /** Approximate number of decoded PQ codebooks currently resident in cache. */
  /**
   * @return approximate number of decoded PQ codebooks resident in the cache
   */
  long getCodebookCacheSize();

  /** Approximate number of adjacency lists currently resident in cache. */
  /**
   * @return approximate number of adjacency lists resident in the cache
   */
  long getAdjacencyCacheSize();

  /**
   * Closes the index and stops any auto-started worker pools.
   *
   * <p>Does not drop data; it only releases resources tied to this instance.
   */
  @Override
  void close();
}
