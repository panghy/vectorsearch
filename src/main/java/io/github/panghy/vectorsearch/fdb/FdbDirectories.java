package io.github.panghy.vectorsearch.fdb;

import com.apple.foundationdb.ReadTransactionContext;
import com.apple.foundationdb.TransactionContext;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Directory layout helper for an index, backed by the FoundationDB DirectoryLayer.
 *
 * <p>Why a dedicated helper?</p>
 * <ul>
 *   <li><b>Shortest possible prefixes:</b> Using {@link DirectorySubspace}s created via
 *       {@code createOrOpen(...)} ensures stable and compact tuple prefixes rather than manually
 *       packing long, ad‑hoc path components.</li>
 *   <li><b>Moveability:</b> Because callers acquire subspaces from DirectoryLayer, it is possible to
 *       move or rename parts of the tree later without rewriting every call site that previously
 *       packed raw tuples.</li>
 *   <li><b>Performance:</b> Single‑value keys that live under the index root (e.g. {@code meta},
 *       {@code currentSegment}, {@code maxSegmentId}) are memoized to avoid repeated tuple packing.
 *       Per‑segment subspaces are memoized per process to avoid redundant DirectoryLayer open calls
 *       for the same segment id.</li>
 *   <li><b>Thread‑safety:</b> This class is immutable after construction; per‑segment lookups are
 *       memoized through a {@code ConcurrentHashMap}. Returned {@link SegmentKeys} instances are
 *       safe to share across threads.</li>
 * </ul>
 *
 * <p>Important: All methods that open subspaces require a live {@link TransactionContext}. Do not
 * cache or share raw {@link com.apple.foundationdb.Transaction} objects across async boundaries; use
 * the context provided by the caller when composing futures.</p>
 */
public final class FdbDirectories {

  private FdbDirectories() {}

  /**
   * Aggregates top‑level subspaces for an index and exposes memoized single‑value keys and
   * a cached async factory for per‑segment key helpers.
   */
  public static final class IndexDirectories {
    private final DirectorySubspace indexDir;
    private final DirectorySubspace segmentsDir;
    private final DirectorySubspace tasksDir;
    private final DirectorySubspace segmentsIndexDir;

    // Memoized single-value keys
    private final byte[] metaKeyMemo;
    private final byte[] currentSegKeyMemo;
    private final byte[] maxSegKeyMemo;

    // Per-segment async memoization
    private final Map<Integer, CompletableFuture<SegmentKeys>> segCache = new ConcurrentHashMap<>();

    /**
     * Constructs an IndexDirectories view with fully‑opened subspaces.
     *
     * @param indexDir         root directory for the index (e.g. /indexes/<name>)
     * @param segmentsDir      container for all segments (e.g. /indexes/<name>/segments)
     * @param tasksDir         container for background task queues (e.g. /indexes/<name>/tasks)
     * @param segmentsIndexDir registry that holds one key per existing segment id
     */
    public IndexDirectories(
        DirectorySubspace indexDir,
        DirectorySubspace segmentsDir,
        DirectorySubspace tasksDir,
        DirectorySubspace segmentsIndexDir) {
      this.indexDir = indexDir;
      this.segmentsDir = segmentsDir;
      this.tasksDir = tasksDir;
      this.segmentsIndexDir = segmentsIndexDir;
      this.metaKeyMemo = indexDir.pack(Tuple.from(FdbPathUtil.META));
      this.currentSegKeyMemo = indexDir.pack(Tuple.from(FdbPathUtil.CURRENT_SEGMENT));
      this.maxSegKeyMemo = indexDir.pack(Tuple.from(FdbPathUtil.MAX_SEGMENT));
    }

    // Accessors (mirroring the previous record API for minimal churn)
    public DirectorySubspace indexDir() {
      return indexDir;
    }

    public DirectorySubspace segmentsDir() {
      return segmentsDir;
    }

    public DirectorySubspace tasksDir() {
      return tasksDir;
    }

    public DirectorySubspace segmentsIndexDir() {
      return segmentsIndexDir;
    }

    /** Key for index-level metadata (single value). */
    public byte[] metaKey() {
      return metaKeyMemo;
    }

    /** Key for the current ACTIVE segment pointer (single value). */
    public byte[] currentSegmentKey() {
      return currentSegKeyMemo;
    }

    /** Key for the maximum known segment id (single value, monotonic). */
    public byte[] maxSegmentKey() {
      return maxSegKeyMemo;
    }

    /** Registry key to mark a segment as existent (under segmentsIndexDir). */
    public byte[] segmentsIndexKey(int segId) {
      return segmentsIndexDir.pack(Tuple.from(segId));
    }

    /** Subspace for the segments index prefix. */
    public DirectorySubspace segmentsIndexSubspace() {
      return segmentsIndexDir;
    }

    /**
     * Packed key for a segment's meta record, opening only the segment directory (no children).
     */
    public CompletableFuture<byte[]> segmentMetaKey(ReadTransactionContext ctx, int segId) {
      String seg = Integer.toString(segId);
      return segmentsDir.open(ctx, List.of(seg)).thenApply(d -> d.pack(Tuple.from(FdbPathUtil.META)));
    }

    /**
     * Opens (or creates) the directory structure for a specific segment and returns key helpers.
     *
     * <p>The result is memoized per‑process per segment id. Multiple concurrent callers for the
     * same segId share the same future. This keeps DirectoryLayer traffic low while still allowing
     * parallelism.</p>
     *
     * @param ctx   transaction context used to create or open subspaces
     * @param segId numeric segment identifier
     * @return a future that completes with a {@link SegmentKeys} for {@code segId}
     */
    public CompletableFuture<SegmentKeys> segmentKeys(ReadTransactionContext ctx, int segId) {
      return segCache.computeIfAbsent(segId, id -> {
        String seg = Integer.toString(id);
        CompletableFuture<DirectorySubspace> segDirF = segmentsDir.open(ctx, List.of(seg));
        CompletableFuture<DirectorySubspace> vectorsF =
            segmentsDir.open(ctx, List.of(seg, FdbPathUtil.VECTORS));
        CompletableFuture<DirectorySubspace> pqF = segmentsDir.open(ctx, List.of(seg, FdbPathUtil.PQ));
        CompletableFuture<DirectorySubspace> pqCodesF =
            segmentsDir.open(ctx, List.of(seg, FdbPathUtil.PQ, FdbPathUtil.CODES));
        CompletableFuture<DirectorySubspace> graphF = segmentsDir.open(ctx, List.of(seg, FdbPathUtil.GRAPH));
        return CompletableFuture.allOf(segDirF, vectorsF, pqF, pqCodesF, graphF)
            .thenApply(v -> new SegmentKeys(
                id, segDirF.join(), vectorsF.join(), pqF.join(), pqCodesF.join(), graphF.join()));
      });
    }

    /** Creates or opens the directory structure for a segment. */
    public CompletableFuture<SegmentKeys> segmentKeys(TransactionContext ctx, int segId) {
      String seg = Integer.toString(segId);
      // Create the segment directory first, then create/open children relative to it to avoid
      // DirectoryLayer races where a child is created concurrently before the parent exists.
      return segmentsDir.createOrOpen(ctx, List.of(seg)).thenCompose(segDir -> {
        CompletableFuture<DirectorySubspace> vectorsF = segDir.createOrOpen(ctx, List.of(FdbPathUtil.VECTORS));
        CompletableFuture<DirectorySubspace> pqF = segDir.createOrOpen(ctx, List.of(FdbPathUtil.PQ));
        CompletableFuture<DirectorySubspace> pqCodesF =
            pqF.thenCompose(pq -> pq.createOrOpen(ctx, List.of(FdbPathUtil.CODES)));
        CompletableFuture<DirectorySubspace> graphF = segDir.createOrOpen(ctx, List.of(FdbPathUtil.GRAPH));
        return CompletableFuture.allOf(vectorsF, pqF, pqCodesF, graphF)
            .thenApply(v -> new SegmentKeys(
                segId, segDir, vectorsF.join(), pqF.join(), pqCodesF.join(), graphF.join()));
      });
    }
  }

  /**
   * Key helper for a specific segment built on top of DirectoryLayer subspaces.
   *
   * <p>This helper exposes strongly‑typed key builders for common segment entities (meta record,
   * vectors, PQ codebook, PQ codes, adjacency). It memoizes static keys that are accessed
   * frequently (e.g., meta and codebook) to reduce tuple packing overhead.</p>
   */
  public static final class SegmentKeys {
    private final int segId;
    private final DirectorySubspace segmentDir;
    private final DirectorySubspace vectorsDir;
    private final DirectorySubspace pqDir;
    private final DirectorySubspace pqCodesDir;
    private final DirectorySubspace graphDir;

    // Memoized static keys under the segment namespace (computed eagerly)
    private final byte[] metaKeyMemo;
    private final byte[] codebookKeyMemo;

    public SegmentKeys(
        int segId,
        DirectorySubspace segmentDir,
        DirectorySubspace vectorsDir,
        DirectorySubspace pqDir,
        DirectorySubspace pqCodesDir,
        DirectorySubspace graphDir) {
      this.segId = segId;
      this.segmentDir = segmentDir;
      this.vectorsDir = vectorsDir;
      this.pqDir = pqDir;
      this.pqCodesDir = pqCodesDir;
      this.graphDir = graphDir;
      this.metaKeyMemo = this.segmentDir.pack(Tuple.from(FdbPathUtil.META));
      this.codebookKeyMemo = this.pqDir.pack(Tuple.from(FdbPathUtil.CODEBOOK));
    }

    // Accessors (mirroring the previous record API for minimal churn)
    public int segId() {
      return segId;
    }

    public DirectorySubspace segmentDir() {
      return segmentDir;
    }

    public DirectorySubspace vectorsDir() {
      return vectorsDir;
    }

    public DirectorySubspace pqDir() {
      return pqDir;
    }

    public DirectorySubspace pqCodesDir() {
      return pqCodesDir;
    }

    public DirectorySubspace graphDir() {
      return graphDir;
    }

    public byte[] metaKey() {
      return metaKeyMemo;
    }

    /** Returns the fully‑qualified key for {@code vecId} under the vectors subspace. */
    public byte[] vectorKey(int vecId) {
      return vectorsDir.pack(Tuple.from(vecId));
    }

    public byte[] pqCodebookKey() {
      return codebookKeyMemo;
    }

    /** Returns the fully‑qualified key for {@code vecId} under the PQ codes subspace. */
    public byte[] pqCodeKey(int vecId) {
      return pqCodesDir.pack(Tuple.from(vecId));
    }

    /** Returns the fully‑qualified key for {@code vecId} under the adjacency subspace. */
    public byte[] graphKey(int vecId) {
      return graphDir.pack(Tuple.from(vecId));
    }
  }

  /**
   * Opens/creates child directories (segments/, tasks/, segmentsIndex/) under the provided index
   * root directory.
   */
  public static CompletableFuture<IndexDirectories> openIndex(DirectorySubspace indexDir, TransactionContext ctx) {
    CompletableFuture<DirectorySubspace> segmentsDirF = indexDir.createOrOpen(ctx, List.of(FdbPathUtil.SEGMENTS));
    CompletableFuture<DirectorySubspace> tasksDirF = indexDir.createOrOpen(ctx, List.of(FdbPathUtil.TASKS));
    CompletableFuture<DirectorySubspace> segmentsIndexF =
        indexDir.createOrOpen(ctx, List.of(FdbPathUtil.SEGMENTS_INDEX));
    return CompletableFuture.allOf(segmentsDirF, tasksDirF, segmentsIndexF)
        .thenApply(v ->
            new IndexDirectories(indexDir, segmentsDirF.join(), tasksDirF.join(), segmentsIndexF.join()));
  }
}
