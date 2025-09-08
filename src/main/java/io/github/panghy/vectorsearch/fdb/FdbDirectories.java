package io.github.panghy.vectorsearch.fdb;

import com.apple.foundationdb.TransactionContext;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;

/**
 * Helpers to create/open the directory structure for an index using DirectoryLayer.
 *
 * <p>All operations are non-blocking and return fully opened {@link DirectorySubspace} instances.</p>
 */
public final class FdbDirectories {

  private FdbDirectories() {}

  /**
   * Group of subspaces for an index root and its shared containers.
   */
  public record IndexDirectories(
      DirectorySubspace indexDir, DirectorySubspace segmentsDir, DirectorySubspace tasksDir) {
    /**
     * Key for index-level metadata.
     */
    public byte[] metaKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.META));
    }

    /**
     * Key for the current ACTIVE segment pointer.
     */
    public byte[] currentSegmentKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.CURRENT_SEGMENT));
    }

    /** Key for the maximum known segment id (monotonic). */
    public byte[] maxSegmentKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.MAX_SEGMENT));
    }

    /**
     * Creates a segment key helper for the given zero-padded segment id.
     */
    public SegmentKeys segmentKeys(int segId) {
      return new SegmentKeys(segmentsDir, segId);
    }

    /** Registry key to mark a segment as existent. */
    public byte[] segmentsIndexKey(int segId) {
      return indexDir.pack(Tuple.from(FdbPathUtil.SEGMENTS_INDEX, segId));
    }

    /** Subspace for the segments index prefix. */
    public Subspace segmentsIndexSubspace() {
      return new Subspace(indexDir.pack(Tuple.from(FdbPathUtil.SEGMENTS_INDEX)));
    }
  }

  /**
   * Lightweight key helper for a segment using the segments container and a segment id string.
   * Avoids DirectoryLayer calls and packs keys using the container subspace.
   */
  public record SegmentKeys(DirectorySubspace segmentsDir, int segId) {

    public byte[] metaKey() {
      return segmentsDir.pack(Tuple.from(segId, FdbPathUtil.META));
    }

    public byte[] vectorKey(int vecId) {
      return segmentsDir.pack(Tuple.from(segId, FdbPathUtil.VECTORS, vecId));
    }

    public byte[] pqCodebookKey() {
      return segmentsDir.pack(Tuple.from(segId, FdbPathUtil.PQ, FdbPathUtil.CODEBOOK));
    }

    public byte[] pqCodeKey(int vecId) {
      return segmentsDir.pack(Tuple.from(segId, FdbPathUtil.PQ, FdbPathUtil.CODES, vecId));
    }

    public byte[] graphKey(int vecId) {
      return segmentsDir.pack(Tuple.from(segId, FdbPathUtil.GRAPH, vecId));
    }
  }

  /**
   * Opens/creates the child directories under a provided index root.
   */
  public static CompletableFuture<IndexDirectories> openIndex(DirectorySubspace indexDir, TransactionContext ctx) {
    CompletableFuture<DirectorySubspace> segmentsDirF =
        indexDir.createOrOpen(ctx, Collections.singletonList(FdbPathUtil.SEGMENTS));
    CompletableFuture<DirectorySubspace> tasksDirF =
        indexDir.createOrOpen(ctx, Collections.singletonList(FdbPathUtil.TASKS));
    return CompletableFuture.allOf(segmentsDirF, tasksDirF)
        .thenApply(v -> new IndexDirectories(indexDir, segmentsDirF.join(), tasksDirF.join()));
  }

  // Deprecated static helpers retained above were removed to reduce error-prone usage.
}
