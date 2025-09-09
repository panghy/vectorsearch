package io.github.panghy.vectorsearch.fdb;

import com.apple.foundationdb.TransactionContext;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Helpers to create/open the directory structure for an index using DirectoryLayer.
 *
 * <p>This refactor moves away from ad-hoc tuple packing for segment-relative keys and instead
 * relies on DirectoryLayer subspaces created via {@code createOrOpen}. This makes segment prefixes
 * stable, short, and movable while keeping single-key values (like {@code meta}) under the index
 * root.</p>
 */
public final class FdbDirectories {

  private FdbDirectories() {}

  /**
   * Group of subspaces for an index root and its shared containers.
   */
  public record IndexDirectories(
      DirectorySubspace indexDir,
      DirectorySubspace segmentsDir,
      DirectorySubspace tasksDir,
      DirectorySubspace segmentsIndexDir) {

    /** Key for index-level metadata (single value). */
    public byte[] metaKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.META));
    }

    /** Key for the current ACTIVE segment pointer (single value). */
    public byte[] currentSegmentKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.CURRENT_SEGMENT));
    }

    /** Key for the maximum known segment id (single value, monotonic). */
    public byte[] maxSegmentKey() {
      return indexDir.pack(Tuple.from(FdbPathUtil.MAX_SEGMENT));
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
     * Opens (or creates) the directory structure for a specific segment and returns key helpers.
     */
    public CompletableFuture<SegmentKeys> segmentKeys(TransactionContext ctx, int segId) {
      String seg = Integer.toString(segId);
      CompletableFuture<DirectorySubspace> segDirF = segmentsDir.createOrOpen(ctx, List.of(seg));
      CompletableFuture<DirectorySubspace> vectorsF =
          segmentsDir.createOrOpen(ctx, List.of(seg, FdbPathUtil.VECTORS));
      CompletableFuture<DirectorySubspace> pqF = segmentsDir.createOrOpen(ctx, List.of(seg, FdbPathUtil.PQ));
      CompletableFuture<DirectorySubspace> pqCodesF =
          segmentsDir.createOrOpen(ctx, List.of(seg, FdbPathUtil.PQ, FdbPathUtil.CODES));
      CompletableFuture<DirectorySubspace> graphF =
          segmentsDir.createOrOpen(ctx, List.of(seg, FdbPathUtil.GRAPH));
      return CompletableFuture.allOf(segDirF, vectorsF, pqF, pqCodesF, graphF)
          .thenApply(v -> new SegmentKeys(
              segId, segDirF.join(), vectorsF.join(), pqF.join(), pqCodesF.join(), graphF.join()));
    }
  }

  /**
   * Key helper backed by DirectoryLayer subspaces for a specific segment.
   */
  public record SegmentKeys(
      int segId,
      DirectorySubspace segmentDir,
      DirectorySubspace vectorsDir,
      DirectorySubspace pqDir,
      DirectorySubspace pqCodesDir,
      DirectorySubspace graphDir) {

    public byte[] metaKey() {
      return segmentDir.pack(Tuple.from(FdbPathUtil.META));
    }

    public byte[] vectorKey(int vecId) {
      return vectorsDir.pack(Tuple.from(vecId));
    }

    public byte[] pqCodebookKey() {
      return pqDir.pack(Tuple.from(FdbPathUtil.CODEBOOK));
    }

    public byte[] pqCodeKey(int vecId) {
      return pqCodesDir.pack(Tuple.from(vecId));
    }

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
