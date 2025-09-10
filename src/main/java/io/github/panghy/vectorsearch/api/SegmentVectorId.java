package io.github.panghy.vectorsearch.api;

/**
 * Strongly-typed identifier for a vector stored in a segmented index.
 *
 * <p>Semantics:
 * - segmentId: the immutable segment identifier where the vector currently lives.
 *   Segments are append-only within their lifecycle; after compaction/merge the
 *   vector may be rewritten to a new segment with a new id (callers should treat
 *   this as a new {@code SegmentVectorId}).
 * - vectorId: the zero-based id within the segment at insert time.
 *   It is stable for the lifetime of the segment; deletions tombstone but do not renumber.
 */
public record SegmentVectorId(int segmentId, int vectorId) {
  @Override
  public String toString() {
    return "SegmentVectorId{" + "segmentId=" + segmentId + ", vectorId=" + vectorId + '}';
  }
}
