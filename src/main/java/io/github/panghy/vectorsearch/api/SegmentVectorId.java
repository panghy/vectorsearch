package io.github.panghy.vectorsearch.api;

/**
 * Utilities for packing a (segmentId, vectorId) pair into a single 64-bit identifier and
 * unpacking it. This makes it easy to persist or pass around a single ID.
 *
 * <p>Encoding:
 * - High 32 bits: segmentId (signed int)
 * - Low 32 bits:  vectorId (signed int)
 *
 * <p>Semantics:
 * - segmentId: immutable identifier of the segment where the vector currently lives. After
 *   compaction/merge, vectors may be rewritten to a new segment with a new packed id.
 * - vectorId: zero-based id within the segment at insert time. Stable for the segment's lifetime;
 *   deletions tombstone but do not renumber existing ids.
 */
public final class SegmentVectorId {
  private SegmentVectorId() {}

  public static long pack(int segmentId, int vectorId) {
    return (((long) segmentId) << 32) | (vectorId & 0xffffffffL);
  }

  public static int segmentId(long packed) {
    return (int) (packed >>> 32);
  }

  public static int vectorId(long packed) {
    return (int) packed;
  }
}
