package io.github.panghy.vectorsearch.fdb;

import java.util.List;

/**
 * Utility to compute DirectoryLayer path components and well-known key labels for the index.
 *
 * <p>Methods here return path components relative to an application root Directory. Callers
 * should pass these lists to DirectoryLayer's createOrOpen to obtain DirectorySubspaces.</p>
 */
public final class FdbPathUtil {
  private FdbPathUtil() {}

  public static final String INDEXES = "indexes";
  public static final String SEGMENTS = "segments";
  public static final String VECTORS = "vectors";
  public static final String PQ = "pq";
  public static final String CODES = "codes";
  public static final String GRAPH = "graph";
  public static final String TASKS = "tasks";
  public static final String BUILD_REQUESTED = "buildRequested";
  public static final String SEGMENTS_INDEX = "segmentsIndex";

  public static final String META = "meta";
  public static final String CURRENT_SEGMENT = "currentSegment";
  public static final String MAX_SEGMENT = "maxSegmentId";
  public static final String CODEBOOK = "codebook";

  /** Returns path components to the index root directory. */
  public static List<String> indexPath(String indexName) {
    return List.of(INDEXES, indexName);
  }

  /** Returns path components to the segments container directory for an index. */
  public static List<String> segmentsPath(String indexName) {
    return List.of(INDEXES, indexName, SEGMENTS);
  }

  /** Returns path components to a specific segment directory. */
  public static List<String> segmentPath(String indexName, String segIdStr) {
    return List.of(INDEXES, indexName, SEGMENTS, segIdStr);
  }

  /** Returns path components to a segment's vectors directory. */
  public static List<String> vectorsPath(String indexName, String segIdStr) {
    return List.of(INDEXES, indexName, SEGMENTS, segIdStr, VECTORS);
  }

  /** Returns path components to a segment's PQ directory. */
  public static List<String> pqPath(String indexName, String segIdStr) {
    return List.of(INDEXES, indexName, SEGMENTS, segIdStr, PQ);
  }

  /** Returns path components to a segment's PQ codes directory. */
  public static List<String> pqCodesPath(String indexName, String segIdStr) {
    return List.of(INDEXES, indexName, SEGMENTS, segIdStr, PQ, CODES);
  }

  /** Returns path components to a segment's graph directory. */
  public static List<String> graphPath(String indexName, String segIdStr) {
    return List.of(INDEXES, indexName, SEGMENTS, segIdStr, GRAPH);
  }

  /** Returns path components to an index's background tasks directory. */
  public static List<String> tasksPath(String indexName) {
    return List.of(INDEXES, indexName, TASKS);
  }
}
