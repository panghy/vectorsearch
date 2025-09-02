package io.github.panghy.vectorsearch.storage;

import static com.apple.foundationdb.tuple.ByteArrayUtil.printable;

import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import lombok.Getter;

/**
 * Key builders for vector index storage in FoundationDB.
 * All keys follow the pattern: /C/{collection}/...
 * Uses FDB tuple encoding for efficient key construction and range scans.
 */
public class VectorIndexKeys {

  // Root prefix for all collections
  private static final String COLLECTIONS_PREFIX = "C";

  // Metadata keys
  private static final String META = "meta";
  private static final String CONFIG = "config";
  private static final String CBV_ACTIVE = "cbv_active";
  private static final String STATS = "stats";

  // Entry list key
  private static final String ENTRY = "entry";

  // PQ storage keys
  private static final String PQ = "pq";
  private static final String CODEBOOK = "codebook";
  private static final String BLOCK = "block";

  // Graph storage keys
  private static final String GRAPH = "graph";
  private static final String NODE = "node";
  private static final String CONNECTIVITY = "connectivity";

  // Sketch storage keys
  private static final String SKETCH = "sketch";

  /**
   * -- GETTER --
   * Gets the collection subspace.
   */
  @Getter
  private final DirectorySubspace collectionSubspace;

  /**
   * Creates a key builder for a specific collection.
   *
   * @param collectionSubspace the FDB directory subspace for this collection
   */
  public VectorIndexKeys(DirectorySubspace collectionSubspace) {
    this.collectionSubspace = collectionSubspace;
  }

  // ==================== Metadata Keys ====================

  /**
   * Key for collection configuration.
   * Path: /C/{collection}/meta/config
   */
  public byte[] configKey() {
    return collectionSubspace.pack(Tuple.from(META, CONFIG));
  }

  /**
   * Key for active codebook version.
   * Path: /C/{collection}/meta/cbv_active
   */
  public byte[] activeCodebookVersionKey() {
    return collectionSubspace.pack(Tuple.from(META, CBV_ACTIVE));
  }

  /**
   * Key for collection statistics.
   * Path: /C/{collection}/meta/stats
   */
  public byte[] statsKey() {
    return collectionSubspace.pack(Tuple.from(META, STATS));
  }

  // ==================== Entry List Keys ====================

  /**
   * Key for search entry points.
   * Path: /C/{collection}/entry
   */
  public byte[] entryListKey() {
    return collectionSubspace.pack(Tuple.from(ENTRY));
  }

  // ==================== PQ Codebook Keys ====================

  /**
   * Key for a PQ codebook subspace.
   * Path: /C/{collection}/pq/codebook/{version}/{subspace}
   *
   * @param version       codebook version
   * @param subspaceIndex PQ subspace index [0, m-1]
   */
  public byte[] codebookKey(long version, int subspaceIndex) {
    return collectionSubspace.pack(Tuple.from(PQ, CODEBOOK, version, subspaceIndex));
  }

  /**
   * Key prefix for all codebooks of a specific version.
   * Path: /C/{collection}/pq/codebook/{version}/
   *
   * @param version codebook version
   */
  public byte[] codebookPrefixForVersion(long version) {
    return collectionSubspace.pack(Tuple.from(PQ, CODEBOOK, version));
  }

  /**
   * Key prefix for all codebooks (all versions).
   * Path: /C/{collection}/pq/codebook/
   */
  public byte[] allCodebooksPrefix() {
    return collectionSubspace.pack(Tuple.from(PQ, CODEBOOK));
  }

  // ==================== PQ Block Keys ====================

  /**
   * Key for a PQ codes block.
   * Path: /C/{collection}/pq/block/{version}/{blockNumber}
   *
   * @param version     codebook version these codes are encoded with
   * @param blockNumber block number (nodeId / codesPerBlock)
   */
  public byte[] pqBlockKey(int version, long blockNumber) {
    return collectionSubspace.pack(Tuple.from(PQ, BLOCK, version, blockNumber));
  }

  /**
   * Key for a striped PQ codes block (for hotspot mitigation).
   * Path: /C/{collection}/pq/block/{version}/{blockNumber}/{stripe}
   *
   * @param version     codebook version
   * @param blockNumber block number
   * @param stripe      stripe index for load distribution
   */
  public byte[] pqBlockKeyStriped(int version, long blockNumber, int stripe) {
    return collectionSubspace.pack(Tuple.from(PQ, BLOCK, version, blockNumber, stripe));
  }

  /**
   * Key prefix for all PQ blocks of a specific version.
   * Path: /C/{collection}/pq/block/{version}/
   *
   * @param version codebook version
   */
  public byte[] pqBlockPrefixForVersion(int version) {
    return collectionSubspace.pack(Tuple.from(PQ, BLOCK, version));
  }

  /**
   * Key prefix for all PQ blocks (all versions).
   * Path: /C/{collection}/pq/block/
   */
  public byte[] allPqBlocksPrefix() {
    return collectionSubspace.pack(Tuple.from(PQ, BLOCK));
  }

  // ==================== Graph Keys ====================

  /**
   * Key for a node's adjacency list.
   * Path: /C/{collection}/graph/node/{nodeId}
   *
   * @param nodeId the node ID
   */
  public byte[] nodeAdjacencyKey(long nodeId) {
    return collectionSubspace.pack(Tuple.from(GRAPH, NODE, nodeId));
  }

  /**
   * Key prefix for all graph nodes.
   * Path: /C/{collection}/graph/node/
   */
  public byte[] allNodesPrefix() {
    return collectionSubspace.pack(Tuple.from(GRAPH, NODE));
  }

  /**
   * Key for graph connectivity metadata.
   * Path: /C/{collection}/graph/meta/connectivity
   */
  public byte[] graphConnectivityKey() {
    return collectionSubspace.pack(Tuple.from(GRAPH, META, CONNECTIVITY));
  }

  // ==================== Sketch Keys ====================

  /**
   * Key for a vector sketch.
   * Path: /C/{collection}/sketch/{nodeId}
   *
   * @param nodeId the node ID
   */
  public byte[] vectorSketchKey(long nodeId) {
    return collectionSubspace.pack(Tuple.from(SKETCH, nodeId));
  }

  /**
   * Key prefix for all vector sketches.
   * Path: /C/{collection}/sketch/
   */
  public byte[] allSketchesPrefix() {
    return collectionSubspace.pack(Tuple.from(SKETCH));
  }

  /**
   * Range for all vector sketches.
   * Path: /C/{collection}/sketch/
   *
   * @return Range for scanning all vector sketches
   */
  public com.apple.foundationdb.Range vectorSketchRange() {
    byte[] prefix = allSketchesPrefix();
    return new com.apple.foundationdb.Range(prefix, com.apple.foundationdb.tuple.ByteArrayUtil.strinc(prefix));
  }

  // ==================== Range Keys ====================

  /**
   * Returns a key range for scanning all keys with a given prefix.
   *
   * @param prefix the key prefix
   * @return tuple of [startKey, endKey) for range scan
   */
  public static byte[][] prefixRange(byte[] prefix) {
    return new byte[][] {prefix, com.apple.foundationdb.tuple.ByteArrayUtil.strinc(prefix)};
  }

  // ==================== Utility Methods ====================

  /**
   * Calculates the block number for a given node ID.
   *
   * @param nodeId        the node ID
   * @param codesPerBlock number of codes per block
   * @return block number
   */
  public static long blockNumber(long nodeId, int codesPerBlock) {
    return nodeId / codesPerBlock;
  }

  /**
   * Calculates the offset within a block for a given node ID.
   *
   * @param nodeId        the node ID
   * @param codesPerBlock number of codes per block
   * @return offset within the block
   */
  public static int blockOffset(long nodeId, int codesPerBlock) {
    return (int) (nodeId % codesPerBlock);
  }

  /**
   * Calculates the stripe index for load distribution.
   *
   * @param nodeId     the node ID
   * @param numStripes number of stripes
   * @return stripe index
   */
  public static int stripeIndex(long nodeId, int numStripes) {
    // Use a simple hash for stripe distribution
    return (int) (Math.abs(nodeId * 2654435761L) % numStripes);
  }

  /**
   * Key for graph metadata.
   * Path: /C/{collection}/graph/meta
   */
  public byte[] graphMetaKey() {
    return collectionSubspace.pack(Tuple.from(GRAPH, META));
  }

  /**
   * Extracts node ID from an adjacency key.
   * Key format: /C/{collection}/graph/node/{nodeId}
   *
   * @param key the adjacency key
   * @return the node ID
   * @throws IllegalArgumentException if the key is not a valid adjacency key
   */
  public long extractNodeIdFromAdjacencyKey(byte[] key) {
    Tuple tuple = collectionSubspace.unpack(key);
    // Tuple should be (GRAPH, NODE, nodeId)
    if (tuple.size() >= 3 && GRAPH.equals(tuple.getString(0)) && NODE.equals(tuple.getString(1))) {
      return tuple.getLong(2);
    }
    throw new IllegalArgumentException("Invalid adjacency key: " + printable(key));
  }
}
