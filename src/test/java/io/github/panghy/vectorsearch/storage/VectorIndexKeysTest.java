package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import java.util.Arrays;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorIndexKeysTest {

  private Database db;
  private DirectorySubspace collectionSubspace;
  private VectorIndexKeys keys;
  private String testCollectionName;

  @BeforeEach
  void setUp() {
    // Create FDB connection for testing
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    // Create a unique test collection name
    testCollectionName = "test_collection_" + UUID.randomUUID().toString().substring(0, 8);

    // Create the directory structure
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      collectionSubspace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "vectorsearch", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(collectionSubspace, testCollectionName);
  }

  @AfterEach
  void tearDown() {
    // Clean up test directory
    if (db != null && collectionSubspace != null) {
      db.run(tr -> {
        DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
        directoryLayer
            .removeIfExists(tr, Arrays.asList("test", "vectorsearch", testCollectionName))
            .join();
        return null;
      });
      db.close();
    }
  }

  @Test
  void testConfigKey() {
    byte[] key = keys.configKey();
    assertThat(key).isNotNull();

    // Verify key structure
    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("meta");
    assertThat(decoded.getString(1)).isEqualTo("config");
  }

  @Test
  void testActiveCodebookVersionKey() {
    byte[] key = keys.activeCodebookVersionKey();
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("meta");
    assertThat(decoded.getString(1)).isEqualTo("cbv_active");
  }

  @Test
  void testStatsKey() {
    byte[] key = keys.statsKey();
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("meta");
    assertThat(decoded.getString(1)).isEqualTo("stats");
  }

  @Test
  void testEntryListKey() {
    byte[] key = keys.entryListKey();
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("entry");
  }

  @Test
  void testCodebookKey() {
    byte[] key = keys.codebookKey(1, 5);
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("codebook");
    assertThat(decoded.getLong(2)).isEqualTo(1);
    assertThat(decoded.getLong(3)).isEqualTo(5);
  }

  @Test
  void testCodebookPrefixForVersion() {
    byte[] prefix = keys.codebookPrefixForVersion(2);
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("codebook");
    assertThat(decoded.getLong(2)).isEqualTo(2);
  }

  @Test
  void testAllCodebooksPrefix() {
    byte[] prefix = keys.allCodebooksPrefix();
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("codebook");
  }

  @Test
  void testPqBlockKey() {
    byte[] key = keys.pqBlockKey(3, 100L);
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("block");
    assertThat(decoded.getLong(2)).isEqualTo(3);
    assertThat(decoded.getLong(3)).isEqualTo(100);
  }

  @Test
  void testPqBlockKeyStriped() {
    byte[] key = keys.pqBlockKeyStriped(3, 100L, 7);
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("block");
    assertThat(decoded.getLong(2)).isEqualTo(3);
    assertThat(decoded.getLong(3)).isEqualTo(100);
    assertThat(decoded.getLong(4)).isEqualTo(7);
  }

  @Test
  void testPqBlockPrefixForVersion() {
    byte[] prefix = keys.pqBlockPrefixForVersion(4);
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("block");
    assertThat(decoded.getLong(2)).isEqualTo(4);
  }

  @Test
  void testAllPqBlocksPrefix() {
    byte[] prefix = keys.allPqBlocksPrefix();
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("pq");
    assertThat(decoded.getString(1)).isEqualTo("block");
  }

  @Test
  void testNodeAdjacencyKey() {
    byte[] key = keys.nodeAdjacencyKey(12345L);
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("graph");
    assertThat(decoded.getString(1)).isEqualTo("node");
    assertThat(decoded.getLong(2)).isEqualTo(12345);
  }

  @Test
  void testAllNodesPrefix() {
    byte[] prefix = keys.allNodesPrefix();
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("graph");
    assertThat(decoded.getString(1)).isEqualTo("node");
  }

  @Test
  void testGraphConnectivityKey() {
    byte[] key = keys.graphConnectivityKey();
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("graph");
    assertThat(decoded.getString(1)).isEqualTo("meta");
    assertThat(decoded.getString(2)).isEqualTo("connectivity");
  }

  @Test
  void testVectorSketchKey() {
    byte[] key = keys.vectorSketchKey(9999L);
    assertThat(key).isNotNull();

    Tuple decoded = collectionSubspace.unpack(key);
    assertThat(decoded.getString(0)).isEqualTo("sketch");
    assertThat(decoded.getLong(1)).isEqualTo(9999);
  }

  @Test
  void testAllSketchesPrefix() {
    byte[] prefix = keys.allSketchesPrefix();
    assertThat(prefix).isNotNull();

    Tuple decoded = collectionSubspace.unpack(prefix);
    assertThat(decoded.getString(0)).isEqualTo("sketch");
  }

  @Test
  void testPrefixRange() {
    byte[] prefix = keys.allNodesPrefix();
    byte[][] range = VectorIndexKeys.prefixRange(prefix);

    assertThat(range.length).isEqualTo(2);
    assertThat(range[0]).isEqualTo(prefix);
    assertThat(range[1]).isNotEqualTo(prefix);

    // End key should be lexicographically greater
    assertThat(Arrays.compare(range[1], range[0])).isGreaterThan(0);
  }

  @Test
  void testBlockNumber() {
    assertThat(VectorIndexKeys.blockNumber(0, 256)).isEqualTo(0);
    assertThat(VectorIndexKeys.blockNumber(255, 256)).isEqualTo(0);
    assertThat(VectorIndexKeys.blockNumber(256, 256)).isEqualTo(1);
    assertThat(VectorIndexKeys.blockNumber(512, 256)).isEqualTo(2);
    assertThat(VectorIndexKeys.blockNumber(1000, 256)).isEqualTo(3);
  }

  @Test
  void testBlockOffset() {
    assertThat(VectorIndexKeys.blockOffset(0, 256)).isEqualTo(0);
    assertThat(VectorIndexKeys.blockOffset(255, 256)).isEqualTo(255);
    assertThat(VectorIndexKeys.blockOffset(256, 256)).isEqualTo(0);
    assertThat(VectorIndexKeys.blockOffset(257, 256)).isEqualTo(1);
    assertThat(VectorIndexKeys.blockOffset(1000, 256)).isEqualTo(232);
  }

  @Test
  void testStripeIndex() {
    // Test deterministic hashing
    int stripes = 8;
    long nodeId = 12345L;

    int stripe1 = VectorIndexKeys.stripeIndex(nodeId, stripes);
    int stripe2 = VectorIndexKeys.stripeIndex(nodeId, stripes);

    assertThat(stripe1).isEqualTo(stripe2);
    assertThat(stripe1).isBetween(0, stripes - 1);

    // Different nodes should generally get different stripes
    int stripe3 = VectorIndexKeys.stripeIndex(12346L, stripes);
    // Not guaranteed to be different, but likely
  }

  @Test
  void testGetters() {
    assertThat(keys.getCollectionSubspace()).isEqualTo(collectionSubspace);
  }

  @Test
  void testKeyUniqueness() {
    // Ensure different keys don't collide
    byte[] configKey = keys.configKey();
    byte[] statsKey = keys.statsKey();
    byte[] entryKey = keys.entryListKey();
    byte[] nodeKey = keys.nodeAdjacencyKey(1L);
    byte[] sketchKey = keys.vectorSketchKey(1L);

    assertThat(configKey).isNotEqualTo(statsKey);
    assertThat(configKey).isNotEqualTo(entryKey);
    assertThat(configKey).isNotEqualTo(nodeKey);
    assertThat(configKey).isNotEqualTo(sketchKey);
    assertThat(nodeKey).isNotEqualTo(sketchKey);
  }

  @Test
  void testVersionedKeyOrdering() {
    // Keys for the same type but different versions should be ordered
    byte[] v1 = keys.codebookKey(1, 0);
    byte[] v2 = keys.codebookKey(2, 0);
    byte[] v10 = keys.codebookKey(10, 0);

    assertThat(Arrays.compare(v2, v1)).isGreaterThan(0);
    assertThat(Arrays.compare(v10, v2)).isGreaterThan(0);
  }

  @Test
  void testSubspaceKeyOrdering() {
    // Keys for the same version but different subspaces should be ordered
    byte[] s0 = keys.codebookKey(1, 0);
    byte[] s1 = keys.codebookKey(1, 1);
    byte[] s10 = keys.codebookKey(1, 10);

    assertThat(Arrays.compare(s1, s0)).isGreaterThan(0);
    assertThat(Arrays.compare(s10, s1)).isGreaterThan(0);
  }
}
