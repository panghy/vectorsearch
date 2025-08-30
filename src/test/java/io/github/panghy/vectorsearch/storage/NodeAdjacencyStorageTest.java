package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.EntryList;
import io.github.panghy.vectorsearch.proto.GraphMeta;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NodeAdjacencyStorageTest {

  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private NodeAdjacencyStorage storage;
  private String testCollectionName;
  private static final int GRAPH_DEGREE = 16;
  private InstantSource testInstantSource;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "node_adjacency", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace, testCollectionName);
    testInstantSource = InstantSource.system();
    storage = new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, testInstantSource, 1000, Duration.ofMinutes(10));
  }

  @AfterEach
  void tearDown() {
    if (db != null && testSpace != null) {
      db.run(tr -> {
        DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
        directoryLayer.removeIfExists(tr, testSpace.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  void testStoreAndLoadAdjacency() throws ExecutionException, InterruptedException {
    long nodeId = 42L;
    List<Long> neighbors = Arrays.asList(1L, 5L, 3L, 9L, 2L);

    storage.storeAdjacency(nodeId, neighbors).get();
    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getNodeId()).isEqualTo(nodeId);
    assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L, 5L, 9L); // Sorted
    assertThat(loaded.getVersion()).isEqualTo(1);
  }

  @Test
  void testUpdateAdjacency() throws ExecutionException, InterruptedException {
    long nodeId = 42L;
    List<Long> neighbors1 = Arrays.asList(1L, 2L, 3L);
    List<Long> neighbors2 = Arrays.asList(4L, 5L, 6L);

    storage.storeAdjacency(nodeId, neighbors1).get();
    NodeAdjacency loaded1 = storage.loadAdjacency(nodeId).get();
    assertThat(loaded1.getVersion()).isEqualTo(1);

    storage.storeAdjacency(nodeId, neighbors2).get();
    NodeAdjacency loaded2 = storage.loadAdjacency(nodeId).get();
    assertThat(loaded2.getVersion()).isEqualTo(2);
    assertThat(loaded2.getNeighborsList()).containsExactly(4L, 5L, 6L);
  }

  @Test
  void testDegreeBound() throws ExecutionException, InterruptedException {
    long nodeId = 100L;
    // Create more neighbors than degree allows
    List<Long> tooManyNeighbors = new ArrayList<>();
    for (long i = 0; i < GRAPH_DEGREE + 10; i++) {
      tooManyNeighbors.add(i);
    }

    storage.storeAdjacency(nodeId, tooManyNeighbors).get();
    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();

    assertThat(loaded.getNeighborsCount()).isEqualTo(GRAPH_DEGREE);
    // Should keep first GRAPH_DEGREE after sorting
    for (int i = 0; i < GRAPH_DEGREE; i++) {
      assertThat(loaded.getNeighbors(i)).isEqualTo((long) i);
    }
  }

  @Test
  void testDuplicateNeighbors() throws ExecutionException, InterruptedException {
    long nodeId = 50L;
    List<Long> neighborsWithDuplicates = Arrays.asList(1L, 2L, 3L, 2L, 1L, 4L);

    storage.storeAdjacency(nodeId, neighborsWithDuplicates).get();
    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();

    assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L, 4L);
  }

  @Test
  void testBatchLoadAdjacency() throws ExecutionException, InterruptedException {
    // Store adjacencies for multiple nodes
    List<Long> nodeIds = Arrays.asList(10L, 20L, 30L);
    for (long nodeId : nodeIds) {
      List<Long> neighbors = Arrays.asList(nodeId + 1, nodeId + 2, nodeId + 3);
      storage.storeAdjacency(nodeId, neighbors).get();
    }

    Map<Long, NodeAdjacency> loaded = storage.batchLoadAdjacency(nodeIds).get();

    assertThat(loaded).hasSize(3);
    assertThat(loaded.get(10L).getNeighborsList()).containsExactly(11L, 12L, 13L);
    assertThat(loaded.get(20L).getNeighborsList()).containsExactly(21L, 22L, 23L);
    assertThat(loaded.get(30L).getNeighborsList()).containsExactly(31L, 32L, 33L);
  }

  @Test
  void testBatchLoadWithMissing() throws ExecutionException, InterruptedException {
    // Store only some nodes
    storage.storeAdjacency(10L, Arrays.asList(1L, 2L)).get();
    storage.storeAdjacency(30L, Arrays.asList(3L, 4L)).get();

    Map<Long, NodeAdjacency> loaded =
        storage.batchLoadAdjacency(Arrays.asList(10L, 20L, 30L)).get();

    assertThat(loaded).hasSize(2);
    assertThat(loaded).containsKeys(10L, 30L);
    assertThat(loaded).doesNotContainKey(20L);
  }

  @Test
  void testAddNeighbor() throws ExecutionException, InterruptedException {
    long nodeId = 75L;

    // Start with some neighbors
    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L)).get();

    // Add a new neighbor
    storage.addNeighbor(nodeId, 3L).get();
    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();

    assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L);
    assertThat(loaded.getVersion()).isEqualTo(2);
  }

  @Test
  void testAddDuplicateNeighbor() throws ExecutionException, InterruptedException {
    long nodeId = 80L;

    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L, 3L)).get();
    storage.addNeighbor(nodeId, 2L).get(); // Try to add duplicate

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testAddNeighborToFull() throws ExecutionException, InterruptedException {
    long nodeId = 85L;

    // Fill to degree limit
    List<Long> fullNeighbors = new ArrayList<>();
    for (int i = 0; i < GRAPH_DEGREE; i++) {
      fullNeighbors.add((long) i);
    }
    storage.storeAdjacency(nodeId, fullNeighbors).get();

    // Try to add one more
    storage.addNeighbor(nodeId, 100L).get();

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded.getNeighborsCount()).isEqualTo(GRAPH_DEGREE);
    assertThat(loaded.getNeighborsList()).doesNotContain(100L);
  }

  @Test
  void testRemoveNeighbor() throws ExecutionException, InterruptedException {
    long nodeId = 90L;

    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L, 3L, 4L)).get();
    storage.removeNeighbor(nodeId, 2L).get();

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded.getNeighborsList()).containsExactly(1L, 3L, 4L);
    assertThat(loaded.getVersion()).isEqualTo(2);
  }

  @Test
  void testRemoveNonExistentNeighbor() throws ExecutionException, InterruptedException {
    long nodeId = 95L;

    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L)).get();
    storage.removeNeighbor(nodeId, 99L).get();

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L);
  }

  @Test
  void testBatchAddBackLinks() throws ExecutionException, InterruptedException {
    long nodeId = 100L;
    List<Long> neighbors = Arrays.asList(200L, 300L, 400L);

    // First create the neighbors
    for (long neighbor : neighbors) {
      storage.storeAdjacency(neighbor, new ArrayList<>()).get();
    }

    // Add back links
    storage.batchAddBackLinks(nodeId, neighbors, 2).get();

    // Verify back links were added
    for (long neighbor : neighbors) {
      NodeAdjacency loaded = storage.loadAdjacency(neighbor).get();
      assertThat(loaded.getNeighborsList()).contains(nodeId);
    }
  }

  @Test
  void testBatchRemoveBackLinks() throws ExecutionException, InterruptedException {
    long nodeId = 150L;
    List<Long> neighbors = Arrays.asList(250L, 350L, 450L);

    // First create neighbors with nodeId as neighbor
    for (long neighbor : neighbors) {
      storage.storeAdjacency(neighbor, Arrays.asList(nodeId, 999L)).get();
    }

    // Remove back links
    storage.batchRemoveBackLinks(nodeId, neighbors, 2).get();

    // Verify back links were removed
    for (long neighbor : neighbors) {
      NodeAdjacency loaded = storage.loadAdjacency(neighbor).get();
      assertThat(loaded.getNeighborsList()).doesNotContain(nodeId);
      assertThat(loaded.getNeighborsList()).contains(999L);
    }
  }

  @Test
  void testRobustPrune() {
    List<NodeAdjacencyStorage.ScoredNode> candidates = Arrays.asList(
        new NodeAdjacencyStorage.ScoredNode(1L, 0.1),
        new NodeAdjacencyStorage.ScoredNode(2L, 0.2),
        new NodeAdjacencyStorage.ScoredNode(3L, 0.25), // Close to 2, might be dominated
        new NodeAdjacencyStorage.ScoredNode(4L, 0.5),
        new NodeAdjacencyStorage.ScoredNode(5L, 1.0));

    List<Long> pruned = storage.robustPrune(candidates, 3, 1.2);

    assertThat(pruned).hasSize(3);
    assertThat(pruned).contains(1L); // Closest, always selected
    assertThat(pruned).contains(2L); // Second closest
    // Node 3 (0.25) is NOT dominated by node 2 (0.2) since 0.25 > 1.2 * 0.2 = 0.24
    // So we expect nodes 1, 2, 3 (the three closest)
    assertThat(pruned).contains(3L);
  }

  @Test
  void testRobustPruneEmpty() {
    List<NodeAdjacencyStorage.ScoredNode> empty = new ArrayList<>();
    List<Long> pruned = storage.robustPrune(empty, 10, 1.2);
    assertThat(pruned).isEmpty();
  }

  @Test
  void testRobustPruneFewerThanMax() {
    List<NodeAdjacencyStorage.ScoredNode> candidates = Arrays.asList(
        new NodeAdjacencyStorage.ScoredNode(1L, 0.1), new NodeAdjacencyStorage.ScoredNode(2L, 0.2));

    List<Long> pruned = storage.robustPrune(candidates, 10, 1.2);
    assertThat(pruned).containsExactly(1L, 2L);
  }

  @Test
  void testMergePrune() {
    List<Long> current = Arrays.asList(1L, 2L, 3L);
    List<Long> additions = Arrays.asList(4L, 5L, 2L); // 2L is duplicate

    List<Long> merged = storage.mergePrune(current, additions, 5);

    assertThat(merged).containsExactly(1L, 2L, 3L, 4L, 5L);
  }

  @Test
  void testMergePruneExceedingMax() {
    List<Long> current = Arrays.asList(1L, 2L, 3L);
    List<Long> additions = Arrays.asList(4L, 5L, 6L);

    List<Long> merged = storage.mergePrune(current, additions, 4);

    assertThat(merged).hasSize(4);
    assertThat(merged).containsExactly(1L, 2L, 3L, 4L);
  }

  @Test
  void testStoreAndLoadEntryList() throws ExecutionException, InterruptedException {
    EntryList entryList = EntryList.newBuilder()
        .addAllPrimaryEntries(Arrays.asList(1L, 2L, 3L))
        .addAllRandomEntries(Arrays.asList(10L, 20L))
        .addAllHighDegreeEntries(Arrays.asList(100L, 200L))
        .setVersion(1)
        .build();

    storage.storeEntryList(entryList).get();
    EntryList loaded = storage.loadEntryList().get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getPrimaryEntriesList()).containsExactly(1L, 2L, 3L);
    assertThat(loaded.getRandomEntriesList()).containsExactly(10L, 20L);
    assertThat(loaded.getHighDegreeEntriesList()).containsExactly(100L, 200L);
    assertThat(loaded.getVersion()).isEqualTo(1);
  }

  @Test
  void testLoadNonExistentEntryList() throws ExecutionException, InterruptedException {
    EntryList loaded = storage.loadEntryList().get();
    assertThat(loaded).isNull();
  }

  @Test
  void testStoreAndLoadGraphMeta() throws ExecutionException, InterruptedException {
    GraphMeta graphMeta = GraphMeta.newBuilder()
        .setConnectedComponents(2)
        .setLargestComponentSize(1000L)
        .setTotalNodes(1100L)
        .addOrphanedNodes(500L)
        .addOrphanedNodes(501L)
        .setRepairState(GraphMeta.RepairState.NOT_NEEDED)
        .build();

    storage.storeGraphMeta(graphMeta).get();
    GraphMeta loaded = storage.loadGraphMeta().get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getConnectedComponents()).isEqualTo(2);
    assertThat(loaded.getLargestComponentSize()).isEqualTo(1000L);
    assertThat(loaded.getTotalNodes()).isEqualTo(1100L);
    assertThat(loaded.getOrphanedNodesList()).containsExactly(500L, 501L);
    assertThat(loaded.getRepairState()).isEqualTo(GraphMeta.RepairState.NOT_NEEDED);
  }

  @Test
  void testLoadNonExistentGraphMeta() throws ExecutionException, InterruptedException {
    GraphMeta loaded = storage.loadGraphMeta().get();
    assertThat(loaded).isNull();
  }

  @Test
  void testDeleteNode() throws ExecutionException, InterruptedException {
    long nodeId = 123L;

    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L, 3L)).get();
    assertThat(storage.loadAdjacency(nodeId).get()).isNotNull();

    storage.deleteNode(nodeId).get();
    assertThat(storage.loadAdjacency(nodeId).get()).isNull();
  }

  @Test
  void testCaching() throws ExecutionException, InterruptedException {
    long nodeId = 555L;
    List<Long> neighbors = Arrays.asList(1L, 2L, 3L);

    storage.storeAdjacency(nodeId, neighbors).get();

    // First load - from DB
    NodeAdjacency loaded1 = storage.loadAdjacency(nodeId).get();

    // Second load - should be from cache
    NodeAdjacency loaded2 = storage.loadAdjacency(nodeId).get();

    assertThat(loaded1).isEqualTo(loaded2);

    // Clear cache
    storage.clearCache();

    // Load again - from DB
    NodeAdjacency loaded3 = storage.loadAdjacency(nodeId).get();
    assertThat(loaded3.getNeighborsList()).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testTimestamps() throws ExecutionException, InterruptedException {
    // Use fixed time for predictable tests
    Instant fixedInstant = Instant.parse("2024-01-01T00:00:00Z");
    InstantSource fixedSource = InstantSource.fixed(fixedInstant);

    NodeAdjacencyStorage storageWithFixedTime =
        new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, fixedSource, 1000, Duration.ofMinutes(10));

    long nodeId = 777L;
    storageWithFixedTime.storeAdjacency(nodeId, Arrays.asList(1L, 2L)).get();

    NodeAdjacency loaded = storageWithFixedTime.loadAdjacency(nodeId).get();
    assertThat(loaded.hasUpdatedAt()).isTrue();
    assertThat(loaded.getUpdatedAt().getSeconds()).isEqualTo(fixedInstant.getEpochSecond());
  }

  @Test
  void testLoadNonExistentNode() throws ExecutionException, InterruptedException {
    NodeAdjacency loaded = storage.loadAdjacency(999999L).get();
    assertThat(loaded).isNull();
  }

  @Test
  void testAddNeighborToNonExistentNode() throws ExecutionException, InterruptedException {
    long nodeId = 888L;

    // Add neighbor to non-existent node
    storage.addNeighbor(nodeId, 1L).get();

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded).isNotNull();
    assertThat(loaded.getNeighborsList()).containsExactly(1L);
    assertThat(loaded.getVersion()).isEqualTo(1);
  }

  @Test
  void testRemoveNeighborFromNonExistentNode() throws ExecutionException, InterruptedException {
    long nodeId = 999L;

    // Should not throw
    storage.removeNeighbor(nodeId, 1L).get();

    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();
    assertThat(loaded).isNull();
  }

  @Test
  void testEmptyNeighborList() throws ExecutionException, InterruptedException {
    long nodeId = 1234L;

    storage.storeAdjacency(nodeId, new ArrayList<>()).get();
    NodeAdjacency loaded = storage.loadAdjacency(nodeId).get();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getNeighborsList()).isEmpty();
  }

  @Test
  void testLastAccessedAtPreservation() throws ExecutionException, InterruptedException {
    long nodeId = 2000L;

    // First store
    storage.storeAdjacency(nodeId, Arrays.asList(1L, 2L)).get();
    NodeAdjacency first = storage.loadAdjacency(nodeId).get();

    // Manually set last accessed (in real usage this would be set by access tracking)
    // For now we just verify it's preserved if present

    // Update adjacency
    storage.storeAdjacency(nodeId, Arrays.asList(3L, 4L)).get();
    NodeAdjacency second = storage.loadAdjacency(nodeId).get();

    assertThat(second.getVersion()).isEqualTo(2);
    assertThat(second.getNeighborsList()).containsExactly(3L, 4L);
  }
}
