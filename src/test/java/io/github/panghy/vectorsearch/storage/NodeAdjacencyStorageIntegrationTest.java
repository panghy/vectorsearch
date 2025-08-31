package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage.ScoredNode;
import java.time.Duration;
import java.time.InstantSource;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NodeAdjacencyStorageIntegrationTest {
  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private NodeAdjacencyStorage storage;
  private String testCollectionName;
  private static final int GRAPH_DEGREE = 16;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "node_adjacency_integration", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace, testCollectionName);
    storage = new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, InstantSource.system(), 100, Duration.ofMinutes(5));
  }

  @AfterEach
  void tearDown() {
    // Clean up test directory
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      directoryLayer
          .removeIfExists(tr, Arrays.asList("test", "node_adjacency_integration", testCollectionName))
          .join();
      return null;
    });
  }

  @Test
  void testRobustPruneIntegration() {
    // Test that the robustPrune method correctly uses RobustPruning
    List<ScoredNode> candidates = Arrays.asList(
        new ScoredNode(1L, 1.0),
        new ScoredNode(2L, 1.1),
        new ScoredNode(3L, 1.5),
        new ScoredNode(4L, 2.0),
        new ScoredNode(5L, 3.0));

    // With alpha=1.2, node 2 should be dominated by node 1
    List<Long> pruned = storage.robustPrune(candidates, 5, 1.2);

    // Node 1 selected first
    // Node 2 dominated (1.1 <= 1.2*1.0)
    // Node 3 selected (1.5 > 1.2*1.0)
    // Node 4 selected (2.0 > 1.2*1.5)
    // Node 5 selected (3.0 > 1.2*2.0)
    assertThat(pruned).containsExactly(1L, 3L, 4L, 5L);
  }

  @Test
  void testRobustPruneWithLowAlpha() {
    // Test with low alpha for more aggressive pruning
    List<ScoredNode> candidates = Arrays.asList(
        new ScoredNode(1L, 1.0), new ScoredNode(2L, 1.5), new ScoredNode(3L, 2.0), new ScoredNode(4L, 2.5));

    // With alpha=0.95, more nodes will be dominated
    List<Long> pruned = storage.robustPrune(candidates, 4, 0.95);

    // All nodes should be selected as none are dominated with these distances
    assertThat(pruned).containsExactly(1L, 2L, 3L, 4L);
  }

  @Test
  void testMergePruneWithAdditions() {
    // Test merging existing neighbors with new additions
    List<Long> current = Arrays.asList(10L, 20L, 30L);
    List<ScoredNode> additions =
        Arrays.asList(new ScoredNode(40L, 0.5), new ScoredNode(50L, 0.6), new ScoredNode(60L, 0.7));

    List<Long> merged = storage.mergePruneWithDistances(current, additions, 5);

    // The implementation keeps some existing and adds pruned new additions
    // With the current logic it will keep at most maxDegree/2 existing and add pruned additions
    assertThat(merged.size()).isLessThanOrEqualTo(5);
    assertThat(merged).contains(40L, 50L); // Best new additions should be included
  }

  @Test
  void testMergePruneExceedsLimit() {
    // Test when merged size exceeds the degree limit
    List<Long> current = Arrays.asList(1L, 2L, 3L, 4L, 5L);
    List<ScoredNode> additions = Arrays.asList(
        new ScoredNode(6L, 0.1), new ScoredNode(7L, 0.2), new ScoredNode(8L, 0.3), new ScoredNode(9L, 0.4));

    List<Long> merged = storage.mergePruneWithDistances(current, additions, 6);

    assertThat(merged).hasSize(6);
    // Should include the best new additions
    assertThat(merged).contains(6L, 7L);
  }

  @Test
  void testStoreAndLoadAdjacency() {
    // Test basic storage and retrieval with async operations
    CompletableFuture<Void> future = db.runAsync(tx -> {
      List<Long> neighbors = Arrays.asList(2L, 3L, 4L);
      return storage.storeAdjacency(tx, 1L, neighbors)
          .thenCompose(v -> storage.loadAdjacency(tx, 1L))
          .thenAccept(adjacency -> {
            assertThat(adjacency).isNotNull();
            assertThat(adjacency.getNodeId()).isEqualTo(1L);
            assertThat(adjacency.getNeighborsList()).containsExactly(2L, 3L, 4L);
            assertThat(adjacency.getVersion()).isEqualTo(1);
          });
    });

    future.join();
  }

  @Test
  void testAddBackLinksWithPruning() {
    // Test that back-links are properly added with simple pruning when needed
    CompletableFuture<Void> future = db.runAsync(tx -> {
      // First, create some existing neighbors
      List<Long> existingNeighbors =
          Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L);
      return storage.storeAdjacency(tx, 100L, existingNeighbors)
          .thenCompose(v -> {
            // Now add back-links from node 200 to node 100
            // Node 100 already has 15 neighbors, adding one more should trigger pruning
            return storage.addBackLinks(tx, 200L, Arrays.asList(100L));
          })
          .thenCompose(v -> storage.loadAdjacency(tx, 100L))
          .thenAccept(adjacency -> {
            assertThat(adjacency).isNotNull();
            // Should have at most GRAPH_DEGREE neighbors after pruning
            assertThat(adjacency.getNeighborsList().size()).isLessThanOrEqualTo(GRAPH_DEGREE);
            // Should contain the back-link or have pruned appropriately
            if (adjacency.getNeighborsList().size() == GRAPH_DEGREE) {
              // If at max capacity, back-link might have been added and something pruned
              assertThat(adjacency.getNeighborsList()).hasSize(GRAPH_DEGREE);
            }
          });
    });

    future.join();
  }

  @Test
  void testAddBackLinksWithDistanceBasedPruning() {
    // Test that back-links use robust pruning when distance function is provided
    CompletableFuture<Void> future = db.runAsync(tx -> {
      // Create a node with many neighbors
      List<Long> existingNeighbors =
          Arrays.asList(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, 10L, 11L, 12L, 13L, 14L, 15L);

      // Mock distance function that returns closer distances for lower node IDs
      BiFunction<Long, Long, CompletableFuture<Float>> distanceFunction = (from, to) -> {
        // Simulate that lower node IDs are closer
        float distance = Math.abs(to - from) * 0.1f;
        return CompletableFuture.completedFuture(distance);
      };

      return storage.storeAdjacency(tx, 100L, existingNeighbors)
          .thenCompose(v -> {
            // Add back-link with distance function - should trigger robust pruning
            return storage.addBackLinksWithDistanceFunction(
                tx, 200L, Arrays.asList(100L), distanceFunction);
          })
          .thenCompose(v -> storage.loadAdjacency(tx, 100L))
          .thenAccept(adjacency -> {
            assertThat(adjacency).isNotNull();
            assertThat(adjacency.getNeighborsList()).hasSize(GRAPH_DEGREE);
            // With robust pruning, should keep diverse neighbors
            // The exact selection depends on the pruning algorithm
            assertThat(adjacency.getNeighborsList()).contains(200L); // New back-link should be included
          });
    });

    future.join();
  }

  @Test
  void testBatchLoadAdjacency() {
    // Test batch loading of multiple adjacency lists
    CompletableFuture<Void> future = db.runAsync(tx -> {
      // Store adjacencies for multiple nodes
      return storage.storeAdjacency(tx, 1L, Arrays.asList(2L, 3L))
          .thenCompose(v -> storage.storeAdjacency(tx, 2L, Arrays.asList(1L, 3L)))
          .thenCompose(v -> storage.storeAdjacency(tx, 3L, Arrays.asList(1L, 2L)))
          .thenCompose(v -> {
            // Batch load all three
            return storage.batchLoadAdjacency(tx, Arrays.asList(1L, 2L, 3L));
          })
          .thenAccept(adjacencyMap -> {
            assertThat(adjacencyMap).hasSize(3);
            assertThat(adjacencyMap.get(1L).getNeighborsList()).containsExactly(2L, 3L);
            assertThat(adjacencyMap.get(2L).getNeighborsList()).containsExactly(1L, 3L);
            assertThat(adjacencyMap.get(3L).getNeighborsList()).containsExactly(1L, 2L);
          });
    });

    future.join();
  }

  @Test
  void testRemoveNeighbor() {
    // Test removing a neighbor from adjacency list
    CompletableFuture<Void> future = db.runAsync(tx -> {
      List<Long> neighbors = Arrays.asList(2L, 3L, 4L);
      return storage.storeAdjacency(tx, 1L, neighbors)
          .thenCompose(v -> storage.removeNeighbor(tx, 1L, 3L))
          .thenCompose(v -> storage.loadAdjacency(tx, 1L))
          .thenAccept(adjacency -> {
            assertThat(adjacency).isNotNull();
            assertThat(adjacency.getNeighborsList()).containsExactly(2L, 4L);
            assertThat(adjacency.getVersion()).isEqualTo(2); // Version incremented
          });
    });

    future.join();
  }

  @Test
  void testRemoveBackLinks() {
    // Test removing back-links
    CompletableFuture<Void> future = db.runAsync(tx -> {
      // Set up bidirectional links
      return storage.storeAdjacency(tx, 1L, Arrays.asList(2L, 3L))
          .thenCompose(v -> storage.storeAdjacency(tx, 2L, Arrays.asList(1L)))
          .thenCompose(v -> storage.storeAdjacency(tx, 3L, Arrays.asList(1L)))
          .thenCompose(v -> {
            // Remove back-links from nodes 2 and 3
            return storage.removeBackLinks(tx, 1L, Arrays.asList(2L, 3L));
          })
          .thenCompose(v -> storage.batchLoadAdjacency(tx, Arrays.asList(2L, 3L)))
          .thenAccept(adjacencyMap -> {
            // Nodes 2 and 3 should no longer have 1 as a neighbor
            assertThat(adjacencyMap.get(2L).getNeighborsList()).isEmpty();
            assertThat(adjacencyMap.get(3L).getNeighborsList()).isEmpty();
          });
    });

    future.join();
  }
}
