package io.github.panghy.vectorsearch.storage;

import static java.util.concurrent.CompletableFuture.completedFuture;
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
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
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
  void testStoreAndLoadAdjacency() {
    long nodeId = 42L;
    List<Long> neighbors = Arrays.asList(1L, 5L, 3L, 9L, 2L);

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, neighbors)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded).isNotNull();
                assertThat(loaded.getNodeId()).isEqualTo(nodeId);
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L, 5L, 9L); // Sorted
                assertThat(loaded.getVersion()).isEqualTo(1);
                return null;
              });
        })
        .join();
  }

  @Test
  void testUpdateAdjacency() {
    long nodeId = 42L;
    List<Long> neighbors1 = Arrays.asList(1L, 2L, 3L);
    List<Long> neighbors2 = Arrays.asList(4L, 5L, 6L);

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, neighbors1)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded1 -> {
                assertThat(loaded1.getVersion()).isEqualTo(1);
                return null;
              });
        })
        .join();

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, neighbors2)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded2 -> {
                assertThat(loaded2.getVersion()).isEqualTo(2);
                assertThat(loaded2.getNeighborsList()).containsExactly(4L, 5L, 6L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testDegreeBound() {
    long nodeId = 100L;
    // Create more neighbors than degree allows
    List<Long> tooManyNeighbors = new ArrayList<>();
    for (long i = 0; i < GRAPH_DEGREE + 10; i++) {
      tooManyNeighbors.add(i);
    }

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, tooManyNeighbors)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsCount()).isEqualTo(GRAPH_DEGREE);
                // Should keep first GRAPH_DEGREE after sorting
                for (int i = 0; i < GRAPH_DEGREE; i++) {
                  assertThat(loaded.getNeighbors(i)).isEqualTo((long) i);
                }
                return null;
              });
        })
        .join();
  }

  @Test
  void testDuplicateNeighbors() {
    long nodeId = 50L;
    List<Long> neighborsWithDuplicates = Arrays.asList(1L, 2L, 3L, 2L, 1L, 4L);

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, neighborsWithDuplicates)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L, 4L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testBatchLoadAdjacency() {
    // Store adjacencies for multiple nodes
    List<Long> nodeIds = Arrays.asList(10L, 20L, 30L);

    db.runAsync(tr -> {
          List<CompletableFuture<Void>> futures = new ArrayList<>();
          for (long nodeId : nodeIds) {
            List<Long> neighbors = Arrays.asList(nodeId + 1, nodeId + 2, nodeId + 3);
            futures.add(storage.storeAdjacency(tr, nodeId, neighbors));
          }
          return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
        })
        .join();

    db.runAsync(tr -> {
          return storage.batchLoadAdjacency(tr, nodeIds).thenApply(loaded -> {
            assertThat(loaded).hasSize(3);
            assertThat(loaded.get(10L).getNeighborsList()).containsExactly(11L, 12L, 13L);
            assertThat(loaded.get(20L).getNeighborsList()).containsExactly(21L, 22L, 23L);
            assertThat(loaded.get(30L).getNeighborsList()).containsExactly(31L, 32L, 33L);
            return null;
          });
        })
        .join();
  }

  @Test
  void testBatchLoadWithMissing() {
    // Store only some nodes
    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, 10L, Arrays.asList(1L, 2L))
              .thenCompose(v -> storage.storeAdjacency(tr, 30L, Arrays.asList(3L, 4L)));
        })
        .join();

    db.runAsync(tr -> {
          return storage.batchLoadAdjacency(tr, Arrays.asList(10L, 20L, 30L))
              .thenApply(loaded -> {
                assertThat(loaded).hasSize(2);
                assertThat(loaded).containsKeys(10L, 30L);
                assertThat(loaded).doesNotContainKey(20L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testAddNeighbor() {
    long nodeId = 75L;

    db.runAsync(tr -> {
          // Start with some neighbors
          return storage.storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L))
              .thenCompose(v -> storage.addNeighbor(tr, nodeId, 3L))
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L);
                assertThat(loaded.getVersion()).isEqualTo(2);
                return null;
              });
        })
        .join();
  }

  @Test
  void testAddDuplicateNeighbor() {
    long nodeId = 80L;

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L, 3L))
              .thenCompose(v -> storage.addNeighbor(tr, nodeId, 2L)) // Try to add duplicate
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L, 3L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testAddNeighborToFull() {
    long nodeId = 85L;

    db.runAsync(tr -> {
          // Fill to degree limit
          List<Long> fullNeighbors = new ArrayList<>();
          for (int i = 0; i < GRAPH_DEGREE; i++) {
            fullNeighbors.add((long) i);
          }
          return storage.storeAdjacency(tr, nodeId, fullNeighbors)
              .thenCompose(v -> storage.addNeighbor(tr, nodeId, 100L)) // Try to add one more
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsCount()).isEqualTo(GRAPH_DEGREE);
                assertThat(loaded.getNeighborsList()).doesNotContain(100L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testRemoveNeighbor() {
    long nodeId = 90L;

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L, 3L, 4L))
              .thenCompose(v -> storage.removeNeighbor(tr, nodeId, 2L))
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 3L, 4L);
                assertThat(loaded.getVersion()).isEqualTo(2);
                return null;
              });
        })
        .join();
  }

  @Test
  void testRemoveNonExistentNeighbor() {
    long nodeId = 95L;

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L))
              .thenCompose(v -> storage.removeNeighbor(tr, nodeId, 99L))
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.getNeighborsList()).containsExactly(1L, 2L);
                return null;
              });
        })
        .join();
  }

  @Test
  void testAddBackLinks() {
    long nodeId = 100L;
    List<Long> neighbors = Arrays.asList(200L, 300L, 400L);

    db.runAsync(tr -> {
          // First create the neighbors
          List<CompletableFuture<Void>> createFutures = new ArrayList<>();
          for (long neighbor : neighbors) {
            createFutures.add(storage.storeAdjacency(tr, neighbor, new ArrayList<>()));
          }
          return CompletableFuture.allOf(createFutures.toArray(CompletableFuture[]::new))
              .thenCompose(v -> storage.addBackLinks(tr, nodeId, neighbors))
              .thenCompose(v -> {
                // Verify back links were added
                List<CompletableFuture<NodeAdjacency>> loadFutures = new ArrayList<>();
                for (long neighbor : neighbors) {
                  loadFutures.add(storage.loadAdjacency(tr, neighbor));
                }
                return CompletableFuture.allOf(loadFutures.toArray(CompletableFuture[]::new))
                    .thenApply(unused -> {
                      for (int i = 0; i < neighbors.size(); i++) {
                        NodeAdjacency loaded =
                            loadFutures.get(i).join();
                        assertThat(loaded.getNeighborsList())
                            .contains(nodeId);
                      }
                      return null;
                    });
              });
        })
        .join();
  }

  @Test
  void testRemoveBackLinks() {
    long nodeId = 150L;
    List<Long> neighbors = Arrays.asList(250L, 350L, 450L);

    db.runAsync(tr -> {
          // First create neighbors with nodeId as neighbor
          List<CompletableFuture<Void>> createFutures = new ArrayList<>();
          for (long neighbor : neighbors) {
            createFutures.add(storage.storeAdjacency(tr, neighbor, Arrays.asList(nodeId, 999L)));
          }
          return CompletableFuture.allOf(createFutures.toArray(CompletableFuture[]::new))
              .thenCompose(v -> storage.removeBackLinks(tr, nodeId, neighbors))
              .thenCompose(v -> {
                // Verify back links were removed
                List<CompletableFuture<NodeAdjacency>> loadFutures = new ArrayList<>();
                for (long neighbor : neighbors) {
                  loadFutures.add(storage.loadAdjacency(tr, neighbor));
                }
                return CompletableFuture.allOf(loadFutures.toArray(CompletableFuture[]::new))
                    .thenApply(unused -> {
                      for (int i = 0; i < neighbors.size(); i++) {
                        NodeAdjacency loaded =
                            loadFutures.get(i).join();
                        assertThat(loaded.getNeighborsList())
                            .doesNotContain(nodeId);
                        assertThat(loaded.getNeighborsList())
                            .contains(999L);
                      }
                      return null;
                    });
              });
        })
        .join();
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
  void testStoreAndLoadEntryList() {
    EntryList entryList = EntryList.newBuilder()
        .addAllPrimaryEntries(Arrays.asList(1L, 2L, 3L))
        .addAllRandomEntries(Arrays.asList(10L, 20L))
        .addAllHighDegreeEntries(Arrays.asList(100L, 200L))
        .setVersion(1)
        .build();

    db.runAsync(tr -> {
          storage.storeEntryList(tr, entryList);
          return completedFuture(null);
        })
        .join();

    db.runAsync(tr -> storage.loadEntryList(tr).thenApply(loaded -> {
          assertThat(loaded).isNotNull();
          assertThat(loaded.getPrimaryEntriesList()).containsExactly(1L, 2L, 3L);
          assertThat(loaded.getRandomEntriesList()).containsExactly(10L, 20L);
          assertThat(loaded.getHighDegreeEntriesList()).containsExactly(100L, 200L);
          assertThat(loaded.getVersion()).isEqualTo(1);
          return null;
        }))
        .join();
  }

  @Test
  void testLoadNonExistentEntryList() {
    db.runAsync(tr -> {
          return storage.loadEntryList(tr).thenApply(loaded -> {
            assertThat(loaded).isNull();
            return null;
          });
        })
        .join();
  }

  @Test
  void testStoreAndLoadGraphMeta() {
    GraphMeta graphMeta = GraphMeta.newBuilder()
        .setConnectedComponents(2)
        .setLargestComponentSize(1000L)
        .setTotalNodes(1100L)
        .addOrphanedNodes(500L)
        .addOrphanedNodes(501L)
        .setRepairState(GraphMeta.RepairState.NOT_NEEDED)
        .build();

    db.runAsync(tr -> {
          storage.storeGraphMeta(tr, graphMeta);
          return completedFuture(null);
        })
        .join();

    db.runAsync(tr -> storage.loadGraphMeta(tr).thenApply(loaded -> {
          assertThat(loaded).isNotNull();
          assertThat(loaded.getConnectedComponents()).isEqualTo(2);
          assertThat(loaded.getLargestComponentSize()).isEqualTo(1000L);
          assertThat(loaded.getTotalNodes()).isEqualTo(1100L);
          assertThat(loaded.getOrphanedNodesList()).containsExactly(500L, 501L);
          assertThat(loaded.getRepairState()).isEqualTo(GraphMeta.RepairState.NOT_NEEDED);
          return null;
        }))
        .join();
  }

  @Test
  void testLoadNonExistentGraphMeta() {
    db.runAsync(tr -> {
          return storage.loadGraphMeta(tr).thenApply(loaded -> {
            assertThat(loaded).isNull();
            return null;
          });
        })
        .join();
  }

  @Test
  void testDeleteNode() {
    long nodeId = 123L;

    db.runAsync(tr -> storage.storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L, 3L)))
        .join();

    db.runAsync(tr -> storage.loadAdjacency(tr, nodeId)
            .thenApply(loaded -> {
              assertThat(loaded).isNotNull();
              return loaded;
            })
            .thenAccept(v -> storage.deleteNode(tr, nodeId)))
        .join();

    db.runAsync(tr -> storage.loadAdjacency(tr, nodeId).thenApply(loaded -> {
          assertThat(loaded).isNull();
          return null;
        }))
        .join();
  }

  @Test
  void testCaching() throws ExecutionException, InterruptedException {
    long nodeId = 555L;
    List<Long> neighbors = Arrays.asList(1L, 2L, 3L);

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, neighbors);
        })
        .join();

    // Load from cache (async method)
    NodeAdjacency loaded1 = storage.loadAdjacencyAsync(nodeId).get();

    // Second load - should be from cache
    NodeAdjacency loaded2 = storage.loadAdjacencyAsync(nodeId).get();

    assertThat(loaded1).isEqualTo(loaded2);

    // Clear cache
    storage.clearCache();

    // Load again - from DB
    NodeAdjacency loaded3 = storage.loadAdjacencyAsync(nodeId).get();
    assertThat(loaded3.getNeighborsList()).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testTimestamps() {
    // Use fixed time for predictable tests
    Instant fixedInstant = Instant.parse("2024-01-01T00:00:00Z");
    InstantSource fixedSource = InstantSource.fixed(fixedInstant);

    NodeAdjacencyStorage storageWithFixedTime =
        new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, fixedSource, 1000, Duration.ofMinutes(10));

    long nodeId = 777L;

    db.runAsync(tr -> {
          return storageWithFixedTime
              .storeAdjacency(tr, nodeId, Arrays.asList(1L, 2L))
              .thenCompose(v -> storageWithFixedTime.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded.hasUpdatedAt()).isTrue();
                assertThat(loaded.getUpdatedAt().getSeconds()).isEqualTo(fixedInstant.getEpochSecond());
                return null;
              });
        })
        .join();
  }

  @Test
  void testLoadNonExistentNode() {
    db.runAsync(tr -> storage.loadAdjacency(tr, 999999L).thenApply(loaded -> {
          assertThat(loaded).isNull();
          return null;
        }))
        .join();
  }

  @Test
  void testAddNeighborToNonExistentNode() {
    long nodeId = 888L;

    db.runAsync(tr -> {
          // Add neighbor to non-existent node
          return storage.addNeighbor(tr, nodeId, 1L)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded).isNotNull();
                assertThat(loaded.getNeighborsList()).containsExactly(1L);
                assertThat(loaded.getVersion()).isEqualTo(1);
                return null;
              });
        })
        .join();
  }

  @Test
  void testRemoveNeighborFromNonExistentNode() {
    long nodeId = 999L;

    db.runAsync(tr -> {
          // Should not throw
          return storage.removeNeighbor(tr, nodeId, 1L)
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded).isNull();
                return null;
              });
        })
        .join();
  }

  @Test
  void testEmptyNeighborList() {
    long nodeId = 1234L;

    db.runAsync(tr -> {
          return storage.storeAdjacency(tr, nodeId, new ArrayList<>())
              .thenCompose(v -> storage.loadAdjacency(tr, nodeId))
              .thenApply(loaded -> {
                assertThat(loaded).isNotNull();
                assertThat(loaded.getNeighborsList()).isEmpty();
                return null;
              });
        })
        .join();
  }

  @Test
  void testCompositeOperationsInSingleTransaction() {
    // Test that multiple operations can be composed in a single transaction
    long mainNode = 1000L;
    List<Long> neighbors = Arrays.asList(2000L, 3000L, 4000L);

    db.runAsync(tr -> {
          // Create main node
          return storage.storeAdjacency(tr, mainNode, neighbors)
              .thenCompose(v -> {
                // Create neighbors and add back-links in same transaction
                List<CompletableFuture<Void>> futures = new ArrayList<>();
                for (long neighbor : neighbors) {
                  futures.add(storage.storeAdjacency(tr, neighbor, List.of(mainNode)));
                }
                return CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new));
              })
              .thenAccept(v -> {
                // Update metadata
                GraphMeta meta = GraphMeta.newBuilder()
                    .setTotalNodes(4)
                    .setConnectedComponents(1)
                    .setLargestComponentSize(4)
                    .build();
                storage.storeGraphMeta(tr, meta);
              })
              .thenCompose(v -> {
                // Update entry list
                EntryList entries = EntryList.newBuilder()
                    .addPrimaryEntries(mainNode)
                    .build();
                storage.storeEntryList(tr, entries);
                return completedFuture(null);
              });
        })
        .join();

    // Verify everything was saved atomically
    db.runAsync(tr -> {
          return storage.loadAdjacency(tr, mainNode)
              .thenCompose(main -> {
                assertThat(main.getNeighborsList()).containsExactly(2000L, 3000L, 4000L);

                List<CompletableFuture<NodeAdjacency>> loadFutures = new ArrayList<>();
                for (long neighbor : neighbors) {
                  loadFutures.add(storage.loadAdjacency(tr, neighbor));
                }
                return CompletableFuture.allOf(loadFutures.toArray(CompletableFuture[]::new))
                    .thenApply(v -> {
                      for (int i = 0; i < neighbors.size(); i++) {
                        NodeAdjacency adj =
                            loadFutures.get(i).join();
                        assertThat(adj.getNeighborsList())
                            .contains(mainNode);
                      }
                      return null;
                    });
              })
              .thenCompose(v -> storage.loadGraphMeta(tr))
              .thenApply(meta -> {
                assertThat(meta.getTotalNodes()).isEqualTo(4);
                return null;
              })
              .thenCompose(v -> storage.loadEntryList(tr))
              .thenApply(entries -> {
                assertThat(entries.getPrimaryEntriesList()).contains(mainNode);
                return null;
              });
        })
        .join();
  }
}
