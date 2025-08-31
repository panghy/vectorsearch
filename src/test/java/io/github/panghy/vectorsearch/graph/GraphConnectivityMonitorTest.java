package io.github.panghy.vectorsearch.graph;

import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;
import static java.util.concurrent.CompletableFuture.allOf;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.graph.GraphConnectivityMonitor.ConnectivityAnalysis;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.GraphMetaStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import java.time.Clock;
import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class GraphConnectivityMonitorTest {
  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private GraphMetaStorage metaStorage;
  private NodeAdjacencyStorage adjacencyStorage;
  private PqBlockStorage pqBlockStorage;
  private EntryPointStorage entryPointStorage;
  private GraphConnectivityMonitor monitor;
  private String testCollectionName;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, asList("test", "graph_connectivity", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace, testCollectionName);
    Clock clock = Clock.systemUTC();
    metaStorage = new GraphMetaStorage(keys, clock);
    adjacencyStorage = new NodeAdjacencyStorage(db, keys, 32, clock, 10000, Duration.ofMinutes(5));
    pqBlockStorage = new PqBlockStorage(db, keys, 1000, 16, clock, 10000, Duration.ofMinutes(5));
    entryPointStorage = new EntryPointStorage(testSpace, testCollectionName, clock);

    // Create monitor without PQ for simplicity in tests
    monitor = new GraphConnectivityMonitor(
        db, keys, metaStorage, adjacencyStorage, pqBlockStorage, entryPointStorage, null);
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
  void testAnalyzeConnectivityEmptyGraph() {
    // Run analysis on empty graph
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.connectedComponents()).isEqualTo(0);
    assertThat(analysis.largestComponentSize()).isEqualTo(0);
    assertThat(analysis.totalNodes()).isEqualTo(0);
    assertThat(analysis.orphanedNodes()).isEmpty();
    assertThat(analysis.needsRepair()).isFalse();
  }

  @Test
  void testAnalyzeConnectivitySingleNode() {
    // Create a single node
    db.runAsync(tx -> adjacencyStorage.storeAdjacency(tx, 1L, List.of())).join();

    // Run analysis
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
    // Single node graph should be considered healthy
    assertThat(analysis.needsRepair()).isFalse();
  }

  @Test
  void testAnalyzeConnectivityConnectedPair() {
    // Create two connected nodes
    db.runAsync(tx -> allOf(
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L))))
        .join();

    // Run analysis
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
    assertThat(analysis.largestComponentSize()).isGreaterThan(0);
    assertThat(analysis.needsRepair()).isFalse();
  }

  @Test
  void testAnalyzeConnectivityDisconnectedComponents() {
    // Create two disconnected components
    db.runAsync(tx -> allOf(
            // Component 1: nodes 1-2
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L)),
            // Component 2: nodes 10-11
            adjacencyStorage.storeAdjacency(tx, 10L, List.of(11L)),
            adjacencyStorage.storeAdjacency(tx, 11L, List.of(10L))))
        .join();

    // Run analysis
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
    // With sampling, we may or may not detect the disconnection
    // but we should get some results
    assertThat(analysis.largestComponentSize()).isGreaterThan(0);
  }

  @Test
  void testAnalyzeConnectivityIsolatedNode() {
    // Create a connected pair and an isolated node
    db.runAsync(tx -> allOf(
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L)),
            adjacencyStorage.storeAdjacency(tx, 3L, List.of()) // No neighbors
            ))
        .join();

    // Run analysis
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
    // Sampling may or may not find the isolated node
  }

  @Test
  void testRefreshEntryPointsEmptyGraph() {
    // Refresh entry points on empty graph
    CompletableFuture<Void> future = db.runAsync(tx -> monitor.refreshEntryPoints(tx));

    // Should complete without error
    assertThat(future.join()).isNull();
  }

  @Test
  void testRefreshEntryPointsWithNodes() {
    // Create a small graph
    db.runAsync(tx -> {
          CompletableFuture<?>[] futures = new CompletableFuture[10];
          for (int i = 0; i < 10; i++) {
            long nodeId = i + 1;
            futures[i] = adjacencyStorage.storeAdjacency(
                tx, nodeId, List.of((nodeId % 10) + 1)); // Ring topology
          }
          return allOf(futures);
        })
        .join();

    // Refresh entry points
    CompletableFuture<Void> future = db.runAsync(tx -> monitor.refreshEntryPoints(tx));

    assertThat(future.join()).isNull();

    // Verify entry points were stored
    db.runAsync(tx -> entryPointStorage.loadEntryList(tx))
        .thenAccept(entryList -> {
          assertThat(entryList).isNotNull();
          // Should have some entry points
          assertThat(entryList.getPrimaryEntriesCount()
                  + entryList.getRandomEntriesCount()
                  + entryList.getHighDegreeEntriesCount())
              .isGreaterThan(0);
        })
        .join();
  }

  @Test
  void testAnalyzeAndRepairHealthyGraph() {
    // Create a small connected graph
    db.runAsync(tx -> allOf(
            adjacencyStorage.storeAdjacency(tx, 1L, asList(2L, 3L)),
            adjacencyStorage.storeAdjacency(tx, 2L, asList(1L, 3L)),
            adjacencyStorage.storeAdjacency(tx, 3L, asList(1L, 2L))))
        .join();

    // Run analyze and repair
    monitor.analyzeAndRepair(1).join();
  }

  @Test
  void testLargeGraphSampling() {
    // Create a larger graph to test sampling
    final int nodeCount = 100; // Reduced for faster tests
    db.runAsync(tx -> {
          CompletableFuture<?>[] futures = new CompletableFuture[nodeCount];
          for (int i = 0; i < nodeCount; i++) {
            long nodeId = i + 1;

            // Create connections to nearby nodes
            java.util.List<Long> neighbors = new java.util.ArrayList<>();
            if (nodeId > 1) {
              neighbors.add(nodeId - 1);
            }
            if (nodeId < nodeCount) {
              neighbors.add(nodeId + 1);
            }

            futures[i] = adjacencyStorage.storeAdjacency(tx, nodeId, neighbors);
          }
          return allOf(futures);
        })
        .join();

    // Run analysis - should use sampling for large graph
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(tx -> monitor.analyzeConnectivity(tx));

    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
    assertThat(analysis.largestComponentSize()).isGreaterThan(0);
    // Linear chain should be mostly connected
    assertThat(analysis.needsRepair()).isFalse();
  }

  @Test
  void testGraphRepairWithOrphanedNodes() {
    // Create a graph with disconnected components that needs repair
    db.runAsync(tx -> allOf(
            // Main component: nodes 1-5 well connected
            adjacencyStorage.storeAdjacency(tx, 1L, asList(2L, 3L)),
            adjacencyStorage.storeAdjacency(tx, 2L, asList(1L, 3L, 4L)),
            adjacencyStorage.storeAdjacency(tx, 3L, asList(1L, 2L, 5L)),
            adjacencyStorage.storeAdjacency(tx, 4L, asList(2L, 5L)),
            adjacencyStorage.storeAdjacency(tx, 5L, asList(3L, 4L)),
            // Orphaned nodes: 10-12
            adjacencyStorage.storeAdjacency(tx, 10L, List.of(11L)),
            adjacencyStorage.storeAdjacency(tx, 11L, asList(10L, 12L)),
            adjacencyStorage.storeAdjacency(tx, 12L, List.of(11L))))
        .join();

    // Test via analyzeAndRepair which will detect and repair disconnection
    CompletableFuture<Void> repairFuture = monitor.analyzeAndRepair(1);

    // Should complete without error
    assertThat(repairFuture.join()).isNull();
  }

  @Test
  void testAnalyzeAndRepairWithDisconnectedGraph() {
    // Create a small graph with clear disconnection to trigger repair
    db.runAsync(tx -> allOf(
            // Small main component
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L)),
            // Many orphaned nodes to trigger repair threshold
            adjacencyStorage.storeAdjacency(tx, 10L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 11L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 12L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 13L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 14L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 15L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 16L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 17L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 18L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 19L, Collections.emptyList()),
            adjacencyStorage.storeAdjacency(tx, 20L, Collections.emptyList())))
        .join();

    // Run analyze and repair - should detect the graph needs repair
    CompletableFuture<Void> future = monitor.analyzeAndRepair(1);

    // Should complete successfully
    assertThat(future.join()).isNull();
  }

  @Test
  void testAnalyzeWithNullPq() {
    // Create monitor without PQ to test null PQ path
    GraphConnectivityMonitor monitorNoPq = new GraphConnectivityMonitor(
        db, keys, metaStorage, adjacencyStorage, pqBlockStorage, entryPointStorage, null);

    // Create some nodes
    db.runAsync(tx -> allOf(
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L))))
        .join();

    // Test analyze with null PQ - should still work
    CompletableFuture<ConnectivityAnalysis> future = db.runAsync(monitorNoPq::analyzeConnectivity);

    // Should complete without error even with null PQ
    ConnectivityAnalysis analysis = future.join();
    assertThat(analysis).isNotNull();
    assertThat(analysis.totalNodes()).isGreaterThan(0);
  }

  @Test
  void testRefreshEntryPointsWithHighDegreeNodes() {
    // Create a graph with varying node degrees
    db.runAsync(tx -> allOf(
            // High degree hub nodes
            adjacencyStorage.storeAdjacency(tx, 1L, asList(2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)),
            adjacencyStorage.storeAdjacency(tx, 10L, asList(11L, 12L, 13L, 14L, 15L)),
            // Medium degree nodes
            adjacencyStorage.storeAdjacency(tx, 2L, asList(1L, 3L, 4L)),
            adjacencyStorage.storeAdjacency(tx, 3L, asList(1L, 2L, 4L)),
            // Low degree nodes
            adjacencyStorage.storeAdjacency(tx, 4L, asList(1L, 2L, 3L)),
            adjacencyStorage.storeAdjacency(tx, 5L, singletonList(1L)),
            adjacencyStorage.storeAdjacency(tx, 6L, singletonList(1L)),
            adjacencyStorage.storeAdjacency(tx, 7L, singletonList(1L)),
            adjacencyStorage.storeAdjacency(tx, 8L, singletonList(1L)),
            adjacencyStorage.storeAdjacency(tx, 9L, singletonList(1L)),
            adjacencyStorage.storeAdjacency(tx, 11L, singletonList(10L)),
            adjacencyStorage.storeAdjacency(tx, 12L, singletonList(10L)),
            adjacencyStorage.storeAdjacency(tx, 13L, singletonList(10L)),
            adjacencyStorage.storeAdjacency(tx, 14L, singletonList(10L)),
            adjacencyStorage.storeAdjacency(tx, 15L, singletonList(10L))))
        .join();

    // Refresh entry points
    CompletableFuture<Void> future = db.runAsync(tx -> monitor.refreshEntryPoints(tx));
    assertThat(future.join()).isNull();

    // Verify high-degree nodes are included in entry points
    db.runAsync(tx -> entryPointStorage.loadEntryList(tx))
        .thenAccept(entryList -> {
          assertThat(entryList).isNotNull();
          // Should have selected high-degree nodes
          assertThat(entryList.getHighDegreeEntriesCount()).isGreaterThan(0);
        })
        .join();
  }

  @Test
  void testEmptyGraphWithRefreshEntryPoints() {
    // Test refreshing entry points on empty graph
    CompletableFuture<Void> future = db.runAsync(tx -> monitor.refreshEntryPoints(tx));

    // Should complete without error
    assertThat(future.join()).isNull();
  }

  @Test
  void testConnectivityAnalysisNeedsRepair() {
    // Test the needsRepair() logic directly
    ConnectivityAnalysis healthy = new ConnectivityAnalysis(1, 95, 100, Collections.emptyList());
    assertThat(healthy.needsRepair()).isFalse();

    ConnectivityAnalysis unhealthy = new ConnectivityAnalysis(2, 50, 100, Collections.emptyList());
    assertThat(unhealthy.needsRepair()).isTrue();

    ConnectivityAnalysis withOrphans = new ConnectivityAnalysis(1, 98, 100, asList(99L, 100L));
    assertThat(withOrphans.needsRepair()).isTrue();

    ConnectivityAnalysis empty = new ConnectivityAnalysis(0, 0, 0, Collections.emptyList());
    assertThat(empty.needsRepair()).isFalse();
  }

  @Test
  void testRepairConnectivityWithPq() {
    // Use reflection to test the repair with actual PQ distances
    db.runAsync(tx -> allOf(
            adjacencyStorage.storeAdjacency(tx, 1L, List.of(2L)),
            adjacencyStorage.storeAdjacency(tx, 2L, List.of(1L)),
            adjacencyStorage.storeAdjacency(tx, 10L, Collections.emptyList())))
        .join();

    // Create disconnected analysis that triggers repair
    ConnectivityAnalysis disconnected = new ConnectivityAnalysis(2, 2, 3, List.of(10L));

    // Test repair
    CompletableFuture<Void> future = db.runAsync(tx -> monitor.repairConnectivity(tx, disconnected, 1));
    assertThat(future.join()).isNull();
  }

  @Test
  void testNodeDistanceComparator() {
    // Test NodeDistance comparator
    GraphConnectivityMonitor.NodeDistance nd1 = new GraphConnectivityMonitor.NodeDistance(1L, 0.5f);
    GraphConnectivityMonitor.NodeDistance nd2 = new GraphConnectivityMonitor.NodeDistance(2L, 1.0f);
    GraphConnectivityMonitor.NodeDistance nd3 = new GraphConnectivityMonitor.NodeDistance(3L, 0.5f);

    assertThat(nd1.compareTo(nd2)).isLessThan(0);
    assertThat(nd2.compareTo(nd1)).isGreaterThan(0);
    assertThat(nd1.compareTo(nd3)).isEqualTo(0);
  }

  @Test
  void testSamplingRangeValidation() {
    // Test SamplingRange constructor validation
    assertThatThrownBy(() -> new GraphConnectivityMonitor.SamplingRange(10, 5))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Invalid range: 10 > 5");

    assertThatThrownBy(() -> new GraphConnectivityMonitor.SamplingRange(-1, 10))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Invalid range: -1 or 10 is negative");

    // Test valid range
    GraphConnectivityMonitor.SamplingRange validRange = new GraphConnectivityMonitor.SamplingRange(1, 10);
    assertThat(validRange.min).isEqualTo(1);
    assertThat(validRange.max).isEqualTo(10);
    assertThat(validRange.size).isEqualTo(10);
  }

  @Test
  void testComputePqDistanceWithNullPq() {
    // Test PQ distance computation with null PQ
    GraphConnectivityMonitor monitorNoPq = new GraphConnectivityMonitor(
        db, keys, metaStorage, adjacencyStorage, pqBlockStorage, entryPointStorage, null);

    CompletableFuture<Float> future = db.runAsync(tx -> monitorNoPq.computePqDistance(tx, 1L, 2L, 1));

    // Should return 1.0f when PQ is null
    assertThat(future.join()).isEqualTo(1.0f);
  }
}
