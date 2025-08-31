package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.apple.foundationdb.Transaction;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.GraphMeta;
import java.time.Clock;
import java.time.Instant;
import java.time.InstantSource;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

class GraphMetaStorageTest {
  private VectorIndexKeys keys;
  private InstantSource instantSource;
  private GraphMetaStorage storage;
  private Transaction tx;
  private byte[] metaKey;

  @BeforeEach
  void setUp() {
    keys = mock(VectorIndexKeys.class);
    instantSource = Clock.fixed(Instant.ofEpochSecond(1000000), ZoneOffset.UTC);
    storage = new GraphMetaStorage(keys, instantSource);
    tx = mock(Transaction.class);

    metaKey = new byte[] {1, 2, 3, 4};
    when(keys.graphMetaKey()).thenReturn(metaKey);
  }

  @Test
  void testStoreGraphMeta() {
    // Create a GraphMeta
    GraphMeta meta = GraphMeta.newBuilder()
        .setConnectedComponents(2)
        .setLargestComponentSize(950)
        .setTotalNodes(1000)
        .addOrphanedNodes(100)
        .addOrphanedNodes(101)
        .build();

    // Store it
    CompletableFuture<Void> future = storage.storeGraphMeta(tx, meta);

    // Verify storage
    assertThat(future).isCompletedWithValue(null);
    verify(tx).set(eq(metaKey), eq(meta.toByteArray()));
  }

  @Test
  void testLoadGraphMeta() {
    // Create expected meta
    GraphMeta expectedMeta = GraphMeta.newBuilder()
        .setConnectedComponents(1)
        .setLargestComponentSize(1000)
        .setTotalNodes(1000)
        .build();

    // Mock the transaction get
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(expectedMeta.toByteArray()));

    // Load it
    CompletableFuture<GraphMeta> future = storage.loadGraphMeta(tx);

    // Verify
    assertThat(future)
        .isCompletedWithValueMatching(meta -> meta.getConnectedComponents() == 1
            && meta.getLargestComponentSize() == 1000
            && meta.getTotalNodes() == 1000);
  }

  @Test
  void testLoadGraphMetaNotFound() {
    // Mock the transaction get to return null
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(null));

    // Load it
    CompletableFuture<GraphMeta> future = storage.loadGraphMeta(tx);

    // Verify null is returned
    assertThat(future).isCompletedWithValue(null);
  }

  @Test
  void testUpdateConnectivityStats() {
    // Mock existing meta
    GraphMeta existingMeta = GraphMeta.newBuilder().setAverageDegree(10.5).build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(existingMeta.toByteArray()));

    // Update connectivity stats
    List<Long> orphans = Arrays.asList(100L, 101L, 102L);
    CompletableFuture<Void> future = storage.updateConnectivityStats(tx, 3, 950L, 1000L, orphans);

    // Verify update
    assertThat(future).isCompletedWithValue(null);

    ArgumentCaptor<byte[]> dataCaptor = ArgumentCaptor.forClass(byte[].class);
    verify(tx).set(eq(metaKey), dataCaptor.capture());

    // Parse the stored data
    GraphMeta storedMeta;
    try {
      storedMeta = GraphMeta.parseFrom(dataCaptor.getValue());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Verify the updated fields
    assertThat(storedMeta.getConnectedComponents()).isEqualTo(3);
    assertThat(storedMeta.getLargestComponentSize()).isEqualTo(950);
    assertThat(storedMeta.getTotalNodes()).isEqualTo(1000);
    assertThat(storedMeta.getOrphanedNodesList()).containsExactly(100L, 101L, 102L);

    // Verify preserved fields
    assertThat(storedMeta.getAverageDegree()).isEqualTo(10.5);
    // Max degree is not in the current proto

    // Verify timestamp was set
    assertThat(storedMeta.hasLastAnalysisTimestamp()).isTrue();
    assertThat(storedMeta.getLastAnalysisTimestamp().getSeconds()).isEqualTo(1000000);
  }

  @Test
  void testUpdateConnectivityStatsNoExisting() {
    // Mock no existing meta
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(null));

    // Update connectivity stats
    List<Long> orphans = Arrays.asList(100L);
    CompletableFuture<Void> future = storage.updateConnectivityStats(tx, 2, 500L, 501L, orphans);

    // Verify update
    assertThat(future).isCompletedWithValue(null);

    ArgumentCaptor<byte[]> dataCaptor = ArgumentCaptor.forClass(byte[].class);
    verify(tx).set(eq(metaKey), dataCaptor.capture());

    // Parse the stored data
    GraphMeta storedMeta;
    try {
      storedMeta = GraphMeta.parseFrom(dataCaptor.getValue());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Verify the fields
    assertThat(storedMeta.getConnectedComponents()).isEqualTo(2);
    assertThat(storedMeta.getLargestComponentSize()).isEqualTo(500);
    assertThat(storedMeta.getTotalNodes()).isEqualTo(501);
    assertThat(storedMeta.getOrphanedNodesList()).containsExactly(100L);
  }

  @Test
  void testUpdateGraphStatistics() {
    // Mock existing meta
    GraphMeta existingMeta = GraphMeta.newBuilder()
        .setConnectedComponents(1)
        .setLargestComponentSize(1000)
        .build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(existingMeta.toByteArray()));

    // Update graph statistics
    CompletableFuture<Void> future = storage.updateGraphStatistics(tx, 15.5, 30, 5, 3);

    // Verify update
    assertThat(future).isCompletedWithValue(null);

    ArgumentCaptor<byte[]> dataCaptor = ArgumentCaptor.forClass(byte[].class);
    verify(tx).set(eq(metaKey), dataCaptor.capture());

    // Parse the stored data
    GraphMeta storedMeta;
    try {
      storedMeta = GraphMeta.parseFrom(dataCaptor.getValue());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Verify the updated fields
    assertThat(storedMeta.getAverageDegree()).isEqualTo(15.5);
    // Max/min degree and isolated nodes are not in the current proto

    // Verify preserved fields
    assertThat(storedMeta.getConnectedComponents()).isEqualTo(1);
    assertThat(storedMeta.getLargestComponentSize()).isEqualTo(1000);
  }

  @Test
  void testMarkRepairCompleted() {
    // Mock existing meta with orphans
    GraphMeta existingMeta = GraphMeta.newBuilder()
        .setConnectedComponents(2)
        .addOrphanedNodes(100)
        .addOrphanedNodes(101)
        .build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(existingMeta.toByteArray()));

    // Mark repair completed
    CompletableFuture<Void> future = storage.markRepairCompleted(tx, 2);

    // Verify update
    assertThat(future).isCompletedWithValue(null);

    ArgumentCaptor<byte[]> dataCaptor = ArgumentCaptor.forClass(byte[].class);
    verify(tx).set(eq(metaKey), dataCaptor.capture());

    // Parse the stored data
    GraphMeta storedMeta;
    try {
      storedMeta = GraphMeta.parseFrom(dataCaptor.getValue());
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Verify orphans were cleared
    assertThat(storedMeta.getOrphanedNodesList()).isEmpty();

    // Verify timestamp was updated (using last_analysis_timestamp for repair tracking)
    assertThat(storedMeta.hasLastAnalysisTimestamp()).isTrue();
    assertThat(storedMeta.getLastAnalysisTimestamp().getSeconds()).isEqualTo(1000000);
  }

  @Test
  void testMarkRepairCompletedNoExisting() {
    // Mock no existing meta
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(null));

    // Mark repair completed
    CompletableFuture<Void> future = storage.markRepairCompleted(tx, 2);

    // Should complete without error but do nothing
    assertThat(future).isCompletedWithValue(null);

    // Verify no set was called
    verify(tx).get(metaKey);
  }

  @Test
  void testIsAnalysisNeeded() {
    // Test with no existing meta
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(null));

    CompletableFuture<Boolean> future = storage.isAnalysisNeeded(tx, 3600);
    assertThat(future).isCompletedWithValue(true);

    // Test with recent analysis
    Timestamp recentTimestamp = Timestamp.newBuilder()
        .setSeconds(999000) // 1000 seconds ago
        .build();

    GraphMeta recentMeta =
        GraphMeta.newBuilder().setLastAnalysisTimestamp(recentTimestamp).build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(recentMeta.toByteArray()));

    future = storage.isAnalysisNeeded(tx, 3600); // 1 hour max age
    assertThat(future).isCompletedWithValue(false); // Only 1000 seconds old

    // Test with old analysis
    Timestamp oldTimestamp = Timestamp.newBuilder()
        .setSeconds(900000) // 100000 seconds ago
        .build();

    GraphMeta oldMeta =
        GraphMeta.newBuilder().setLastAnalysisTimestamp(oldTimestamp).build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(oldMeta.toByteArray()));

    future = storage.isAnalysisNeeded(tx, 3600); // 1 hour max age
    assertThat(future).isCompletedWithValue(true); // Much older than 1 hour
  }

  @Test
  void testIsGraphHealthy() {
    // Test with no existing meta
    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(null));

    CompletableFuture<Boolean> future = storage.isGraphHealthy(tx, 0.95);
    assertThat(future).isCompletedWithValue(true); // Empty graph is healthy

    // Test with healthy graph
    GraphMeta healthyMeta = GraphMeta.newBuilder()
        .setLargestComponentSize(960)
        .setTotalNodes(1000)
        .build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(healthyMeta.toByteArray()));

    future = storage.isGraphHealthy(tx, 0.95);
    assertThat(future).isCompletedWithValue(true); // 96% > 95%

    // Test with unhealthy graph
    GraphMeta unhealthyMeta = GraphMeta.newBuilder()
        .setLargestComponentSize(900)
        .setTotalNodes(1000)
        .build();

    when(tx.get(metaKey)).thenReturn(CompletableFuture.completedFuture(unhealthyMeta.toByteArray()));

    future = storage.isGraphHealthy(tx, 0.95);
    assertThat(future).isCompletedWithValue(false); // 90% < 95%
  }

  @Test
  void testClearGraphMeta() {
    CompletableFuture<Void> future = storage.clearGraphMeta(tx);

    assertThat(future).isCompletedWithValue(null);
    verify(tx).clear(metaKey);
  }
}
