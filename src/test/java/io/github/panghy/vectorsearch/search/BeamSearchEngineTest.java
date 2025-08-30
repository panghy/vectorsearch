package io.github.panghy.vectorsearch.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.apple.foundationdb.Transaction;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class BeamSearchEngineTest {

  @Mock
  private NodeAdjacencyStorage adjacencyStorage;

  @Mock
  private PqBlockStorage pqBlockStorage;

  @Mock
  private EntryPointStorage entryPointStorage;

  @Mock
  private Transaction tx;

  private ProductQuantizer pq;
  private BeamSearchEngine searchEngine;

  private static final int VECTOR_DIM = 8;
  private static final int PQ_SUBVECTORS = 4;

  @BeforeEach
  void setUp() {
    // Create a real ProductQuantizer with trained codebooks
    pq = new ProductQuantizer(VECTOR_DIM, PQ_SUBVECTORS, DistanceMetrics.Metric.L2);

    // Manually set codebooks for testing (bypassing training)
    float[][][] codebooks = new float[PQ_SUBVECTORS][256][VECTOR_DIM / PQ_SUBVECTORS];
    for (int s = 0; s < PQ_SUBVECTORS; s++) {
      for (int c = 0; c < 256; c++) {
        for (int d = 0; d < VECTOR_DIM / PQ_SUBVECTORS; d++) {
          codebooks[s][c][d] = (s * 256 + c + d) * 0.01f;
        }
      }
    }
    pq.setCodebooks(codebooks);
    searchEngine = new BeamSearchEngine(adjacencyStorage, pqBlockStorage, entryPointStorage, pq);
  }

  @Test
  void testSearchWithNoEntryPoints() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    Arrays.fill(queryVector, 1.0f);

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Collections.emptyList()));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 5, 10, 100, 1).join();

    // Then
    assertThat(results).isEmpty();
    verify(entryPointStorage).getHierarchicalEntryPoints(tx, 10);
  }

  @Test
  void testSearchWithSingleNodeNoNeighbors() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    Arrays.fill(queryVector, 1.0f);
    long nodeId = 42L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(nodeId)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    for (int i = 0; i < PQ_SUBVECTORS; i++) {
      pqCode[i] = (byte) i;
    }

    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(nodeId)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    when(adjacencyStorage.loadAdjacency(any(), eq(nodeId))).thenReturn(CompletableFuture.completedFuture(null));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 5, 10, 100, 1).join();

    // Then
    assertThat(results).hasSize(1);
    assertThat(results.get(0).getNodeId()).isEqualTo(nodeId);
    assertThat(results.get(0).getDistance()).isGreaterThanOrEqualTo(0);
  }

  @Test
  void testSearchWithMultipleNodesAndNeighbors() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    Arrays.fill(queryVector, 1.0f);

    long node1 = 10L;
    long node2 = 20L;
    long node3 = 30L;

    // Entry points
    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(node1)));

    // PQ codes for different nodes
    byte[] pqCode1 = new byte[PQ_SUBVECTORS];
    byte[] pqCode2 = new byte[PQ_SUBVECTORS];
    byte[] pqCode3 = new byte[PQ_SUBVECTORS];
    for (int i = 0; i < PQ_SUBVECTORS; i++) {
      pqCode1[i] = (byte) (i * 1);
      pqCode2[i] = (byte) (i * 2);
      pqCode3[i] = (byte) (i * 3);
    }

    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node1)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode1)));
    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node2, node3)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode2, pqCode3)));

    // Adjacency lists
    NodeAdjacency adjacency1 = NodeAdjacency.newBuilder()
        .setNodeId(node1)
        .addAllNeighbors(Arrays.asList(node2, node3))
        .build();

    NodeAdjacency adjacency2 = NodeAdjacency.newBuilder().setNodeId(node2).build();

    NodeAdjacency adjacency3 = NodeAdjacency.newBuilder().setNodeId(node3).build();

    when(adjacencyStorage.loadAdjacency(any(), eq(node1)))
        .thenReturn(CompletableFuture.completedFuture(adjacency1));
    when(adjacencyStorage.loadAdjacency(any(), eq(node2)))
        .thenReturn(CompletableFuture.completedFuture(adjacency2));
    when(adjacencyStorage.loadAdjacency(any(), eq(node3)))
        .thenReturn(CompletableFuture.completedFuture(adjacency3));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 2, 10, 100, 1).join();

    // Then
    assertThat(results).hasSize(2);
    assertThat(results.get(0).getDistance())
        .isLessThanOrEqualTo(results.get(1).getDistance());

    // Verify all nodes were explored
    verify(adjacencyStorage).loadAdjacency(any(), eq(node1));
    verify(adjacencyStorage).loadAdjacency(any(), eq(node2));
  }

  @Test
  void testSearchWithVisitLimit() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    Arrays.fill(queryVector, 1.0f);

    long node1 = 1L;
    long node2 = 2L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(node1)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node1)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));
    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node2)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    NodeAdjacency adjacency1 =
        NodeAdjacency.newBuilder().setNodeId(node1).addNeighbors(node2).build();

    when(adjacencyStorage.loadAdjacency(any(), eq(node1)))
        .thenReturn(CompletableFuture.completedFuture(adjacency1));

    // When - search with visit limit of 1
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 5, 10, 1, 1).join();

    // Then
    assertThat(results).hasSizeLessThanOrEqualTo(1);
    verify(adjacencyStorage, times(1)).loadAdjacency(any(), anyLong());
  }

  @Test
  void testSearchDefaultBeamSize() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    long nodeId = 100L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), eq(16)))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(nodeId)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    when(adjacencyStorage.loadAdjacency(any(), anyLong())).thenReturn(CompletableFuture.completedFuture(null));

    // When - search with searchListSize = 0 (should default to max(16, topK))
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 5, 0, 100, 1).join();

    // Then
    assertThat(results).isNotNull();
    verify(entryPointStorage).getHierarchicalEntryPoints(tx, 16); // max(16, 5) = 16
  }

  @Test
  void testSearchWithLargeTopK() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    long nodeId = 200L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), eq(50)))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(nodeId)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    when(adjacencyStorage.loadAdjacency(any(), anyLong())).thenReturn(CompletableFuture.completedFuture(null));

    // When - search with topK = 50, searchListSize = 0
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 50, 0, 100, 1).join();

    // Then
    assertThat(results).isNotNull();
    verify(entryPointStorage).getHierarchicalEntryPoints(tx, 50); // max(16, 50) = 50
  }

  @Test
  void testSearchWithNullPqCode() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    long node1 = 1L;
    long node2 = 2L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(node1)));

    // Node1 has valid PQ code, node2 has null (missing)
    byte[] pqCode1 = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node1)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode1)));
    when(pqBlockStorage.batchLoadPqCodes(eq(Arrays.asList(node2)), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList((byte[]) null)));

    NodeAdjacency adjacency =
        NodeAdjacency.newBuilder().setNodeId(node1).addNeighbors(node2).build();

    when(adjacencyStorage.loadAdjacency(any(), eq(node1))).thenReturn(CompletableFuture.completedFuture(adjacency));
    when(adjacencyStorage.loadAdjacency(any(), eq(node2))).thenReturn(CompletableFuture.completedFuture(null));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 2, 10, 100, 1).join();

    // Then
    assertThat(results).hasSize(2);
    // Node with null PQ code should have MAX_VALUE distance
    SearchResult nodeWithNullCode =
        results.stream().filter(r -> r.getNodeId() == node2).findFirst().orElse(null);
    assertThat(nodeWithNullCode).isNotNull();
    assertThat(nodeWithNullCode.getDistance()).isEqualTo(Float.MAX_VALUE);
  }

  @Test
  void testSearchWithCyclicGraph() {
    // Given - create a cycle: node1 -> node2 -> node3 -> node1
    float[] queryVector = new float[VECTOR_DIM];
    long node1 = 1L;
    long node2 = 2L;
    long node3 = 3L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(node1)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    NodeAdjacency adjacency1 =
        NodeAdjacency.newBuilder().setNodeId(node1).addNeighbors(node2).build();

    NodeAdjacency adjacency2 =
        NodeAdjacency.newBuilder().setNodeId(node2).addNeighbors(node3).build();

    NodeAdjacency adjacency3 = NodeAdjacency.newBuilder()
        .setNodeId(node3)
        .addNeighbors(node1) // Cycle back to node1
        .build();

    when(adjacencyStorage.loadAdjacency(any(), eq(node1)))
        .thenReturn(CompletableFuture.completedFuture(adjacency1));
    when(adjacencyStorage.loadAdjacency(any(), eq(node2)))
        .thenReturn(CompletableFuture.completedFuture(adjacency2));
    when(adjacencyStorage.loadAdjacency(any(), eq(node3)))
        .thenReturn(CompletableFuture.completedFuture(adjacency3));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 3, 10, 100, 1).join();

    // Then - should handle cycle correctly without infinite loop
    assertThat(results).hasSize(3);
    assertThat(results.stream().map(SearchResult::getNodeId)).containsExactlyInAnyOrder(node1, node2, node3);
  }

  @Test
  void testSearchWithEmptyNeighborsList() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    long nodeId = 42L;

    when(entryPointStorage.getHierarchicalEntryPoints(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(nodeId)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    // Adjacency with empty neighbors list
    NodeAdjacency adjacency = NodeAdjacency.newBuilder().setNodeId(nodeId).build();

    when(adjacencyStorage.loadAdjacency(any(), eq(nodeId)))
        .thenReturn(CompletableFuture.completedFuture(adjacency));

    // When
    List<SearchResult> results =
        searchEngine.search(tx, queryVector, 5, 10, 100, 1).join();

    // Then
    assertThat(results).hasSize(1);
    assertThat(results.get(0).getNodeId()).isEqualTo(nodeId);
  }

  @Test
  void testSearchBeamSizeEnforcement() {
    // Given
    float[] queryVector = new float[VECTOR_DIM];
    int topK = 20;
    int searchListSize = 10; // Smaller than topK

    long nodeId = 1L;
    when(entryPointStorage.getHierarchicalEntryPoints(any(), eq(20))) // Should use max(10, 20) = 20
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(nodeId)));

    byte[] pqCode = new byte[PQ_SUBVECTORS];
    when(pqBlockStorage.batchLoadPqCodes(any(), anyInt()))
        .thenReturn(CompletableFuture.completedFuture(Arrays.asList(pqCode)));

    when(adjacencyStorage.loadAdjacency(any(), anyLong())).thenReturn(CompletableFuture.completedFuture(null));

    // When
    List<SearchResult> results = searchEngine
        .search(tx, queryVector, topK, searchListSize, 100, 1)
        .join();

    // Then
    assertThat(results).isNotNull();
    verify(entryPointStorage).getHierarchicalEntryPoints(tx, 20); // Beam size should be max(searchListSize, topK)
  }
}
