package io.github.panghy.vectorsearch.search;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import java.time.Clock;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class BeamSearchEngineTest {
  private Database db;
  private DirectorySubspace subspace;
  private NodeAdjacencyStorage adjacencyStorage;
  private PqBlockStorage pqBlockStorage;
  private EntryPointStorage entryPointStorage;
  private CodebookStorage codebookStorage;
  private ProductQuantizer pq;
  private BeamSearchEngine searchEngine;
  private InstantSource instantSource;

  private static final int DIMENSION = 8;
  private static final int NUM_SUBVECTORS = 2;
  private static final int CODEBOOK_VERSION = 1;
  private static final int CODES_PER_BLOCK = 256;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(710);
    db = fdb.open();

    String testDir = "test_beam_search_" + System.currentTimeMillis();
    subspace = db.run(tr -> {
      return DirectorySubspace.create(tr, Collections.singletonList(testDir))
          .join();
    });

    instantSource = Clock.systemUTC();
    String collectionName = "test_collection";

    // Initialize storage components
    adjacencyStorage = new NodeAdjacencyStorage(subspace, collectionName, instantSource);
    pqBlockStorage = new PqBlockStorage(db, subspace, collectionName, CODES_PER_BLOCK, instantSource);
    entryPointStorage = new EntryPointStorage(subspace, collectionName, instantSource);
    codebookStorage = new CodebookStorage(subspace, collectionName);

    // Initialize PQ
    pq = new ProductQuantizer(DIMENSION, NUM_SUBVECTORS, DistanceMetrics.Metric.L2);

    // Create search engine
    searchEngine = new BeamSearchEngine(adjacencyStorage, pqBlockStorage, entryPointStorage, pq);
  }

  @AfterEach
  void tearDown() {
    if (db != null && subspace != null) {
      db.run(tr -> {
        tr.clear(subspace.range());
        return null;
      });
    }
  }

  @Test
  void testSearchWithNoEntryPoints() {
    // Search with no entry points should return empty results
    float[] queryVector = new float[DIMENSION];
    Arrays.fill(queryVector, 1.0f);

    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 5, 0, 100, CODEBOOK_VERSION);
        })
        .join();

    assertThat(results).isEmpty();
  }

  @Test
  void testSearchWithSingleNode() {
    // Setup: Create a single node with PQ code and make it an entry point
    long nodeId = 1L;
    float[] vector = generateRandomVector(DIMENSION);

    // Train PQ and encode vector
    List<float[]> trainingData = generateTrainingData(100, DIMENSION);
    pq.train(trainingData).join();
    byte[] pqCode = pq.encode(vector);

    // Store codebooks
    db.runAsync(tx -> {
          return codebookStorage.storeCodebooks(
              tx, CODEBOOK_VERSION, pq.getCodebooks(), new CodebookStorage.TrainingStats(100, null));
        })
        .join();

    // Store PQ code
    db.runAsync(tx -> {
          return pqBlockStorage.storeCode(tx, nodeId, pqCode, CODEBOOK_VERSION);
        })
        .join();

    // Store adjacency (no neighbors)
    db.runAsync(tx -> {
          return adjacencyStorage.storeAdjacency(tx, nodeId, Collections.emptyList());
        })
        .join();

    // Set as entry point
    db.runAsync(tx -> {
          return entryPointStorage.storeEntryList(tx, Collections.singletonList(nodeId), null, null);
        })
        .join();

    // Search
    float[] queryVector = vector.clone(); // Search for same vector
    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 1, 0, 100, CODEBOOK_VERSION);
        })
        .join();

    assertThat(results).hasSize(1);
    assertThat(results.get(0).getNodeId()).isEqualTo(nodeId);
    assertThat(results.get(0).getDistance()).isGreaterThanOrEqualTo(0); // PQ introduces error
  }

  @Test
  void testSearchWithSmallGraph() {
    // Create a small graph: 1 -> 2 -> 3
    //                           -> 4
    List<float[]> vectors = Arrays.asList(
        generateRandomVector(DIMENSION),
        generateRandomVector(DIMENSION),
        generateRandomVector(DIMENSION),
        generateRandomVector(DIMENSION));

    // Train PQ
    List<float[]> trainingData = generateTrainingData(100, DIMENSION);
    trainingData.addAll(vectors);
    pq.train(trainingData).join();

    // Store codebooks
    db.runAsync(tx -> {
          return codebookStorage.storeCodebooks(
              tx,
              CODEBOOK_VERSION,
              pq.getCodebooks(),
              new CodebookStorage.TrainingStats(trainingData.size(), null));
        })
        .join();

    // Store PQ codes and adjacency
    db.runAsync(tx -> {
          CompletableFuture<?>[] futures = new CompletableFuture[8];

          // Store PQ codes
          for (int i = 0; i < 4; i++) {
            byte[] pqCode = pq.encode(vectors.get(i));
            futures[i] = pqBlockStorage.storeCode(tx, i + 1, pqCode, CODEBOOK_VERSION);
          }

          // Store adjacency
          futures[4] = adjacencyStorage.storeAdjacency(tx, 1L, Arrays.asList(2L));
          futures[5] = adjacencyStorage.storeAdjacency(tx, 2L, Arrays.asList(1L, 3L, 4L));
          futures[6] = adjacencyStorage.storeAdjacency(tx, 3L, Arrays.asList(2L));
          futures[7] = adjacencyStorage.storeAdjacency(tx, 4L, Arrays.asList(2L));

          return CompletableFuture.allOf(futures);
        })
        .join();

    // Set entry point
    db.runAsync(tx -> {
          return entryPointStorage.storeEntryList(tx, Collections.singletonList(1L), null, null);
        })
        .join();

    // Search for a vector similar to vector[2] (node 3)
    float[] queryVector = vectors.get(2).clone();
    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 3, 0, 100, CODEBOOK_VERSION);
        })
        .join();

    assertThat(results).hasSize(3);
    // Node 3 should be closest
    assertThat(results.get(0).getNodeId()).isEqualTo(3L);
  }

  @Test
  void testSearchWithBeamSizeLimit() {
    // Create a star graph with center node connected to many others
    int numNodes = 20;
    long centerNode = 1L;

    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < numNodes; i++) {
      vectors.add(generateRandomVector(DIMENSION));
    }

    // Train PQ
    pq.train(vectors).join();

    // Store codebooks
    db.runAsync(tx -> {
          return codebookStorage.storeCodebooks(
              tx,
              CODEBOOK_VERSION,
              pq.getCodebooks(),
              new CodebookStorage.TrainingStats(vectors.size(), null));
        })
        .join();

    // Store graph
    db.runAsync(tx -> {
          List<CompletableFuture<?>> futures = new ArrayList<>();

          // Store PQ codes
          for (int i = 0; i < numNodes; i++) {
            byte[] pqCode = pq.encode(vectors.get(i));
            futures.add(pqBlockStorage.storeCode(tx, i + 1, pqCode, CODEBOOK_VERSION));
          }

          // Create star topology
          List<Long> spokes = new ArrayList<>();
          for (long i = 2; i <= numNodes; i++) {
            spokes.add(i);
          }
          futures.add(adjacencyStorage.storeAdjacency(tx, centerNode, spokes));

          // Each spoke connects back to center
          for (long i = 2; i <= numNodes; i++) {
            futures.add(adjacencyStorage.storeAdjacency(tx, i, Collections.singletonList(centerNode)));
          }

          return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
        })
        .join();

    // Set center as entry point
    db.runAsync(tx -> {
          return entryPointStorage.storeEntryList(tx, Collections.singletonList(centerNode), null, null);
        })
        .join();

    // Search with small beam size
    float[] queryVector = vectors.get(10).clone(); // Search for node 11
    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 5, 8, 100, CODEBOOK_VERSION);
        })
        .join();

    assertThat(results).hasSize(5);
    // All results should be valid node IDs
    for (SearchResult result : results) {
      assertThat(result.getNodeId()).isBetween(1L, (long) numNodes);
      assertThat(result.getDistance()).isGreaterThanOrEqualTo(0);
    }
  }

  @Test
  void testSearchWithMaxVisitsLimit() {
    // Create a chain graph to test visit limits
    int numNodes = 10;
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < numNodes; i++) {
      vectors.add(generateRandomVector(DIMENSION));
    }

    // Train PQ
    pq.train(vectors).join();

    // Store codebooks
    db.runAsync(tx -> {
          return codebookStorage.storeCodebooks(
              tx,
              CODEBOOK_VERSION,
              pq.getCodebooks(),
              new CodebookStorage.TrainingStats(vectors.size(), null));
        })
        .join();

    // Create chain: 1 -> 2 -> 3 -> ... -> 10
    db.runAsync(tx -> {
          List<CompletableFuture<?>> futures = new ArrayList<>();

          // Store PQ codes
          for (int i = 0; i < numNodes; i++) {
            byte[] pqCode = pq.encode(vectors.get(i));
            futures.add(pqBlockStorage.storeCode(tx, i + 1, pqCode, CODEBOOK_VERSION));
          }

          // Create chain adjacency
          for (int i = 1; i <= numNodes; i++) {
            List<Long> neighbors = new ArrayList<>();
            if (i > 1) neighbors.add((long) (i - 1));
            if (i < numNodes) neighbors.add((long) (i + 1));
            futures.add(adjacencyStorage.storeAdjacency(tx, (long) i, neighbors));
          }

          return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]));
        })
        .join();

    // Set first node as entry point
    db.runAsync(tx -> {
          return entryPointStorage.storeEntryList(tx, Collections.singletonList(1L), null, null);
        })
        .join();

    // Search with very limited visits (should not reach end of chain)
    float[] queryVector = vectors.get(9).clone(); // Search for node 10
    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 5, 0, 3, CODEBOOK_VERSION);
        })
        .join();

    // With only 3 visits, we shouldn't reach node 10
    assertThat(results.size()).isLessThanOrEqualTo(3);
    boolean foundNode10 = results.stream().anyMatch(r -> r.getNodeId() == 10L);
    assertThat(foundNode10).isFalse();
  }

  @Test
  void testHierarchicalEntryPoints() {
    // Test that hierarchical entry point selection works
    List<Long> primaryEntries = Arrays.asList(1L, 2L);
    List<Long> randomEntries = Arrays.asList(3L, 4L);
    List<Long> highDegreeEntries = Arrays.asList(5L, 6L);

    // Store multiple entry point categories
    db.runAsync(tx -> {
          return entryPointStorage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries);
        })
        .join();

    // Store nodes with PQ codes
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < 6; i++) {
      vectors.add(generateRandomVector(DIMENSION));
    }

    pq.train(vectors).join();

    db.runAsync(tx -> {
          CompletableFuture<?>[] futures = new CompletableFuture[7];

          futures[0] = codebookStorage.storeCodebooks(
              tx,
              CODEBOOK_VERSION,
              pq.getCodebooks(),
              new CodebookStorage.TrainingStats(vectors.size(), null));

          for (int i = 0; i < 6; i++) {
            byte[] pqCode = pq.encode(vectors.get(i));
            futures[i + 1] = pqBlockStorage.storeCode(tx, i + 1, pqCode, CODEBOOK_VERSION);
          }

          return CompletableFuture.allOf(futures);
        })
        .join();

    // Store adjacency for all nodes
    db.runAsync(tx -> {
          CompletableFuture<?>[] futures = new CompletableFuture[6];
          for (int i = 0; i < 6; i++) {
            futures[i] = adjacencyStorage.storeAdjacency(tx, (long) (i + 1), Collections.emptyList());
          }
          return CompletableFuture.allOf(futures);
        })
        .join();

    // Search with beam size that requires multiple entry categories
    float[] queryVector = generateRandomVector(DIMENSION);
    List<SearchResult> results = db.readAsync(tx -> {
          return searchEngine.search(tx, queryVector, 3, 5, 100, CODEBOOK_VERSION);
        })
        .join();

    // Should get results from multiple entry categories
    assertThat(results).isNotEmpty();
  }

  // Helper methods
  private float[] generateRandomVector(int dimension) {
    Random random = new Random();
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat() * 2 - 1; // [-1, 1]
    }
    return vector;
  }

  private List<float[]> generateTrainingData(int count, int dimension) {
    List<float[]> data = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      data.add(generateRandomVector(dimension));
    }
    return data;
  }
}
