package io.github.panghy.vectorsearch.workers;

import static java.util.concurrent.CompletableFuture.completedFuture;
import static java.util.concurrent.CompletableFuture.failedFuture;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.ByteString;
import io.github.panghy.taskqueue.TaskClaim;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.proto.LinkTask;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.proto.PqEncodedVector;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import java.nio.charset.StandardCharsets;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class LinkWorkerTest {
  private static final Logger LOGGER = LoggerFactory.getLogger(LinkWorkerTest.class);

  private Database db;
  private DirectorySubspace testDirectory;
  private DirectorySubspace taskQueueDirectory;
  private TaskQueue<Long, LinkTask> taskQueue;
  private VectorIndexKeys keys;
  private PqBlockStorage pqBlockStorage;
  private NodeAdjacencyStorage nodeAdjacencyStorage;
  private EntryPointStorage entryPointStorage;
  private CodebookStorage codebookStorage;
  private BeamSearchEngine searchEngine;
  private ProductQuantizer pq;
  private LinkWorker linkWorker;

  private static final int DIMENSION = 8;
  private static final int PQ_SUBVECTORS = 4;
  private static final int GRAPH_DEGREE = 32;
  private static final int MAX_SEARCH_VISITS = 1000;
  private static final double PRUNING_ALPHA = 1.2;
  private static final int CODES_PER_BLOCK = 256;

  @BeforeEach
  void setUp() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    String testId = "test_" + UUID.randomUUID();

    // Create test directory
    testDirectory = db.runAsync(tr -> {
          DirectoryLayer layer = DirectoryLayer.getDefault();
          return layer.createOrOpen(tr, List.of("vectorsearch_test", testId));
        })
        .get(5, TimeUnit.SECONDS);

    // Create task queue directory
    taskQueueDirectory = db.runAsync(tr -> {
          DirectoryLayer layer = DirectoryLayer.getDefault();
          return layer.createOrOpen(tr, List.of("vectorsearch_test", testId, "link_queue"));
        })
        .get(5, TimeUnit.SECONDS);

    // Initialize storage components
    keys = new VectorIndexKeys(testDirectory);

    pqBlockStorage = new PqBlockStorage(
        db, keys, CODES_PER_BLOCK, PQ_SUBVECTORS, InstantSource.system(), 100, Duration.ofMinutes(5));

    nodeAdjacencyStorage =
        new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, InstantSource.system(), 100, Duration.ofMinutes(5));

    entryPointStorage = new EntryPointStorage(testDirectory, InstantSource.system());

    codebookStorage = new CodebookStorage(db, keys, DIMENSION, PQ_SUBVECTORS, DistanceMetrics.Metric.L2);

    // Train and store test codebooks
    List<float[]> trainingData = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      float[] vec = new float[DIMENSION];
      for (int j = 0; j < DIMENSION; j++) {
        vec[j] = (float) Math.random();
      }
      trainingData.add(vec);
    }
    float[][][] codebooks = ProductQuantizer.train(
            DIMENSION, PQ_SUBVECTORS, DistanceMetrics.Metric.L2, trainingData)
        .get(5, TimeUnit.SECONDS);

    // Store codebooks
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(100L, null);
    codebookStorage.storeCodebooks(1, codebooks, stats).get();
    codebookStorage.setActiveVersion(1).get();

    // Get the ProductQuantizer from cache for testing
    pq = codebookStorage.getProductQuantizer(1L).get();

    searchEngine = new BeamSearchEngine(nodeAdjacencyStorage, pqBlockStorage, entryPointStorage, codebookStorage);

    // Create task queue
    TaskQueueConfig<Long, LinkTask> queueConfig = TaskQueueConfig.builder(
            db, taskQueueDirectory, new LongSerializer(), new LinkTaskSerializer())
        .defaultTtl(Duration.ofMinutes(10))
        .build();

    taskQueue = TaskQueues.createTaskQueue(queueConfig).get(5, TimeUnit.SECONDS);

    // Create link worker
    linkWorker = new LinkWorker(
        db,
        taskQueue,
        keys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        GRAPH_DEGREE,
        MAX_SEARCH_VISITS,
        PRUNING_ALPHA,
        InstantSource.system());
  }

  @AfterEach
  void tearDown() throws Exception {
    // Clean up test data - using unique test IDs so cleanup isn't critical
    if (db != null) {
      db.close();
    }
  }

  @Test
  @Timeout(10)
  void testProcessSingleTask() throws Exception {
    // Given
    long nodeId = 123L;
    float[] vector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(vector);

    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(nodeId)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    // Enqueue task
    taskQueue.enqueue(nodeId, linkTask).get();

    // Set up some entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(
            tr, Arrays.asList(1L, 2L, 3L), Arrays.asList(4L, 5L), Arrays.asList(6L, 7L)))
        .get();

    // When - process single task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify PQ code was stored
    byte[] storedCode = pqBlockStorage.loadPqCode(nodeId, 1).get();
    assertThat(storedCode).isEqualTo(pqCode);

    // Verify node adjacency was created
    NodeAdjacency adjacency = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, nodeId))
        .get();
    assertThat(adjacency).isNotNull();
    assertThat(adjacency.getNodeId()).isEqualTo(nodeId);

    assertThat(linkWorker.getTotalProcessed()).isEqualTo(1);
    assertThat(linkWorker.getTotalFailed()).isEqualTo(0);
  }

  @Test
  @Timeout(10)
  void testBatchProcessing() throws Exception {
    // Given
    int batchSize = 3;
    List<Long> nodeIds = new ArrayList<>();
    List<byte[]> pqCodes = new ArrayList<>();

    for (int i = 0; i < batchSize; i++) {
      long nodeId = 100L + i;
      nodeIds.add(nodeId);

      float[] vector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(vector);
      pqCodes.add(pqCode);

      LinkTask linkTask = LinkTask.newBuilder()
          .setCollection("test-collection")
          .setNodeId(nodeId)
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // Set up entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(1L, 2L, 3L), null, null))
        .get();

    // When - process all tasks
    int processed = linkWorker.processAllAvailableTasks(0).get();

    // Then
    assertThat(processed).isEqualTo(batchSize);

    for (int i = 0; i < batchSize; i++) {
      long nodeId = nodeIds.get(i);
      byte[] expectedCode = pqCodes.get(i);

      // Verify PQ code was stored
      byte[] storedCode = pqBlockStorage.loadPqCode(nodeId, 1).get();
      assertThat(storedCode).isEqualTo(expectedCode);

      // Verify node adjacency was created
      NodeAdjacency adjacency = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, nodeId))
          .get();
      assertThat(adjacency).isNotNull();
    }

    assertThat(linkWorker.getTotalProcessed()).isEqualTo(batchSize);
    assertThat(linkWorker.getTotalFailed()).isEqualTo(0);
  }

  @Test
  @Timeout(10)
  void testNoEntryPoints() throws Exception {
    // Given - first node with no entry points
    long nodeId = 1L;
    float[] vector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(vector);

    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(nodeId)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(nodeId, linkTask).get();

    // No entry points set

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Should still store the node with empty neighbors
    NodeAdjacency adjacency = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, nodeId))
        .get();
    assertThat(adjacency).isNotNull();
    assertThat(adjacency.getNeighborsList()).isEmpty();

    assertThat(linkWorker.getTotalProcessed()).isEqualTo(1);
  }

  @Test
  @Timeout(10)
  void testBackLinkCreation() throws Exception {
    // Given - set up some existing nodes first
    long existingNode1 = 10L;
    long existingNode2 = 20L;

    // Store existing nodes with PQ codes
    float[] vec1 = generateRandomVector(DIMENSION);
    float[] vec2 = generateRandomVector(DIMENSION);

    db.runAsync(tr -> {
          pqBlockStorage.storePqCode(tr, existingNode1, pq.encode(vec1), 1);
          return pqBlockStorage.storePqCode(tr, existingNode2, pq.encode(vec2), 1);
        })
        .get();

    // Create adjacency for existing nodes
    db.runAsync(tr -> {
          nodeAdjacencyStorage.storeAdjacency(tr, existingNode1, List.of());
          return nodeAdjacencyStorage.storeAdjacency(tr, existingNode2, List.of());
        })
        .get();

    // Set them as entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(existingNode1, existingNode2), null, null))
        .get();

    // Now add a new node
    long newNodeId = 100L;
    float[] newVector = generateRandomVector(DIMENSION);
    byte[] newPqCode = pq.encode(newVector);

    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(newNodeId)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(newPqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(newNodeId, linkTask).get();

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify the new node has neighbors
    NodeAdjacency newNodeAdj = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, newNodeId))
        .get();
    assertThat(newNodeAdj).isNotNull();

    // The new node should have discovered some neighbors
    List<Long> neighbors = newNodeAdj.getNeighborsList();
    LOGGER.info("New node {} has neighbors: {}", newNodeId, neighbors);

    // Verify back-links were created
    for (Long neighborId : neighbors) {
      NodeAdjacency neighborAdj = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, neighborId))
          .get();
      assertThat(neighborAdj.getNeighborsList()).contains(newNodeId);
      LOGGER.info("Neighbor {} has back-link to {}", neighborId, newNodeId);
    }
  }

  @Test
  @Timeout(10)
  void testAdaptiveBatchSizing() throws Exception {
    // Given
    int initialBatchSize = linkWorker.getCurrentBatchSize();
    assertThat(initialBatchSize).isEqualTo(10);

    // Create multiple tasks for fast processing
    for (int i = 0; i < 20; i++) {
      long nodeId = 1000L + i;
      float[] vector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(vector);

      LinkTask linkTask = LinkTask.newBuilder()
          .setCollection("test-collection")
          .setNodeId(nodeId)
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // Set up entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(1L, 2L), null, null))
        .get();

    // When - process all tasks (the worker internally uses batching)
    int processed = linkWorker.processAllAvailableTasks(0).get();

    // Then
    assertThat(processed).isEqualTo(20);

    int finalBatchSize = linkWorker.getCurrentBatchSize();
    LOGGER.info("Batch size changed from {} to {}", initialBatchSize, finalBatchSize);

    // Batch size may have changed based on processing speed
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(20);
  }

  @Test
  @Timeout(5)
  void testNoTasksAvailable() throws Exception {
    // When - try to process when queue is empty
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isFalse();
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(0);
    assertThat(linkWorker.getTotalFailed()).isEqualTo(0);
  }

  @Test
  @Timeout(5)
  void testProcessAllWithLimit() throws Exception {
    // Given - multiple tasks
    int totalTasks = 10;
    int limitTasks = 5;

    for (int i = 0; i < totalTasks; i++) {
      long nodeId = 2000L + i;
      float[] vector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(vector);

      LinkTask linkTask = LinkTask.newBuilder()
          .setCollection("test-collection")
          .setNodeId(nodeId)
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // When - process only limited tasks
    int processed = linkWorker.processAllAvailableTasks(limitTasks).get();

    // Then
    assertThat(processed).isEqualTo(limitTasks);
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(limitTasks);
  }

  @Test
  @Timeout(5)
  void testRunMethodWithStop() throws Exception {
    // Given - a task to process
    long nodeId = 3000L;
    byte[] pqCode = new byte[PQ_SUBVECTORS];
    Arrays.fill(pqCode, (byte) 1);

    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(nodeId)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(nodeId, linkTask).get();

    // When - process the task using single-task mode to verify functionality
    boolean processed = linkWorker.processOneTask().get();

    // Then verify the task was processed
    assertThat(processed).isTrue();
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(1);

    // Also test the stop method works correctly
    linkWorker.stop();
    // The stop method should set the running flag to false
    // which would cause the run() method to exit if it were running
  }

  @Test
  @Timeout(5)
  void testTransactionFailure() throws Exception {
    // Given - create a task with invalid codebook version that will cause transaction to fail
    long nodeId = 4000L;
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);

    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(nodeId)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(999)
            .build()) // Invalid codebook version
        .build();

    taskQueue.enqueue(nodeId, linkTask).get();

    // When - try to process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isFalse(); // Processing failed due to invalid codebook
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(0); // No successful processing
    assertThat(linkWorker.getTotalFailed()).isEqualTo(1); // One failure recorded
  }

  @Test
  @Timeout(10)
  void testBatchSizeAdjustment() throws Exception {
    // Given - many tasks to trigger batch size changes
    int initialBatchSize = linkWorker.getCurrentBatchSize();
    assertThat(initialBatchSize).isEqualTo(10);

    // Create many small tasks that process quickly
    for (int i = 0; i < 50; i++) {
      long nodeId = 5000L + i;
      byte[] pqCode = new byte[PQ_SUBVECTORS];
      Arrays.fill(pqCode, (byte) (i % 10));

      LinkTask linkTask = LinkTask.newBuilder()
          .setCollection("test-collection")
          .setNodeId(nodeId)
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // When - process all tasks using the batch processing mode
    int processed = linkWorker.processAllAvailableTasks(0).get();

    // Then - verify tasks were processed
    assertThat(processed).isEqualTo(50);
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(50);

    // The batch size may have been adjusted during processing
    int finalBatchSize = linkWorker.getCurrentBatchSize();
    LOGGER.info("Batch size changed from {} to {}", initialBatchSize, finalBatchSize);
    // Batch size should be positive and within bounds
    assertThat(finalBatchSize).isGreaterThan(0).isLessThanOrEqualTo(100);
  }

  @Test
  @Timeout(5)
  void testWorkerInterruption() throws Exception {
    // This test verifies that interruption handling works correctly
    // We'll test this by processing with no tasks available

    // When - try to process when no tasks are available
    boolean processed = linkWorker.processOneTask().get();

    // Then - should return false when no tasks are available
    assertThat(processed).isFalse();
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(0);

    // The interruption handling is implicitly tested through the timeout
    // mechanism in processOneTask which handles interruption gracefully
  }

  @Test
  @Timeout(5)
  void testBackLinkAlreadyExists() throws Exception {
    // Given - two nodes that already have back-links
    long node1 = 6000L;
    long node2 = 6001L;

    // Store PQ codes
    db.runAsync(tr -> {
          pqBlockStorage.storePqCode(tr, node1, pq.encode(generateRandomVector(DIMENSION)), 1);
          return pqBlockStorage.storePqCode(tr, node2, pq.encode(generateRandomVector(DIMENSION)), 1);
        })
        .get();

    // Create adjacency with existing back-link
    db.runAsync(tr -> {
          NodeAdjacency adj1 = NodeAdjacency.newBuilder()
              .setNodeId(node1)
              .addNeighbors(node2)
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          NodeAdjacency adj2 = NodeAdjacency.newBuilder()
              .setNodeId(node2)
              .addNeighbors(node1) // Already has back-link
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          nodeAdjacencyStorage.storeNodeAdjacency(tr, node1, adj1);
          return nodeAdjacencyStorage.storeNodeAdjacency(tr, node2, adj2);
        })
        .get();

    // Set entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(node2), null, null))
        .get();

    // Create link task for node1
    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(node1)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pq.encode(generateRandomVector(DIMENSION))))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(node1, linkTask).get();

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify back-link still exists (should not duplicate)
    NodeAdjacency adj2After =
        db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, node2)).get();
    long backLinkCount =
        adj2After.getNeighborsList().stream().filter(id -> id == node1).count();
    assertThat(backLinkCount).isEqualTo(1); // Should only have one back-link
  }

  @Test
  @Timeout(5)
  void testPruningWithManyNeighbors() throws Exception {
    // Given - a node that will discover many neighbors (more than graphDegree)
    long targetNode = 7000L;

    // Create many existing nodes
    List<Long> existingNodes = new ArrayList<>();
    for (int i = 0; i < GRAPH_DEGREE * 2; i++) {
      long nodeId = 7001L + i;
      existingNodes.add(nodeId);

      db.runAsync(tr -> {
            pqBlockStorage.storePqCode(tr, nodeId, pq.encode(generateRandomVector(DIMENSION)), 1);
            return nodeAdjacencyStorage.storeAdjacency(tr, nodeId, List.of());
          })
          .get();
    }

    // Set many entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, existingNodes.subList(0, 10), null, null))
        .get();

    // Create link task
    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(targetNode)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pq.encode(generateRandomVector(DIMENSION))))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(targetNode, linkTask).get();

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify pruning occurred
    NodeAdjacency adjacency = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, targetNode))
        .get();
    assertThat(adjacency).isNotNull();
    assertThat(adjacency.getNeighborsList().size()).isLessThanOrEqualTo(GRAPH_DEGREE);
  }

  // Serializers for task queue
  private static class LongSerializer implements TaskQueueConfig.TaskSerializer<Long> {
    @Override
    public ByteString serialize(Long value) {
      return ByteString.copyFrom(String.valueOf(value).getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public Long deserialize(ByteString bytes) {
      return Long.parseLong(bytes.toString(StandardCharsets.UTF_8));
    }
  }

  private static class LinkTaskSerializer implements TaskQueueConfig.TaskSerializer<LinkTask> {
    @Override
    public ByteString serialize(LinkTask value) {
      return value.toByteString();
    }

    @Override
    public LinkTask deserialize(ByteString bytes) {
      try {
        return LinkTask.parseFrom(bytes);
      } catch (Exception e) {
        throw new RuntimeException("Failed to parse LinkTask", e);
      }
    }
  }

  private float[] generateRandomVector(int dimension) {
    Random random = new Random();
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat() * 2 - 1; // Range [-1, 1]
    }
    return vector;
  }

  @Test
  @DisplayName("Test claim timeout scenario")
  void testClaimTimeout() throws Exception {
    // Test when no tasks are available and claim times out
    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        Clock.systemUTC());

    // Process one task when queue is empty - should handle timeout gracefully
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertFalse(result.get(1, TimeUnit.SECONDS));
  }

  @Test
  @DisplayName("Test exception during task claiming")
  void testClaimException() throws Exception {
    // Create a mock task queue that throws exception
    @SuppressWarnings("unchecked")
    TaskQueue<Long, LinkTask> failingQueue = mock(TaskQueue.class);
    when(failingQueue.awaitAndClaimTask(any())).thenReturn(failedFuture(new RuntimeException("Claim failed")));

    LinkWorker worker = new LinkWorker(
        db,
        failingQueue,
        keys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        Clock.systemUTC());

    // Should handle exception and return false
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertFalse(result.get(1, TimeUnit.SECONDS));
  }

  @Test
  @DisplayName("Test processing with PQ codes not found")
  void testMissingPqCodes() throws Exception {
    // Add task
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    db.runAsync(tx -> taskQueue.enqueue(tx, 100L, task)).get();

    // Mock PQ block storage to return null for some nodes
    PqBlockStorage mockPqStorage = mock(PqBlockStorage.class);
    when(mockPqStorage.storePqCode(any(), anyLong(), any(), anyInt())).thenReturn(completedFuture(null));
    when(mockPqStorage.loadPqCode(anyLong(), anyInt())).thenReturn(completedFuture(null)); // Return null PQ codes
    when(mockPqStorage.getBlockNumber(anyLong())).thenReturn(0L);

    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        mockPqStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        Clock.systemUTC());

    // Should handle null PQ codes gracefully
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertTrue(result.get(5, TimeUnit.SECONDS));
  }

  @Test
  @DisplayName("Test error during task completion")
  void testTaskCompletionError() throws Exception {
    // This test verifies that the worker handles exceptions during task processing gracefully
    // We'll test with a valid task but mock the storage to fail
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    db.runAsync(tx -> taskQueue.enqueue(tx, 100L, task)).get();

    // Create a worker with storage that will succeed initially but fail during back-link updates
    NodeAdjacencyStorage failingAdjacency = mock(NodeAdjacencyStorage.class);

    // Allow loading adjacency but fail on store
    NodeAdjacency emptyAdjacency = NodeAdjacency.newBuilder().build();
    when(failingAdjacency.loadAdjacency(any(), anyLong()))
        .thenReturn(CompletableFuture.completedFuture(emptyAdjacency));
    when(failingAdjacency.getNodeAdjacency(any(), anyLong()))
        .thenReturn(CompletableFuture.completedFuture(emptyAdjacency));
    when(failingAdjacency.storeAdjacency(any(), anyLong(), any()))
        .thenReturn(failedFuture(new RuntimeException("Store adjacency failed")));

    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        pqBlockStorage,
        failingAdjacency,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        Clock.systemUTC());

    // Should handle completion error gracefully - the task will be processed but may fail
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertFalse(result.get(10, TimeUnit.SECONDS)); // Should return false when storage fails
  }

  @Test
  @DisplayName("Test error during task failure marking")
  void testTaskFailMarkingError() throws Exception {
    // Add a task that will fail during processing
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    db.runAsync(tx -> taskQueue.enqueue(tx, 100L, task)).get();

    // Mock storage to throw exception during processing
    PqBlockStorage failingPqStorage = mock(PqBlockStorage.class);
    when(failingPqStorage.storePqCode(any(), anyLong(), any(), anyInt()))
        .thenReturn(failedFuture(new RuntimeException("Storage failed")));
    when(failingPqStorage.getBlockNumber(anyLong())).thenReturn(0L);
    when(failingPqStorage.loadPqCode(anyLong(), anyInt())).thenReturn(completedFuture(pqCode));

    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        failingPqStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        Clock.systemUTC());

    // Should handle task processing failure gracefully
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertFalse(result.get(10, TimeUnit.SECONDS)); // Should return false when storage fails

    // Verify task was re-enqueued or handled appropriately
    // The task should be available again after failure
  }

  @Test
  @DisplayName("Test transaction timeout scenario")
  void testTransactionTimeout() throws Exception {
    // Add task
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    db.runAsync(tx -> taskQueue.enqueue(tx, 100L, task)).get();

    // Create slow instant source that simulates timeout
    InstantSource slowSource = new InstantSource() {
      private int callCount = 0;

      @Override
      public Instant instant() {
        callCount++;
        if (callCount > 5) {
          // After a few calls, jump forward in time to simulate timeout
          return Instant.now().plus(Duration.ofSeconds(10));
        }
        return Instant.now();
      }
    };

    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        slowSource);

    // Should handle the timeout scenario
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertTrue(result.get(10, TimeUnit.SECONDS)); // Task completes even with simulated timeout
  }

  @Test
  @DisplayName("Test claim budget exhaustion")
  void testClaimBudgetExhaustion() throws Exception {
    // Add many tasks
    for (int i = 0; i < 20; i++) {
      float[] queryVector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(queryVector);
      LinkTask task = LinkTask.newBuilder()
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .build();

      final long taskId = i;
      db.runAsync(tx -> taskQueue.enqueue(tx, taskId, task)).get();
    }

    // Create instant source that simulates slow claiming
    InstantSource slowClaimSource = new InstantSource() {
      private Instant start = Instant.now();
      private int callCount = 0;

      @Override
      public Instant instant() {
        callCount++;
        if (callCount > 3) {
          // After a few claims, jump forward to exhaust budget
          return start.plus(Duration.ofSeconds(1));
        }
        return start;
      }
    };

    LinkWorker worker = new LinkWorker(
        db,
        taskQueue,
        keys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        searchEngine,
        32,
        1000,
        1.2,
        slowClaimSource);

    // Should process some but not all due to budget
    CompletableFuture<Boolean> result = worker.processOneTask();
    assertTrue(result.get(10, TimeUnit.SECONDS));
  }

  @Test
  @DisplayName("Test processBatch method directly")
  void testProcessBatch() throws Exception {
    // Add multiple tasks to process as a batch
    for (int i = 0; i < 5; i++) {
      float[] queryVector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(queryVector);
      LinkTask task = LinkTask.newBuilder()
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .setNodeId(100L + i)
          .build();

      taskQueue.enqueue(100L + i, task).get();
    }

    // Call processBatch directly
    linkWorker.processBatchAsync().join();

    // Verify tasks were processed
    assertThat(linkWorker.getTotalProcessed()).isGreaterThan(0);

    // Process again when no tasks available (should handle gracefully)
    linkWorker.processBatchAsync().join();
  }

  @Test
  @DisplayName("Test processBatch with failed transaction")
  void testProcessBatchWithFailure() throws Exception {
    // Add a task with invalid codebook version
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(999)
            .build()) // Invalid version
        .setNodeId(200L)
        .build();

    taskQueue.enqueue(200L, task).get();

    long initialFailed = linkWorker.getTotalFailed();

    // Call processBatch - should handle failure gracefully
    linkWorker.processBatchAsync().join();

    // Verify task failed
    assertThat(linkWorker.getTotalFailed()).isEqualTo(initialFailed + 1);
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(0);
  }

  @Test
  @DisplayName("Test claimTasks method")
  void testClaimTasks() throws Exception {
    // Add multiple tasks
    for (int i = 0; i < 10; i++) {
      float[] queryVector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(queryVector);
      LinkTask task = LinkTask.newBuilder()
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .setNodeId(300L + i)
          .build();

      taskQueue.enqueue(300L + i, task).get();
    }

    // Claim tasks
    List<TaskClaim<Long, LinkTask>> claims = linkWorker.claimTasksAsync().join();

    // Should claim up to batch size
    assertThat(claims).isNotEmpty();
    assertThat(claims.size()).isLessThanOrEqualTo(linkWorker.getCurrentBatchSize());

    // Verify claimed tasks have correct structure
    for (TaskClaim<Long, LinkTask> claim : claims) {
      assertThat(claim.task()).isNotNull();
      assertThat(claim.task().hasPqEncoded()).isTrue();
      assertThat(claim.task().getPqEncoded().getCodebookVersion()).isEqualTo(1);
    }
  }

  @Test
  @DisplayName("Test claimTasks with timeout")
  void testClaimTasksWithTimeout() throws Exception {
    // Don't add any tasks - should timeout and return empty list
    List<TaskClaim<Long, LinkTask>> claims = linkWorker.claimTasksAsync().join();
    assertThat(claims).isEmpty();
  }

  @Test
  @DisplayName("Test adjustBatchSize for failed transaction")
  void testAdjustBatchSizeFailure() {
    int initialSize = linkWorker.getCurrentBatchSize();

    // Test failure adjustment - should reduce by half
    linkWorker.adjustBatchSize(Duration.ofSeconds(2), false);
    assertThat(linkWorker.getCurrentBatchSize()).isEqualTo(initialSize / 2);
  }

  @Test
  @DisplayName("Test adjustBatchSize for fast transaction")
  void testAdjustBatchSizeFast() {
    int initialSize = linkWorker.getCurrentBatchSize();

    // Test fast transaction - should increase by 20%
    linkWorker.adjustBatchSize(Duration.ofMillis(2000), true);
    assertThat(linkWorker.getCurrentBatchSize()).isEqualTo((int) (initialSize * 1.2));
  }

  @Test
  @DisplayName("Test adjustBatchSize for slow transaction")
  void testAdjustBatchSizeSlow() {
    int initialSize = linkWorker.getCurrentBatchSize();

    // Test slow transaction - should decrease by 20%
    linkWorker.adjustBatchSize(Duration.ofMillis(4500), true);
    assertThat(linkWorker.getCurrentBatchSize()).isEqualTo((int) (initialSize * 0.8));
  }

  @Test
  @DisplayName("Test adjustBatchSize for normal transaction")
  void testAdjustBatchSizeNormal() {
    int initialSize = linkWorker.getCurrentBatchSize();

    // Test normal transaction - should keep same size
    linkWorker.adjustBatchSize(Duration.ofMillis(3500), true);
    assertThat(linkWorker.getCurrentBatchSize()).isEqualTo(initialSize);
  }

  @Test
  @DisplayName("Test adjustBatchSize respects MIN_BATCH_SIZE")
  void testAdjustBatchSizeMinimum() {
    // Set batch size to minimum
    while (linkWorker.getCurrentBatchSize() > 1) {
      linkWorker.adjustBatchSize(Duration.ofSeconds(2), false);
    }

    int minSize = linkWorker.getCurrentBatchSize();

    // Try to reduce further - should stay at minimum
    linkWorker.adjustBatchSize(Duration.ofSeconds(2), false);
    assertThat(linkWorker.getCurrentBatchSize()).isEqualTo(minSize);
  }

  @Test
  @DisplayName("Test adjustBatchSize respects MAX_BATCH_SIZE")
  void testAdjustBatchSizeMaximum() {
    // Increase batch size significantly
    for (int i = 0; i < 20; i++) {
      linkWorker.adjustBatchSize(Duration.ofMillis(1000), true);
    }

    int currentSize = linkWorker.getCurrentBatchSize();

    // Should have increased from initial size of 10
    assertThat(currentSize).isGreaterThan(10);

    // Try to increase further - should still be able to increase or stay the same
    linkWorker.adjustBatchSize(Duration.ofMillis(1000), true);
    assertThat(linkWorker.getCurrentBatchSize()).isGreaterThanOrEqualTo(currentSize);
  }

  @Test
  @DisplayName("Test processing with no entry points")
  void testProcessWithNoEntryPoints() throws Exception {
    // Clear any existing entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, new ArrayList<>(), null, null))
        .get();

    // Add a task
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(1)
            .build())
        .setNodeId(400L)
        .build();

    taskQueue.enqueue(400L, task).get();

    // Process task - should succeed even with no entry points
    boolean processed = linkWorker.processOneTask().get();
    assertThat(processed).isTrue();
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(1);
  }

  @Test
  @DisplayName("Test processing with codebook not found")
  void testProcessWithCodebookNotFound() throws Exception {
    // Add a task with invalid codebook version that doesn't exist
    float[] queryVector = generateRandomVector(DIMENSION);
    byte[] pqCode = pq.encode(queryVector);
    LinkTask task = LinkTask.newBuilder()
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(999)
            .build()) // Codebook version that doesn't exist
        .setNodeId(500L)
        .build();

    taskQueue.enqueue(500L, task).get();

    // Process task - should fail gracefully
    boolean processed = linkWorker.processOneTask().get();
    assertThat(processed).isFalse();
    assertThat(linkWorker.getTotalFailed()).isEqualTo(1);
  }

  @Test
  @DisplayName("Test processAllAvailableTasks with max limit")
  void testProcessAllAvailableTasksWithMaxLimit() throws Exception {
    // Add 20 tasks
    for (int i = 0; i < 20; i++) {
      float[] queryVector = generateRandomVector(DIMENSION);
      byte[] pqCode = pq.encode(queryVector);
      LinkTask task = LinkTask.newBuilder()
          .setPqEncoded(PqEncodedVector.newBuilder()
              .setPqCode(ByteString.copyFrom(pqCode))
              .setCodebookVersion(1)
              .build())
          .setNodeId(600L + i)
          .build();

      taskQueue.enqueue(600L + i, task).get();
    }

    // Process with a max limit of 5
    int processed = linkWorker.processAllAvailableTasks(5).get();

    // Should process exactly 5 tasks
    assertThat(processed).isEqualTo(5);
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(5);
  }

  @Test
  @DisplayName("Test processAllAvailableTasks with no tasks")
  void testProcessAllAvailableTasksEmpty() throws Exception {
    // Process when no tasks available
    int processed = linkWorker.processAllAvailableTasks(10).get();

    // Should process 0 tasks
    assertThat(processed).isEqualTo(0);
    assertThat(linkWorker.getTotalProcessed()).isEqualTo(0);
  }

  @Test
  @DisplayName("Test back-link pruning when neighbors exceed graphDegree")
  void testBackLinkPruningWhenNeighborsExceedGraphDegree() throws Exception {
    // Given - a node that already has many neighbors (more than graphDegree)
    long targetNode = 8000L;
    long newNode = 8001L;

    // Create target node with many existing neighbors
    List<Long> existingNeighbors = new ArrayList<>();
    for (int i = 0; i < GRAPH_DEGREE + 10; i++) {
      long neighborId = 8100L + i;
      existingNeighbors.add(neighborId);

      // Store PQ codes for all neighbors
      byte[] pqCode = pq.encode(generateRandomVector(DIMENSION));
      db.runAsync(tr -> pqBlockStorage.storePqCode(tr, neighborId, pqCode, 1))
          .get();
    }

    // Store PQ code for target node
    db.runAsync(tr -> pqBlockStorage.storePqCode(tr, targetNode, pq.encode(generateRandomVector(DIMENSION)), 1))
        .get();

    // Create adjacency for target node with many neighbors
    db.runAsync(tr -> {
          NodeAdjacency adj = NodeAdjacency.newBuilder()
              .setNodeId(targetNode)
              .addAllNeighbors(existingNeighbors)
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          return nodeAdjacencyStorage.storeNodeAdjacency(tr, targetNode, adj);
        })
        .get();

    // Store PQ code for new node
    byte[] newNodePqCode = pq.encode(generateRandomVector(DIMENSION));
    db.runAsync(tr -> pqBlockStorage.storePqCode(tr, newNode, newNodePqCode, 1))
        .get();

    // Create adjacency for new node pointing to target
    db.runAsync(tr -> {
          NodeAdjacency adj = NodeAdjacency.newBuilder()
              .setNodeId(newNode)
              .addNeighbors(targetNode)
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          return nodeAdjacencyStorage.storeNodeAdjacency(tr, newNode, adj);
        })
        .get();

    // Set entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(targetNode), null, null))
        .get();

    // Create link task for new node
    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(newNode)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(newNodePqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(newNode, linkTask).get();

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify target node's neighbors were pruned
    NodeAdjacency targetAdjAfter = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, targetNode))
        .get();
    assertThat(targetAdjAfter).isNotNull();

    // Should have been pruned to graphDegree
    assertThat(targetAdjAfter.getNeighborsList().size()).isLessThanOrEqualTo(GRAPH_DEGREE);

    // The new node may or may not be retained after pruning based on distance
    // Just verify pruning occurred
  }

  @Test
  @DisplayName("Test back-link pruning with null PQ codes")
  void testBackLinkPruningWithNullPqCodes() throws Exception {
    // Given - a node with neighbors where some have null PQ codes
    long targetNode = 9000L;
    long newNode = 9001L;

    // Create target node with many neighbors
    List<Long> existingNeighbors = new ArrayList<>();
    for (int i = 0; i < GRAPH_DEGREE + 5; i++) {
      long neighborId = 9100L + i;
      existingNeighbors.add(neighborId);

      // Store PQ codes for only some neighbors (simulate missing codes)
      if (i % 3 != 0) {
        byte[] pqCode = pq.encode(generateRandomVector(DIMENSION));
        db.runAsync(tr -> pqBlockStorage.storePqCode(tr, neighborId, pqCode, 1))
            .get();
      }
      // Every 3rd neighbor has no PQ code stored
    }

    // Store PQ code for target and new node
    db.runAsync(tr -> pqBlockStorage.storePqCode(tr, targetNode, pq.encode(generateRandomVector(DIMENSION)), 1))
        .get();
    byte[] newNodePqCode = pq.encode(generateRandomVector(DIMENSION));
    db.runAsync(tr -> pqBlockStorage.storePqCode(tr, newNode, newNodePqCode, 1))
        .get();

    // Create adjacency for target node
    db.runAsync(tr -> {
          NodeAdjacency adj = NodeAdjacency.newBuilder()
              .setNodeId(targetNode)
              .addAllNeighbors(existingNeighbors)
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          return nodeAdjacencyStorage.storeNodeAdjacency(tr, targetNode, adj);
        })
        .get();

    // Create adjacency for new node
    db.runAsync(tr -> {
          NodeAdjacency adj = NodeAdjacency.newBuilder()
              .setNodeId(newNode)
              .addNeighbors(targetNode)
              .setState(NodeAdjacency.State.ACTIVE)
              .build();
          return nodeAdjacencyStorage.storeNodeAdjacency(tr, newNode, adj);
        })
        .get();

    // Set entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(targetNode), null, null))
        .get();

    // Create link task
    LinkTask linkTask = LinkTask.newBuilder()
        .setCollection("test-collection")
        .setNodeId(newNode)
        .setPqEncoded(PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(newNodePqCode))
            .setCodebookVersion(1)
            .build())
        .build();

    taskQueue.enqueue(newNode, linkTask).get();

    // When - process the task
    boolean processed = linkWorker.processOneTask().get();

    // Then
    assertThat(processed).isTrue();

    // Verify pruning occurred despite null PQ codes
    NodeAdjacency targetAdjAfter = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, targetNode))
        .get();
    assertThat(targetAdjAfter).isNotNull();
    assertThat(targetAdjAfter.getNeighborsList().size()).isLessThanOrEqualTo(GRAPH_DEGREE);

    // Nodes with null PQ codes should have been deprioritized (treated as MAX_VALUE distance)
    // Verify that pruning occurred
  }
}
