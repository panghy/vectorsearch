package io.github.panghy.vectorsearch.workers;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.ByteString;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.proto.LinkTask;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class LinkWorkerTest {
  private static final Logger LOGGER = Logger.getLogger(LinkWorkerTest.class.getName());

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
  private ExecutorService executorService;

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
    keys = new VectorIndexKeys(testDirectory, testId);

    pqBlockStorage = new PqBlockStorage(
        db, keys, CODES_PER_BLOCK, PQ_SUBVECTORS, InstantSource.system(), 100, Duration.ofMinutes(5));

    nodeAdjacencyStorage =
        new NodeAdjacencyStorage(db, keys, GRAPH_DEGREE, InstantSource.system(), 100, Duration.ofMinutes(5));

    entryPointStorage = new EntryPointStorage(testDirectory, testId, InstantSource.system());

    codebookStorage = new CodebookStorage(db, keys);

    // Initialize PQ with test codebooks
    pq = new ProductQuantizer(DIMENSION, PQ_SUBVECTORS, DistanceMetrics.Metric.L2);
    List<float[]> trainingData = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      float[] vec = new float[DIMENSION];
      for (int j = 0; j < DIMENSION; j++) {
        vec[j] = (float) Math.random();
      }
      trainingData.add(vec);
    }
    pq.train(trainingData).get(5, TimeUnit.SECONDS);

    // Store codebooks
    CodebookStorage.TrainingStats stats = new CodebookStorage.TrainingStats(100L, null);
    codebookStorage.storeCodebooks(1, pq.getCodebooks(), stats).get();

    searchEngine = new BeamSearchEngine(nodeAdjacencyStorage, pqBlockStorage, entryPointStorage, pq);

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

    executorService = Executors.newSingleThreadExecutor();
  }

  @AfterEach
  void tearDown() throws Exception {
    if (linkWorker != null) {
      linkWorker.stop();
    }
    if (executorService != null) {
      executorService.shutdown();
      executorService.awaitTermination(5, TimeUnit.SECONDS);
    }

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
        .setPqCode(ByteString.copyFrom(pqCode))
        .setCodebookVersion(1)
        .build();

    // Enqueue task
    taskQueue.enqueue(nodeId, linkTask).get();

    // Set up some entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(
            tr, Arrays.asList(1L, 2L, 3L), Arrays.asList(4L, 5L), Arrays.asList(6L, 7L)))
        .get();

    // When
    AtomicBoolean processed = new AtomicBoolean(false);
    executorService.submit(() -> {
      try {
        // Run for a short time
        Thread.sleep(2000);
        processed.set(true);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    });

    // Start worker in background
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();
    workerExecutor.submit(linkWorker);

    // Wait for processing
    Thread.sleep(3000);
    linkWorker.stop();
    workerExecutor.shutdown();
    workerExecutor.awaitTermination(5, TimeUnit.SECONDS);

    // Then
    // Verify PQ code was stored
    byte[] storedCode = pqBlockStorage.loadPqCode(nodeId, 1).get();
    assertThat(storedCode).isEqualTo(pqCode);

    // Verify node adjacency was created
    NodeAdjacency adjacency = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, nodeId))
        .get();
    assertThat(adjacency).isNotNull();
    assertThat(adjacency.getNodeId()).isEqualTo(nodeId);

    assertThat(linkWorker.getTotalProcessed()).isGreaterThan(0);
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
          .setPqCode(ByteString.copyFrom(pqCode))
          .setCodebookVersion(1)
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // Set up entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(1L, 2L, 3L), null, null))
        .get();

    // When
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();
    workerExecutor.submit(linkWorker);

    // Wait for processing
    Thread.sleep(3000);
    linkWorker.stop();
    workerExecutor.shutdown();
    workerExecutor.awaitTermination(5, TimeUnit.SECONDS);

    // Then
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
        .setPqCode(ByteString.copyFrom(pqCode))
        .setCodebookVersion(1)
        .build();

    taskQueue.enqueue(nodeId, linkTask).get();

    // No entry points set

    // When
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();
    workerExecutor.submit(linkWorker);

    Thread.sleep(2000);
    linkWorker.stop();
    workerExecutor.shutdown();
    workerExecutor.awaitTermination(5, TimeUnit.SECONDS);

    // Then
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
        .setPqCode(ByteString.copyFrom(newPqCode))
        .setCodebookVersion(1)
        .build();

    taskQueue.enqueue(newNodeId, linkTask).get();

    // When
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();
    workerExecutor.submit(linkWorker);

    Thread.sleep(3000);
    linkWorker.stop();
    workerExecutor.shutdown();
    workerExecutor.awaitTermination(5, TimeUnit.SECONDS);

    // Then - verify the new node has neighbors
    NodeAdjacency newNodeAdj = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, newNodeId))
        .get();
    assertThat(newNodeAdj).isNotNull();

    // The new node should have discovered some neighbors
    List<Long> neighbors = newNodeAdj.getNeighborsList();
    LOGGER.info("New node " + newNodeId + " has neighbors: " + neighbors);

    // Verify back-links were created
    for (Long neighborId : neighbors) {
      NodeAdjacency neighborAdj = db.runAsync(tr -> nodeAdjacencyStorage.loadAdjacency(tr, neighborId))
          .get();
      assertThat(neighborAdj.getNeighborsList()).contains(newNodeId);
      LOGGER.info("Neighbor " + neighborId + " has back-link to " + newNodeId);
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
          .setPqCode(ByteString.copyFrom(pqCode))
          .setCodebookVersion(1)
          .build();

      taskQueue.enqueue(nodeId, linkTask).get();
    }

    // Set up entry points
    db.runAsync(tr -> entryPointStorage.storeEntryList(tr, Arrays.asList(1L, 2L), null, null))
        .get();

    // When
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();
    workerExecutor.submit(linkWorker);

    // Let it process for a while
    Thread.sleep(5000);
    linkWorker.stop();
    workerExecutor.shutdown();
    workerExecutor.awaitTermination(5, TimeUnit.SECONDS);

    // Then
    int finalBatchSize = linkWorker.getCurrentBatchSize();
    LOGGER.info("Batch size changed from " + initialBatchSize + " to " + finalBatchSize);

    // Batch size may have changed based on processing speed
    assertThat(linkWorker.getTotalProcessed()).isGreaterThan(0);
  }

  @Test
  @Timeout(5)
  void testStopMethod() throws Exception {
    // Given - no tasks to process
    ExecutorService workerExecutor = Executors.newSingleThreadExecutor();

    // When
    CountDownLatch started = new CountDownLatch(1);
    CountDownLatch stopped = new CountDownLatch(1);

    workerExecutor.submit(() -> {
      started.countDown();
      linkWorker.run();
      stopped.countDown();
    });

    started.await();
    Thread.sleep(500);
    linkWorker.stop();

    // Then
    boolean stoppedInTime = stopped.await(2, TimeUnit.SECONDS);
    assertThat(stoppedInTime).isTrue();

    workerExecutor.shutdown();
    workerExecutor.awaitTermination(1, TimeUnit.SECONDS);
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
}
