package io.github.panghy.vectorsearch;

import static com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt;
import static com.apple.foundationdb.tuple.ByteArrayUtil.encodeInt;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;
import static java.util.concurrent.CompletableFuture.failedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.TransactionContext;
import com.apple.foundationdb.async.AsyncUtil;
import com.apple.foundationdb.directory.Directory;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.Weigher;
import com.google.protobuf.ByteString;
import com.google.protobuf.Message;
import com.google.protobuf.Parser;
import com.ibm.asyncutil.locks.AsyncSemaphore;
import com.ibm.asyncutil.locks.FairAsyncSemaphore;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.graph.GraphConnectivityMonitor;
import io.github.panghy.vectorsearch.pq.DistanceMetrics;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.proto.CodebookSub;
import io.github.panghy.vectorsearch.proto.Config;
import io.github.panghy.vectorsearch.proto.LinkTask;
import io.github.panghy.vectorsearch.proto.NodeAdjacency;
import io.github.panghy.vectorsearch.proto.PqCodesBlock;
import io.github.panghy.vectorsearch.proto.PqEncodedVector;
import io.github.panghy.vectorsearch.proto.RawVector;
import io.github.panghy.vectorsearch.proto.UnlinkTask;
import io.github.panghy.vectorsearch.search.BeamSearchEngine;
import io.github.panghy.vectorsearch.search.SearchResult;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.GraphMetaStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import io.github.panghy.vectorsearch.storage.VectorIndexKeys;
import io.github.panghy.vectorsearch.storage.VectorSketchStorage;
import io.github.panghy.vectorsearch.workers.LinkWorker;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.metrics.LongCounter;
import io.opentelemetry.api.metrics.LongHistogram;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.trace.Tracer;
import java.time.Duration;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * FoundationDB-backed vector search index implementing DiskANN-style graph traversal
 * with Product Quantization (PQ) compression for approximate nearest neighbor search.
 *
 * <p>This index provides:
 * <ul>
 *   <li>Online vector insertion and deletion with async graph construction</li>
 *   <li>Millisecond-latency similarity search using PQ-compressed vectors</li>
 *   <li>Automatic graph connectivity maintenance and repair</li>
 *   <li>Two-phase codebook rotation for PQ retraining</li>
 * </ul>
 *
 * <p>The index stores all data in FoundationDB with the following structure:
 * <ul>
 *   <li>Configuration and metadata in /meta/</li>
 *   <li>PQ codebooks and compressed codes in /pq/</li>
 *   <li>Graph adjacency lists in /graph/</li>
 *   <li>Vector sketches for reconstruction in /sketch/</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * VectorSearchConfig config = VectorSearchConfig.builder(database, directory)
 *     .dimension(768)
 *     .distanceMetric(DistanceMetric.COSINE)
 *     .build();
 *
 * FdbVectorSearch index = FdbVectorSearch.createOrOpen(config, database)
 *     .join();
 *
 * // Insert vectors
 * index.upsert(nodeId, vector).join();
 *
 * // Search for similar vectors
 * List<SearchResult> results = index.search(queryVector, topK).join();
 * }</pre>
 */
public class FdbVectorSearch implements VectorSearch, AutoCloseable {

  private static final Logger LOGGER = LoggerFactory.getLogger(FdbVectorSearch.class);

  // OpenTelemetry instrumentation
  private static final Tracer TRACER = GlobalOpenTelemetry.getTracer("io.github.panghy.vectorsearch", "0.1.0");
  private static final Meter METER = GlobalOpenTelemetry.getMeter("io.github.panghy.vectorsearch");

  // Metrics for monitoring
  private static final LongCounter VECTORS_INSERTED = METER.counterBuilder("vectorsearch.vectors.inserted")
      .setDescription("Number of vectors inserted")
      .setUnit("vectors")
      .build();

  private static final LongCounter VECTORS_DELETED = METER.counterBuilder("vectorsearch.vectors.deleted")
      .setDescription("Number of vectors deleted")
      .setUnit("vectors")
      .build();

  private static final LongCounter SEARCHES_PERFORMED = METER.counterBuilder("vectorsearch.searches.performed")
      .setDescription("Number of searches performed")
      .setUnit("searches")
      .build();

  private static final LongHistogram SEARCH_LATENCY = METER.histogramBuilder("vectorsearch.search.latency")
      .setDescription("Search latency in milliseconds")
      .setUnit("ms")
      .ofLongs()
      .build();

  private static final LongHistogram NODES_VISITED = METER.histogramBuilder("vectorsearch.search.nodes.visited")
      .setDescription("Number of nodes visited during search")
      .setUnit("nodes")
      .ofLongs()
      .build();

  // ============= Core Fields =============

  /**
   * Configuration for this index instance
   */
  private final VectorSearchConfig config;

  /**
   * FDB database instance
   */
  private final Database database;

  /**
   * Root directory for all index data
   */
  private final Directory rootDirectory;

  // ============= Subspaces for Data Storage =============

  /**
   * /meta/ - Configuration and metadata
   */
  private final DirectorySubspace metaSubspace;

  /**
   * /pq/codebook/ - PQ codebooks by version and subspace
   */
  private final DirectorySubspace codebookSubspace;

  /**
   * /pq/block/ - PQ code blocks by version and block number
   */
  private final DirectorySubspace pqBlockSubspace;

  /**
   * /graph/node/ - Node adjacency lists
   */
  private final DirectorySubspace graphNodeSubspace;

  /**
   * /graph/meta/ - Graph metadata and connectivity info
   */
  private final DirectorySubspace graphMetaSubspace;

  /**
   * /sketch/ - Vector sketches for reconstruction
   */
  private final DirectorySubspace sketchSubspace;

  /**
   * /entry - Entry points for search initialization
   */
  private final DirectorySubspace entrySubspace;

  // ============= Caching =============

  /**
   * In-memory cache of PQ codebooks (always kept in memory if configured)
   */
  private final Map<CodebookCacheKey, CodebookSub> codebookCache;

  /**
   * LRU cache for PQ code blocks
   */
  private final Cache<PqBlockCacheKey, PqCodesBlock> pqBlockCache;

  /**
   * LRU cache for node adjacency lists
   */
  private final Cache<Long, NodeAdjacency> adjacencyCache;

  /**
   * Cached entry points for search initialization
   */
  private final AtomicReference<List<Long>> cachedEntryPoints;

  // ============= Background Workers =============

  /**
   * Task queue for async link operations
   */
  private final TaskQueue<Long, LinkTask> linkTaskQueue;

  /**
   * Task queue for async unlink operations
   */
  private final TaskQueue<Long, UnlinkTask> unlinkTaskQueue;

  /**
   * Scheduler for periodic maintenance tasks
   */
  private final ScheduledExecutorService maintenanceScheduler;

  /**
   * Time source for consistent time operations
   */
  private final InstantSource instantSource;

  // ============= Storage Components =============

  /**
   * Storage for PQ codebooks
   */
  private CodebookStorage codebookStorage;

  /**
   * Storage for PQ code blocks
   */
  private PqBlockStorage pqBlockStorage;

  /**
   * Storage for node adjacency lists
   */
  private NodeAdjacencyStorage nodeAdjacencyStorage;

  /**
   * Storage for entry points
   */
  private EntryPointStorage entryPointStorage;

  /**
   * Storage for graph metadata
   */
  private GraphMetaStorage graphMetaStorage;

  /**
   * Storage for vector sketches
   */
  private VectorSketchStorage vectorSketchStorage;

  /**
   * Graph connectivity monitor
   */
  private GraphConnectivityMonitor connectivityMonitor;

  /**
   * Beam search engine for graph traversal
   */
  private BeamSearchEngine beamSearchEngine;

  /**
   * Vector index keys helper
   */
  private VectorIndexKeys vectorIndexKeys;

  /**
   * Link worker thread for processing link tasks
   */
  private LinkWorker linkWorker;

  /**
   * Thread running the link worker
   */
  private Thread linkWorkerThread;

  // ============= Runtime State =============

  /**
   * Whether the index has been initialized (config loaded from FDB)
   */
  private volatile boolean initialized;

  /**
   * Stored configuration from FDB (for validation)
   */
  private volatile Config storedConfig;

  /**
   * Semaphore to ensure only one thread is loading codebooks at a time
   */
  private final AsyncSemaphore codeBooksLoadingSemaphore = new FairAsyncSemaphore(1);

  /**
   * Private constructor. Use {@link #createOrOpen(VectorSearchConfig, TransactionContext)} to create an instance.
   *
   * @param config            the configuration for this index
   * @param database          the FDB database
   * @param rootDirectory     the root directory for all data
   * @param metaSubspace      subspace for metadata
   * @param codebookSubspace  subspace for PQ codebooks
   * @param pqBlockSubspace   subspace for PQ blocks
   * @param graphNodeSubspace subspace for graph nodes
   * @param graphMetaSubspace subspace for graph metadata
   * @param sketchSubspace    subspace for vector sketches
   * @param entrySubspace     subspace for entry points
   * @param linkTaskQueue     task queue for link operations
   * @param unlinkTaskQueue   task queue for unlink operations
   */
  private FdbVectorSearch(
      VectorSearchConfig config,
      Database database,
      Directory rootDirectory,
      DirectorySubspace metaSubspace,
      DirectorySubspace codebookSubspace,
      DirectorySubspace pqBlockSubspace,
      DirectorySubspace graphNodeSubspace,
      DirectorySubspace graphMetaSubspace,
      DirectorySubspace sketchSubspace,
      DirectorySubspace entrySubspace,
      TaskQueue<Long, LinkTask> linkTaskQueue,
      TaskQueue<Long, UnlinkTask> unlinkTaskQueue) {

    this.config = config;
    this.database = database;
    this.rootDirectory = rootDirectory;
    this.metaSubspace = metaSubspace;
    this.codebookSubspace = codebookSubspace;
    this.pqBlockSubspace = pqBlockSubspace;
    this.graphNodeSubspace = graphNodeSubspace;
    this.graphMetaSubspace = graphMetaSubspace;
    this.sketchSubspace = sketchSubspace;
    this.entrySubspace = entrySubspace;
    this.linkTaskQueue = linkTaskQueue;
    this.unlinkTaskQueue = unlinkTaskQueue;
    this.instantSource = config.getInstantSource();

    // Initialize caches
    this.codebookCache = new ConcurrentHashMap<>();

    // PQ block cache with size-based eviction (weighted by bytes)
    this.pqBlockCache = Caffeine.newBuilder()
        .maximumWeight(config.getPqBlockCacheSize())
        .weigher((Weigher<PqBlockCacheKey, PqCodesBlock>)
            (key, block) -> block.getCodes().size())
        .recordStats()
        .build();

    // Adjacency cache with count-based eviction
    this.adjacencyCache = Caffeine.newBuilder()
        .maximumSize(config.getAdjacencyCacheSize())
        .recordStats()
        .build();

    // Initialize state
    this.cachedEntryPoints = new AtomicReference<>(List.of());
    this.initialized = false;
    this.storedConfig = null;

    // Initialize maintenance scheduler
    this.maintenanceScheduler = new ScheduledThreadPoolExecutor(2, r -> {
      Thread t = new Thread(r, "vectorsearch-maintenance");
      t.setDaemon(true);
      return t;
    });

    // Storage components will be initialized after index creation
    this.vectorIndexKeys = null;
    this.codebookStorage = null;
    this.pqBlockStorage = null;
    this.nodeAdjacencyStorage = null;
    this.entryPointStorage = null;
    this.graphMetaStorage = null;
    this.connectivityMonitor = null;
  }

  /**
   * Initializes storage components after the index is created.
   * Must be called before using any storage operations.
   */
  private void initializeStorageComponents() {
    // Use metaSubspace as the collection subspace since all our subspaces are under the same root
    // The VectorIndexKeys will handle the proper key paths
    DirectorySubspace collectionSubspace = metaSubspace;

    // Initialize storage components
    this.vectorIndexKeys = new VectorIndexKeys(collectionSubspace);

    // Get metric and subvectors for CodebookStorage
    DistanceMetrics.Metric metric = storedConfig != null
        ? DistanceMetrics.Metric.valueOf(storedConfig.getMetric())
        : convertDistanceMetric(config.getDistanceMetric());
    int subVectors = storedConfig != null && storedConfig.getPqSubvectors() > 0
        ? storedConfig.getPqSubvectors()
        : config.getPqSubvectors();

    int dimension = storedConfig != null ? storedConfig.getDimension() : config.getDimension();

    this.codebookStorage = new CodebookStorage(database, vectorIndexKeys, dimension, subVectors, metric);

    this.pqBlockStorage = new PqBlockStorage(
        database,
        vectorIndexKeys,
        config.getCodesPerBlock(),
        subVectors,
        instantSource,
        config.getAdjacencyCacheSize(),
        Duration.ofMinutes(5));
    this.nodeAdjacencyStorage = new NodeAdjacencyStorage(
        database,
        vectorIndexKeys,
        config.getGraphDegree(),
        instantSource,
        config.getAdjacencyCacheSize(),
        Duration.ofMinutes(5));
    this.entryPointStorage = new EntryPointStorage(collectionSubspace, instantSource);
    this.graphMetaStorage = new GraphMetaStorage(vectorIndexKeys, instantSource);
    this.vectorSketchStorage = new VectorSketchStorage(vectorIndexKeys, dimension);

    // Initialize connectivity monitor
    this.connectivityMonitor = new GraphConnectivityMonitor(
        database,
        vectorIndexKeys,
        graphMetaStorage,
        nodeAdjacencyStorage,
        pqBlockStorage,
        entryPointStorage,
        codebookStorage);

    // Initialize beam search engine
    this.beamSearchEngine =
        new BeamSearchEngine(nodeAdjacencyStorage, pqBlockStorage, entryPointStorage, codebookStorage);

    // Initialize link worker
    this.linkWorker = new LinkWorker(
        database,
        linkTaskQueue,
        vectorIndexKeys,
        pqBlockStorage,
        nodeAdjacencyStorage,
        entryPointStorage,
        codebookStorage,
        beamSearchEngine,
        config.getGraphDegree(),
        config.getMaxSearchVisits(),
        1.2, // pruning alpha - typical DiskANN value
        instantSource);
  }

  /**
   * Creates or opens a vector search index with the given configuration.
   *
   * <p>If the index already exists, validates that immutable configuration values
   * match the stored configuration. If this is a new index, stores the configuration.
   *
   * @param config  the configuration for the index
   * @param context the transaction context (usually the Database)
   * @return a CompletableFuture that completes with the initialized index
   * @throws IllegalStateException if existing index has incompatible configuration
   */
  public static CompletableFuture<FdbVectorSearch> createOrOpen(
      VectorSearchConfig config, TransactionContext context) {

    // Get the root directory from config
    Directory rootDir = config.getDirectory();

    var metaFuture = rootDir.createOrOpen(context, List.of("meta"));
    var codebookFuture = rootDir.createOrOpen(context, List.of("pq", "codebook"));
    var pqBlockFuture = rootDir.createOrOpen(context, List.of("pq", "block"));
    var graphNodeFuture = rootDir.createOrOpen(context, List.of("graph", "node"));
    var graphMetaFuture = rootDir.createOrOpen(context, List.of("graph", "meta"));
    var sketchFuture = rootDir.createOrOpen(context, List.of("sketch"));
    var entryFuture = rootDir.createOrOpen(context, List.of("entry"));

    // Create directories for task queues
    var linkQueueDirFuture = rootDir.createOrOpen(context, List.of("queue", "link"));
    var unlinkQueueDirFuture = rootDir.createOrOpen(context, List.of("queue", "unlink"));

    // Wait for all subspaces to be created first
    return allOf(
            metaFuture,
            codebookFuture,
            pqBlockFuture,
            graphNodeFuture,
            graphMetaFuture,
            sketchFuture,
            entryFuture,
            linkQueueDirFuture,
            unlinkQueueDirFuture)
        .thenCompose(v -> {
          // Now create the task queues with their own configs
          var linkQueueConfig = TaskQueueConfig.builder(
                  config.getDatabase(),
                  linkQueueDirFuture.join(),
                  new LongSerializer(),
                  new ProtoSerializer<>(LinkTask.parser()))
              .defaultTtl(Duration.ofMinutes(10))
              .maxAttempts(5)
              .estimatedWorkerCount(config.getLinkWorkerCount())
              .instantSource(config.getInstantSource())
              .taskNameExtractor(task -> String.format("LinkTask[nodeId=%d]", task.getNodeId()))
              .build();

          var unlinkQueueConfig = TaskQueueConfig.builder(
                  config.getDatabase(),
                  unlinkQueueDirFuture.join(),
                  new LongSerializer(),
                  new ProtoSerializer<>(UnlinkTask.parser()))
              .defaultTtl(Duration.ofMinutes(10))
              .maxAttempts(5)
              .estimatedWorkerCount(config.getUnlinkWorkerCount())
              .instantSource(config.getInstantSource())
              .taskNameExtractor(task -> String.format("UnlinkTask[nodeId=%d]", task.getNodeId()))
              .build();

          // Create the task queues
          var linkQueueF = TaskQueues.createTaskQueue(linkQueueConfig);
          var unlinkQueueF = TaskQueues.createTaskQueue(unlinkQueueConfig);

          // Continue with index creation
          return allOf(linkQueueF, unlinkQueueF).thenCompose($ -> {
            TaskQueue<Long, LinkTask> linkQueue = linkQueueF.join();
            TaskQueue<Long, UnlinkTask> unlinkQueue = unlinkQueueF.join();

            // Create the index instance
            FdbVectorSearch index = new FdbVectorSearch(
                config,
                config.getDatabase(),
                config.getDirectory(),
                metaFuture.join(),
                codebookFuture.join(),
                pqBlockFuture.join(),
                graphNodeFuture.join(),
                graphMetaFuture.join(),
                sketchFuture.join(),
                entryFuture.join(),
                linkQueue,
                unlinkQueue);

            // Initialize or validate configuration
            return index.initializeOrValidateConfig(context).thenApply(__ -> {
              // Initialize storage components
              index.initializeStorageComponents();

              // Start link worker thread
              index.startLinkWorker();

              // Schedule maintenance tasks if configured
              if (config.isAutoRepairEnabled()) {
                index.scheduleMaintenanceTasks();
              }

              LOGGER.info("Vector search index initialized successfully");
              return index;
            });
          });
        });
  }

  /**
   * Initializes a new index or validates configuration against existing index.
   */
  private CompletableFuture<Void> initializeOrValidateConfig(TransactionContext context) {
    return context.runAsync(tr -> {
      byte[] configKey = metaSubspace.pack(Tuple.from("config"));
      return tr.get(configKey).thenCompose(existingConfig -> {
        if (existingConfig == null) {
          // New index - store configuration
          Config protoConfig = config.toProtoConfig();
          tr.set(configKey, protoConfig.toByteArray());

          // Initialize active codebook version
          byte[] cbvKey = metaSubspace.pack(Tuple.from("cbv_active"));
          tr.set(cbvKey, encodeInt(0));

          this.storedConfig = protoConfig;
          this.initialized = true;

          LOGGER.info("Created new vector search index with dimension={}", config.getDimension());
          return completedFuture(null);
        } else {
          // Existing index - validate configuration
          try {
            Config existingProtoConfig = Config.parseFrom(existingConfig);
            config.validateAgainstStored(existingProtoConfig);

            this.storedConfig = existingProtoConfig;

            // Load active codebook version
            byte[] cbvKey = metaSubspace.pack(Tuple.from("cbv_active"));
            return tr.get(cbvKey).thenAccept(cbvBytes -> {
              if (cbvBytes != null) {
                int cbv = (int) decodeInt(cbvBytes);
              }

              this.initialized = true;

              LOGGER.info("Opened existing vector search index with dimension={}", config.getDimension());
            });
          } catch (IndexConfigurationException e) {
            // Re-throw configuration mismatches directly
            throw new IllegalStateException(e);
          } catch (Exception e) {
            throw new IllegalStateException("Failed to parse existing configuration", e);
          }
        }
      });
    });
  }

  /**
   * Schedules periodic maintenance tasks like entry point refresh and graph repair.
   */
  private void scheduleMaintenanceTasks() {
    // Schedule entry point refresh
    long refreshIntervalMs = config.getEntryPointRefreshInterval().toMillis();
    maintenanceScheduler.scheduleWithFixedDelay(
        this::refreshEntryPoints, refreshIntervalMs, refreshIntervalMs, TimeUnit.MILLISECONDS);

    // Schedule graph connectivity check and repair
    if (config.isAutoRepairEnabled()) {
      long repairIntervalMs = config.getGraphRepairInterval().toMillis();
      maintenanceScheduler.scheduleWithFixedDelay(
          this::checkAndRepairConnectivity, repairIntervalMs, repairIntervalMs, TimeUnit.MILLISECONDS);
    }
  }

  /**
   * Starts the link worker thread for processing link tasks.
   * Only starts if codebooks exist.
   */
  private void startLinkWorker() {
    // Check if we have codebooks available
    CompletableFuture<Integer> versionFuture = codebookStorage.getActiveVersion();
    int activeVersion = versionFuture.join();
    if (linkWorker != null && linkWorkerThread == null && activeVersion > 0) {
      linkWorkerThread = new Thread(linkWorker, "LinkWorker");
      linkWorkerThread.setDaemon(true);
      linkWorkerThread.start();
      LOGGER.info("Started LinkWorker thread with codebook version {}", activeVersion);
    } else if (activeVersion <= 0) {
      LOGGER.info("Not starting LinkWorker - no codebooks available yet");
    }
  }

  /**
   * Stops the link worker thread gracefully.
   */
  private void stopLinkWorker() {
    if (linkWorker != null) {
      linkWorker.stop();
    }
    if (linkWorkerThread != null) {
      try {
        linkWorkerThread.join(5000); // Wait up to 5 seconds
        if (linkWorkerThread.isAlive()) {
          LOGGER.warn("LinkWorker thread did not stop gracefully");
          linkWorkerThread.interrupt();
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        LOGGER.warn("Interrupted while waiting for LinkWorker to stop");
      }
      linkWorkerThread = null;
    }
  }

  // Maintenance task implementations
  @Override
  public CompletableFuture<Void> refreshEntryPoints() {
    if (connectivityMonitor == null) {
      LOGGER.warn("Cannot refresh entry points: storage not initialized");
      return completedFuture(null);
    }

    LOGGER.info("Starting entry point refresh");

    return database.runAsync(
            tx -> connectivityMonitor.refreshEntryPoints(tx).thenAccept(v -> {
              LOGGER.info("Entry points refreshed successfully");
              // Clear cached entry points to force reload
              cachedEntryPoints.set(null);
            }))
        .exceptionally(e -> {
          LOGGER.error("Failed to refresh entry points: {}", e.getMessage());
          return null;
        });
  }

  CompletableFuture<Void> checkAndRepairConnectivity() {
    if (connectivityMonitor == null || graphMetaStorage == null) {
      LOGGER.warn("Cannot check connectivity: storage not initialized");
      return completedFuture(null);
    }

    LOGGER.info("Starting graph connectivity check and repair");

    // Get active codebook version for distance calculations
    // Using cached version for read-only operation (non-transactional)
    // Get the latest ProductQuantizer
    ProductQuantizer pq = codebookStorage.getLatestProductQuantizer().join();
    int codebookVersion = pq != null ? pq.getCodebookVersion() : 0;

    // Check if analysis is needed (default: every 6 hours)
    return database.runAsync(tx -> {
          long maxAgeSeconds = config.getGraphRepairInterval().getSeconds();
          return graphMetaStorage.isAnalysisNeeded(tx, maxAgeSeconds);
        })
        .thenCompose(needsAnalysis -> {
          if (needsAnalysis) {
            return connectivityMonitor
                .analyzeAndRepair(codebookVersion)
                .thenAccept(v -> LOGGER.info("Graph connectivity check completed"));
          } else {
            LOGGER.debug("Graph connectivity check skipped - analysis still fresh");
            return completedFuture(null);
          }
        })
        .exceptionally(e -> {
          LOGGER.error("Failed to check/repair graph connectivity: {}", e.getMessage());
          return null;
        });
  }

  /**
   * Shuts down the index and releases all resources.
   * This method should be called when the index is no longer needed.
   *
   * <p>This will:
   * <ul>
   *   <li>Stop all maintenance tasks</li>
   *   <li>Shut down the maintenance scheduler</li>
   *   <li>Clear all caches</li>
   * </ul>
   *
   * @param awaitTermination if true, waits for running tasks to complete
   * @param timeout          the maximum time to wait for termination
   * @param unit             the time unit of the timeout
   * @return true if termination completed within timeout, false otherwise
   */
  public boolean shutdown(boolean awaitTermination, long timeout, TimeUnit unit) {
    LOGGER.info("Shutting down vector search index");

    // Stop the maintenance scheduler
    maintenanceScheduler.shutdown();

    // Await termination if requested
    boolean terminated = true;
    if (awaitTermination) {
      try {
        terminated = maintenanceScheduler.awaitTermination(timeout, unit);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        LOGGER.warn("Interrupted while awaiting scheduler termination");
        terminated = false;
      }
    } else {
      // Force shutdown if not waiting
      maintenanceScheduler.shutdownNow();
    }

    // Clear caches to free memory
    pqBlockCache.invalidateAll();
    adjacencyCache.invalidateAll();
    codebookCache.clear();
    cachedEntryPoints.set(List.of());

    LOGGER.info("Vector search index shutdown complete");
    return terminated;
  }

  /**
   * Shuts down the index immediately without waiting for running tasks.
   * This is equivalent to calling {@code shutdown(false, 0, TimeUnit.SECONDS)}.
   */
  public void shutdown() {
    shutdown(false, 0, TimeUnit.SECONDS);
  }

  // Package-private for testing
  Cache<PqBlockCacheKey, PqCodesBlock> getPqBlockCache() {
    return pqBlockCache;
  }

  // ============= VectorSearch Interface Implementation =============

  @Override
  public CompletableFuture<List<Long>> insert(List<float[]> vectors) {
    if (!initialized) {
      return failedFuture(new IllegalStateException("Index not initialized"));
    }

    // Validate dimensions
    for (float[] vector : vectors) {
      if (vector.length != config.getDimension()) {
        return failedFuture(new IllegalArgumentException(
            "Vector dimension mismatch: expected " + config.getDimension() + " but got " + vector.length));
      }
    }

    // Allocate IDs and insert vectors transactionally
    return database.runAsync(tr -> {
      // Get and update ID counter
      byte[] counterKey = metaSubspace.pack(Tuple.from("node_id_counter"));

      return tr.get(counterKey).thenCompose(counterBytes -> {
        long startId = counterBytes == null ? 1L : decodeInt(counterBytes) + 1;
        long endId = startId + vectors.size() - 1;

        // Update counter for next batch
        tr.set(counterKey, encodeInt(endId));

        // Create ID to vector mapping
        Map<Long, float[]> idToVector = new LinkedHashMap<>();
        List<Long> assignedIds = new ArrayList<>();
        long currentId = startId;

        for (float[] vector : vectors) {
          idToVector.put(currentId, vector);
          assignedIds.add(currentId);
          currentId++;
        }

        // Enqueue upsert tasks
        return upsertInternal(tr, idToVector).thenApply(v -> assignedIds);
      });
    });
  }

  @Override
  public CompletableFuture<Void> upsert(Map<Long, float[]> vectors) {
    if (!initialized) {
      return failedFuture(new IllegalStateException("Index not initialized"));
    }

    // Validate dimensions
    for (Map.Entry<Long, float[]> entry : vectors.entrySet()) {
      if (entry.getValue().length != config.getDimension()) {
        return failedFuture(new IllegalArgumentException("Vector dimension mismatch for ID " + entry.getKey()
            + ": expected " + config.getDimension()
            + " but got " + entry.getValue().length));
      }
    }

    return database.runAsync(tr -> upsertInternal(tr, vectors));
  }

  /**
   * Internal upsert implementation that runs within a transaction.
   */
  private CompletableFuture<Void> upsertInternal(Transaction tr, Map<Long, float[]> vectors) {
    // Ensure latest codebooks are loaded
    return ensureLatestCodebooksLoaded(tr).thenCompose(loaded -> {
      // If no codebooks exist, check if we should train initial codebooks
      ProductQuantizer currentPq =
          codebookStorage.getLatestProductQuantizer().join();
      if (!loaded && (currentPq == null || currentPq.getCodebookVersion() == 0)) {
        return maybeTrainInitialCodebooks(tr, vectors).thenCompose(trained -> {
          if (trained) {
            // Retry loading after training
            return ensureLatestCodebooksLoaded(tr)
                .thenCompose(reloaded -> processUpsertWithCodebooks(tr, vectors, reloaded));
          } else {
            // Continue without codebooks
            return processUpsertWithCodebooks(tr, vectors, false);
          }
        });
      }

      // Codebooks already exist or loaded
      return processUpsertWithCodebooks(tr, vectors, loaded);
    });
  }

  /**
   * Process the actual upsert operations with or without codebooks.
   */
  private CompletableFuture<Void> processUpsertWithCodebooks(
      Transaction tr, Map<Long, float[]> vectors, boolean loaded) {
    // Get the PQ if codebooks are loaded
    ProductQuantizer pq =
        loaded ? codebookStorage.getLatestProductQuantizer().join() : null;
    int cbv = loaded && pq != null ? pq.getCodebookVersion() : 0;

    List<CompletableFuture<Void>> futures = new ArrayList<>();

    for (Map.Entry<Long, float[]> entry : vectors.entrySet()) {
      long nodeId = entry.getKey();
      float[] vector = entry.getValue();

      // Ensure node adjacency exists
      CompletableFuture<Void> adjFuture = nodeAdjacencyStorage.storeAdjacency(tr, nodeId, List.of());

      // Create link task with either PQ-encoded or raw vector
      LinkTask.Builder taskBuilder = LinkTask.newBuilder().setNodeId(nodeId);

      if (pq != null && cbv > 0) {
        // Encode vector with PQ
        byte[] pqCode = pq.encode(vector);
        PqEncodedVector pqEncoded = PqEncodedVector.newBuilder()
            .setPqCode(ByteString.copyFrom(pqCode))
            .setCodebookVersion(cbv)
            .build();
        taskBuilder.setPqEncoded(pqEncoded);
      } else {
        // Store raw vector for later encoding
        RawVector.Builder rawVectorBuilder = RawVector.newBuilder();
        for (float value : vector) {
          rawVectorBuilder.addValues(value);
        }
        taskBuilder.setRawVector(rawVectorBuilder.build());
      }

      // Store vector sketch for future codebook retraining
      CompletableFuture<Void> sketchFuture = vectorSketchStorage.storeVectorSketch(tr, nodeId, vector);

      // Enqueue link task
      CompletableFuture<Void> enqueueFuture =
          linkTaskQueue.enqueue(tr, nodeId, taskBuilder.build()).thenApply(metadata -> null);

      // Combine all operations for this vector
      futures.add(allOf(adjFuture, sketchFuture, enqueueFuture));
    }

    // Update metrics
    VECTORS_INSERTED.add(vectors.size());

    return allOf(futures.toArray(CompletableFuture[]::new));
  }

  /**
   * Check if we should train initial codebooks and do so if needed.
   * This is called when no codebooks exist and vectors are being inserted.
   *
   * @param tr the transaction to use
   * @param currentVectors the vectors being inserted in this batch
   * @return true if codebooks were trained and stored
   */
  private CompletableFuture<Boolean> maybeTrainInitialCodebooks(Transaction tr, Map<Long, float[]> currentVectors) {
    // Check if we have enough vectors to train
    // We need at least some vectors for training, but we'll also check if there are existing sketches

    return vectorSketchStorage.countVectorSketches(tr).thenCompose(existingCount -> {
      int totalVectors = existingCount + currentVectors.size();

      // We need at least k=256 vectors to train meaningful codebooks
      // since each subspace uses k-means with k=256 centroids
      final int minVectorsForTraining = 256;
      if (totalVectors < minVectorsForTraining) {
        LOGGER.info(
            "Not enough vectors for training codebooks. Have {}, need at least {}",
            totalVectors,
            minVectorsForTraining);
        return completedFuture(false);
      }

      LOGGER.info("Training initial codebooks with {} vectors", totalVectors);

      // Collect training vectors: current batch + sample of existing sketches
      List<float[]> trainingVectors = new ArrayList<>(currentVectors.values());

      // If we have existing sketches, load a sample of them
      if (existingCount > 0) {
        int sampleSize = Math.min(existingCount, 1000 - currentVectors.size());
        return vectorSketchStorage.sampleVectorSketches(tr, sampleSize).thenCompose(sketches -> {
          trainingVectors.addAll(sketches);
          return trainAndStoreCodebooks(tr, trainingVectors);
        });
      } else {
        return trainAndStoreCodebooks(tr, trainingVectors);
      }
    });
  }

  /**
   * Train codebooks and store them in the database.
   *
   * @param tr the transaction to use
   * @param trainingVectors the vectors to use for training
   * @return true if successful
   */
  private CompletableFuture<Boolean> trainAndStoreCodebooks(Transaction tr, List<float[]> trainingVectors) {
    // Train the product quantizer
    // Get metrics and parameters from config
    int numSubvectors = config.getPqSubvectors();
    DistanceMetrics.Metric metric = convertDistanceMetric(config.getDistanceMetric());

    return ProductQuantizer.train(numSubvectors, metric, trainingVectors)
        .thenCompose(codebooks -> {
          // Get the trained codebooks

          if (codebooks == null) {
            LOGGER.error("Failed to train codebooks");
            return completedFuture(false);
          }

          // Get the next available codebook version
          return codebookStorage.listVersions().thenCompose(versions -> {
            int nextVersion = versions.isEmpty()
                ? 1
                : versions.stream().max(Integer::compare).get() + 1;

            CodebookStorage.TrainingStats stats =
                new CodebookStorage.TrainingStats(trainingVectors.size(), null);

            // Store codebooks with the next available version
            return codebookStorage
                .storeCodebooks(nextVersion, codebooks, stats)
                .thenCompose(storeResult -> {
                  // Update active codebook version in both database and CodebookStorage
                  byte[] cbvKey = vectorIndexKeys.activeCodebookVersionKey();
                  tr.set(cbvKey, encodeInt(nextVersion));

                  // Also update the active version in CodebookStorage
                  return codebookStorage.setActiveVersion(nextVersion);
                })
                .thenApply(v -> {
                  LOGGER.info(
                      "Successfully trained and stored initial codebooks (version {})",
                      nextVersion);

                  // Start LinkWorker now that we have codebooks
                  startLinkWorker();

                  return true;
                });
          });
        })
        .exceptionally(ex -> {
          LOGGER.error("Error training codebooks: {}", ex.getMessage());
          return false;
        });
  }

  /**
   * Reads the active codebook version from the database.
   *
   * @param tr the transaction to use
   * @return the active codebook version, or 0 if not set
   */
  private CompletableFuture<Long> readActiveCodebookVersion(Transaction tr) {
    // Read the active codebook version key
    byte[] key = vectorIndexKeys.activeCodebookVersionKey();
    return tr.get(key).thenApply(value -> {
      if (value == null) {
        return 0L;
      }
      return decodeInt(value);
    });
  }

  /**
   * Converts VectorSearchConfig.DistanceMetric to DistanceMetrics.Metric.
   */
  private DistanceMetrics.Metric convertDistanceMetric(VectorSearchConfig.DistanceMetric metric) {
    return switch (metric) {
      case L2 -> DistanceMetrics.Metric.L2;
      case INNER_PRODUCT -> DistanceMetrics.Metric.INNER_PRODUCT;
      case COSINE -> DistanceMetrics.Metric.COSINE;
    };
  }

  /**
   * Ensures the latest codebooks are loaded into the ProductQuantizer.
   * This method checks if we already have the latest version loaded to minimize contention.
   * If a transaction is provided, it reads the version within that transaction for consistency.
   *
   * @param tr transaction for reading the latest version
   * @return future containing true if successfully loaded, false otherwise
   */
  private CompletableFuture<Boolean> ensureLatestCodebooksLoaded(Transaction tr) {
    // Read the active codebook version within the transaction
    return readActiveCodebookVersion(tr).thenCompose(latestVersion -> {
      if (latestVersion == 0) {
        // No codebooks exist yet
        LOGGER.debug("No active codebook version found (version=0)");
        return completedFuture(false);
      }

      LOGGER.debug("Active codebook version: {}", latestVersion);

      // Use semaphore to prevent concurrent loading of the same version
      return codeBooksLoadingSemaphore
          .acquire()
          .thenCompose($ -> {
            // Double-check after acquiring the lock
            ProductQuantizer cachedPq = codebookStorage
                .getProductQuantizer(latestVersion)
                .join();
            if (cachedPq != null && cachedPq.getCodebookVersion() == latestVersion) {
              return completedFuture(true);
            }
            return codebookStorage
                .getProductQuantizer(latestVersion)
                .thenApply(productQuantizer -> {
                  if (productQuantizer == null) {
                    LOGGER.warn("Failed to load codebooks for version {}", latestVersion);
                    return false;
                  }

                  // Load into ProductQuantizer
                  // Codebooks are now loaded via cache
                  LOGGER.debug("Loaded codebooks version {} into ProductQuantizer", latestVersion);
                  return true;
                })
                .exceptionally(ex -> {
                  LOGGER.error("Error loading codebooks: {}", ex.getMessage());
                  return false;
                });
          })
          .whenComplete((v, ex) -> codeBooksLoadingSemaphore.release());
    });
  }

  @Override
  public CompletableFuture<List<SearchResult>> search(float[] queryVector, int k) {
    // Use default search list (max(16, k))
    int searchList = Math.max(16, k);
    return search(queryVector, k, searchList, 1500);
  }

  @Override
  public CompletableFuture<List<SearchResult>> search(float[] queryVector, int k, int searchList, int maxVisits) {
    if (!initialized) {
      return failedFuture(new IllegalStateException("Index not initialized"));
    }

    if (queryVector.length != config.getDimension()) {
      return failedFuture(new IllegalArgumentException("Query vector dimension mismatch: expected "
          + config.getDimension() + " but got " + queryVector.length));
    }

    SEARCHES_PERFORMED.add(1);

    // Use snapshot read for consistent search results
    return database.runAsync(tr -> {
      // Ensure latest codebooks are loaded and perform search
      return ensureLatestCodebooksLoaded(tr).thenCompose(loaded -> {
        if (!loaded) {
          // No codebooks available or failed to load
          LOGGER.warn("No codebooks loaded for search, returning empty results");
          return completedFuture(List.of());
        }

        LOGGER.debug("Codebooks loaded, performing search");

        // Get ProductQuantizer and perform beam search
        return codebookStorage.getLatestProductQuantizer().thenCompose(pq -> {
          if (pq == null) {
            LOGGER.warn("No ProductQuantizer available for search");
            return completedFuture(List.of());
          }
          return beamSearchEngine.search(tr, queryVector, k, searchList, maxVisits, pq);
        });
      });
    });
  }

  @Override
  public CompletableFuture<Void> delete(List<Long> ids) {
    if (!initialized) {
      return failedFuture(new IllegalStateException("Index not initialized"));
    }

    return database.runAsync(tr -> {
      List<CompletableFuture<Void>> futures = new ArrayList<>();

      for (Long nodeId : ids) {
        // Create unlink task
        UnlinkTask task = UnlinkTask.newBuilder().setNodeId(nodeId).build();

        // Enqueue unlink task
        futures.add(unlinkTaskQueue.enqueue(tr, nodeId, task).thenApply(metadata -> null));
      }

      // Update metrics
      VECTORS_DELETED.add(ids.size());

      return allOf(futures.toArray(new CompletableFuture[0]));
    });
  }

  @Override
  public CompletableFuture<IndexStats> getStats() {
    // TODO: Gather actual statistics
    return completedFuture(new IndexStats(0, 0, 0.0, 0, 0.0));
  }

  @Override
  public CompletableFuture<Boolean> isHealthy() {
    // Check basic health indicators
    return completedFuture(initialized && database != null);
  }

  @Override
  public void close() {
    LOGGER.info("Closing FdbVectorSearch index");

    // Stop the link worker thread
    stopLinkWorker();

    // Shutdown maintenance scheduler
    if (maintenanceScheduler != null && !maintenanceScheduler.isShutdown()) {
      maintenanceScheduler.shutdown();
      try {
        if (!maintenanceScheduler.awaitTermination(5, TimeUnit.SECONDS)) {
          maintenanceScheduler.shutdownNow();
        }
      } catch (InterruptedException e) {
        maintenanceScheduler.shutdownNow();
        Thread.currentThread().interrupt();
      }
    }

    // Clear caches
    if (codebookCache != null) {
      codebookCache.clear();
    }
    // PqBlockStorage cache will be cleared when the storage is garbage collected

    LOGGER.info("FdbVectorSearch index closed");
  }

  /**
   * Waits for pending indexing operations to complete by checking task queue status.
   * This is primarily useful for testing to ensure vectors are indexed before searching.
   *
   * <p>Note: This method provides a best-effort wait. In production, consider using
   * eventual consistency patterns rather than blocking on indexing completion.
   *
   * @param maxWait maximum time to wait in milliseconds
   * @return future that completes when indexing is likely complete or timeout is reached
   */
  public CompletableFuture<Void> waitForIndexing(Duration maxWait) {
    if (!initialized) {
      return failedFuture(new IllegalStateException("Index not initialized"));
    }

    LOGGER.info("Waiting for indexing to complete");

    // Check if LinkWorker is connected and queues are available
    if (linkTaskQueue == null || unlinkTaskQueue == null) {
      // If queues are not available, just return immediately
      return completedFuture(null);
    }

    Instant startTime = Instant.now();
    return AsyncUtil.whileTrue(() -> {
      CompletableFuture<Boolean> hasLinkTasksF = linkTaskQueue.hasVisibleUnclaimedTasks();
      CompletableFuture<Boolean> hasUnlinkTasksF = unlinkTaskQueue.hasVisibleUnclaimedTasks();
      return allOf(hasLinkTasksF, hasUnlinkTasksF)
          .thenApply($ -> {
            if (Instant.now().isAfter(startTime.plus(maxWait))) {
              return false;
            }
            if (hasLinkTasksF.join() || hasUnlinkTasksF.join()) {
              LOGGER.info(
                  "Waiting for indexing to complete. link_queue has_tasks: {}, unlink_queue"
                      + " has_tasks: {}",
                  hasLinkTasksF.join(),
                  hasUnlinkTasksF.join());
            } else {
              LOGGER.info("Indexing complete");
            }
            return hasLinkTasksF.join() || hasUnlinkTasksF.join();
          })
          .thenCompose(shouldLoop -> {
            if (shouldLoop) {
              CompletableFuture<Boolean> toReturn = new CompletableFuture<>();
              return toReturn.completeOnTimeout(true, 1, TimeUnit.SECONDS);
            }
            return completedFuture(false);
          });
    });
  }

  // Helper classes for cache keys
  record CodebookCacheKey(int version, int subspace) {}

  record PqBlockCacheKey(int version, long blockNumber) {}

  // Serializers for task queues
  static class LongSerializer implements TaskQueueConfig.TaskSerializer<Long> {
    @Override
    public ByteString serialize(Long value) {
      return ByteString.copyFrom(encodeInt(value));
    }

    @Override
    public Long deserialize(ByteString bytes) {
      return decodeInt(bytes.toByteArray());
    }
  }

  // Generic protobuf message serializer for task queues
  record ProtoSerializer<T extends Message>(Parser<T> parser) implements TaskQueueConfig.TaskSerializer<T> {

    @Override
    public ByteString serialize(T value) {
      return value.toByteString();
    }

    @Override
    public T deserialize(ByteString bytes) {
      try {
        return parser.parseFrom(bytes);
      } catch (Exception e) {
        throw new RuntimeException("Failed to parse protobuf message", e);
      }
    }
  }
}
