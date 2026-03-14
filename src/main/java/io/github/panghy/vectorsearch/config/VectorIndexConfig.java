package io.github.panghy.vectorsearch.config;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.time.Duration;
import java.time.InstantSource;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for a vector index instance, rooted in an FDB Directory (DirectoryLayer namespace).
 *
 * <p>Modeled after TaskQueueConfig: uses a builder, validates inputs, and exposes getters only.
 * The config carries both algorithmic parameters (dimension, metric, PQ, graph) and operational
 * settings (worker count, TTL, throttle, time source).</p>
 */
public final class VectorIndexConfig {

  public enum Metric {
    L2,
    COSINE
  }

  private final Database database;
  private final DirectorySubspace indexDir;

  private final int dimension;
  private final Metric metric;

  private final int maxSegmentSize;
  private final int pqM;
  private final int pqK;
  private final int graphDegree;
  private final int oversample;
  private final int graphBuildBreadth;
  private final double graphAlpha;

  private final int localWorkerThreads;
  private final int localMaintenanceWorkerThreads;

  // All operational settings are delegated to WorkerConfig
  private final WorkerConfig workerConfig;

  // Optional global task queue config
  private final GlobalTaskQueueConfig globalTaskQueueConfig;

  private VectorIndexConfig(Builder b) {
    this.database = Objects.requireNonNull(b.database, "database must not be null");
    this.indexDir = Objects.requireNonNull(b.indexDir, "indexDir must not be null");

    if (b.dimension <= 0) throw new IllegalArgumentException("dimension must be positive");
    this.dimension = b.dimension;
    this.metric = Objects.requireNonNull(b.metric, "metric must not be null");

    if (b.maxSegmentSize <= 0) throw new IllegalArgumentException("maxSegmentSize must be positive");
    this.maxSegmentSize = b.maxSegmentSize;
    if (b.pqM <= 0) throw new IllegalArgumentException("pqM must be positive");
    this.pqM = b.pqM;
    if (b.pqK <= 1) throw new IllegalArgumentException("pqK must be > 1");
    this.pqK = b.pqK;
    if (b.graphDegree <= 0) throw new IllegalArgumentException("graphDegree must be positive");
    this.graphDegree = b.graphDegree;
    if (b.graphBuildBreadth < b.graphDegree)
      throw new IllegalArgumentException("graphBuildBreadth must be >= graphDegree");
    if (b.graphAlpha < 0.0) throw new IllegalArgumentException("graphAlpha must be >= 0");
    this.graphBuildBreadth = b.graphBuildBreadth;
    this.graphAlpha = b.graphAlpha;
    if (b.oversample <= 0) throw new IllegalArgumentException("oversample must be positive");
    this.oversample = b.oversample;

    if (b.localWorkerThreads < 0) throw new IllegalArgumentException("localWorkerThreads must be >= 0");
    this.localWorkerThreads = b.localWorkerThreads;
    if (b.localMaintenanceWorkerThreads < 0)
      throw new IllegalArgumentException("localMaintenanceWorkerThreads must be >= 0");
    this.localMaintenanceWorkerThreads = b.localMaintenanceWorkerThreads;

    // Build or use the provided WorkerConfig for all operational settings
    if (b.workerConfig != null) {
      this.workerConfig = b.workerConfig;
    } else {
      // Validate operational fields before building WorkerConfig
      if (b.estimatedWorkerCount <= 0)
        throw new IllegalArgumentException("estimatedWorkerCount must be positive");
      if (b.maxConcurrentCompactions < 0)
        throw new IllegalArgumentException("maxConcurrentCompactions must be >= 0");
      Objects.requireNonNull(b.vacuumCooldown, "vacuumCooldown must not be null");
      if (b.vacuumCooldown.isNegative()) throw new IllegalArgumentException("vacuumCooldown must be >= 0");
      if (!(b.vacuumMinDeletedRatio >= 0.0 && b.vacuumMinDeletedRatio <= 1.0))
        throw new IllegalArgumentException("vacuumMinDeletedRatio must be in [0,1]");
      requirePositive(b.defaultTtl, "defaultTtl");
      Objects.requireNonNull(b.defaultThrottle, "defaultThrottle must not be null");
      if (b.defaultThrottle.isNegative())
        throw new IllegalArgumentException("defaultThrottle must not be negative");
      Objects.requireNonNull(b.instantSource, "instantSource must not be null");
      if (b.codebookBatchLoadSize <= 0)
        throw new IllegalArgumentException("codebookBatchLoadSize must be positive");
      if (b.adjacencyBatchLoadSize <= 0)
        throw new IllegalArgumentException("adjacencyBatchLoadSize must be positive");
      if (b.compactionMinSegments < 2) throw new IllegalArgumentException("compactionMinSegments must be >= 2");
      if (b.compactionMaxSegments < b.compactionMinSegments)
        throw new IllegalArgumentException("compactionMaxSegments must be >= compactionMinSegments");
      if (!(b.compactionMinFragmentation >= 0.0 && b.compactionMinFragmentation <= 1.0))
        throw new IllegalArgumentException("compactionMinFragmentation must be in [0,1]");
      if (b.compactionAgeBiasWeight < 0.0)
        throw new IllegalArgumentException("compactionAgeBiasWeight must be >= 0");
      if (b.compactionSizeBiasWeight < 0.0)
        throw new IllegalArgumentException("compactionSizeBiasWeight must be >= 0");
      if (b.compactionFragBiasWeight < 0.0)
        throw new IllegalArgumentException("compactionFragBiasWeight must be >= 0");
      if (b.buildTxnLimitBytes <= 0) throw new IllegalArgumentException("buildTxnLimitBytes must be positive");
      if (!(b.buildTxnSoftLimitRatio > 0.0 && b.buildTxnSoftLimitRatio < 1.0))
        throw new IllegalArgumentException("buildTxnSoftLimitRatio must be in (0,1)");
      if (b.buildSizeCheckEvery <= 0) throw new IllegalArgumentException("buildSizeCheckEvery must be positive");

      this.workerConfig = WorkerConfig.builder()
          .estimatedWorkerCount(b.estimatedWorkerCount)
          .defaultTtl(b.defaultTtl)
          .defaultThrottle(b.defaultThrottle)
          .maxConcurrentCompactions(b.maxConcurrentCompactions)
          .buildTxnLimitBytes(b.buildTxnLimitBytes)
          .buildTxnSoftLimitRatio(b.buildTxnSoftLimitRatio)
          .buildSizeCheckEvery(b.buildSizeCheckEvery)
          .vacuumCooldown(b.vacuumCooldown)
          .vacuumMinDeletedRatio(b.vacuumMinDeletedRatio)
          .autoFindCompactionCandidates(b.autoFindCompactionCandidates)
          .compactionMinSegments(b.compactionMinSegments)
          .compactionMaxSegments(b.compactionMaxSegments)
          .compactionMinFragmentation(b.compactionMinFragmentation)
          .compactionAgeBiasWeight(b.compactionAgeBiasWeight)
          .compactionSizeBiasWeight(b.compactionSizeBiasWeight)
          .compactionFragBiasWeight(b.compactionFragBiasWeight)
          .codebookBatchLoadSize(b.codebookBatchLoadSize)
          .adjacencyBatchLoadSize(b.adjacencyBatchLoadSize)
          .prefetchCodebooksEnabled(b.prefetchCodebooksEnabled)
          .prefetchCodebooksSync(b.prefetchCodebooksSync)
          .instantSource(b.instantSource)
          .metricAttributes(b.metricAttributes)
          .defaultMaxSegmentSize(b.maxSegmentSize)
          .defaultPqM(b.pqM)
          .defaultPqK(b.pqK)
          .defaultGraphDegree(b.graphDegree)
          .defaultOversample(b.oversample)
          .defaultGraphBuildBreadth(b.graphBuildBreadth)
          .defaultGraphAlpha(b.graphAlpha)
          .build();
    }

    this.globalTaskQueueConfig = b.globalTaskQueueConfig; // nullable
  }

  private static Duration requirePositive(Duration d, String name) {
    if (d == null) throw new IllegalArgumentException(name + " must not be null");
    if (d.isZero() || d.isNegative()) throw new IllegalArgumentException(name + " must be positive");
    return d;
  }

  /**
   * Returns the FoundationDB database handle.
   */
  public Database getDatabase() {
    return database;
  }

  /**
   * Returns the DirectorySubspace representing this index's root directory.
   */
  public DirectorySubspace getIndexDir() {
    return indexDir;
  }

  /**
   * Returns the fixed embedding dimension.
   */
  public int getDimension() {
    return dimension;
  }

  /**
   * Returns the distance metric.
   */
  public Metric getMetric() {
    return metric;
  }

  /**
   * Returns the maximum number of vectors per segment.
   */
  public int getMaxSegmentSize() {
    return maxSegmentSize;
  }

  /**
   * Returns the number of PQ subspaces (M).
   */
  public int getPqM() {
    return pqM;
  }

  /**
   * Returns the number of PQ centroids per subspace (K).
   */
  public int getPqK() {
    return pqK;
  }

  /**
   * Returns the target graph out-degree (R).
   */
  public int getGraphDegree() {
    return graphDegree;
  }

  /** Returns the graph build breadth (L_build). */
  public int getGraphBuildBreadth() {
    return graphBuildBreadth;
  }

  /** Returns the pruning alpha parameter (>1.0 enables pruning; <= 1.0 disables). */
  public double getGraphAlpha() {
    return graphAlpha;
  }

  /**
   * Returns the oversample multiplier for merging candidates.
   */
  public int getOversample() {
    return oversample;
  }

  /**
   * Returns the {@link WorkerConfig} holding all operational settings.
   */
  public WorkerConfig getWorkerConfig() {
    return workerConfig;
  }

  /** Returns the estimated number of background workers. */
  public int getEstimatedWorkerCount() {
    return workerConfig.getEstimatedWorkerCount();
  }

  /** Returns the number of local worker threads to auto-start. */
  public int getLocalWorkerThreads() {
    return localWorkerThreads;
  }

  /** Returns the number of local maintenance worker threads to auto-start. */
  public int getLocalMaintenanceWorkerThreads() {
    return localMaintenanceWorkerThreads;
  }

  /** Maximum number of in-flight compactions per index. */
  public int getMaxConcurrentCompactions() {
    return workerConfig.getMaxConcurrentCompactions();
  }

  /** Returns the cooldown between repeated vacuum enqueues for the same segment. */
  public Duration getVacuumCooldown() {
    return workerConfig.getVacuumCooldown();
  }

  /**
   * Minimum deleted ratio [0, 1] that triggers auto-enqueue of a vacuum task after deletes.
   */
  public double getVacuumMinDeletedRatio() {
    return workerConfig.getVacuumMinDeletedRatio();
  }

  /** Returns the default claim TTL used by background tasks. */
  public Duration getDefaultTtl() {
    return workerConfig.getDefaultTtl();
  }

  /** Returns the default throttle between tasks for the same key. */
  public Duration getDefaultThrottle() {
    return workerConfig.getDefaultThrottle();
  }

  /** Returns the time source (injectable for tests). */
  public InstantSource getInstantSource() {
    return workerConfig.getInstantSource();
  }

  /** Returns batch size for async codebook bulk loads. */
  public int getCodebookBatchLoadSize() {
    return workerConfig.getCodebookBatchLoadSize();
  }

  /** Returns batch size for async adjacency bulk loads. */
  public int getAdjacencyBatchLoadSize() {
    return workerConfig.getAdjacencyBatchLoadSize();
  }

  /** Additional metric attributes to add to emitted metrics/spans. */
  public Map<String, String> getMetricAttributes() {
    return workerConfig.getMetricAttributes();
  }

  /** Returns whether query-time codebook prefetch is enabled. */
  public boolean isPrefetchCodebooksEnabled() {
    return workerConfig.isPrefetchCodebooksEnabled();
  }

  /** When true, query waits for codebook prefetch to complete before searching (test-only). */
  public boolean isPrefetchCodebooksSync() {
    return workerConfig.isPrefetchCodebooksSync();
  }

  /** When true, vacuum enqueues FindCompactionCandidates for small sealed segments. */
  public boolean isAutoFindCompactionCandidates() {
    return workerConfig.isAutoFindCompactionCandidates();
  }

  /** Minimum number of segments required to trigger compaction (default 2). */
  public int getCompactionMinSegments() {
    return workerConfig.getCompactionMinSegments();
  }

  /** Maximum number of segments to merge at once (default 8). */
  public int getCompactionMaxSegments() {
    return workerConfig.getCompactionMaxSegments();
  }

  /** Minimum average deleted ratio across candidates to proceed with compaction (default 0.1). */
  public double getCompactionMinFragmentation() {
    return workerConfig.getCompactionMinFragmentation();
  }

  /** Weight for age score in composite compaction ranking (default 0.3). */
  public double getCompactionAgeBiasWeight() {
    return workerConfig.getCompactionAgeBiasWeight();
  }

  /** Weight for size score (smaller = higher) in composite compaction ranking (default 0.5). */
  public double getCompactionSizeBiasWeight() {
    return workerConfig.getCompactionSizeBiasWeight();
  }

  /** Weight for fragmentation score in composite compaction ranking (default 0.2). */
  public double getCompactionFragBiasWeight() {
    return workerConfig.getCompactionFragBiasWeight();
  }

  /** Upper bound for FDB transaction size (bytes) used by segment build batching. */
  public long getBuildTxnLimitBytes() {
    return workerConfig.getBuildTxnLimitBytes();
  }

  /** Ratio of limit where we split (e.g., 0.9 => leave 10% headroom). */
  public double getBuildTxnSoftLimitRatio() {
    return workerConfig.getBuildTxnSoftLimitRatio();
  }

  /** Writes between approximate-size checks during build batching. */
  public int getBuildSizeCheckEvery() {
    return workerConfig.getBuildSizeCheckEvery();
  }

  /**
   * Returns the optional global task queue configuration, or {@code null} if not set.
   */
  public GlobalTaskQueueConfig getGlobalTaskQueueConfig() {
    return globalTaskQueueConfig;
  }

  /**
   * Returns {@code true} if a global task queue configuration has been set,
   * indicating that local queues/workers should be skipped.
   */
  public boolean isGlobalTaskQueueEnabled() {
    return globalTaskQueueConfig != null;
  }

  /**
   * Creates a new builder for {@link VectorIndexConfig}.
   */
  public static Builder builder(Database database, DirectorySubspace indexDir) {
    return new Builder(database, indexDir);
  }

  /**
   * Builder for {@link VectorIndexConfig}.
   */
  public static final class Builder {
    private final Database database;
    private final DirectorySubspace indexDir;

    private int dimension = 768;
    private Metric metric = Metric.L2;
    private int maxSegmentSize = 100_000;
    private int pqM = 16;
    private int pqK = 256;
    private int graphDegree = 64;
    private int oversample = 2;
    private int graphBuildBreadth = 256;
    private double graphAlpha = 1.2;
    private int estimatedWorkerCount = 1;
    private int localWorkerThreads = 0;
    private int localMaintenanceWorkerThreads = 0;
    private int maxConcurrentCompactions = 1;
    private Duration vacuumCooldown = Duration.ZERO;
    private double vacuumMinDeletedRatio = 0.25;
    private Duration defaultTtl = Duration.ofMinutes(5);
    private Duration defaultThrottle = Duration.ofSeconds(1);
    private InstantSource instantSource = InstantSource.system();
    private int codebookBatchLoadSize = 10_000;
    private int adjacencyBatchLoadSize = 10_000;
    private final Map<String, String> metricAttributes = new HashMap<>();
    private boolean prefetchCodebooksEnabled = true;
    private boolean prefetchCodebooksSync = false;
    private boolean autoFindCompactionCandidates = true;
    private int compactionMinSegments = 2;
    private int compactionMaxSegments = 8;
    private double compactionMinFragmentation = 0.1;
    private double compactionAgeBiasWeight = 0.3;
    private double compactionSizeBiasWeight = 0.5;
    private double compactionFragBiasWeight = 0.2;
    private long buildTxnLimitBytes = 10L * 1024 * 1024; // 10 MB
    private double buildTxnSoftLimitRatio = 0.9; // leave 10% headroom
    private int buildSizeCheckEvery = 32;
    private GlobalTaskQueueConfig globalTaskQueueConfig;
    private WorkerConfig workerConfig; // nullable; when set, overrides individual operational fields

    private Builder(Database database, DirectorySubspace indexDir) {
      this.database = database;
      this.indexDir = indexDir;
    }

    /**
     * Sets a pre-built {@link WorkerConfig} for all operational settings.
     * When set, individual operational setters on this builder are ignored.
     */
    public Builder workerConfig(WorkerConfig workerConfig) {
      this.workerConfig = workerConfig;
      return this;
    }

    /**
     * Sets the embedding dimension.
     */
    public Builder dimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    /**
     * Sets the distance metric.
     */
    public Builder metric(Metric metric) {
      this.metric = metric;
      return this;
    }

    /**
     * Sets the maximum number of vectors per segment.
     */
    public Builder maxSegmentSize(int maxSegmentSize) {
      this.maxSegmentSize = maxSegmentSize;
      return this;
    }

    /**
     * Sets the PQ subspace count (M).
     */
    public Builder pqM(int pqM) {
      this.pqM = pqM;
      return this;
    }

    /**
     * Sets the PQ centroid count per subspace (K).
     */
    public Builder pqK(int pqK) {
      this.pqK = pqK;
      return this;
    }

    /**
     * Sets the target graph out-degree (R).
     */
    public Builder graphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /** Sets the graph build breadth (L_build). Must be >= graphDegree. */
    public Builder graphBuildBreadth(int breadth) {
      this.graphBuildBreadth = breadth;
      return this;
    }

    /** Sets the Vamana-like pruning alpha (>= 0). alpha <= 1.0 disables pruning. */
    public Builder graphAlpha(double alpha) {
      this.graphAlpha = alpha;
      return this;
    }

    /**
     * Sets the oversample multiplier for candidate merging.
     */
    public Builder oversample(int oversample) {
      this.oversample = oversample;
      return this;
    }

    /**
     * Sets the estimated number of background workers.
     */
    public Builder estimatedWorkerCount(int estimatedWorkerCount) {
      this.estimatedWorkerCount = estimatedWorkerCount;
      return this;
    }

    /**
     * Sets how many local worker threads to auto-start (0 to disable).
     */
    public Builder localWorkerThreads(int localWorkerThreads) {
      this.localWorkerThreads = localWorkerThreads;
      return this;
    }

    /**
     * Sets how many local maintenance worker threads to auto-start (0 to disable).
     */
    public Builder localMaintenanceWorkerThreads(int localMaintenanceWorkerThreads) {
      this.localMaintenanceWorkerThreads = localMaintenanceWorkerThreads;
      return this;
    }

    /** Sets maximum number of in-flight compactions per index (default 1). */
    public Builder maxConcurrentCompactions(int max) {
      this.maxConcurrentCompactions = max;
      return this;
    }

    /**
     * Sets min deleted ratio [0,1] to auto-enqueue vacuum after deletes.
     */
    public Builder vacuumMinDeletedRatio(double ratio) {
      this.vacuumMinDeletedRatio = ratio;
      return this;
    }

    /** Sets the cooldown between repeated vacuum enqueues for the same segment. */
    public Builder vacuumCooldown(Duration cooldown) {
      this.vacuumCooldown = cooldown;
      return this;
    }

    /**
     * Sets the default claim TTL used by background tasks.
     */
    public Builder defaultTtl(Duration defaultTtl) {
      this.defaultTtl = defaultTtl;
      return this;
    }

    /**
     * Sets the default throttle between tasks for the same key.
     */
    public Builder defaultThrottle(Duration defaultThrottle) {
      this.defaultThrottle = defaultThrottle;
      return this;
    }

    /**
     * Sets the time source used to obtain the current time.
     */
    public Builder instantSource(InstantSource instantSource) {
      this.instantSource = instantSource;
      return this;
    }

    /**
     * Sets batch size for async codebook bulk loads.
     */
    public Builder codebookBatchLoadSize(int size) {
      this.codebookBatchLoadSize = size;
      return this;
    }

    /**
     * Sets batch size for async adjacency bulk loads.
     */
    public Builder adjacencyBatchLoadSize(int size) {
      this.adjacencyBatchLoadSize = size;
      return this;
    }

    /**
     * Adds a metric attribute (key/value) to be included on metrics/spans.
     */
    public Builder metricAttribute(String key, String value) {
      this.metricAttributes.put(key, value);
      return this;
    }

    /**
     * Sets metric attributes in bulk (copied).
     */
    public Builder metricAttributes(Map<String, String> attrs) {
      this.metricAttributes.clear();
      if (attrs != null) this.metricAttributes.putAll(attrs);
      return this;
    }

    /**
     * Enables/disables query-time codebook prefetch for SEALED segments.
     */
    public Builder prefetchCodebooksEnabled(boolean enabled) {
      this.prefetchCodebooksEnabled = enabled;
      return this;
    }

    /**
     * If enabled, query blocks until codebook prefetch completes. Useful in tests to avoid
     * polling/sleeps. Default false.
     */
    public Builder prefetchCodebooksSync(boolean sync) {
      this.prefetchCodebooksSync = sync;
      return this;
    }

    /** Enables/disables auto-enqueue of find-compaction-candidates after vacuum. */
    public Builder autoFindCompactionCandidates(boolean enabled) {
      this.autoFindCompactionCandidates = enabled;
      return this;
    }

    /** Sets minimum segments required to trigger compaction (default 2, must be >= 2). */
    public Builder compactionMinSegments(int min) {
      this.compactionMinSegments = min;
      return this;
    }

    /** Sets maximum segments to merge at once (default 8). */
    public Builder compactionMaxSegments(int max) {
      this.compactionMaxSegments = max;
      return this;
    }

    /** Sets minimum average deleted ratio across candidates to proceed (default 0.1). */
    public Builder compactionMinFragmentation(double ratio) {
      this.compactionMinFragmentation = ratio;
      return this;
    }

    /** Sets weight for age score in composite ranking (default 0.3). */
    public Builder compactionAgeBiasWeight(double weight) {
      this.compactionAgeBiasWeight = weight;
      return this;
    }

    /** Sets weight for size score in composite ranking (default 0.5). */
    public Builder compactionSizeBiasWeight(double weight) {
      this.compactionSizeBiasWeight = weight;
      return this;
    }

    /** Sets weight for fragmentation score in composite ranking (default 0.2). */
    public Builder compactionFragBiasWeight(double weight) {
      this.compactionFragBiasWeight = weight;
      return this;
    }

    /**
     * Sets the assumed FDB transaction hard limit in bytes (default 10MB).
     */
    public Builder buildTxnLimitBytes(long bytes) {
      this.buildTxnLimitBytes = bytes;
      return this;
    }

    /**
     * Sets the soft ratio of the limit where we split the build write (default 0.9).
     */
    public Builder buildTxnSoftLimitRatio(double ratio) {
      this.buildTxnSoftLimitRatio = ratio;
      return this;
    }

    /**
     * Sets how many writes between approximate-size checks during build (default 32).
     */
    public Builder buildSizeCheckEvery(int n) {
      this.buildSizeCheckEvery = n;
      return this;
    }

    /**
     * Sets the optional global task queue configuration. When set, the index will enqueue
     * tasks into the shared global queues and skip creating local queues/workers.
     *
     * @param config the global task queue config (may be {@code null} to disable)
     * @return this builder
     */
    public Builder globalTaskQueueConfig(GlobalTaskQueueConfig config) {
      this.globalTaskQueueConfig = config;
      return this;
    }

    /**
     * Builds the immutable {@link VectorIndexConfig}.
     */
    public VectorIndexConfig build() {
      return new VectorIndexConfig(this);
    }
  }
}
