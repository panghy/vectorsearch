package io.github.panghy.vectorsearch.config;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.time.Duration;
import java.time.InstantSource;
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

  private final int estimatedWorkerCount;
  private final int localWorkerThreads;
  private final Duration defaultTtl;
  private final Duration defaultThrottle;
  private final InstantSource instantSource;

  // Batch sizes for async cache bulk loads
  private final int codebookBatchLoadSize;
  private final int adjacencyBatchLoadSize;
  private final java.util.Map<String, String> metricAttributes;
  private final boolean prefetchCodebooksEnabled;

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
    if (b.oversample <= 0) throw new IllegalArgumentException("oversample must be positive");
    this.oversample = b.oversample;

    if (b.estimatedWorkerCount <= 0) throw new IllegalArgumentException("estimatedWorkerCount must be positive");
    this.estimatedWorkerCount = b.estimatedWorkerCount;
    if (b.localWorkerThreads < 0) throw new IllegalArgumentException("localWorkerThreads must be >= 0");
    this.localWorkerThreads = b.localWorkerThreads;

    this.defaultTtl = requirePositive(b.defaultTtl, "defaultTtl");
    if (b.defaultThrottle.isNegative()) {
      throw new IllegalArgumentException("defaultThrottle must not be negative");
    }
    this.defaultThrottle = b.defaultThrottle;
    this.instantSource = Objects.requireNonNull(b.instantSource, "instantSource must not be null");

    if (b.codebookBatchLoadSize <= 0) throw new IllegalArgumentException("codebookBatchLoadSize must be positive");
    if (b.adjacencyBatchLoadSize <= 0)
      throw new IllegalArgumentException("adjacencyBatchLoadSize must be positive");
    this.codebookBatchLoadSize = b.codebookBatchLoadSize;
    this.adjacencyBatchLoadSize = b.adjacencyBatchLoadSize;
    this.metricAttributes = java.util.Map.copyOf(b.metricAttributes);
    this.prefetchCodebooksEnabled = b.prefetchCodebooksEnabled;
  }

  private static String requireNonBlank(String s, String name) {
    if (s == null || s.isBlank()) throw new IllegalArgumentException(name + " must not be blank");
    return s;
  }

  private static Duration requirePositive(Duration d, String name) {
    if (d == null) throw new IllegalArgumentException(name + " must not be null");
    if (d.isZero() || d.isNegative()) throw new IllegalArgumentException(name + " must be positive");
    return d;
  }

  /** Returns the FoundationDB database handle. */
  public Database getDatabase() {
    return database;
  }

  /** Returns the DirectorySubspace representing this index's root directory. */
  public DirectorySubspace getIndexDir() {
    return indexDir;
  }

  /** Returns the fixed embedding dimension. */
  public int getDimension() {
    return dimension;
  }

  /** Returns the distance metric. */
  public Metric getMetric() {
    return metric;
  }

  /** Returns the maximum number of vectors per segment. */
  public int getMaxSegmentSize() {
    return maxSegmentSize;
  }

  /** Returns the number of PQ subspaces (M). */
  public int getPqM() {
    return pqM;
  }

  /** Returns the number of PQ centroids per subspace (K). */
  public int getPqK() {
    return pqK;
  }

  /** Returns the target graph out-degree (R). */
  public int getGraphDegree() {
    return graphDegree;
  }

  /** Returns the oversample multiplier for merging candidates. */
  public int getOversample() {
    return oversample;
  }

  /** Returns the estimated number of background workers. */
  public int getEstimatedWorkerCount() {
    return estimatedWorkerCount;
  }

  /** Returns the number of local worker threads to auto-start. */
  public int getLocalWorkerThreads() {
    return localWorkerThreads;
  }

  /** Returns the default claim TTL used by background tasks. */
  public Duration getDefaultTtl() {
    return defaultTtl;
  }

  /** Returns the default throttle between tasks for the same key. */
  public Duration getDefaultThrottle() {
    return defaultThrottle;
  }

  /** Returns the time source (injectable for tests). */
  public InstantSource getInstantSource() {
    return instantSource;
  }

  /** Returns batch size for async codebook bulk loads. */
  public int getCodebookBatchLoadSize() {
    return codebookBatchLoadSize;
  }

  /** Returns batch size for async adjacency bulk loads. */
  public int getAdjacencyBatchLoadSize() {
    return adjacencyBatchLoadSize;
  }

  /** Additional metric attributes to add to emitted metrics/spans. */
  public java.util.Map<String, String> getMetricAttributes() {
    return metricAttributes;
  }

  /** Returns whether query-time codebook prefetch is enabled. */
  public boolean isPrefetchCodebooksEnabled() {
    return prefetchCodebooksEnabled;
  }

  /** Creates a new builder for {@link VectorIndexConfig}. */
  public static Builder builder(Database database, DirectorySubspace indexDir) {
    return new Builder(database, indexDir);
  }

  /** Builder for {@link VectorIndexConfig}. */
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
    private int estimatedWorkerCount = 1;
    private int localWorkerThreads = 0;
    private Duration defaultTtl = Duration.ofMinutes(5);
    private Duration defaultThrottle = Duration.ofSeconds(1);
    private InstantSource instantSource = InstantSource.system();
    private int codebookBatchLoadSize = 10_000;
    private int adjacencyBatchLoadSize = 10_000;
    private java.util.Map<String, String> metricAttributes = new java.util.HashMap<>();
    private boolean prefetchCodebooksEnabled = true;

    private Builder(Database database, DirectorySubspace indexDir) {
      this.database = database;
      this.indexDir = indexDir;
    }

    /** Sets the embedding dimension. */
    public Builder dimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    /** Sets the distance metric. */
    public Builder metric(Metric metric) {
      this.metric = metric;
      return this;
    }

    /** Sets the maximum number of vectors per segment. */
    public Builder maxSegmentSize(int maxSegmentSize) {
      this.maxSegmentSize = maxSegmentSize;
      return this;
    }

    /** Sets the PQ subspace count (M). */
    public Builder pqM(int pqM) {
      this.pqM = pqM;
      return this;
    }

    /** Sets the PQ centroid count per subspace (K). */
    public Builder pqK(int pqK) {
      this.pqK = pqK;
      return this;
    }

    /** Sets the target graph out-degree (R). */
    public Builder graphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /** Sets the oversample multiplier for candidate merging. */
    public Builder oversample(int oversample) {
      this.oversample = oversample;
      return this;
    }

    /** Sets the estimated number of background workers. */
    public Builder estimatedWorkerCount(int estimatedWorkerCount) {
      this.estimatedWorkerCount = estimatedWorkerCount;
      return this;
    }

    /** Sets how many local worker threads to auto-start (0 to disable). */
    public Builder localWorkerThreads(int localWorkerThreads) {
      this.localWorkerThreads = localWorkerThreads;
      return this;
    }

    /** Sets the default claim TTL used by background tasks. */
    public Builder defaultTtl(Duration defaultTtl) {
      this.defaultTtl = defaultTtl;
      return this;
    }

    /** Sets the default throttle between tasks for the same key. */
    public Builder defaultThrottle(Duration defaultThrottle) {
      this.defaultThrottle = defaultThrottle;
      return this;
    }

    /** Sets the time source used to obtain the current time. */
    public Builder instantSource(InstantSource instantSource) {
      this.instantSource = instantSource;
      return this;
    }

    /** Sets batch size for async codebook bulk loads. */
    public Builder codebookBatchLoadSize(int size) {
      this.codebookBatchLoadSize = size;
      return this;
    }

    /** Sets batch size for async adjacency bulk loads. */
    public Builder adjacencyBatchLoadSize(int size) {
      this.adjacencyBatchLoadSize = size;
      return this;
    }

    /** Adds a metric attribute (key/value) to be included on metrics/spans. */
    public Builder metricAttribute(String key, String value) {
      this.metricAttributes.put(key, value);
      return this;
    }

    /** Sets metric attributes in bulk (copied). */
    public Builder metricAttributes(java.util.Map<String, String> attrs) {
      this.metricAttributes.clear();
      if (attrs != null) this.metricAttributes.putAll(attrs);
      return this;
    }

    /** Enables/disables query-time codebook prefetch for SEALED segments. */
    public Builder prefetchCodebooksEnabled(boolean enabled) {
      this.prefetchCodebooksEnabled = enabled;
      return this;
    }

    /** Builds the immutable {@link VectorIndexConfig}. */
    public VectorIndexConfig build() {
      return new VectorIndexConfig(this);
    }
  }
}
