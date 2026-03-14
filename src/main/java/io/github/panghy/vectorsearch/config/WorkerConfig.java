package io.github.panghy.vectorsearch.config;

import java.time.Duration;
import java.time.InstantSource;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable configuration for global worker operational settings.
 *
 * <p>Holds all operational knobs that control how background workers behave (TTL, throttle,
 * compaction planner, vacuum, build batching, caches, etc.) plus data-format fallback defaults
 * used by {@code GlobalWorkerRunner} when {@code IndexMeta} has zero/default values.</p>
 *
 * <p>This class eliminates the need to construct a full {@link VectorIndexConfig} (with its
 * required {@code database} and {@code indexDir}) just to supply operational settings to
 * {@code GlobalWorkerRunner}.</p>
 *
 * @see VectorIndexConfig
 */
public final class WorkerConfig {

  // Operational settings
  private final int estimatedWorkerCount;
  private final Duration defaultTtl;
  private final Duration defaultThrottle;
  private final int maxConcurrentCompactions;
  private final long buildTxnLimitBytes;
  private final double buildTxnSoftLimitRatio;
  private final int buildSizeCheckEvery;
  private final Duration vacuumCooldown;
  private final double vacuumMinDeletedRatio;
  private final boolean autoFindCompactionCandidates;
  private final int compactionMinSegments;
  private final int compactionMaxSegments;
  private final double compactionMinFragmentation;
  private final double compactionAgeBiasWeight;
  private final double compactionSizeBiasWeight;
  private final double compactionFragBiasWeight;
  private final int codebookBatchLoadSize;
  private final int adjacencyBatchLoadSize;
  private final boolean prefetchCodebooksEnabled;
  private final boolean prefetchCodebooksSync;
  private final InstantSource instantSource;
  private final Map<String, String> metricAttributes;

  // Data-format fallback defaults
  private final int defaultMaxSegmentSize;
  private final int defaultPqM;
  private final int defaultPqK;
  private final int defaultGraphDegree;
  private final int defaultOversample;
  private final int defaultGraphBuildBreadth;
  private final double defaultGraphAlpha;

  private WorkerConfig(Builder b) {
    this.estimatedWorkerCount = b.estimatedWorkerCount;
    this.defaultTtl = Objects.requireNonNull(b.defaultTtl, "defaultTtl");
    this.defaultThrottle = Objects.requireNonNull(b.defaultThrottle, "defaultThrottle");
    this.maxConcurrentCompactions = b.maxConcurrentCompactions;
    this.buildTxnLimitBytes = b.buildTxnLimitBytes;
    this.buildTxnSoftLimitRatio = b.buildTxnSoftLimitRatio;
    this.buildSizeCheckEvery = b.buildSizeCheckEvery;
    this.vacuumCooldown = Objects.requireNonNull(b.vacuumCooldown, "vacuumCooldown");
    this.vacuumMinDeletedRatio = b.vacuumMinDeletedRatio;
    this.autoFindCompactionCandidates = b.autoFindCompactionCandidates;
    this.compactionMinSegments = b.compactionMinSegments;
    this.compactionMaxSegments = b.compactionMaxSegments;
    this.compactionMinFragmentation = b.compactionMinFragmentation;
    this.compactionAgeBiasWeight = b.compactionAgeBiasWeight;
    this.compactionSizeBiasWeight = b.compactionSizeBiasWeight;
    this.compactionFragBiasWeight = b.compactionFragBiasWeight;
    this.codebookBatchLoadSize = b.codebookBatchLoadSize;
    this.adjacencyBatchLoadSize = b.adjacencyBatchLoadSize;
    this.prefetchCodebooksEnabled = b.prefetchCodebooksEnabled;
    this.prefetchCodebooksSync = b.prefetchCodebooksSync;
    this.instantSource = Objects.requireNonNull(b.instantSource, "instantSource");
    this.metricAttributes = Map.copyOf(b.metricAttributes);

    this.defaultMaxSegmentSize = b.defaultMaxSegmentSize;
    this.defaultPqM = b.defaultPqM;
    this.defaultPqK = b.defaultPqK;
    this.defaultGraphDegree = b.defaultGraphDegree;
    this.defaultOversample = b.defaultOversample;
    this.defaultGraphBuildBreadth = b.defaultGraphBuildBreadth;
    this.defaultGraphAlpha = b.defaultGraphAlpha;
  }

  /** Creates a new builder with sensible defaults matching {@link VectorIndexConfig.Builder}. */
  public static Builder builder() {
    return new Builder();
  }

  // ---- Operational getters ----

  public int getEstimatedWorkerCount() {
    return estimatedWorkerCount;
  }

  public Duration getDefaultTtl() {
    return defaultTtl;
  }

  public Duration getDefaultThrottle() {
    return defaultThrottle;
  }

  public int getMaxConcurrentCompactions() {
    return maxConcurrentCompactions;
  }

  public long getBuildTxnLimitBytes() {
    return buildTxnLimitBytes;
  }

  public double getBuildTxnSoftLimitRatio() {
    return buildTxnSoftLimitRatio;
  }

  public int getBuildSizeCheckEvery() {
    return buildSizeCheckEvery;
  }

  public Duration getVacuumCooldown() {
    return vacuumCooldown;
  }

  public double getVacuumMinDeletedRatio() {
    return vacuumMinDeletedRatio;
  }

  public boolean isAutoFindCompactionCandidates() {
    return autoFindCompactionCandidates;
  }

  public int getCompactionMinSegments() {
    return compactionMinSegments;
  }

  public int getCompactionMaxSegments() {
    return compactionMaxSegments;
  }

  public double getCompactionMinFragmentation() {
    return compactionMinFragmentation;
  }

  public double getCompactionAgeBiasWeight() {
    return compactionAgeBiasWeight;
  }

  public double getCompactionSizeBiasWeight() {
    return compactionSizeBiasWeight;
  }

  public double getCompactionFragBiasWeight() {
    return compactionFragBiasWeight;
  }

  public int getCodebookBatchLoadSize() {
    return codebookBatchLoadSize;
  }

  public int getAdjacencyBatchLoadSize() {
    return adjacencyBatchLoadSize;
  }

  public boolean isPrefetchCodebooksEnabled() {
    return prefetchCodebooksEnabled;
  }

  public boolean isPrefetchCodebooksSync() {
    return prefetchCodebooksSync;
  }

  public InstantSource getInstantSource() {
    return instantSource;
  }

  public Map<String, String> getMetricAttributes() {
    return metricAttributes;
  }

  // ---- Data-format fallback getters ----

  public int getDefaultMaxSegmentSize() {
    return defaultMaxSegmentSize;
  }

  public int getDefaultPqM() {
    return defaultPqM;
  }

  public int getDefaultPqK() {
    return defaultPqK;
  }

  public int getDefaultGraphDegree() {
    return defaultGraphDegree;
  }

  public int getDefaultOversample() {
    return defaultOversample;
  }

  public int getDefaultGraphBuildBreadth() {
    return defaultGraphBuildBreadth;
  }

  public double getDefaultGraphAlpha() {
    return defaultGraphAlpha;
  }

  /** Builder for {@link WorkerConfig}. */
  public static final class Builder {
    // Operational — defaults match VectorIndexConfig.Builder
    private int estimatedWorkerCount = 1;
    private Duration defaultTtl = Duration.ofMinutes(5);
    private Duration defaultThrottle = Duration.ofSeconds(1);
    private int maxConcurrentCompactions = 1;
    private long buildTxnLimitBytes = 10L * 1024 * 1024;
    private double buildTxnSoftLimitRatio = 0.9;
    private int buildSizeCheckEvery = 32;
    private Duration vacuumCooldown = Duration.ZERO;
    private double vacuumMinDeletedRatio = 0.25;
    private boolean autoFindCompactionCandidates = true;
    private int compactionMinSegments = 2;
    private int compactionMaxSegments = 8;
    private double compactionMinFragmentation = 0.1;
    private double compactionAgeBiasWeight = 0.3;
    private double compactionSizeBiasWeight = 0.5;
    private double compactionFragBiasWeight = 0.2;
    private int codebookBatchLoadSize = 10_000;
    private int adjacencyBatchLoadSize = 10_000;
    private boolean prefetchCodebooksEnabled = true;
    private boolean prefetchCodebooksSync = false;
    private InstantSource instantSource = InstantSource.system();
    private final Map<String, String> metricAttributes = new HashMap<>();

    // Data-format fallback defaults — match VectorIndexConfig.Builder
    private int defaultMaxSegmentSize = 100_000;
    private int defaultPqM = 16;
    private int defaultPqK = 256;
    private int defaultGraphDegree = 64;
    private int defaultOversample = 2;
    private int defaultGraphBuildBreadth = 256;
    private double defaultGraphAlpha = 1.2;

    private Builder() {}

    // ---- Operational setters ----

    public Builder estimatedWorkerCount(int v) {
      this.estimatedWorkerCount = v;
      return this;
    }

    public Builder defaultTtl(Duration v) {
      this.defaultTtl = v;
      return this;
    }

    public Builder defaultThrottle(Duration v) {
      this.defaultThrottle = v;
      return this;
    }

    public Builder maxConcurrentCompactions(int v) {
      this.maxConcurrentCompactions = v;
      return this;
    }

    public Builder buildTxnLimitBytes(long v) {
      this.buildTxnLimitBytes = v;
      return this;
    }

    public Builder buildTxnSoftLimitRatio(double v) {
      this.buildTxnSoftLimitRatio = v;
      return this;
    }

    public Builder buildSizeCheckEvery(int v) {
      this.buildSizeCheckEvery = v;
      return this;
    }

    public Builder vacuumCooldown(Duration v) {
      this.vacuumCooldown = v;
      return this;
    }

    public Builder vacuumMinDeletedRatio(double v) {
      this.vacuumMinDeletedRatio = v;
      return this;
    }

    public Builder autoFindCompactionCandidates(boolean v) {
      this.autoFindCompactionCandidates = v;
      return this;
    }

    public Builder compactionMinSegments(int v) {
      this.compactionMinSegments = v;
      return this;
    }

    public Builder compactionMaxSegments(int v) {
      this.compactionMaxSegments = v;
      return this;
    }

    public Builder compactionMinFragmentation(double v) {
      this.compactionMinFragmentation = v;
      return this;
    }

    public Builder compactionAgeBiasWeight(double v) {
      this.compactionAgeBiasWeight = v;
      return this;
    }

    public Builder compactionSizeBiasWeight(double v) {
      this.compactionSizeBiasWeight = v;
      return this;
    }

    public Builder compactionFragBiasWeight(double v) {
      this.compactionFragBiasWeight = v;
      return this;
    }

    public Builder codebookBatchLoadSize(int v) {
      this.codebookBatchLoadSize = v;
      return this;
    }

    public Builder adjacencyBatchLoadSize(int v) {
      this.adjacencyBatchLoadSize = v;
      return this;
    }

    public Builder prefetchCodebooksEnabled(boolean v) {
      this.prefetchCodebooksEnabled = v;
      return this;
    }

    public Builder prefetchCodebooksSync(boolean v) {
      this.prefetchCodebooksSync = v;
      return this;
    }

    public Builder instantSource(InstantSource v) {
      this.instantSource = v;
      return this;
    }

    /** Adds a single metric attribute. */
    public Builder metricAttribute(String key, String value) {
      this.metricAttributes.put(key, value);
      return this;
    }

    /** Sets metric attributes in bulk (copied). */
    public Builder metricAttributes(Map<String, String> attrs) {
      this.metricAttributes.clear();
      if (attrs != null) this.metricAttributes.putAll(attrs);
      return this;
    }

    // ---- Data-format fallback setters ----

    public Builder defaultMaxSegmentSize(int v) {
      this.defaultMaxSegmentSize = v;
      return this;
    }

    public Builder defaultPqM(int v) {
      this.defaultPqM = v;
      return this;
    }

    public Builder defaultPqK(int v) {
      this.defaultPqK = v;
      return this;
    }

    public Builder defaultGraphDegree(int v) {
      this.defaultGraphDegree = v;
      return this;
    }

    public Builder defaultOversample(int v) {
      this.defaultOversample = v;
      return this;
    }

    public Builder defaultGraphBuildBreadth(int v) {
      this.defaultGraphBuildBreadth = v;
      return this;
    }

    public Builder defaultGraphAlpha(double v) {
      this.defaultGraphAlpha = v;
      return this;
    }

    /** Builds the immutable {@link WorkerConfig}. */
    public WorkerConfig build() {
      return new WorkerConfig(this);
    }
  }
}
