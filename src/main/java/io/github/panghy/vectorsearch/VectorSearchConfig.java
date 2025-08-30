package io.github.panghy.vectorsearch;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.directory.Directory;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.vectorsearch.proto.Config;
import java.time.Duration;
import java.time.InstantSource;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import lombok.Getter;

/**
 * Configuration for a VectorSearch instance that manages approximate nearest neighbor search
 * using FoundationDB as the storage backend with DiskANN-style graph traversal and Product
 * Quantization (PQ) for vector compression.
 *
 * <p>This configuration controls all aspects of the vector search system including:
 * <ul>
 *   <li>Vector dimensionality and distance metrics</li>
 *   <li>Product Quantization parameters for compression</li>
 *   <li>Graph construction and search parameters</li>
 *   <li>Storage optimization and caching strategies</li>
 *   <li>Background worker configurations</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * VectorSearchConfig config = VectorSearchConfig.builder(database, directory)
 *     .dimension(768)  // BERT embeddings
 *     .distanceMetric(DistanceMetric.COSINE)
 *     .pqEnabled(true)
 *     .graphDegree(64)
 *     .build();
 * }</pre>
 */
public class VectorSearchConfig {

  /**
   * Distance metric used for similarity computation between vectors.
   */
  public enum DistanceMetric {
    /**
     * Euclidean (L2) distance - sum of squared differences
     */
    L2,
    /**
     * Inner product (dot product) - for maximum similarity
     */
    INNER_PRODUCT,
    /**
     * Cosine similarity - normalized inner product, angle-based
     */
    COSINE
  }

  /**
   * -- GETTER --
   * Gets the FoundationDB database for this vector search instance.
   * All vector data, PQ codes, and graph structures are stored in this database.
   */
  @Getter
  private final Database database;

  /**
   * -- GETTER --
   * Gets the FoundationDB directory for this vector search instance.
   * All keys will be prefixed with this directory's prefix, allowing multiple
   * collections to coexist in the same database.
   */
  @Getter
  private final Directory directory;

  /**
   * -- GETTER --
   * Gets the dimensionality of vectors in this collection.
   * This must be set at creation time and cannot be changed.
   * Common values: 128, 256, 512, 768 (BERT), 1536 (OpenAI ada-002).
   */
  @Getter
  private final int dimension;

  /**
   * -- GETTER --
   * Gets the distance metric used for similarity calculations.
   * This affects both PQ training and graph construction.
   */
  @Getter
  private final DistanceMetric distanceMetric;

  // ============= Product Quantization Configuration =============

  /**
   * -- GETTER --
   * Whether Product Quantization is enabled for vector compression.
   * When enabled, vectors are compressed to ~dimension/2 bytes using PQ codes.
   * Disabling PQ requires storing full vectors (4*dimension bytes per vector).
   */
  @Getter
  private final boolean pqEnabled;

  /**
   * -- GETTER --
   * Gets the number of subvectors for PQ decomposition.
   * Must divide dimension evenly. Higher values give better accuracy but use more storage.
   * Default: dimension/2 (following common practice).
   * Typical range: dimension/4 to dimension/2.
   */
  @Getter
  private final int pqSubvectors;

  /**
   * -- GETTER --
   * Gets the number of bits per PQ code (per subvector).
   * Currently only 8 is supported (256 centroids per subspace).
   * This gives 1 byte per subvector, aligning with Milvus defaults.
   */
  @Getter
  private final int pqNbits;

  /**
   * -- GETTER --
   * Gets the number of PQ codes to pack in a single storage block.
   * This affects read-modify-write efficiency in FoundationDB.
   * Recommended: 256-512 to keep block size at 24-64KB (optimal for FDB).
   * Block size in bytes = codesPerBlock * pqSubvectors.
   */
  @Getter
  private final int codesPerBlock;

  // ============= Graph Configuration =============

  /**
   * -- GETTER --
   * Gets the maximum number of neighbors per node in the proximity graph.
   * This is equivalent to DiskANN's R parameter.
   * Higher values improve recall but increase memory usage and write amplification.
   * Default: 64 (DiskANN default).
   */
  @Getter
  private final int graphDegree;

  /**
   * -- GETTER --
   * Gets the default search list size for queries.
   * This controls the beam width during graph traversal.
   * Will be automatically adjusted to max(searchList, topK) per Milvus behavior.
   * Default: 16.
   */
  @Getter
  private final int defaultSearchList;

  /**
   * -- GETTER --
   * Gets the maximum number of nodes to visit during a single search.
   * This is a safety limit to prevent runaway queries on poorly connected graphs.
   * Default: 1500.
   */
  @Getter
  private final int maxSearchVisits;

  /**
   * -- GETTER --
   * Gets the number of primary entry points for search initialization.
   * These are high-quality seed nodes (medoids or high-degree nodes).
   * More entry points improve recall but increase search initialization cost.
   * Default: 32.
   */
  @Getter
  private final int entryPointCount;

  /**
   * -- GETTER --
   * Gets the interval for refreshing entry points.
   * Entry points are recomputed periodically to maintain search quality.
   * Default: 1 hour.
   */
  @Getter
  private final Duration entryPointRefreshInterval;

  // ============= Cache Configuration =============

  /**
   * -- GETTER --
   * Gets the maximum size of the PQ block cache in bytes.
   * This cache stores frequently accessed PQ code blocks to reduce FDB reads.
   * Larger cache improves query performance but uses more memory.
   * Default: 1GB.
   */
  @Getter
  private final long pqBlockCacheSize;

  /**
   * -- GETTER --
   * Gets the maximum number of adjacency lists to cache.
   * Each adjacency list is small (~512 bytes), so this can be generous.
   * Default: 100,000 nodes.
   */
  @Getter
  private final int adjacencyCacheSize;

  /**
   * -- GETTER --
   * Gets whether to keep all PQ codebooks in memory.
   * Codebooks are required for every query and are relatively small
   * (256 * dimension * 2 bytes per subspace).
   * Default: true.
   */
  @Getter
  private final boolean keepCodebooksInMemory;

  // ============= Worker Configuration =============

  /**
   * -- GETTER --
   * Gets the task queue configuration for background link/unlink operations.
   * This queue handles asynchronous graph construction after vector insertion.
   * If null, a default configuration will be created.
   */
  @Getter
  private final TaskQueueConfig<Long, ?> taskQueueConfig;

  /**
   * -- GETTER --
   * Gets the number of parallel link workers for graph construction.
   * More workers speed up index building but increase FDB load.
   * Default: 4.
   */
  @Getter
  private final int linkWorkerCount;

  /**
   * -- GETTER --
   * Gets the number of parallel unlink workers for node deletion.
   * Unlink operations are typically faster than links.
   * Default: 2.
   */
  @Getter
  private final int unlinkWorkerCount;

  /**
   * -- GETTER --
   * Gets the executor for background operations.
   * This is used for all async work including graph construction and maintenance.
   * Default: ForkJoinPool.commonPool().
   */
  @Getter
  private final Executor backgroundExecutor;

  // ============= Graph Maintenance Configuration =============

  /**
   * -- GETTER --
   * Gets whether automatic graph repair is enabled.
   * When enabled, the system periodically checks for and repairs graph disconnections.
   * Default: true.
   */
  @Getter
  private final boolean autoRepairEnabled;

  /**
   * -- GETTER --
   * Gets the interval for graph connectivity checks.
   * The system will periodically analyze graph connectivity and trigger repairs if needed.
   * Default: 6 hours.
   */
  @Getter
  private final Duration graphRepairInterval;

  /**
   * -- GETTER --
   * Gets the minimum percentage of nodes that must be in the main component.
   * If connectivity drops below this threshold, repair is triggered.
   * Default: 0.95 (95%).
   */
  @Getter
  private final double minConnectivityThreshold;

  // ============= Storage Configuration =============

  /**
   * -- GETTER --
   * Gets whether to use key striping for hot PQ blocks.
   * When enabled, PQ blocks are striped across multiple keys to reduce contention.
   * Stripe count is determined by stripeCount parameter.
   * Default: false (enable only if experiencing contention).
   */
  @Getter
  private final boolean useBlockStriping;

  /**
   * -- GETTER --
   * Gets the number of stripes for PQ block striping.
   * Only used if useBlockStriping is true.
   * Default: 4.
   */
  @Getter
  private final int stripeCount;

  /**
   * -- GETTER --
   * Gets whether to store vector sketches for codebook rotation.
   * Sketches allow approximate vector reconstruction without storing full vectors.
   * Adds ~32 bytes per vector of storage overhead.
   * Default: true.
   */
  @Getter
  private final boolean storeVectorSketches;

  /**
   * -- GETTER --
   * Gets the instant source for time-based operations.
   * Used for timestamps, TTLs, and scheduling.
   * Default: InstantSource.system().
   */
  @Getter
  private final InstantSource instantSource;

  private VectorSearchConfig(Builder builder) {
    // Core configuration
    this.database = Objects.requireNonNull(builder.database, "database must not be null");
    this.directory = Objects.requireNonNull(builder.directory, "directory must not be null");
    this.dimension = builder.dimension;
    this.distanceMetric = Objects.requireNonNull(builder.distanceMetric, "distanceMetric must not be null");

    // PQ configuration
    this.pqEnabled = builder.pqEnabled;
    this.pqSubvectors = builder.pqSubvectors;
    this.pqNbits = builder.pqNbits;
    this.codesPerBlock = builder.codesPerBlock;

    // Graph configuration
    this.graphDegree = builder.graphDegree;
    this.defaultSearchList = builder.defaultSearchList;
    this.maxSearchVisits = builder.maxSearchVisits;
    this.entryPointCount = builder.entryPointCount;
    this.entryPointRefreshInterval = Objects.requireNonNull(builder.entryPointRefreshInterval);

    // Cache configuration
    this.pqBlockCacheSize = builder.pqBlockCacheSize;
    this.adjacencyCacheSize = builder.adjacencyCacheSize;
    this.keepCodebooksInMemory = builder.keepCodebooksInMemory;

    // Worker configuration
    this.taskQueueConfig = builder.taskQueueConfig;
    this.linkWorkerCount = builder.linkWorkerCount;
    this.unlinkWorkerCount = builder.unlinkWorkerCount;
    this.backgroundExecutor = Objects.requireNonNull(builder.backgroundExecutor);

    // Graph maintenance
    this.autoRepairEnabled = builder.autoRepairEnabled;
    this.graphRepairInterval = Objects.requireNonNull(builder.graphRepairInterval);
    this.minConnectivityThreshold = builder.minConnectivityThreshold;

    // Storage configuration
    this.useBlockStriping = builder.useBlockStriping;
    this.stripeCount = builder.stripeCount;
    this.storeVectorSketches = builder.storeVectorSketches;
    this.instantSource = Objects.requireNonNull(builder.instantSource);

    // Validation
    validateConfiguration();
  }

  /**
   * Validates that all configuration parameters are within acceptable ranges
   * and that interdependent parameters are compatible.
   */
  private void validateConfiguration() {
    // Dimension validation
    if (dimension <= 0) {
      throw new IllegalArgumentException("dimension must be positive, got: " + dimension);
    }
    if (dimension > 4096) {
      throw new IllegalArgumentException("dimension too large (max 4096), got: " + dimension);
    }

    // PQ validation
    if (pqEnabled) {
      if (pqSubvectors <= 0 || pqSubvectors > dimension) {
        throw new IllegalArgumentException(
            "pqSubvectors must be between 1 and dimension, got: " + pqSubvectors);
      }
      if (dimension % pqSubvectors != 0) {
        throw new IllegalArgumentException("dimension must be divisible by pqSubvectors, got dimension="
            + dimension + ", pqSubvectors=" + pqSubvectors);
      }
      if (pqNbits != 8) {
        throw new IllegalArgumentException("only 8-bit PQ is currently supported, got: " + pqNbits);
      }
      if (codesPerBlock <= 0 || codesPerBlock > 1024) {
        throw new IllegalArgumentException("codesPerBlock must be between 1 and 1024, got: " + codesPerBlock);
      }
    }

    // Graph validation
    if (graphDegree <= 0 || graphDegree > 256) {
      throw new IllegalArgumentException("graphDegree must be between 1 and 256, got: " + graphDegree);
    }
    if (defaultSearchList <= 0) {
      throw new IllegalArgumentException("defaultSearchList must be positive, got: " + defaultSearchList);
    }
    if (maxSearchVisits <= 0) {
      throw new IllegalArgumentException("maxSearchVisits must be positive, got: " + maxSearchVisits);
    }
    if (entryPointCount <= 0 || entryPointCount > 256) {
      throw new IllegalArgumentException("entryPointCount must be between 1 and 256, got: " + entryPointCount);
    }

    // Cache validation
    if (pqBlockCacheSize < 0) {
      throw new IllegalArgumentException("pqBlockCacheSize must be non-negative, got: " + pqBlockCacheSize);
    }
    if (adjacencyCacheSize < 0) {
      throw new IllegalArgumentException("adjacencyCacheSize must be non-negative, got: " + adjacencyCacheSize);
    }

    // Worker validation
    if (linkWorkerCount <= 0) {
      throw new IllegalArgumentException("linkWorkerCount must be positive, got: " + linkWorkerCount);
    }
    if (unlinkWorkerCount <= 0) {
      throw new IllegalArgumentException("unlinkWorkerCount must be positive, got: " + unlinkWorkerCount);
    }

    // Graph maintenance validation
    if (minConnectivityThreshold < 0 || minConnectivityThreshold > 1) {
      throw new IllegalArgumentException(
          "minConnectivityThreshold must be between 0 and 1, got: " + minConnectivityThreshold);
    }
    if (graphRepairInterval.isNegative() || graphRepairInterval.isZero()) {
      throw new IllegalArgumentException("graphRepairInterval must be positive");
    }

    // Storage validation
    if (useBlockStriping && (stripeCount <= 0 || stripeCount > 16)) {
      throw new IllegalArgumentException("stripeCount must be between 1 and 16, got: " + stripeCount);
    }
  }

  /**
   * Validates this configuration against a stored configuration from FoundationDB.
   * <p>This method checks that all immutable values match the stored configuration.
   * If any immutable value differs, an exception is thrown.
   *
   * @param storedConfig the configuration stored in FoundationDB (protobuf Config message)
   * @throws IndexConfigurationException if any immutable configuration value doesn't match
   */
  public void validateAgainstStored(Config storedConfig) throws IndexConfigurationException {
    // Validate dimension (IMMUTABLE)
    if (storedConfig.getDimension() != this.dimension) {
      throw new IndexConfigurationException(String.format(
          "Dimension mismatch: configured=%d, stored=%d. "
              + "Dimension cannot be changed after collection creation.",
          this.dimension, storedConfig.getDimension()));
    }

    // Validate distance metric (IMMUTABLE)
    String storedMetric = storedConfig.getMetric();
    String configuredMetric = this.distanceMetric.name();
    if (storedMetric.equals("L2")) storedMetric = "L2";
    if (storedMetric.equals("IP")) storedMetric = "INNER_PRODUCT";
    if (!storedMetric.equalsIgnoreCase(configuredMetric)) {
      throw new IndexConfigurationException(String.format(
          "Distance metric mismatch: configured=%s, stored=%s. "
              + "Distance metric cannot be changed after collection creation.",
          configuredMetric, storedMetric));
    }

    // Validate PQ configuration (ALL IMMUTABLE)
    if (this.pqEnabled) {
      if (storedConfig.getPqSubvectors() != this.pqSubvectors) {
        throw new IndexConfigurationException(String.format(
            "PQ subvectors mismatch: configured=%d, stored=%d. "
                + "PQ configuration cannot be changed after collection creation.",
            this.pqSubvectors, storedConfig.getPqSubvectors()));
      }

      if (storedConfig.getPqNbits() != this.pqNbits) {
        throw new IndexConfigurationException(String.format(
            "PQ nbits mismatch: configured=%d, stored=%d. "
                + "PQ configuration cannot be changed after collection creation.",
            this.pqNbits, storedConfig.getPqNbits()));
      }

      if (storedConfig.getCodesPerBlock() != this.codesPerBlock) {
        throw new IndexConfigurationException(String.format(
            "Codes per block mismatch: configured=%d, stored=%d. "
                + "Storage block size cannot be changed after collection creation.",
            this.codesPerBlock, storedConfig.getCodesPerBlock()));
      }
    }

    // Validate graph degree (IMMUTABLE)
    if (storedConfig.getGraphDegree() != this.graphDegree) {
      throw new IndexConfigurationException(String.format(
          "Graph degree mismatch: configured=%d, stored=%d. "
              + "Graph degree cannot be changed after collection creation (would require graph rebuild).",
          this.graphDegree, storedConfig.getGraphDegree()));
    }

    // Note: Search parameters, cache sizes, worker counts are MUTABLE and not validated
  }

  /**
   * Creates a protobuf Config message from this configuration for storage in FDB.
   * <p>Only immutable configuration values are stored. Mutable values like cache sizes
   * and worker counts are not persisted as they can change between runs.
   *
   * @return a Config protobuf message containing immutable configuration
   */
  public Config toProtoConfig() {
    Config.Builder builder = Config.newBuilder()
        .setDimension(dimension)
        .setMetric(
            distanceMetric == DistanceMetric.INNER_PRODUCT
                ? "IP"
                : distanceMetric == DistanceMetric.COSINE ? "COSINE" : "L2")
        .setGraphDegree(graphDegree)
        .setCodesPerBlock(codesPerBlock);

    if (pqEnabled) {
      builder.setPqSubvectors(pqSubvectors).setPqNbits(pqNbits);
    }

    // Set defaults for search parameters (these are stored but can be overridden)
    builder.setDefaultSearchList(defaultSearchList).setMaxSearchVisits(maxSearchVisits);

    return builder.build();
  }

  /**
   * Creates a new builder for VectorSearchConfig.
   *
   * @param database  the FoundationDB database to use
   * @param directory the FDB directory for this collection
   * @return a new builder instance
   */
  public static Builder builder(Database database, Directory directory) {
    return new Builder(database, directory);
  }

  /**
   * Builder for VectorSearchConfig with fluent API and sensible defaults.
   */
  public static class Builder {
    private final Database database;
    private final Directory directory;

    // Required parameters (no defaults)
    private int dimension = -1; // Must be set explicitly

    // Core configuration with defaults
    private DistanceMetric distanceMetric = DistanceMetric.L2;

    // PQ configuration defaults (following Milvus/DiskANN)
    private boolean pqEnabled = true;
    private int pqSubvectors = -1; // Will default to dimension/2
    private int pqNbits = 8;
    private int codesPerBlock = 512;

    // Graph configuration defaults
    private int graphDegree = 64; // DiskANN default R
    private int defaultSearchList = 16; // Milvus default
    private int maxSearchVisits = 1500;
    private int entryPointCount = 32;
    private Duration entryPointRefreshInterval = Duration.ofHours(1);

    // Cache configuration defaults
    private long pqBlockCacheSize = 1024L * 1024 * 1024; // 1GB
    private int adjacencyCacheSize = 100_000;
    private boolean keepCodebooksInMemory = true;

    // Worker configuration defaults
    private TaskQueueConfig<Long, ?> taskQueueConfig = null;
    private int linkWorkerCount = 4;
    private int unlinkWorkerCount = 2;
    private Executor backgroundExecutor = ForkJoinPool.commonPool();

    // Graph maintenance defaults
    private boolean autoRepairEnabled = true;
    private Duration graphRepairInterval = Duration.ofHours(6);
    private double minConnectivityThreshold = 0.95;

    // Storage configuration defaults
    private boolean useBlockStriping = false;
    private int stripeCount = 4;
    private boolean storeVectorSketches = true;
    private InstantSource instantSource = InstantSource.system();

    private Builder(Database database, Directory directory) {
      this.database = database;
      this.directory = directory;
    }

    /**
     * Sets the vector dimension (required).
     * Common values: 128, 256, 512, 768 (BERT), 1536 (OpenAI).
     */
    public Builder dimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    /**
     * Sets the distance metric for similarity calculations.
     * Default: L2 (Euclidean distance).
     */
    public Builder distanceMetric(DistanceMetric distanceMetric) {
      this.distanceMetric = distanceMetric;
      return this;
    }

    /**
     * Enables or disables Product Quantization.
     * Default: true (enabled for compression).
     */
    public Builder pqEnabled(boolean pqEnabled) {
      this.pqEnabled = pqEnabled;
      return this;
    }

    /**
     * Sets the number of PQ subvectors.
     * Default: dimension/2 (set automatically if not specified).
     */
    public Builder pqSubvectors(int pqSubvectors) {
      this.pqSubvectors = pqSubvectors;
      return this;
    }

    /**
     * Sets the number of bits per PQ code.
     * Default: 8 (only 8 is currently supported).
     */
    public Builder pqNbits(int pqNbits) {
      this.pqNbits = pqNbits;
      return this;
    }

    /**
     * Sets the number of codes per storage block.
     * Default: 512 (optimal for FDB).
     */
    public Builder codesPerBlock(int codesPerBlock) {
      this.codesPerBlock = codesPerBlock;
      return this;
    }

    /**
     * Sets the maximum graph degree (neighbors per node).
     * Default: 64 (DiskANN default).
     */
    public Builder graphDegree(int graphDegree) {
      this.graphDegree = graphDegree;
      return this;
    }

    /**
     * Sets the default search list size.
     * Default: 16.
     */
    public Builder defaultSearchList(int defaultSearchList) {
      this.defaultSearchList = defaultSearchList;
      return this;
    }

    /**
     * Sets the maximum nodes to visit during search.
     * Default: 1500.
     */
    public Builder maxSearchVisits(int maxSearchVisits) {
      this.maxSearchVisits = maxSearchVisits;
      return this;
    }

    /**
     * Sets the number of entry points for search.
     * Default: 32.
     */
    public Builder entryPointCount(int entryPointCount) {
      this.entryPointCount = entryPointCount;
      return this;
    }

    /**
     * Sets the entry point refresh interval.
     * Default: 1 hour.
     */
    public Builder entryPointRefreshInterval(Duration entryPointRefreshInterval) {
      this.entryPointRefreshInterval = entryPointRefreshInterval;
      return this;
    }

    /**
     * Sets the PQ block cache size in bytes.
     * Default: 1GB.
     */
    public Builder pqBlockCacheSize(long pqBlockCacheSize) {
      this.pqBlockCacheSize = pqBlockCacheSize;
      return this;
    }

    /**
     * Sets the adjacency cache size (number of nodes).
     * Default: 100,000.
     */
    public Builder adjacencyCacheSize(int adjacencyCacheSize) {
      this.adjacencyCacheSize = adjacencyCacheSize;
      return this;
    }

    /**
     * Sets whether to keep codebooks in memory.
     * Default: true.
     */
    public Builder keepCodebooksInMemory(boolean keepCodebooksInMemory) {
      this.keepCodebooksInMemory = keepCodebooksInMemory;
      return this;
    }

    /**
     * Sets the task queue configuration for background operations.
     * Default: null (will create default configuration).
     */
    public Builder taskQueueConfig(TaskQueueConfig<Long, ?> taskQueueConfig) {
      this.taskQueueConfig = taskQueueConfig;
      return this;
    }

    /**
     * Sets the number of link workers.
     * Default: 4.
     */
    public Builder linkWorkerCount(int linkWorkerCount) {
      this.linkWorkerCount = linkWorkerCount;
      return this;
    }

    /**
     * Sets the number of unlink workers.
     * Default: 2.
     */
    public Builder unlinkWorkerCount(int unlinkWorkerCount) {
      this.unlinkWorkerCount = unlinkWorkerCount;
      return this;
    }

    /**
     * Sets the executor for background operations.
     * Default: ForkJoinPool.commonPool().
     */
    public Builder backgroundExecutor(Executor backgroundExecutor) {
      this.backgroundExecutor = backgroundExecutor;
      return this;
    }

    /**
     * Enables or disables automatic graph repair.
     * Default: true.
     */
    public Builder autoRepairEnabled(boolean autoRepairEnabled) {
      this.autoRepairEnabled = autoRepairEnabled;
      return this;
    }

    /**
     * Sets the graph repair check interval.
     * Default: 6 hours.
     */
    public Builder graphRepairInterval(Duration graphRepairInterval) {
      this.graphRepairInterval = graphRepairInterval;
      return this;
    }

    /**
     * Sets the minimum connectivity threshold for graph health.
     * Default: 0.95 (95%).
     */
    public Builder minConnectivityThreshold(double minConnectivityThreshold) {
      this.minConnectivityThreshold = minConnectivityThreshold;
      return this;
    }

    /**
     * Enables or disables PQ block striping for hot blocks.
     * Default: false.
     */
    public Builder useBlockStriping(boolean useBlockStriping) {
      this.useBlockStriping = useBlockStriping;
      return this;
    }

    /**
     * Sets the number of stripes for block striping.
     * Default: 4.
     */
    public Builder stripeCount(int stripeCount) {
      this.stripeCount = stripeCount;
      return this;
    }

    /**
     * Enables or disables vector sketch storage.
     * Default: true (recommended for codebook rotation).
     */
    public Builder storeVectorSketches(boolean storeVectorSketches) {
      this.storeVectorSketches = storeVectorSketches;
      return this;
    }

    /**
     * Sets the instant source for time operations.
     * Default: InstantSource.system().
     */
    public Builder instantSource(InstantSource instantSource) {
      this.instantSource = instantSource;
      return this;
    }

    /**
     * Builds the VectorSearchConfig with validation.
     *
     * @return the configured VectorSearchConfig instance
     * @throws IllegalArgumentException if required parameters are missing or invalid
     */
    public VectorSearchConfig build() {
      // Check required parameters
      if (dimension <= 0) {
        throw new IllegalArgumentException("dimension must be set and positive");
      }

      // Auto-configure PQ subvectors if not set
      if (pqEnabled && pqSubvectors == -1) {
        pqSubvectors = Math.max(1, dimension / 2); // Ensure at least 1
        // Ensure it divides evenly
        while (pqSubvectors > 1 && dimension % pqSubvectors != 0) {
          pqSubvectors--;
        }
      }

      return new VectorSearchConfig(this);
    }
  }
}
