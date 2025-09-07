package io.github.panghy.vectorsearch.api;

/**
 * Per-search tuning knobs inspired by DiskANN, provided per call.
 *
 * <ul>
 *   <li>efSearch: target number of candidates explored via graph before exact rerank</li>
 *   <li>beamWidth: number of frontier nodes to expand each iteration (for BEAM mode)</li>
 *   <li>maxIters: maximum number of frontier expansion iterations</li>
 *   <li>maxExplore: hard cap on explored candidates</li>
 *   <li>refineFrontier: consider prior frontier membership when picking next frontier</li>
 *   <li>minHops: minimum hops before early-exit</li>
 *   <li>pivots: number of diversified seed pivots (used when seedStrategy allows)</li>
 *   <li>seedStrategy: seeding behavior (PQ-only or randomized pivots)</li>
 *   <li>perSegmentLimitMultiplier: per-segment fan-in cap = K * multiplier for multi-segment merge</li>
 *   <li>normalizeOnRead: optional cosine normalization during exact rerank</li>
 *   <li>mode: traversal strategy (BEAM or BEST_FIRST)</li>
 * </ul>
 */
public record SearchParams(
    int efSearch,
    int beamWidth,
    int maxIters,
    int maxExplore,
    boolean refineFrontier,
    int minHops,
    int pivots,
    SeedStrategy seedStrategy,
    int perSegmentLimitMultiplier,
    boolean normalizeOnRead,
    Mode mode) {
  /** Traversal strategy. */
  public enum Mode {
    @Deprecated
    BEAM,
    BEST_FIRST
  }
  /** Seed strategy for initial frontier. */
  public enum SeedStrategy {
    PQ_SEED_ONLY,
    RANDOM_PIVOTS
  }

  public static SearchParams of(int efSearch, int beamWidth, int maxIters) {
    return of(efSearch, beamWidth, maxIters, Math.max(efSearch * 4, 1024), true, Mode.BEAM);
  }

  public static SearchParams of(int efSearch, int beamWidth, int maxIters, Mode mode) {
    return of(efSearch, beamWidth, maxIters, Math.max(efSearch * 4, 1024), true, mode);
  }

  public static SearchParams of(
      int efSearch, int beamWidth, int maxIters, int maxExplore, boolean refineFrontier, Mode mode) {
    if (efSearch <= 0) throw new IllegalArgumentException("efSearch must be positive");
    if (beamWidth <= 0) throw new IllegalArgumentException("beamWidth must be positive");
    if (maxIters <= 0) throw new IllegalArgumentException("maxIters must be positive");
    if (maxExplore <= 0) throw new IllegalArgumentException("maxExplore must be positive");
    if (mode == null) throw new IllegalArgumentException("mode must not be null");
    return new SearchParams(
        efSearch,
        beamWidth,
        maxIters,
        maxExplore,
        refineFrontier,
        2, // minHops default
        2, // pivots default
        SeedStrategy.PQ_SEED_ONLY,
        2, // per-seg cap multiplier default
        false, // normalizeOnRead default
        mode);
  }

  /** Reasonable defaults inspired by Milvus/HNSW-style search: BEST_FIRST with higher ef. */
  public static SearchParams defaults(int k, int oversample) {
    int ef = Math.max(100, k * Math.max(1, oversample) * 4);
    int beam = Math.min(64, Math.max(8, k * 2)); // ignored by BEST_FIRST but kept for explicit mode
    int iters = 6;
    int maxExplore = Math.max(ef * 4, 4096);
    boolean refine = true;
    return new SearchParams(
        ef, beam, iters, maxExplore, refine, 2, 2, SeedStrategy.PQ_SEED_ONLY, 2, false, Mode.BEST_FIRST);
  }

  /** Fluent builder for per-call tuning. */
  public static final class Builder {
    private int efSearch;
    private int beamWidth;
    private int maxIters = 4;
    private int maxExplore = 2048;
    private boolean refineFrontier = true;
    private int minHops = 2;
    private int pivots = 2;
    private SeedStrategy seedStrategy = SeedStrategy.PQ_SEED_ONLY;
    private int perSegmentLimitMultiplier = 2;
    private boolean normalizeOnRead = false;
    private Mode mode = Mode.BEAM;

    public Builder efSearch(int v) {
      this.efSearch = v;
      return this;
    }

    public Builder beamWidth(int v) {
      this.beamWidth = v;
      return this;
    }

    public Builder maxIters(int v) {
      this.maxIters = v;
      return this;
    }

    public Builder maxExplore(int v) {
      this.maxExplore = v;
      return this;
    }

    public Builder refineFrontier(boolean v) {
      this.refineFrontier = v;
      return this;
    }

    public Builder minHops(int v) {
      this.minHops = v;
      return this;
    }

    public Builder pivots(int v) {
      this.pivots = v;
      return this;
    }

    public Builder seedStrategy(SeedStrategy v) {
      this.seedStrategy = v;
      return this;
    }

    public Builder perSegmentLimitMultiplier(int v) {
      this.perSegmentLimitMultiplier = v;
      return this;
    }

    public Builder normalizeOnRead(boolean v) {
      this.normalizeOnRead = v;
      return this;
    }

    public Builder mode(Mode v) {
      this.mode = v;
      return this;
    }

    public SearchParams build() {
      return SearchParams.of(efSearch, beamWidth, maxIters, maxExplore, refineFrontier, mode)
          .with(minHops, pivots, seedStrategy, perSegmentLimitMultiplier, normalizeOnRead);
    }
  }

  private SearchParams with(int minHops, int pivots, SeedStrategy ss, int perSegMult, boolean norm) {
    return new SearchParams(
        efSearch, beamWidth, maxIters, maxExplore, refineFrontier, minHops, pivots, ss, perSegMult, norm, mode);
  }
}
