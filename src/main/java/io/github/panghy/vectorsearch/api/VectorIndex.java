package io.github.panghy.vectorsearch.api;

import static com.apple.foundationdb.tuple.ByteArrayUtil.decodeInt;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.async.AsyncUtil;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.cache.SegmentCaches;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.fdb.FdbVectorStore;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers.BuildTaskSerializer;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers.StringSerializer;
import io.github.panghy.vectorsearch.tasks.SegmentBuildWorkerPool;
import io.github.panghy.vectorsearch.util.Distances;
import io.github.panghy.vectorsearch.util.FloatPacker;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Public API façade for performing vector operations against an index.
 *
 * <p>Provides async methods for index initialization, insertion, and KNN queries.
 * For now, queries perform a brute-force scan across segments and will be
 * upgraded to use graph+PQ for SEALED segments in later milestones.</p>
 */
public class VectorIndex implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(VectorIndex.class);
  private final VectorIndexConfig config;
  private final FdbVectorStore store;
  private final SegmentCaches caches;
  private final FdbDirectories.IndexDirectories indexDirs;
  private TaskQueue<String, BuildTask> taskQueue;

  private SegmentBuildWorkerPool workerPool;

  private VectorIndex(VectorIndexConfig config, FdbVectorStore store, FdbDirectories.IndexDirectories indexDirs) {
    this.config = config;
    this.store = store;
    this.indexDirs = requireNonNull(indexDirs, "indexDirs");
    this.caches = new SegmentCaches(config, indexDirs);
    this.caches.registerMetrics(config);
  }

  /**
   * Creates or opens the index asynchronously and returns a ready VectorIndex.
   */
  public static CompletableFuture<VectorIndex> createOrOpen(VectorIndexConfig config) {
    return FdbDirectories.openIndex(config.getIndexDir(), config.getDatabase())
        .thenCompose(dirs -> {
          // Single TaskQueue for all background build tasks
          var tqc = TaskQueueConfig.builder(
                  config.getDatabase(),
                  dirs.tasksDir(),
                  new StringSerializer(),
                  new BuildTaskSerializer())
              .estimatedWorkerCount(config.getEstimatedWorkerCount())
              .defaultTtl(config.getDefaultTtl())
              .defaultThrottle(config.getDefaultThrottle())
              .taskNameExtractor(bt -> "build-segment:" + bt.getSegId())
              .build();
          return TaskQueues.createTaskQueue(tqc).thenCompose(queue -> {
            FdbVectorStore fdbVectorStore = new FdbVectorStore(config, dirs, queue);
            return fdbVectorStore.createOrOpenIndex().thenApply($ -> {
              VectorIndex ix = new VectorIndex(config, fdbVectorStore, dirs);
              ix.taskQueue = queue;
              int n = config.getLocalWorkerThreads();
              if (n > 0) {
                ix.workerPool = new SegmentBuildWorkerPool(config, dirs, queue);
                ix.workerPool.start(n);
                LOG.info("Started local SegmentBuildWorkerPool with threads={}.", n);
              }
              return ix;
            });
          });
        });
  }

  /**
   * Shuts down any auto-started background workers.
   */
  @Override
  public void close() {
    if (workerPool != null) {
      workerPool.close();
      workerPool = null;
    }
  }

  /**
   * Inserts a vector into the current ACTIVE segment, rotating when the segment threshold is exceeded.
   * Also, enqueues a background build task when a segment flips to PENDING.
   *
   * @param embedding the vector embedding (length must equal configured dimension)
   * @param payload   optional payload bytes to store alongside the embedding
   * @return a future with [segmentId, vectorId]
   */
  public CompletableFuture<int[]> add(float[] embedding, byte[] payload) {
    return store.add(embedding, payload);
  }

  /**
   * Batched insert API to reduce contention and improve throughput.
   *
   * <p>Writes are chunked per transaction by ACTIVE segment capacity; rotates and enqueues
   * builds as needed.</p>
   */
  public CompletableFuture<List<int[]>> addAll(float[][] embeddings, byte[][] payloads) {
    return store.addBatch(embeddings, payloads);
  }

  /**
   * Queries the index for the top-K nearest neighbors to a query vector.
   *
   * @param q the query vector (length must equal configured dimension)
   * @param k the number of results to return
   * @return a future with up to K results ordered by descending score
   */
  public CompletableFuture<List<SearchResult>> query(float[] q, int k) {
    return query(q, k, SearchParams.defaults(k, config.getOversample()));
  }

  /**
   * Queries with per-call traversal knobs.
   */
  public CompletableFuture<List<SearchResult>> query(float[] q, int k, SearchParams params) {
    Database db = config.getDatabase();
    LOG.debug(
        "Query start: k={} metric={} dim={} oversample={}",
        k,
        config.getMetric(),
        config.getDimension(),
        config.getOversample());
    //noinspection deprecation
    if (params.mode() == SearchParams.Mode.BEAM && BeamWarn.once()) {
      LOG.warn("Search mode BEAM is deprecated; prefer BEST_FIRST.");
    }
    return listSegmentsWithMeta(indexDirs).thenCompose(segIds -> {
      LOG.debug("Discovered {} segment(s) for search: {}", segIds.size(), segIds);
      // Prefetch codebooks for SEALED segments via async cache getAll (batched inside loader)
      CompletableFuture<?> prefetch = !config.isPrefetchCodebooksEnabled()
          ? completedFuture(null)
          : db.readAsync(tr -> {
                List<CompletableFuture<byte[]>> gets = new ArrayList<>();
                List<Integer> ids = new ArrayList<>();
                for (int segId : segIds) {
                  String segStr = String.format("%06d", segId);
                  gets.add(
                      tr.get(indexDirs.segmentKeys(segStr).metaKey()));
                  ids.add(segId);
                }
                return allOf(gets.toArray(CompletableFuture[]::new))
                    .thenApply(v -> {
                      Set<Integer> sealedSegs = new HashSet<>();
                      for (int i = 0; i < ids.size(); i++) {
                        byte[] b = gets.get(i).join();
                        if (b == null) continue;
                        try {
                          SegmentMeta sm = SegmentMeta.parseFrom(b);
                          if (sm.getState() == SegmentMeta.State.SEALED)
                            sealedSegs.add(ids.get(i));
                        } catch (InvalidProtocolBufferException e) {
                          throw new RuntimeException(e);
                        }
                      }
                      return sealedSegs;
                    });
              })
              .thenCompose(sealedSegs -> {
                if (sealedSegs.isEmpty()) {
                  return completedFuture(null);
                }
                return caches.getCodebookCacheAsync()
                    .getAll(sealedSegs)
                    .thenApply(m -> null);
              })
              .exceptionally(ex -> null);
      prefetch.whenComplete((v, ex) -> {
        /* fire-and-forget */
      });
      List<CompletableFuture<List<SearchResult>>> perSeg = new ArrayList<>();
      for (int segId : segIds) {
        int perSegLimit = Math.max(k, k * Math.max(1, config.getOversample()));
        LOG.debug(
            "Scheduling per-segment search: segId={} perSegLimit={} mode={} ef={} beam={} maxExplore={}",
            segId,
            perSegLimit,
            params.mode(),
            params.efSearch(),
            params.beamWidth(),
            params.maxExplore());
        perSeg.add(searchSegment(db, indexDirs, segId, q, perSegLimit, params));
      }
      return allOf(perSeg.toArray(CompletableFuture[]::new)).thenApply(v -> {
        List<SearchResult> merged = new ArrayList<>();
        for (CompletableFuture<List<SearchResult>> f : perSeg) merged.addAll(f.join());
        merged.sort(Comparator.comparingDouble(SearchResult::score)
            .reversed()
            .thenComparingInt(SearchResult::vectorId));
        if (merged.size() > k) merged = merged.subList(0, k);
        if (!merged.isEmpty()) {
          int limit = Math.min(5, merged.size());
          LOG.debug(
              "Merged top {} results: {}",
              limit,
              merged.subList(0, limit).stream()
                  .map(r -> String.format(
                      "(seg=%d,id=%d,score=%.4f)", r.segmentId(), r.vectorId(), r.score()))
                  .toList());
        }
        return merged;
      });
    });
  }

  private static final class BeamWarn {
    private static final AtomicBoolean WARNED = new AtomicBoolean(false);

    static boolean once() {
      return WARNED.compareAndSet(false, true);
    }
  }

  /**
   * Approximate size of the codebook cache (for tests/observability).
   */
  public long getCodebookCacheSize() {
    return caches.getCodebookCacheAsync().synchronous().estimatedSize();
  }

  /**
   * Approximate size of the adjacency cache (for tests/observability).
   */
  public long getAdjacencyCacheSize() {
    return caches.getAdjacencyCache().estimatedSize();
  }

  /**
   * Asynchronously waits until the background indexing queue is drained.
   *
   * <p>Loops using {@code hasVisibleUnclaimedTasks} without sleeps; returns when the queue reports
   * no visible unclaimed tasks. This does not wait for currently claimed tasks to complete, but
   * in typical usage with local workers running, claimed tasks should quickly finish and unclaimed
   * will reappear if needed.</p>
   */
  public CompletableFuture<Void> awaitIndexingComplete() {
    if (taskQueue == null) return completedFuture(null);
    Database db = config.getDatabase();
    final long pollDelayMs = 50L;
    return AsyncUtil.whileTrue(() -> db.runAsync(tr -> taskQueue.hasVisibleUnclaimedTasks(tr))
            .thenCompose(has -> Boolean.TRUE.equals(has)
                ? CompletableFuture.supplyAsync(
                    () -> true,
                    CompletableFuture.delayedExecutor(pollDelayMs, TimeUnit.MILLISECONDS))
                : completedFuture(false)))
        .thenApply(v -> null);
  }

  /**
   * Lists known segment ids by scanning for meta keys under the segments directory.
   */
  private CompletableFuture<List<Integer>> listSegmentsWithMeta(FdbDirectories.IndexDirectories dirs) {
    Database db = config.getDatabase();
    byte[] maxK = dirs.maxSegmentKey();
    return db.readAsync(tr -> tr.get(maxK).thenCompose(maxB -> {
      if (maxB == null) return completedFuture(List.of());
      int maxSeg = Math.toIntExact(decodeInt(maxB));
      // Batch get meta keys for [0..maxSeg]
      List<byte[]> keys = new ArrayList<>(maxSeg + 1);
      for (int i = 0; i <= maxSeg; i++)
        keys.add(dirs.segmentKeys(String.format("%06d", i)).metaKey());
      List<CompletableFuture<byte[]>> gets = new ArrayList<>(keys.size());
      for (byte[] k : keys) gets.add(tr.get(k));
      return allOf(gets.toArray(CompletableFuture[]::new)).thenApply(v -> {
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < gets.size(); i++) if (gets.get(i).join() != null) ids.add(i);
        return ids;
      });
    }));
  }

  /**
   * Searches a single segment, dispatching to SEALED or ACTIVE/PENDING strategies.
   */
  private CompletableFuture<List<SearchResult>> searchSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k, SearchParams params) {
    String segStr = String.format("%06d", segId);
    return db.readAsync(tr -> tr.get(dirs.segmentKeys(segStr).metaKey()).thenCompose(metaB -> {
      if (metaB == null) return completedFuture(List.of());
      try {
        SegmentMeta sm = SegmentMeta.parseFrom(metaB);
        LOG.debug("searchSegment segId={} state={} count={}", segId, sm.getState(), sm.getCount());
        if (sm.getState() == SegmentMeta.State.SEALED) {
          return searchSealedSegment(db, dirs, segId, q, k, params);
        }
        return searchBruteForceSegment(db, dirs, segId, q, k);
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
    }));
  }

  /**
   * Searches a single segment via brute-force and returns its top-K results.
   */
  private CompletableFuture<List<SearchResult>> searchBruteForceSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k) {
    String segStr = String.format("%06d", segId);
    Subspace vectorsPrefix = new Subspace(dirs.segmentsDir().pack(Tuple.from(segStr, "vectors")));
    Range vr = vectorsPrefix.range();
    return db.readAsync(tr -> tr.getRange(vr).asList().thenApply(kvs -> {
      LOG.debug("Brute-force segId={} loaded {} vector records", segId, kvs.size());
      List<SearchResult> results = new ArrayList<>();
      for (KeyValue kv : kvs) {
        try {
          VectorRecord rec = VectorRecord.parseFrom(kv.getValue());
          if (rec.getDeleted()) continue;
          float[] emb = FloatPacker.bytesToFloats(rec.getEmbedding().toByteArray());
          double score;
          double distance;
          if (config.getMetric() == VectorIndexConfig.Metric.COSINE) {
            double sim = Distances.cosine(q, emb);
            score = sim;
            distance = 1.0 - sim;
          } else {
            double d = Distances.l2(q, emb);
            score = -d;
            distance = d;
          }
          results.add(SearchResult.builder()
              .segmentId(segId)
              .vectorId(rec.getVecId())
              .score(score)
              .distance(distance)
              .payload(rec.getPayload().toByteArray())
              .build());
        } catch (InvalidProtocolBufferException e) {
          throw new RuntimeException(e);
        }
      }
      results.sort(Comparator.comparingDouble(SearchResult::score).reversed());
      if (!results.isEmpty()) {
        int limit = Math.min(3, results.size());
        LOG.debug(
            "Brute-force segId={} top {}: {}",
            segId,
            limit,
            results.subList(0, limit).stream()
                .map(r -> String.format("(id=%d,score=%.4f)", r.vectorId(), r.score()))
                .toList());
      }
      if (results.size() > k) return results.subList(0, k);
      return results;
    }));
  }

  /**
   * Searches a SEALED segment using PQ approximate scoring then exact rerank of top candidates.
   */
  private CompletableFuture<List<SearchResult>> searchSealedSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k, SearchParams params) {
    String segStr = String.format("%06d", segId);
    return caches.getCodebookCacheAsync().get(segId).thenCompose(centroids -> {
      if (centroids == null) {
        LOG.warn("Missing PQ codebook for sealed segment segId={}", segId);
        return completedFuture(List.of());
      }
      double[][] lut = buildLut(centroids, q);
      Subspace codesPrefix = new Subspace(dirs.segmentsDir().pack(Tuple.from(segStr, "pq", "codes")));
      Range cr = codesPrefix.range();
      return config.getDatabase()
          .readAsync(tr -> tr.getRange(cr).asList())
          .thenCompose(codeKvs -> {
            LOG.debug("Sealed segId={} has {} PQ code entries", segId, codeKvs.size());
            // Build approx distances for all codes and retain a map for quick lookup
            List<Approx> approxAll = new ArrayList<>();
            Map<Integer, byte[]> codeMap = new HashMap<>();
            int m = centroids.length;
            int kCent = centroids[0].length;
            for (KeyValue kv : codeKvs) {
              int vecId =
                  (int) dirs.segmentsDir().unpack(kv.getKey()).getLong(3);
              byte[] codes = kv.getValue();
              if (codes == null || codes.length < m) continue;
              codeMap.put(vecId, codes);
              double ad = 0.0;
              for (int s = 0; s < m; s++) {
                int ci = codes[s] & 0xFF;
                if (ci >= kCent) continue;
                ad += lut[s][ci];
              }
              approxAll.add(new Approx(vecId, ad));
            }
            if (approxAll.isEmpty()) return completedFuture(List.of());

            approxAll.sort(Comparator.comparingDouble(a -> a.approx));
            int nCodes = approxAll.size();
            int baseEf = Math.max(params.efSearch(), k * Math.max(1, params.perSegmentLimitMultiplier()));
            int scale = (int) Math.max(1, Math.round(Math.sqrt(Math.max(1, nCodes) / 1000.0)));
            int tunedEf = Math.min(params.maxExplore(), Math.max(baseEf, Math.min(nCodes, baseEf * scale)));
            int tunedBeam = Math.max(1, Math.min(nCodes, Math.max(params.beamWidth(), Math.min(64, (int)
                Math.ceil(Math.sqrt(nCodes))))));
            SearchParams eff = SearchParams.of(
                tunedEf,
                tunedBeam,
                params.maxIters(),
                params.maxExplore(),
                params.refineFrontier(),
                params.mode());
            LOG.debug(
                "sealed-search segId={} nCodes={} ef={} beam={} maxExplore={}",
                segId,
                nCodes,
                eff.efSearch(),
                eff.beamWidth(),
                eff.maxExplore());
            int preview = Math.min(3, approxAll.size());
            LOG.debug(
                "sealed segId={} approx top {}: {}",
                segId,
                preview,
                approxAll.subList(0, preview).stream()
                    .map(a -> String.format("(id=%d,approx=%.4f)", a.vecId(), a.approx))
                    .toList());

            int beam = Math.max(1, Math.min(nCodes, tunedBeam));
            List<Approx> seeds = new ArrayList<>(approxAll.subList(0, beam));
            // Optional deterministic pivot seeds
            if (params.seedStrategy() == SearchParams.SeedStrategy.RANDOM_PIVOTS && nCodes > beam) {
              int pivots = Math.min(params.pivots(), nCodes - beam);
              long seed = ((long) segId << 21) ^ Double.doubleToLongBits(lut[0][0]);
              Random rnd = new Random(seed);
              for (int i = 0; i < pivots; i++) {
                int idx = beam + rnd.nextInt(Math.max(1, nCodes - beam));
                seeds.add(approxAll.get(idx));
              }
            }
            CompletableFuture<List<Approx>> expF = (params.mode() == SearchParams.Mode.BEST_FIRST)
                ? diskannBestFirstExpand(segId, seeds, codeMap, lut, eff)
                : diskannExpand(segId, seeds, codeMap, lut, eff);
            return expF.thenCompose(expanded -> {
              expanded.sort(Comparator.comparingDouble(a -> a.approx));
              int topN = Math.min(expanded.size(), Math.max(eff.efSearch(), k));
              List<Approx> cand = expanded.subList(0, topN);
              double qNorm =
                  (config.getMetric() == VectorIndexConfig.Metric.COSINE && params.normalizeOnRead())
                      ? Distances.norm(q)
                      : 0.0;
              return fetchExactAndScore(db, dirs, segId, q, qNorm, params.normalizeOnRead(), cand, k);
            });
          });
    });
  }

  private record Approx(int vecId, double approx) {}

  private CompletableFuture<List<Approx>> diskannExpand(
      int segId,
      List<Approx> initialFrontier,
      Map<Integer, byte[]> codeMap,
      double[][] lut,
      SearchParams params) {
    Set<Integer> visited = new HashSet<>();
    for (Approx a : initialFrontier) visited.add(a.vecId());
    // Include seeds to ensure we return something even if no neighbors
    List<Approx> expanded = new ArrayList<>(initialFrontier);

    CompletableFuture<List<Approx>> loop = completedFuture(initialFrontier);
    final int MIN_HOPS = Math.max(0, params.minHops());
    for (int iter = 0; iter < params.maxIters(); iter++) {
      final int hop = iter;
      loop = loop.thenCompose(frontier -> {
        if (frontier.isEmpty()
            || expanded.size() >= params.efSearch()
            || expanded.size() >= params.maxExplore()) {
          return completedFuture(List.of());
        }
        // Batch load adjacency via async cache loader for all frontier nodes
        Set<Long> keys = new HashSet<>();
        for (Approx a : frontier) keys.add(SegmentCaches.adjKey(segId, a.vecId()));
        return caches.getAdjacencyCacheAsync().getAll(keys).thenApply(v2 -> {
          List<Approx> newly = new ArrayList<>();
          int m = lut.length;
          int kCent = lut[0].length;
          for (Approx a : frontier) {
            int[] neigh = caches.getAdjacencyCache().getIfPresent(SegmentCaches.adjKey(segId, a.vecId()));
            if (neigh == null) neigh = new int[0];
            for (int nb : neigh) {
              if (expanded.size() + newly.size() >= params.efSearch()
                  || expanded.size() + newly.size() >= params.maxExplore()) break;
              if (!visited.add(nb)) continue;
              byte[] codes = codeMap.get(nb);
              if (codes == null || codes.length < m) continue;
              double ad = 0.0;
              for (int s = 0; s < m; s++) {
                int ci = codes[s] & 0xFF;
                if (ci >= kCent) continue;
                ad += lut[s][ci];
              }
              newly.add(new Approx(nb, ad));
            }
          }
          newly.sort(Comparator.comparingDouble(a -> a.approx));
          if (newly.isEmpty()) {
            if (hop + 1 < MIN_HOPS) return frontier; // force minimum hops
            return List.of();
          }
          int nextSize = Math.min(params.beamWidth(), newly.size());
          List<Approx> next;
          if (params.refineFrontier()) {
            List<Approx> union = new ArrayList<>(newly);
            union.addAll(frontier);
            union.sort(Comparator.comparingDouble(a -> a.approx));
            next = union.subList(0, Math.min(params.beamWidth(), union.size()));
          } else {
            next = newly.subList(0, nextSize);
          }
          expanded.addAll(next);
          return next;
        });
      });
    }
    return loop.thenApply(v -> expanded);
  }

  /**
   * Best-first (priority-queue) expansion akin to DiskANN/HNSW efSearch loop using PQ approx scores.
   * <p>
   * Implements batched adjacency/IO: pops up to {@code beamWidth} best candidates each step and
   * prefetches their adjacency lists together to reduce round-trips (Milvus-like beam behavior).
   */
  private CompletableFuture<List<Approx>> diskannBestFirstExpand(
      int segId, List<Approx> seeds, Map<Integer, byte[]> codeMap, double[][] lut, SearchParams params) {
    PriorityQueue<Approx> pq = new PriorityQueue<>(Comparator.comparingDouble(a -> a.approx));
    Set<Integer> visited = new HashSet<>();
    List<Approx> chosen = new ArrayList<>();
    for (Approx a : seeds) {
      if (visited.add(a.vecId())) {
        pq.offer(a);
        chosen.add(a);
      }
    }

    CompletableFuture<Void> chain = completedFuture(null);
    for (int step = 0; step < params.efSearch(); step++) {
      chain = chain.thenCompose(v -> {
        if (chosen.size() >= params.efSearch() || chosen.size() >= params.maxExplore())
          return completedFuture(null);
        // Pop up to beamWidth best items to expand this step
        int batch = Math.max(1, params.beamWidth());
        List<Approx> expand = new ArrayList<>(batch);
        for (int i = 0; i < batch; i++) {
          Approx cur = pq.poll();
          if (cur == null) break;
          expand.add(cur);
        }
        if (expand.isEmpty()) return completedFuture(null);

        // Batch load adjacency for missing entries using asyncLoadAll via getAll
        java.util.Set<Long> miss2 = new java.util.HashSet<>();
        for (Approx cur : expand) {
          long ckey = SegmentCaches.adjKey(segId, cur.vecId());
          if (caches.getAdjacencyCache().getIfPresent(ckey) == null) miss2.add(ckey);
        }
        java.util.concurrent.CompletableFuture<java.util.Map<Long, int[]>> bulk2 = miss2.isEmpty()
            ? java.util.concurrent.CompletableFuture.completedFuture(java.util.Map.of())
            : caches.getAdjacencyCacheAsync().getAll(miss2);
        return bulk2.thenRun(() -> {
          int m = lut.length;
          int kCent = lut[0].length;
          for (Approx cur : expand) {
            int[] neigh = caches.getAdjacencyCache().getIfPresent(SegmentCaches.adjKey(segId, cur.vecId()));
            if (neigh == null) neigh = new int[0];
            for (int nb : neigh) {
              if (chosen.size() >= params.efSearch()) break;
              if (!visited.add(nb)) continue;
              byte[] codes = codeMap.get(nb);
              if (codes == null || codes.length < m) continue;
              double ad = 0.0;
              for (int s = 0; s < m; s++) {
                int ci = codes[s] & 0xFF;
                if (ci >= kCent) continue;
                ad += lut[s][ci];
              }
              Approx na = new Approx(nb, ad);
              pq.offer(na);
              chosen.add(na);
            }
          }
        });
      });
    }
    return chain.thenApply(v -> chosen);
  }

  private CompletableFuture<List<SearchResult>> fetchExactAndScore(
      Database db,
      FdbDirectories.IndexDirectories dirs,
      int segId,
      float[] q,
      double qNorm,
      boolean normalizeOnRead,
      List<Approx> cand,
      int k) {
    String segStr = String.format("%06d", segId);
    LOG.debug("Exact rerank segId={} candidates={}", segId, cand.size());
    return db.readAsync(tr -> {
      List<CompletableFuture<VectorRecord>> recF = new ArrayList<>();
      for (Approx a : cand) {
        byte[] key = dirs.segmentKeys(segStr).vectorKey(a.vecId());
        recF.add(tr.get(key).thenApply(v -> {
          if (v == null) return null;
          try {
            return VectorRecord.parseFrom(v);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }));
      }
      return allOf(recF.toArray(CompletableFuture[]::new)).thenApply(v -> {
        List<SearchResult> out = new ArrayList<>();
        for (int i = 0; i < cand.size(); i++) {
          VectorRecord rec = recF.get(i).join();
          if (rec == null || rec.getDeleted()) continue;
          float[] emb = FloatPacker.bytesToFloats(rec.getEmbedding().toByteArray());
          double score;
          double distance;
          if (config.getMetric() == VectorIndexConfig.Metric.COSINE) {
            double sim;
            if (normalizeOnRead) {
              double denom = qNorm == 0.0
                  ? Distances.norm(q) * Distances.norm(emb)
                  : qNorm * Distances.norm(emb);
              sim = denom == 0.0 ? 0.0 : Distances.dot(q, emb) / denom;
            } else {
              sim = Distances.cosine(q, emb);
            }
            score = sim;
            distance = 1.0 - sim;
          } else {
            double d = Distances.l2(q, emb);
            score = -d;
            distance = d;
          }
          out.add(SearchResult.builder()
              .segmentId(segId)
              .vectorId(rec.getVecId())
              .score(score)
              .distance(distance)
              .payload(rec.getPayload().toByteArray())
              .build());
        }
        out.sort(Comparator.comparingDouble(SearchResult::score).reversed());
        int limit = Math.min(5, out.size());
        if (limit > 0) {
          LOG.debug(
              "Exact segId={} top {}: {}",
              segId,
              limit,
              out.subList(0, limit).stream()
                  .map(r -> String.format("(id=%d,score=%.4f)", r.vectorId(), r.score()))
                  .toList());
        }
        if (out.size() > k) return out.subList(0, k);
        return out;
      });
    });
  }

  private static double[][] buildLut(float[][][] centroids, float[] q) {
    int m = centroids.length;
    int subDim = centroids[0][0].length;
    int k = centroids[0].length;
    double[][] lut = new double[m][k];
    for (int s = 0; s < m; s++) {
      int off = s * subDim;
      for (int ci = 0; ci < k; ci++) {
        double d = 0.0;
        for (int di = 0; di < subDim; di++) {
          double dd = (double) q[off + di] - centroids[s][ci][di];
          d += dd * dd;
        }
        lut[s][ci] = d;
      }
    }
    return lut;
  }
}
