package io.github.panghy.vectorsearch.fdb;

import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.allOf;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.ByteArrayUtil;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.api.SearchParams;
import io.github.panghy.vectorsearch.api.SearchResult;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.cache.SegmentCaches;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.tasks.MaintenanceWorkerPool;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers.BuildTaskSerializer;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers.StringSerializer;
import io.github.panghy.vectorsearch.tasks.SegmentBuildWorkerPool;
import io.github.panghy.vectorsearch.util.Distances;
import io.github.panghy.vectorsearch.util.FloatPacker;
import io.github.panghy.vectorsearch.util.Metrics;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import java.util.AbstractMap.SimpleEntry;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// Implementation of VectorIndex using FoundationDB. See VectorIndex for API documentation.
public class FdbVectorIndex implements VectorIndex, AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(FdbVectorIndex.class);

  private final VectorIndexConfig config;
  private final FdbVectorStore store;
  private final SegmentCaches caches;
  private final FdbDirectories.IndexDirectories indexDirs;

  private TaskQueue<String, BuildTask> buildQueue;
  private TaskQueue<String, MaintenanceTask> maintenanceQueue;
  private MaintenanceWorkerPool maintenancePool;
  private SegmentBuildWorkerPool workerPool;

  private final io.opentelemetry.api.metrics.Meter meter;
  private final io.opentelemetry.api.metrics.LongCounter vacScheduled;
  private final io.opentelemetry.api.metrics.LongCounter vacSkipped;

  private FdbVectorIndex(VectorIndexConfig config, FdbVectorStore store, FdbDirectories.IndexDirectories indexDirs) {
    this.config = config;
    this.store = store;
    this.indexDirs = requireNonNull(indexDirs, "indexDirs");
    this.caches = new SegmentCaches(config, indexDirs);
    this.caches.registerMetrics(config);
    // Register index-level metrics
    this.meter = io.opentelemetry.api.GlobalOpenTelemetry.getMeter("io.github.panghy.vectorsearch");
    this.vacScheduled = meter.counterBuilder("vectorsearch.maintenance.vacuum.scheduled")
        .build();
    this.vacSkipped =
        meter.counterBuilder("vectorsearch.maintenance.vacuum.skipped").build();
  }

  // Creates or opens the index asynchronously and returns a ready VectorIndex.
  public static CompletableFuture<FdbVectorIndex> createOrOpen(VectorIndexConfig config) {
    Database db = config.getDatabase();
    return FdbDirectories.openIndex(config.getIndexDir(), db)
        .thenCompose(dirs -> createBuildQueue(config, dirs).thenCompose(buildQ -> {
          FdbVectorStore store = new FdbVectorStore(config, dirs, buildQ);
          CompletableFuture<Void> init =
              store.createOrOpenIndex().thenCompose(v -> ensureCurrentSubspaces(config, dirs));
          return init.thenCompose(v -> createMaintenanceQueue(config, dirs))
              .thenApply(maintQ -> {
                FdbVectorIndex ix = new FdbVectorIndex(config, store, dirs);
                ix.buildQueue = buildQ;
                int n = config.getLocalWorkerThreads();
                if (n > 0) {
                  ix.workerPool = new SegmentBuildWorkerPool(config, dirs, buildQ);
                  ix.workerPool.start(n);
                  LOG.info("Started local SegmentBuildWorkerPool with threads={}.", n);
                }
                ix.maintenanceQueue = maintQ;
                int m = config.getLocalMaintenanceWorkerThreads();
                if (m > 0) {
                  ix.maintenancePool = new MaintenanceWorkerPool(config, ix.indexDirs, maintQ);
                  ix.maintenancePool.start(m);
                  LOG.info("Started local MaintenanceWorkerPool with threads={}", m);
                }
                return ix;
              });
        }));
  }

  private static CompletableFuture<Void> ensureCurrentSubspaces(
      VectorIndexConfig config, FdbDirectories.IndexDirectories dirs) {
    Database db = config.getDatabase();
    return db.runAsync(tr -> tr.get(dirs.currentSegmentKey()).thenCompose(cur -> {
      int sid = cur == null ? 0 : Math.toIntExact(ByteArrayUtil.decodeInt(cur));
      return dirs.segmentKeys(tr, sid).thenApply(sk -> null);
    }));
  }

  private static CompletableFuture<TaskQueue<String, BuildTask>> createBuildQueue(
      VectorIndexConfig config, FdbDirectories.IndexDirectories dirs) {
    TaskQueueConfig<String, BuildTask> tqc = TaskQueueConfig.builder(
            config.getDatabase(), dirs.tasksDir(), new StringSerializer(), new BuildTaskSerializer())
        .estimatedWorkerCount(config.getEstimatedWorkerCount())
        .defaultTtl(config.getDefaultTtl())
        .defaultThrottle(config.getDefaultThrottle())
        .taskNameExtractor(bt -> "build-segment:" + bt.getSegId())
        .build();
    return TaskQueues.createTaskQueue(tqc);
  }

  private static CompletableFuture<TaskQueue<String, MaintenanceTask>> createMaintenanceQueue(
      VectorIndexConfig config, FdbDirectories.IndexDirectories dirs) {
    return dirs.tasksDir()
        .createOrOpen(config.getDatabase(), List.of("maint"))
        .thenCompose(maintDir -> {
          TaskQueueConfig<String, MaintenanceTask> tqc = TaskQueueConfig.builder(
                  config.getDatabase(),
                  maintDir,
                  new StringSerializer(),
                  new ProtoSerializers.MaintenanceTaskSerializer())
              .estimatedWorkerCount(config.getEstimatedWorkerCount())
              .defaultTtl(config.getDefaultTtl())
              .defaultThrottle(config.getDefaultThrottle())
              .taskNameExtractor(mt -> mt.hasVacuum()
                  ? ("vacuum-segment:" + mt.getVacuum().getSegId())
                  : "compact")
              .build();
          return TaskQueues.createTaskQueue(tqc);
        });
  }

  // Shuts down any auto-started background workers.
  @Override
  public void close() {
    if (workerPool != null) {
      workerPool.close();
      workerPool = null;
    }
    if (maintenancePool != null) {
      maintenancePool.close();
      maintenancePool = null;
    }
  }

  @Override
  public CompletableFuture<Long> add(float[] embedding, byte[] payload) {
    Database db = config.getDatabase();
    return store.add(embedding, payload)
        .thenCompose(
            a -> db.readAsync(tr -> tr.get(indexDirs.gidRevDir().pack(Tuple.from(a[0], a[1])))
                .thenApply(b -> Tuple.fromBytes(b).getLong(0))));
  }

  @Override
  public CompletableFuture<Long> add(Transaction tx, float[] embedding, byte[] payload) {
    return store.add(tx, embedding, payload)
        .thenCompose(a -> tx.get(indexDirs.gidRevDir().pack(Tuple.from(a[0], a[1])))
            .thenApply(b -> Tuple.fromBytes(b).getLong(0)));
  }

  @Override
  public CompletableFuture<Void> delete(long gid) {
    Database db = config.getDatabase();
    return db.readAsync(tr -> tr.get(indexDirs.gidMapDir().pack(Tuple.from(gid))))
        .thenCompose(bytes -> {
          if (bytes == null) return completedFuture(null);
          Tuple t = Tuple.fromBytes(bytes);
          int segId = Math.toIntExact(t.getLong(0));
          int vecId = Math.toIntExact(t.getLong(1));
          return store.delete(segId, vecId).thenCompose(v -> scheduleVacuumIfNeeded(Set.of(segId)));
        });
  }

  @Override
  public CompletableFuture<Void> delete(Transaction tx, long gid) {
    return tx.get(indexDirs.gidMapDir().pack(Tuple.from(gid))).thenCompose(bytes -> {
      if (bytes == null) return completedFuture(null);
      Tuple t = Tuple.fromBytes(bytes);
      int segId = Math.toIntExact(t.getLong(0));
      int vecId = Math.toIntExact(t.getLong(1));
      return store.delete(tx, segId, vecId).thenCompose(v -> scheduleVacuumIfNeeded(tx, Set.of(segId)));
    });
  }

  @Override
  public CompletableFuture<Void> deleteAll(long[] gids) {
    if (gids == null || gids.length == 0) return completedFuture(null);
    Database db = config.getDatabase();
    return db.readAsync(tr -> {
          List<CompletableFuture<byte[]>> reads = new ArrayList<>();
          for (long gid : gids) reads.add(tr.get(indexDirs.gidMapDir().pack(Tuple.from(gid))));
          return allOf(reads.toArray(CompletableFuture[]::new)).thenApply(v -> {
            List<int[]> out = new ArrayList<>();
            Set<Integer> segs = new HashSet<>();
            for (CompletableFuture<byte[]> f : reads) {
              byte[] b = f.getNow(null);
              if (b == null) continue;
              Tuple t = Tuple.fromBytes(b);
              int s = Math.toIntExact(t.getLong(0));
              int v2 = Math.toIntExact(t.getLong(1));
              out.add(new int[] {s, v2});
              segs.add(s);
            }
            return new SimpleEntry<>(out, segs);
          });
        })
        .thenCompose(entry -> store.deleteBatch(entry.getKey().toArray(new int[0][]))
            .thenCompose(v -> scheduleVacuumIfNeeded(entry.getValue())));
  }

  @Override
  public CompletableFuture<Void> deleteAll(Transaction tx, long[] gids) {
    if (gids == null || gids.length == 0) return completedFuture(null);
    List<CompletableFuture<byte[]>> reads = new ArrayList<>();
    for (long gid : gids) reads.add(tx.get(indexDirs.gidMapDir().pack(Tuple.from(gid))));
    return allOf(reads.toArray(CompletableFuture[]::new)).thenCompose(vv -> {
      List<int[]> out = new ArrayList<>();
      Set<Integer> segs = new HashSet<>();
      for (CompletableFuture<byte[]> f : reads) {
        byte[] b = f.getNow(null);
        if (b == null) continue;
        Tuple t = Tuple.fromBytes(b);
        int s = Math.toIntExact(t.getLong(0));
        int v2 = Math.toIntExact(t.getLong(1));
        out.add(new int[] {s, v2});
        segs.add(s);
      }
      return store.deleteBatch(tx, out.toArray(new int[0][])).thenCompose(v -> scheduleVacuumIfNeeded(tx, segs));
    });
  }

  @Override
  public CompletableFuture<List<Long>> addAll(float[][] embeddings, byte[][] payloads) {
    Database db = config.getDatabase();
    return store.addBatch(embeddings, payloads)
        .thenCompose(list -> db.readAsync(tr -> {
          List<CompletableFuture<byte[]>> reads = new ArrayList<>();
          for (int[] a : list) reads.add(tr.get(indexDirs.gidRevDir().pack(Tuple.from(a[0], a[1]))));
          return allOf(reads.toArray(CompletableFuture[]::new)).thenApply(v -> {
            List<Long> out = new ArrayList<>(list.size());
            for (int i = 0; i < list.size(); i++)
              out.add(Tuple.fromBytes(reads.get(i).getNow(null)).getLong(0));
            return out;
          });
        }));
  }

  @Override
  public CompletableFuture<List<Long>> addAll(Transaction tx, float[][] embeddings, byte[][] payloads) {
    return store.addBatch(tx, embeddings, payloads).thenCompose(list -> {
      List<CompletableFuture<byte[]>> reads = new ArrayList<>();
      for (int[] a : list) reads.add(tx.get(indexDirs.gidRevDir().pack(Tuple.from(a[0], a[1]))));
      return allOf(reads.toArray(CompletableFuture[]::new)).thenApply(v -> {
        List<Long> out = new ArrayList<>(list.size());
        for (int i = 0; i < list.size(); i++)
          out.add(Tuple.fromBytes(reads.get(i).getNow(null)).getLong(0));
        return out;
      });
    });
  }

  @Override
  public CompletableFuture<List<SearchResult>> query(float[] q, int k) {
    return query(q, k, SearchParams.defaults(k, config.getOversample()));
  }

  @Override
  public CompletableFuture<List<SearchResult>> query(float[] q, int k, SearchParams params) {
    Database db = config.getDatabase();
    Tracer tracer = Metrics.tracer();
    Span span = tracer.spanBuilder("vectorsearch.query")
        .setSpanKind(SpanKind.INTERNAL)
        .startSpan();
    long t0 = System.nanoTime();
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
    return listSegmentsWithMeta(indexDirs)
        .thenCompose(segIds -> {
          LOG.debug("Discovered {} segment(s) for search: {}", segIds.size(), segIds);
          // Prefetch codebooks for SEALED segments via async cache getAll (batched inside loader)
          CompletableFuture<?> prefetch = !config.isPrefetchCodebooksEnabled()
              ? completedFuture(null)
              : db.readAsync(tr -> {
                    List<CompletableFuture<byte[]>> keyFs = new ArrayList<>();
                    List<Integer> ids = new ArrayList<>();
                    for (int segId : segIds) {
                      keyFs.add(indexDirs
                          .segmentKeys(tr, segId)
                          .thenCompose(sk -> tr.get(sk.metaKey()))
                          .exceptionally(ex -> null));
                      ids.add(segId);
                    }
                    return allOf(keyFs.toArray(CompletableFuture[]::new))
                        .thenApply(v -> {
                          Set<Integer> sealedSegs = new HashSet<>();
                          for (int i = 0; i < ids.size(); i++) {
                            byte[] b = keyFs.get(i).getNow(null);
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
          CompletableFuture<Void> prefetchBarrier =
              config.isPrefetchCodebooksSync() ? prefetch.thenApply(v -> null) : completedFuture(null);
          return prefetchBarrier.thenCompose(ignored -> {
            List<CompletableFuture<List<SearchResult>>> perSeg = new ArrayList<>();
            for (int segId : segIds) {
              int perSegLimit = Math.max(k, k * Math.max(1, config.getOversample()));
              LOG.debug(
                  "Scheduling per-segment search: segId={} perSegLimit={} mode={} ef={} beam={}"
                      + " maxExplore={}",
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
              for (CompletableFuture<List<SearchResult>> f : perSeg) merged.addAll(f.getNow(List.of()));
              merged.sort(Comparator.comparingDouble(SearchResult::score)
                  .reversed());
              if (merged.size() > k) merged = merged.subList(0, k);
              if (!merged.isEmpty()) {
                int limit = Math.min(5, merged.size());
                LOG.debug(
                    "Merged top {} results: {}",
                    limit,
                    merged.subList(0, limit).stream()
                        .map(r -> String.format(
                            "(seg=%d,id=%d,score=%.4f)",
                            (int) (r.gid() >>> 32), (int) r.gid(), r.score()))
                        .toList());
              }
              long durMs = (System.nanoTime() - t0) / 1_000_000;
              Attributes attrs = Attributes.builder()
                  .put("metric", config.getMetric().name())
                  .put("dim", config.getDimension())
                  .put("k", k)
                  .build();
              Metrics.QUERY_COUNT.add(1, attrs);
              Metrics.QUERY_DURATION_MS.record((double) durMs, attrs);
              span.setAttribute("k", k);
              span.setAttribute("dimension", config.getDimension());
              span.end();
              return merged;
            });
          });
        })
        .whenComplete((r, ex) -> {
          if (ex != null) {
            span.recordException(ex);
            span.setStatus(io.opentelemetry.api.trace.StatusCode.ERROR);
            span.end();
          }
        });
  }

  private static final class BeamWarn {
    private static final AtomicBoolean WARNED = new AtomicBoolean(false);

    static boolean once() {
      return WARNED.compareAndSet(false, true);
    }
  }

  @Override
  public long getCodebookCacheSize() {
    return caches.getCodebookCacheAsync().synchronous().estimatedSize();
  }

  @Override
  public long getAdjacencyCacheSize() {
    return caches.getAdjacencyCache().estimatedSize();
  }

  @Override
  public CompletableFuture<int[][]> resolveIds(long[] segmentVectorIds) {
    if (segmentVectorIds == null || segmentVectorIds.length == 0) return completedFuture(new int[0][0]);
    Database db = config.getDatabase();
    return db.readAsync(tr -> {
      List<CompletableFuture<byte[]>> reads = new ArrayList<>(segmentVectorIds.length);
      for (long gid : segmentVectorIds) {
        byte[] k = indexDirs.gidMapDir().pack(Tuple.from(gid));
        reads.add(tr.get(k));
      }
      return allOf(reads.toArray(CompletableFuture[]::new)).thenApply(v -> {
        int[][] out = new int[segmentVectorIds.length][2];
        for (int i = 0; i < segmentVectorIds.length; i++) {
          byte[] b = reads.get(i).getNow(null);
          if (b == null) {
            out[i][0] = -1;
            out[i][1] = -1;
          } else {
            Tuple t = Tuple.fromBytes(b);
            out[i][0] = Math.toIntExact(t.getLong(0));
            out[i][1] = Math.toIntExact(t.getLong(1));
          }
        }
        return out;
      });
    });
  }

  /**
   * Enqueues a compaction task for the given sealed segments. This is a skeleton operation for M7
   * that logs intent; future versions will merge segments and rewrite PQ/graph artifacts.
   */
  public CompletableFuture<Void> requestCompaction(List<Integer> segIds) {
    if (maintenanceQueue == null || segIds == null || segIds.isEmpty()) return completedFuture(null);
    // Deterministic key for idempotency
    List<Integer> sorted = new ArrayList<>(segIds);
    sorted.sort(Integer::compareTo);
    String key = "compact:" + sorted.toString();
    MaintenanceTask.Compact c =
        MaintenanceTask.Compact.newBuilder().addAllSegIds(sorted).build();
    MaintenanceTask mt = MaintenanceTask.newBuilder().setCompact(c).build();
    return config.getDatabase()
        .runAsync(tr -> maintenanceQueue.enqueueIfNotExists(tr, key, mt))
        .thenApply(x -> null);
  }

  @Override
  public CompletableFuture<Void> awaitIndexingComplete() {
    if (buildQueue == null) return completedFuture(null);
    Database db = config.getDatabase();
    return buildQueue.awaitQueueEmpty(db).thenApply(v -> null);
  }

  private CompletableFuture<Void> scheduleVacuumIfNeeded(Set<Integer> segIds) {
    if (maintenanceQueue == null || segIds == null || segIds.isEmpty()) return completedFuture(null);
    Database db = config.getDatabase();
    double thr = config.getVacuumMinDeletedRatio();
    List<CompletableFuture<Void>> fs = new ArrayList<>();
    for (int segId : segIds) {
      fs.add(db.runAsync(tr -> indexDirs
          .segmentKeys(tr, segId)
          .thenCompose(sk -> tr.get(sk.metaKey()))
          .thenCompose(bytes -> {
            if (bytes == null) return completedFuture(null);
            try {
              SegmentMeta sm = SegmentMeta.parseFrom(bytes);
              long live = sm.getCount();
              long del = sm.getDeletedCount();
              double ratio = (live + del) == 0 ? 0.0 : ((double) del) / ((double) (live + del));
              if (ratio < thr) {
                if (vacSkipped != null) vacSkipped.add(1);
                return completedFuture(null);
              }
              // Cooldown: skip if last vacuum completion is within cooldown window
              long cdMs = Math.max(0, config.getVacuumCooldown().toMillis());
              if (cdMs > 0 && sm.getLastVacuumAtMs() > 0) {
                long now = config.getInstantSource().instant().toEpochMilli();
                if (now - sm.getLastVacuumAtMs() < cdMs) {
                  if (vacSkipped != null) vacSkipped.add(1);
                  return completedFuture(null);
                }
              }
              MaintenanceTask.Vacuum v = MaintenanceTask.Vacuum.newBuilder()
                  .setSegId(segId)
                  .setMinDeletedRatio(thr)
                  .build();
              MaintenanceTask mt =
                  MaintenanceTask.newBuilder().setVacuum(v).build();
              String key = "vacuum-if-needed:" + segId;
              if (vacScheduled != null) vacScheduled.add(1);
              return maintenanceQueue
                  .enqueueIfNotExists(tr, key, mt)
                  .thenApply(x -> null);
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
          })));
    }
    return allOf(fs.toArray(CompletableFuture[]::new)).thenApply(x -> null);
  }

  private CompletableFuture<Void> scheduleVacuumIfNeeded(Transaction tx, Set<Integer> segIds) {
    if (maintenanceQueue == null || segIds == null || segIds.isEmpty()) return completedFuture(null);
    double thr = config.getVacuumMinDeletedRatio();
    List<CompletableFuture<Void>> fs = new ArrayList<>();
    for (int segId : segIds) {
      fs.add(indexDirs
          .segmentKeys(tx, segId)
          .thenCompose(sk -> tx.get(sk.metaKey()))
          .thenCompose(bytes -> {
            if (bytes == null) return completedFuture(null);
            try {
              SegmentMeta sm = SegmentMeta.parseFrom(bytes);
              long live = sm.getCount();
              long del = sm.getDeletedCount();
              double ratio = (live + del) == 0 ? 0.0 : ((double) del) / ((double) (live + del));
              if (ratio < thr) {
                if (vacSkipped != null) vacSkipped.add(1);
                return completedFuture(null);
              }
              long cdMs = Math.max(0, config.getVacuumCooldown().toMillis());
              if (cdMs > 0 && sm.getLastVacuumAtMs() > 0) {
                long now = config.getInstantSource().instant().toEpochMilli();
                if (now - sm.getLastVacuumAtMs() < cdMs) {
                  if (vacSkipped != null) vacSkipped.add(1);
                  return completedFuture(null);
                }
              }
              MaintenanceTask.Vacuum v = MaintenanceTask.Vacuum.newBuilder()
                  .setSegId(segId)
                  .setMinDeletedRatio(thr)
                  .build();
              MaintenanceTask mt =
                  MaintenanceTask.newBuilder().setVacuum(v).build();
              String key = "vacuum-if-needed:" + segId;
              if (vacScheduled != null) vacScheduled.add(1);
              return maintenanceQueue
                  .enqueueIfNotExists(tx, key, mt)
                  .thenApply(x -> null);
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
          }));
    }
    return allOf(fs.toArray(CompletableFuture[]::new)).thenApply(x -> null);
  }

  /**
   * Lists known segment ids by scanning for meta keys under the segments directory.
   */
  private CompletableFuture<List<Integer>> listSegmentsWithMeta(FdbDirectories.IndexDirectories dirs) {
    Database db = config.getDatabase();
    Subspace reg = dirs.segmentsIndexSubspace();
    Range r = reg.range();
    return db.readAsync(tr -> tr.getRange(r).asList()).thenApply(kvs -> {
      List<Integer> ids = new ArrayList<>(kvs.size());
      for (KeyValue kv : kvs) {
        Tuple t = reg.unpack(kv.getKey());
        ids.add(Math.toIntExact(t.getLong(0)));
      }
      ids.sort(Integer::compareTo);
      return ids;
    });
  }

  /**
   * Searches a single segment, dispatching to SEALED or ACTIVE/PENDING strategies.
   */
  private CompletableFuture<List<SearchResult>> searchSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k, SearchParams params) {
    return db.readAsync(tr -> indexDirs.segmentMetaKey(tr, segId).thenCompose(mk -> tr.get(mk)))
        .thenCompose(metaB -> {
          if (metaB == null) {
            LOG.debug("searchSegment segId={} meta missing; returning empty", segId);
            return completedFuture(List.of());
          }
          try {
            SegmentMeta sm = SegmentMeta.parseFrom(metaB);
            LOG.debug("searchSegment segId={} state={} count={}", segId, sm.getState(), sm.getCount());
            if (sm.getState() == SegmentMeta.State.SEALED
                || sm.getState() == SegmentMeta.State.COMPACTING) {
              return searchSealedSegment(db, dirs, segId, q, k, params);
            }
            if (sm.getState() == SegmentMeta.State.WRITING) {
              // Destination segment under construction: not visible to search
              return completedFuture(List.of());
            }
            return searchBruteForceSegment(db, dirs, segId, q, k);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        });
  }

  /**
   * Searches a single segment via brute-force and returns its top-K results.
   */
  private CompletableFuture<List<SearchResult>> searchBruteForceSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k) {
    return db.readAsync(tr -> indexDirs
        .segmentKeys(tr, segId)
        .thenCompose(sk -> tr.getRange(sk.vectorsDir().range()).asList().thenCompose(kvs -> {
          LOG.debug("Brute-force segId={} loaded {} vector records", segId, kvs.size());
          // Prefetch gids for all vecIds
          return indexDirs.segmentKeys(tr, segId).thenCompose(sk2 -> {
            List<Integer> vecIds = new ArrayList<>(kvs.size());
            for (KeyValue kv : kvs)
              vecIds.add(
                  (int) sk2.vectorsDir().unpack(kv.getKey()).getLong(0));
            List<CompletableFuture<byte[]>> gidF = new ArrayList<>(vecIds.size());
            for (int vid : vecIds)
              gidF.add(tr.get(indexDirs.gidRevDir().pack(Tuple.from(segId, vid))));
            return allOf(gidF.toArray(CompletableFuture[]::new)).thenApply(vv -> {
              List<SearchResult> results = new ArrayList<>();
              for (int i = 0; i < kvs.size(); i++) {
                try {
                  VectorRecord rec =
                      VectorRecord.parseFrom(kvs.get(i).getValue());
                  if (rec.getDeleted()) continue;
                  float[] emb = FloatPacker.bytesToFloats(
                      rec.getEmbedding().toByteArray());
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
                  byte[] gb = gidF.get(i).getNow(null);
                  if (gb == null) continue; // skip if mapping missing; gids are opaque
                  long gid = Tuple.fromBytes(gb).getLong(0);
                  results.add(SearchResult.builder()
                      .gid(gid)
                      .score(score)
                      .distance(distance)
                      .payload(rec.getPayload().toByteArray())
                      .build());
                } catch (InvalidProtocolBufferException e) {
                  throw new RuntimeException(e);
                }
              }
              results.sort(Comparator.comparingDouble(SearchResult::score)
                  .reversed());
              if (!results.isEmpty()) {
                int limit = Math.min(3, results.size());
                LOG.debug(
                    "Brute-force segId={} top {}: {}",
                    segId,
                    limit,
                    results.subList(0, limit).stream()
                        .map(r -> String.format("(id=%d,score=%.4f)", (int) r.gid(), r.score()))
                        .toList());
              }
              if (results.size() > k) return results.subList(0, k);
              return results;
            });
          });
        })));
  }

  /**
   * Searches a SEALED segment using PQ approximate scoring then exact rerank of top candidates.
   */
  private CompletableFuture<List<SearchResult>> searchSealedSegment(
      Database db, FdbDirectories.IndexDirectories dirs, int segId, float[] q, int k, SearchParams params) {
    return caches.getCodebookCacheAsync().get(segId).thenCompose(centroids -> {
      if (centroids == null) {
        LOG.warn("Missing PQ codebook for sealed segment segId={}", segId);
        return completedFuture(List.of());
      }
      double[][] lut = buildLut(centroids, q);
      return config.getDatabase()
          .readAsync(tr -> indexDirs.segmentKeys(tr, segId).thenCompose(sk -> tr.getRange(
                  sk.pqCodesDir().range())
              .asList()
              .thenApply(kvs -> new SimpleEntry<>(sk, kvs))))
          .thenCompose(entry -> {
            var sk = entry.getKey();
            var codeKvs = entry.getValue();
            LOG.debug("Sealed segId={} has {} PQ code entries", segId, codeKvs.size());
            // Build approx distances for all codes and retain a map for quick lookup
            List<Approx> approxAll = new ArrayList<>();
            Map<Integer, byte[]> codeMap = new HashMap<>();
            int m = centroids.length;
            int kCent = centroids[0].length;
            for (KeyValue kv : codeKvs) {
              int vecId =
                  (int) sk.pqCodesDir().unpack(kv.getKey()).getLong(0);
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
        Set<Long> miss2 = new HashSet<>();
        for (Approx cur : expand) {
          long ckey = SegmentCaches.adjKey(segId, cur.vecId());
          if (caches.getAdjacencyCache().getIfPresent(ckey) == null) miss2.add(ckey);
        }
        CompletableFuture<Map<Long, int[]>> bulk2 = miss2.isEmpty()
            ? CompletableFuture.completedFuture(Map.of())
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
    LOG.debug("Exact rerank segId={} candidates={}", segId, cand.size());
    return db.readAsync(tr -> indexDirs.segmentKeys(tr, segId).thenCompose(sk -> {
      List<CompletableFuture<VectorRecord>> recF = new ArrayList<>();
      List<CompletableFuture<byte[]>> gidF = new ArrayList<>();
      for (Approx a : cand) {
        recF.add(tr.get(sk.vectorKey(a.vecId())).thenApply(v -> {
          if (v == null) return null;
          try {
            return VectorRecord.parseFrom(v);
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }));
        gidF.add(tr.get(indexDirs.gidRevDir().pack(Tuple.from(segId, a.vecId()))));
      }
      CompletableFuture<Void> waitAll =
          allOf(Stream.concat(recF.stream(), gidF.stream()).toArray(CompletableFuture[]::new));
      return waitAll.thenApply(v -> {
        List<SearchResult> out = new ArrayList<>();
        for (int i = 0; i < cand.size(); i++) {
          VectorRecord rec = recF.get(i).getNow(null);
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
          byte[] gb = gidF.get(i).getNow(null);
          if (gb == null) continue; // gids are opaque; mapping must exist
          long gid = Tuple.fromBytes(gb).getLong(0);
          out.add(SearchResult.builder()
              .gid(gid)
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
                  .map(r -> String.format("(id=%d,score=%.4f)", (int) r.gid(), r.score()))
                  .toList());
        }
        if (out.size() > k) return out.subList(0, k);
        return out;
      });
    }));
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
