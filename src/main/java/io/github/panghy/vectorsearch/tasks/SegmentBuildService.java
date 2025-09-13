package io.github.panghy.vectorsearch.tasks;

import static io.github.panghy.vectorsearch.graph.GraphBuilder.buildL2Neighbors;
import static io.github.panghy.vectorsearch.graph.GraphBuilder.buildPrunedNeighbors;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.fdb.FdbPathUtil;
import io.github.panghy.vectorsearch.pq.PqEncoder;
import io.github.panghy.vectorsearch.pq.PqTrainer;
import io.github.panghy.vectorsearch.proto.Adjacency;
import io.github.panghy.vectorsearch.proto.PQCodebook;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.util.FloatPacker;
import io.github.panghy.vectorsearch.util.Metrics;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds per-segment graph + PQ structures.
 *
 * <p>Initial implementation writes placeholders:
 * <ul>
 *   <li>PQ codebook of zeros</li>
 *   <li>PQ codes of zeros per vector</li>
 *   <li>Empty adjacency per vector</li>
 * </ul>
 * then marks the segment SEALED. Future versions will train real PQ codebooks and construct
 * neighbor graphs.</p>
 */
public class SegmentBuildService {

  public static final Logger LOGGER = LoggerFactory.getLogger(SegmentBuildService.class);

  private final VectorIndexConfig config;
  private final FdbDirectories.IndexDirectories indexDirs;

  /**
   * Dependency-injected constructor with resolved index directories.
   */
  public SegmentBuildService(VectorIndexConfig config, FdbDirectories.IndexDirectories indexDirs) {
    this.config = config;
    this.indexDirs = requireNonNull(indexDirs, "indexDirs");
  }

  /**
   * Builds artifacts for the given segment and seals it.
   *
   * @param segId segment identifier
   * @return future that completes when the build is persisted and the segment is sealed
   */
  public CompletableFuture<Void> build(long segId) {
    Database db = config.getDatabase();
    Tracer tracer = Metrics.tracer();
    Span span = tracer.spanBuilder("vectorsearch.build")
        .setSpanKind(SpanKind.INTERNAL)
        .setAttribute("segId", segId)
        .startSpan();
    long t0 = System.nanoTime();
    FdbDirectories.IndexDirectories dirs = indexDirs;
    LOGGER.debug("Building segment {}", segId);
    // Ensure per-segment subspaces exist (idempotent) before any reads.
    // This covers tests that prewrite meta/vector without creating all child directories.
    // Only build PENDING/WRITING segments. ACTIVE segments must never be sealed here.
    return db.runAsync(tr -> dirs.segmentKeys(tr, (int) segId).thenApply(sk -> null))
        .thenCompose($ ->
            db.readAsync(tr -> dirs.segmentKeys(tr, (int) segId).thenCompose(sk -> tr.get(sk.metaKey()))))
        .thenCompose(metaB -> {
          if (metaB != null) {
            try {
              SegmentMeta sm = SegmentMeta.parseFrom(metaB);
              if (sm.getState() != SegmentMeta.State.PENDING
                  && sm.getState() != SegmentMeta.State.WRITING) {
                LOGGER.debug(
                    "Segment {} not PENDING/WRITING (state={}); skipping build",
                    segId,
                    sm.getState());
                return completedFuture(null);
              }
            } catch (InvalidProtocolBufferException e) {
              throw new RuntimeException(e);
            }
          }
          return db.readAsync(tr -> dirs.segmentKeys(tr, (int) segId)
                  .thenCompose(sk ->
                      tr.getRange(sk.vectorsDir().range()).asList()))
              .exceptionally(ex -> {
                if (ex != null
                    && ex.getCause()
                        instanceof com.apple.foundationdb.directory.NoSuchDirectoryException) {
                  return java.util.List.of();
                }
                throw new java.util.concurrent.CompletionException(ex);
              })
              .thenCompose(kvs -> {
                if (!kvs.isEmpty()) return writeBuildArtifacts(dirs, (int) segId, kvs);
                return completedFuture(null);
              })
              .thenCompose(v -> sealSegment(dirs, (int) segId))
              .thenCompose(v -> ensureCodebookExists((int) segId));
        })
        .whenComplete((v, ex) -> {
          if (ex != null) {
            LOGGER.error("Failed to build segment {}", segId, ex);
            span.recordException(ex);
            span.setStatus(io.opentelemetry.api.trace.StatusCode.ERROR);
          } else {
            LOGGER.debug("Built segment {}", segId);
          }
          long durMs = (System.nanoTime() - t0) / 1_000_000;
          Attributes attrs = Attributes.builder()
              .put("segId", segId)
              .put("dim", config.getDimension())
              .put("degree", config.getGraphDegree())
              .put("index.path", io.github.panghy.vectorsearch.util.Metrics.dirPath(indexDirs.indexDir()))
              .build();
          Metrics.BUILD_COUNT.add(1, attrs);
          Metrics.BUILD_DURATION_MS.record((double) durMs, attrs);
          span.end();
        });
  }

  private CompletableFuture<Void> ensureCodebookExists(int segId) {
    Database db = config.getDatabase();
    return db.runAsync(tr -> indexDirs.segmentKeys(tr, segId).thenCompose(sk -> tr.get(sk.pqCodebookKey())
        .thenApply(cb -> {
          if (cb == null) {
            int m = config.getPqM();
            int k = config.getPqK();
            int subDim = Math.max(1, config.getDimension() / Math.max(1, m));
            float[][][] zeros = new float[m][k][subDim];
            tr.set(sk.pqCodebookKey(), buildCodebookBytes(zeros));
          }
          return null;
        })));
  }

  private CompletableFuture<Void> writeBuildArtifacts(
      FdbDirectories.IndexDirectories dirs, int segId, List<KeyValue> kvs) {
    Database db = config.getDatabase();

    // Decode vectors
    List<float[]> vectors = new ArrayList<>(kvs.size());
    for (KeyValue kv : kvs) {
      try {
        VectorRecord rec = VectorRecord.parseFrom(kv.getValue());
        vectors.add(FloatPacker.bytesToFloats(rec.getEmbedding().toByteArray()));
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
    }

    // 2) Train PQ or fallback to placeholder if invalid configuration
    int m = config.getPqM();
    int k = config.getPqK();
    int d = config.getDimension();
    float[][][] centroids;
    try {
      centroids = PqTrainer.train(vectors, d, m, k, 5, 42L);
    } catch (IllegalArgumentException ex) {
      // Fail the build so TaskQueue can retry later rather than sealing with a degenerate codebook.
      throw new IllegalStateException(
          "PQ training failed for segment " + segId + " (m=" + m + ", k=" + k + ", d=" + d + ")", ex);
    }

    // Build PQCodebook proto
    int subDim = centroids[0][0].length;
    PQCodebook.Builder cb = PQCodebook.newBuilder().setM(m).setK(k);
    for (int s = 0; s < m; s++) {
      ByteBuffer bb = ByteBuffer.allocate(k * subDim * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (int ci = 0; ci < k; ci++) {
        for (int di = 0; di < subDim; di++) bb.putFloat(centroids[s][ci][di]);
      }
      cb.addCentroids(ByteString.copyFrom(bb.array()));
    }

    // 3) Persist: codebook + codes + adjacency per vecId
    // Use FDB's approximate transaction size to split work before hitting the configured limit.
    final long softLimit = (long) (config.getBuildTxnLimitBytes() * config.getBuildTxnSoftLimitRatio());
    final int checkEvery = config.getBuildSizeCheckEvery();

    // Precompute adjacency using raw vectors
    int degree = Math.max(0, Math.min(config.getGraphDegree(), Math.max(0, vectors.size() - 1)));
    int lBuild = Math.max(degree, config.getGraphBuildBreadth());
    double alpha = config.getGraphAlpha();
    final int[][] neighbors = (alpha <= 1.0)
        ? buildL2Neighbors(vectors.toArray(new float[0][]), degree)
        : buildPrunedNeighbors(vectors.toArray(new float[0][]), degree, lBuild, alpha);
    final float[][][] cCentroids = centroids;
    final List<float[]> vVectors = vectors;

    // Asynchronously write chunks until all records (and codebook) are persisted.
    return writeChunkLoop(db, dirs, segId, kvs, vVectors, cCentroids, neighbors, softLimit, checkEvery);
  }

  private CompletableFuture<Void> writeChunkLoop(
      Database db,
      FdbDirectories.IndexDirectories dirs,
      int segId,
      List<KeyValue> kvs,
      List<float[]> vectors,
      float[][][] centroids,
      int[][] neighbors,
      long softLimit,
      int checkEvery) {
    return writeChunk(db, dirs, segId, kvs, vectors, centroids, neighbors, 0, false, softLimit, checkEvery);
  }

  private CompletableFuture<Void> writeChunk(
      Database db,
      FdbDirectories.IndexDirectories dirs,
      int segId,
      List<KeyValue> kvs,
      List<float[]> vectors,
      float[][][] centroids,
      int[][] neighbors,
      int start,
      boolean wroteCodebook,
      long softLimit,
      int checkEvery) {
    if (start >= kvs.size() && wroteCodebook) return completedFuture(null);
    final boolean writeCbThisTxn = !wroteCodebook;
    return db.runAsync(tr -> {
          String seg = Integer.toString(segId);
          DirectorySubspace segRoot = dirs.segmentsDir();
          CompletableFuture<DirectorySubspace> segDirF = segRoot.createOrOpen(tr, java.util.List.of(seg));
          CompletableFuture<DirectorySubspace> vectorsF =
              segRoot.open(tr, java.util.List.of(seg, FdbPathUtil.VECTORS));
          CompletableFuture<DirectorySubspace> pqF =
              segRoot.createOrOpen(tr, java.util.List.of(seg, FdbPathUtil.PQ));
          CompletableFuture<DirectorySubspace> pqCodesF =
              segRoot.createOrOpen(tr, java.util.List.of(seg, FdbPathUtil.PQ, FdbPathUtil.CODES));
          CompletableFuture<DirectorySubspace> graphF =
              segRoot.createOrOpen(tr, java.util.List.of(seg, FdbPathUtil.GRAPH));
          return CompletableFuture.allOf(segDirF, vectorsF, pqF, pqCodesF, graphF)
              .thenCompose(v -> {
                FdbDirectories.SegmentKeys sk = new FdbDirectories.SegmentKeys(
                    segId,
                    segDirF.join(),
                    vectorsF.join(),
                    pqF.join(),
                    pqCodesF.join(),
                    graphF.join());
                if (writeCbThisTxn) {
                  tr.set(sk.pqCodebookKey(), buildCodebookBytes(centroids));
                }
                return writeSome(
                        tr,
                        sk,
                        dirs,
                        kvs,
                        vectors,
                        centroids,
                        neighbors,
                        start,
                        checkEvery,
                        softLimit)
                    .thenApply(next -> next);
              });
        })
        .thenCompose(next -> writeChunk(
            db, dirs, segId, kvs, vectors, centroids, neighbors, next, true, softLimit, checkEvery));
  }

  private CompletableFuture<Integer> writeSome(
      Transaction tr,
      FdbDirectories.SegmentKeys sk,
      FdbDirectories.IndexDirectories dirs,
      List<KeyValue> kvs,
      List<float[]> vectors,
      float[][][] centroids,
      int[][] neighbors,
      int j,
      int checkEvery,
      long softLimit) {
    int wroteSinceCheck = 0;
    while (j < kvs.size()) {
      KeyValue kv = kvs.get(j);
      long vecId = vectorsUnpackId(sk, kv.getKey());
      byte[] codes = PqEncoder.encode(centroids, vectors.get(j));
      tr.set(sk.pqCodeKey((int) vecId), codes);
      Adjacency adj = Adjacency.newBuilder()
          .addAllNeighborIds(java.util.Arrays.stream(neighbors[(int) vecId])
              .boxed()
              .toList())
          .build();
      tr.set(sk.graphKey((int) vecId), adj.toByteArray());
      wroteSinceCheck++;
      j++;
      // Only perform approximate-size checks when a positive cadence is configured.
      if (checkEvery > 0 && (wroteSinceCheck % checkEvery) == 0) {
        final int jNow = j;
        return tr.getApproximateSize().thenCompose(sz -> {
          if (sz != null && sz >= softLimit) {
            return CompletableFuture.completedFuture(jNow);
          }
          return writeSome(tr, sk, dirs, kvs, vectors, centroids, neighbors, jNow, 0, softLimit);
        });
      }
    }
    return CompletableFuture.completedFuture(j);
  }

  private byte[] buildCodebookBytes(float[][][] centroids) {
    int m = centroids.length;
    int k = centroids[0].length;
    int subDim = centroids[0][0].length;
    PQCodebook.Builder cb = PQCodebook.newBuilder().setM(m).setK(k);
    for (int s = 0; s < m; s++) {
      ByteBuffer bb = ByteBuffer.allocate(k * subDim * 4).order(ByteOrder.LITTLE_ENDIAN);
      for (int ci = 0; ci < k; ci++) {
        for (int di = 0; di < subDim; di++) bb.putFloat(centroids[s][ci][di]);
      }
      cb.addCentroids(ByteString.copyFrom(bb.array()));
    }
    return cb.build().toByteArray();
  }

  private long vectorsUnpackId(FdbDirectories.SegmentKeys sk, byte[] key) {
    // key is vectorsDir.pack(Tuple.from(vecId)) + suffix
    Tuple t = sk.vectorsDir().unpack(key);
    return t.getLong(0);
  }

  private CompletableFuture<Void> sealSegment(FdbDirectories.IndexDirectories dirs, int segId) {
    Database db = config.getDatabase();
    return db.runAsync(tr -> dirs.segmentKeys(tr, segId)
        .thenCompose(sk ->
            tr.get(sk.metaKey()).thenApply(bytes -> new java.util.AbstractMap.SimpleEntry<>(sk, bytes)))
        .thenCompose(entry -> {
          var sk = entry.getKey();
          var bytes = entry.getValue();
          if (bytes == null) {
            // Meta missing: synthesize from current vectors count and seal.
            return tr.getRange(sk.vectorsDir().range()).asList().thenApply(kvs -> {
              SegmentMeta sealed = SegmentMeta.newBuilder()
                  .setSegmentId(segId)
                  .setState(SegmentMeta.State.SEALED)
                  .setCount(kvs.size())
                  .setCreatedAtMs(Instant.now().toEpochMilli())
                  .build();
              tr.set(sk.metaKey(), sealed.toByteArray());
              return null;
            });
          }
          try {
            SegmentMeta sm = SegmentMeta.parseFrom(bytes);
            if (sm.getState() != SegmentMeta.State.PENDING && sm.getState() != SegmentMeta.State.WRITING)
              return completedFuture(null); // only seal PENDING/WRITING
            SegmentMeta sealed = sm.toBuilder()
                .setState(SegmentMeta.State.SEALED)
                .setCreatedAtMs(
                    sm.getCreatedAtMs() == 0 ? Instant.now().toEpochMilli() : sm.getCreatedAtMs())
                .build();
            tr.set(sk.metaKey(), sealed.toByteArray());
            // Ensure a PQ codebook exists; if missing, write a trivial zero codebook.
            return tr.get(sk.pqCodebookKey()).thenApply(cb -> {
              if (cb == null) {
                int m = config.getPqM();
                int k = config.getPqK();
                int subDim = Math.max(1, config.getDimension() / Math.max(1, m));
                float[][][] zeros = new float[m][k][subDim];
                tr.set(sk.pqCodebookKey(), buildCodebookBytes(zeros));
              }
              return null;
            });
          } catch (InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
          }
        }));
  }
}
