package io.github.panghy.vectorsearch.tasks;

import static io.github.panghy.vectorsearch.graph.GraphBuilder.buildL2Neighbors;
import static java.util.Objects.requireNonNull;
import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.pq.PqEncoder;
import io.github.panghy.vectorsearch.pq.PqTrainer;
import io.github.panghy.vectorsearch.proto.Adjacency;
import io.github.panghy.vectorsearch.proto.PQCodebook;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.util.FloatPacker;
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
  public CompletableFuture<Void> build(int segId) {
    Database db = config.getDatabase();
    String segStr = String.format("%06d", segId);
    FdbDirectories.IndexDirectories dirs = indexDirs;
    Subspace vectorsPrefix = new Subspace(dirs.segmentsDir().pack(Tuple.from(segStr, "vectors")));
    Range vr = vectorsPrefix.range();
    LOGGER.debug("Building segment {}", segId);
    return db.readAsync(tr -> tr.getRange(vr).asList())
        .thenCompose(kvs -> writeBuildArtifacts(dirs, segStr, kvs))
        .thenCompose(v -> sealSegment(dirs, segStr))
        .whenComplete((v, ex) -> {
          if (ex != null) {
            LOGGER.error("Failed to build segment {}", segId, ex);
          } else {
            LOGGER.debug("Built segment {}", segId);
          }
        });
  }

  private CompletableFuture<Void> writeBuildArtifacts(
      FdbDirectories.IndexDirectories dirs, String segStr, List<KeyValue> kvs) {
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
      int sub = Math.max(1, d / Math.max(1, m));
      centroids = new float[m][k][sub];
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
    // Use FDB's approximate transaction size to split work before hitting the 10 MB limit.
    final long TXN_LIMIT = 10L * 1024 * 1024; // 10 MB
    final long SOFT_LIMIT = (long) (TXN_LIMIT * 0.9); // leave ~10% headroom
    final int CHECK_EVERY = 32; // frequency of size checks to amortize overhead

    // Precompute adjacency using raw vectors
    final int[][] neighbors = buildL2Neighbors(
        vectors.toArray(new float[0][]),
        Math.max(0, Math.min(config.getGraphDegree(), Math.max(0, vectors.size() - 1))));
    final float[][][] cCentroids = centroids;
    final List<float[]> vVectors = vectors;

    int idx = 0;
    boolean wroteCodebook = false;
    while (idx < kvs.size() || !wroteCodebook) {
      final int start = idx;
      final boolean writeCb = !wroteCodebook;
      int next = db.run(tr -> {
        // Optionally write codebook in this transaction
        if (writeCb) {
          tr.set(dirs.segmentKeys(segStr).pqCodebookKey(), cb.build().toByteArray());
        }
        FdbDirectories.SegmentKeys sk = dirs.segmentKeys(segStr);
        int wrote = 0;
        int j = start;
        for (; j < kvs.size(); j++) {
          KeyValue kv = kvs.get(j);
          long vecId = vectorsPrefixUnpackId(dirs, kv.getKey());
          byte[] codes = PqEncoder.encode(cCentroids, vVectors.get(j));
          tr.set(sk.pqCodeKey((int) vecId), codes);
          Adjacency adj = Adjacency.newBuilder()
              .addAllNeighborIds(java.util.Arrays.stream(neighbors[(int) vecId])
                  .boxed()
                  .toList())
              .build();
          tr.set(sk.graphKey((int) vecId), adj.toByteArray());
          wrote++;
          if ((wrote % CHECK_EVERY) == 0) {
            Long approx = tr.getApproximateSize().join();
            if (approx != null && approx >= SOFT_LIMIT) {
              j++; // include this write and split
              break;
            }
          }
        }
        return j;
      });
      wroteCodebook = true;
      idx = next;
    }

    return completedFuture(null);
  }

  private long vectorsPrefixUnpackId(FdbDirectories.IndexDirectories dirs, byte[] key) {
    // key is segmentsDir.pack(Tuple.from(segStr, "vectors", vecId)) + suffix
    Tuple t = dirs.segmentsDir().unpack(key);
    // Expect [segStr, "vectors", vecId]
    return t.getLong(t.size() - 1);
  }

  private CompletableFuture<Void> sealSegment(FdbDirectories.IndexDirectories dirs, String segStr) {
    Database db = config.getDatabase();
    return db.runAsync(tr -> tr.get(dirs.segmentKeys(segStr).metaKey()).thenApply(bytes -> {
      if (bytes == null) return null;
      try {
        SegmentMeta sm = SegmentMeta.parseFrom(bytes);
        if (sm.getState() == SegmentMeta.State.SEALED) return null;
        SegmentMeta sealed = sm.toBuilder()
            .setState(SegmentMeta.State.SEALED)
            .setCreatedAtMs(sm.getCreatedAtMs() == 0 ? Instant.now().toEpochMilli() : sm.getCreatedAtMs())
            .build();
        tr.set(dirs.segmentKeys(segStr).metaKey(), sealed.toByteArray());
        return null;
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
    }));
  }
}
