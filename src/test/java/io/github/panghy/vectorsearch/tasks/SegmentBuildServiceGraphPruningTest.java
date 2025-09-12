package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.Adjacency;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import io.github.panghy.vectorsearch.util.FloatPacker;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class SegmentBuildServiceGraphPruningTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-graph-prune", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void prunes_edges_when_alpha_enabled() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .pqM(1)
        .pqK(2)
        .graphDegree(2)
        .graphBuildBreadth(3)
        .graphAlpha(1.1)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);

    // Write 4 vectors at positions 0,1,2,3 into seg0
    db.run(tr -> {
      try {
        var sk = dirs.segmentKeys(tr, 0).get();
        for (int i = 0; i < 4; i++) {
          VectorRecord rec = VectorRecord.newBuilder()
              .setSegId(0)
              .setVecId(i)
              .setEmbedding(
                  com.google.protobuf.ByteString.copyFrom(FloatPacker.floatsToBytes(new float[] {i})))
              .setDeleted(false)
              .build();
          tr.set(sk.vectorKey(i), rec.toByteArray());
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });

    new SegmentBuildService(cfg, dirs).build(0).get(10, TimeUnit.SECONDS);

    // Read adjacency for vecId 0; with pruning, should keep only neighbor 1
    byte[] adj0 = db.readAsync(tr -> dirs.segmentKeys(tr, 0)
            .thenCompose(sk -> tr.get(sk.graphDir().pack(Tuple.from(0)))))
        .get(5, TimeUnit.SECONDS);
    int[] n0;
    try {
      n0 = Adjacency.parseFrom(adj0).getNeighborIdsList().stream()
          .mapToInt(Integer::intValue)
          .toArray();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    assertThat(n0.length).isEqualTo(1);
    assertThat(n0[0]).isEqualTo(1);
  }

  @Test
  void no_pruning_when_alpha_disabled() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .pqM(1)
        .pqK(2)
        .graphDegree(2)
        .graphBuildBreadth(3)
        .graphAlpha(1.0)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    // Write 3 vectors at positions 0,1,2 into seg0
    db.run(tr -> {
      try {
        var sk = dirs.segmentKeys(tr, 0).get();
        for (int i = 0; i < 3; i++) {
          VectorRecord rec = VectorRecord.newBuilder()
              .setSegId(0)
              .setVecId(i)
              .setEmbedding(
                  com.google.protobuf.ByteString.copyFrom(FloatPacker.floatsToBytes(new float[] {i})))
              .setDeleted(false)
              .build();
          tr.set(sk.vectorKey(i), rec.toByteArray());
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });

    new SegmentBuildService(cfg, dirs).build(0).get(10, TimeUnit.SECONDS);
    byte[] adj0 = db.readAsync(tr -> dirs.segmentKeys(tr, 0)
            .thenCompose(sk -> tr.get(sk.graphDir().pack(Tuple.from(0)))))
        .get(5, TimeUnit.SECONDS);
    int[] n0;
    try {
      n0 = Adjacency.parseFrom(adj0).getNeighborIdsList().stream()
          .mapToInt(Integer::intValue)
          .toArray();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    assertThat(n0.length).isEqualTo(2);
    assertThat(n0[0]).isEqualTo(1);
    assertThat(n0[1]).isEqualTo(2);
  }
}
