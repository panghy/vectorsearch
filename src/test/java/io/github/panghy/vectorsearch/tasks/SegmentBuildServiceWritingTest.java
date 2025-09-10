package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.ByteString;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.proto.VectorRecord;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies SegmentBuildService can seal a WRITING segment (created by compaction) to SEALED.
 */
public class SegmentBuildServiceWritingTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-build-writing", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  void sealsWritingToSealed() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(2)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    int segId = 0;
    // Create WRITING meta and two vectors
    db.run(tr -> {
      SegmentMeta sm = SegmentMeta.newBuilder()
          .setSegmentId(segId)
          .setState(SegmentMeta.State.WRITING)
          .setCount(2)
          .build();
      try {
        var sk = dirs.segmentKeys(tr, segId).get();
        tr.set(sk.metaKey(), sm.toByteArray());
        for (int i = 0; i < 2; i++) {
          VectorRecord rec = VectorRecord.newBuilder()
              .setSegId(segId)
              .setVecId(i)
              .setEmbedding(
                  ByteString.copyFrom(io.github.panghy.vectorsearch.util.FloatPacker.floatsToBytes(
                      new float[] {i, 0, 0, 0})))
              .setDeleted(false)
              .build();
          tr.set(sk.vectorKey(i), rec.toByteArray());
        }
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      // Ensure registry has the segment
      tr.set(dirs.segmentsIndexKey(segId), new byte[0]);
      return null;
    });

    new SegmentBuildService(cfg, dirs).build(segId).get(10, TimeUnit.SECONDS);
    // Sanity: vectors subspace can be opened via DirectoryLayer
    db.readAsync(tr -> dirs.segmentKeys(tr, segId)
            .thenCompose(
                sk2 -> tr.getRange(sk2.vectorsDir().range()).asList()))
        .get(5, TimeUnit.SECONDS);

    SegmentMeta after = db.runAsync(tr -> dirs.segmentKeys(tr, segId)
            .thenCompose(sk -> tr.get(sk.metaKey()))
            .thenApply((byte[] b) -> {
              try {
                return SegmentMeta.parseFrom(b);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(after.getState()).isEqualTo(SegmentMeta.State.SEALED);
    // PQ codebook should exist
    byte[] cb = db.readAsync(tr -> dirs.segmentKeys(tr, segId).thenCompose(sk -> tr.get(sk.pqCodebookKey())))
        .get(5, TimeUnit.SECONDS);
    assertThat(cb).isNotNull();
  }
}
