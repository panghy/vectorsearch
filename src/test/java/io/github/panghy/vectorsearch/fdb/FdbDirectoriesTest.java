package io.github.panghy.vectorsearch.fdb;

/**
 * Unit tests for FdbDirectories key helpers and real packing/unpacking behavior.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.tuple.Tuple;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class FdbDirectoriesTest {

  @Test
  void helperKeyPackers() throws Exception {
    var db = FDB.selectAPIVersion(730).open();
    var root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-keys-pack", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    try {
      var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      // Index-level keys
      Tuple tMeta = dirs.indexDir().unpack(dirs.metaKey());
      Tuple tCur = dirs.indexDir().unpack(dirs.currentSegmentKey());
      Tuple tMax = dirs.indexDir().unpack(dirs.maxSegmentKey());
      assertThat(tMeta.getString(0)).isEqualTo("meta");
      assertThat(tCur.getString(0)).isEqualTo("currentSegment");
      assertThat(tMax.getString(0)).isEqualTo(FdbPathUtil.MAX_SEGMENT);

      // Segment-level helpers
      var sk = dirs.segmentKeys(123);
      assertThat(sk.metaKey()).isNotNull();
      assertThat(sk.vectorKey(1)).isNotNull();
      assertThat(sk.pqCodebookKey()).isNotNull();
      assertThat(sk.pqCodeKey(1)).isNotNull();
      assertThat(sk.graphKey(1)).isNotNull();
    } finally {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  void realSegmentKeysPackExpectedTuples() throws Exception {
    var db = FDB.selectAPIVersion(730).open();
    var root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-keys-merge", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    try {
      var dirs = FdbDirectories.openIndex(root, db).get(5, java.util.concurrent.TimeUnit.SECONDS);
      var sk = dirs.segmentKeys(123);
      byte[] cbook = sk.pqCodebookKey();
      byte[] code5 = sk.pqCodeKey(5);
      byte[] g7 = sk.graphKey(7);
      Tuple t1 = dirs.segmentsDir().unpack(cbook);
      Tuple t2 = dirs.segmentsDir().unpack(code5);
      Tuple t3 = dirs.segmentsDir().unpack(g7);
      assertThat(t1.getLong(0)).isEqualTo(123);
      assertThat(t1.getString(1)).isEqualTo("pq");
      assertThat(t1.getString(2)).isEqualTo("codebook");
      assertThat(t2.getLong(0)).isEqualTo(123);
      assertThat(t2.getString(1)).isEqualTo("pq");
      assertThat(t2.getString(2)).isEqualTo("codes");
      assertThat(t2.getLong(3)).isEqualTo(5);
      assertThat(t3.getLong(0)).isEqualTo(123);
      assertThat(t3.getString(1)).isEqualTo("graph");
      assertThat(t3.getLong(2)).isEqualTo(7);
    } finally {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  void pathUtilPathsAreCorrect() {
    String ix = "myIndex";
    int seg = 123;

    assertThat(FdbPathUtil.indexPath(ix)).containsExactly("indexes", ix);
    assertThat(FdbPathUtil.segmentsPath(ix)).containsExactly("indexes", ix, "segments");
    assertThat(FdbPathUtil.segmentPath(ix, String.valueOf(seg)))
        .containsExactly("indexes", ix, "segments", String.valueOf(seg));
    assertThat(FdbPathUtil.vectorsPath(ix, String.valueOf(seg)))
        .containsExactly("indexes", ix, "segments", String.valueOf(seg), "vectors");
    assertThat(FdbPathUtil.pqPath(ix, String.valueOf(seg)))
        .containsExactly("indexes", ix, "segments", String.valueOf(seg), "pq");
    assertThat(FdbPathUtil.pqCodesPath(ix, String.valueOf(seg)))
        .containsExactly("indexes", ix, "segments", String.valueOf(seg), "pq", "codes");
    assertThat(FdbPathUtil.graphPath(ix, String.valueOf(seg)))
        .containsExactly("indexes", ix, "segments", String.valueOf(seg), "graph");
    assertThat(FdbPathUtil.tasksPath(ix)).containsExactly("indexes", ix, "tasks");
  }
}
