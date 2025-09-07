package io.github.panghy.vectorsearch.fdb;

/**
 * Unit tests for FdbDirectories key helpers and real packing/unpacking behavior.
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class FdbDirectoriesTest {

  @Test
  void helperKeyPackers() {
    // Validate IndexDirectories exposes key helpers and SegmentKeys builds flattened keys
    // Use Mockito to mock DirectorySubspace pack behavior
    DirectorySubspace idx = mock(DirectorySubspace.class);
    DirectorySubspace segs = mock(DirectorySubspace.class);
    when(idx.pack(Tuple.from("meta"))).thenReturn(new byte[] {1});
    when(idx.pack(Tuple.from("currentSegment"))).thenReturn(new byte[] {2});
    when(segs.pack(Tuple.from("000123", "meta"))).thenReturn(new byte[] {3});
    when(segs.pack(Tuple.from("000123", "vectors", 1))).thenReturn(new byte[] {4});
    when(segs.pack(Tuple.from("000123", "pq", "codebook"))).thenReturn(new byte[] {5});
    when(segs.pack(Tuple.from("000123", "pq", "codes", 1))).thenReturn(new byte[] {6});
    when(segs.pack(Tuple.from("000123", "graph", 1))).thenReturn(new byte[] {7});
    FdbDirectories.IndexDirectories dirs = new FdbDirectories.IndexDirectories(idx, segs, idx);
    assertThat(dirs.metaKey()).isNotNull();
    assertThat(dirs.currentSegmentKey()).isNotNull();
    var sk = dirs.segmentKeys("000123");
    assertThat(sk.metaKey()).isNotNull();
    assertThat(sk.vectorKey(1)).isNotNull();
    assertThat(sk.pqCodebookKey()).isNotNull();
    assertThat(sk.pqCodeKey(1)).isNotNull();
    assertThat(sk.graphKey(1)).isNotNull();
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
      var sk = dirs.segmentKeys("000123");
      byte[] cbook = sk.pqCodebookKey();
      byte[] code5 = sk.pqCodeKey(5);
      byte[] g7 = sk.graphKey(7);
      Tuple t1 = dirs.segmentsDir().unpack(cbook);
      Tuple t2 = dirs.segmentsDir().unpack(code5);
      Tuple t3 = dirs.segmentsDir().unpack(g7);
      assertThat(t1.getString(0)).isEqualTo("000123");
      assertThat(t1.getString(1)).isEqualTo("pq");
      assertThat(t1.getString(2)).isEqualTo("codebook");
      assertThat(t2.getString(0)).isEqualTo("000123");
      assertThat(t2.getString(1)).isEqualTo("pq");
      assertThat(t2.getString(2)).isEqualTo("codes");
      assertThat(t2.getLong(3)).isEqualTo(5);
      assertThat(t3.getString(0)).isEqualTo("000123");
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
    String seg = "000123";

    assertThat(FdbPathUtil.indexPath(ix)).containsExactly("indexes", ix);
    assertThat(FdbPathUtil.segmentsPath(ix)).containsExactly("indexes", ix, "segments");
    assertThat(FdbPathUtil.segmentPath(ix, seg)).containsExactly("indexes", ix, "segments", seg);
    assertThat(FdbPathUtil.vectorsPath(ix, seg)).containsExactly("indexes", ix, "segments", seg, "vectors");
    assertThat(FdbPathUtil.pqPath(ix, seg)).containsExactly("indexes", ix, "segments", seg, "pq");
    assertThat(FdbPathUtil.pqCodesPath(ix, seg)).containsExactly("indexes", ix, "segments", seg, "pq", "codes");
    assertThat(FdbPathUtil.graphPath(ix, seg)).containsExactly("indexes", ix, "segments", seg, "graph");
    assertThat(FdbPathUtil.tasksPath(ix)).containsExactly("indexes", ix, "tasks");
  }
}
