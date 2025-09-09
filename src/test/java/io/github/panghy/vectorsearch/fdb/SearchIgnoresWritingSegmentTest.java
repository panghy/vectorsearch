package io.github.panghy.vectorsearch.fdb;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.ByteString;
import io.github.panghy.vectorsearch.api.SearchResult;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
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
 * Ensures that segments in WRITING state are not used by search even if present in the registry.
 */
public class SearchIgnoresWritingSegmentTest {
  Database db;
  DirectorySubspace root;
  VectorIndex index;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-ignore-writing", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    index = VectorIndex.createOrOpen(VectorIndexConfig.builder(db, root)
            .dimension(3)
            .localWorkerThreads(0)
            .build())
        .get(10, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    if (index != null) index.close();
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  void writing_segment_is_invisible_to_search() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    int segId = 100;
    // Create WRITING meta and a single vector record, register in segment index
    db.run(tr -> {
      SegmentMeta sm = SegmentMeta.newBuilder()
          .setSegmentId(segId)
          .setState(SegmentMeta.State.WRITING)
          .setCount(1)
          .build();
      tr.set(dirs.segmentKeys(segId).metaKey(), sm.toByteArray());
      VectorRecord rec = VectorRecord.newBuilder()
          .setSegId(segId)
          .setVecId(0)
          .setEmbedding(ByteString.copyFrom(
              io.github.panghy.vectorsearch.util.FloatPacker.floatsToBytes(new float[] {1f, 0f, 0f})))
          .setDeleted(false)
          .build();
      tr.set(dirs.segmentKeys(segId).vectorKey(0), rec.toByteArray());
      tr.set(dirs.segmentsIndexKey(segId), new byte[0]);
      return null;
    });

    // Query for that vector; it should not be returned because WRITING is ignored
    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).isEmpty();
  }
}
