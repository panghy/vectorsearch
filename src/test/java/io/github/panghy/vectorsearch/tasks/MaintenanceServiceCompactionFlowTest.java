package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.subspace.Subspace;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * End-to-end compaction flow coverage: creates two sealed segments and compacts them into one.
 */
public class MaintenanceServiceCompactionFlowTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-compact-flow", UUID.randomUUID().toString()),
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
  void compactsTwoSealedIntoOne() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(1)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    try {
      // Create three small segments; seal first two
      index.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS); // seg0
      index.add(new float[] {5, 6, 7, 8}, null).get(5, TimeUnit.SECONDS); // seg1
      index.add(new float[] {9, 0, 1, 2}, null).get(5, TimeUnit.SECONDS); // seg2 active
      var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
      new SegmentBuildService(cfg, dirs).build(1).get(5, TimeUnit.SECONDS);

      // Run compaction on [0,1]
      new MaintenanceService(cfg, dirs).compactSegments(List.of(0, 1)).get(30, TimeUnit.SECONDS);

      // Registry should no longer contain 0,1; a new sealed segment should exist
      Subspace reg = dirs.segmentsIndexSubspace();
      List<KeyValue> entries =
          db.readAsync(tr -> tr.getRange(reg.range()).asList()).get(5, TimeUnit.SECONDS);
      List<Integer> segs = entries.stream()
          .map(kv -> Math.toIntExact(reg.unpack(kv.getKey()).getLong(0)))
          .sorted()
          .toList();
      assertThat(segs).doesNotContain(0, 1);
      // At least seg2 and the new compacted seg should be present
      assertThat(segs.size()).isGreaterThanOrEqualTo(2);
    } finally {
      index.close();
    }
  }
}
