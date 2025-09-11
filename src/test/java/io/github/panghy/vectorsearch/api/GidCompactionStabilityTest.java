package io.github.panghy.vectorsearch.api;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.tasks.MaintenanceService;
import io.github.panghy.vectorsearch.tasks.SegmentBuildService;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies that gids remain stable across compaction: the numeric gid returned by add/query
 * does not change after segments are merged, and resolveIds(gid) reflects the new (segment,vec).
 */
class GidCompactionStabilityTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-gid-compact", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void teardown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void gids_stable_and_remap_after_compaction() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    try {
      // Insert 4 vectors across two segments (2 per segment)
      List<Long> gids = new ArrayList<>();
      gids.add(index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS));
      gids.add(index.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS));
      // Seal seg0
      FdbDirectories.IndexDirectories dirs =
          FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
      // Two more in seg1, then seal seg1
      gids.add(index.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS));
      gids.add(index.add(new float[] {1, 1, 0, 0}, null).get(5, TimeUnit.SECONDS));
      new SegmentBuildService(cfg, dirs).build(1).get(5, TimeUnit.SECONDS);

      // Collect existing sealed segment ids from registry (should be 0 and 1)
      var sealed = db.readAsync(tr ->
              tr.getRange(dirs.segmentsIndexSubspace().range()).asList())
          .get(5, TimeUnit.SECONDS);
      assertThat(sealed.stream()
              .map(kv -> (int) dirs.segmentsIndexSubspace()
                  .unpack(kv.getKey())
                  .getLong(0))
              .toList())
          .contains(0, 1);

      // Compact [0,1] -> new segment (e.g., seg2). After compaction, sources removed.
      new MaintenanceService(cfg, dirs).compactSegments(List.of(0, 1)).get(10, TimeUnit.SECONDS);

      // Read registry; ensure a new segment id exists and sources are gone
      List<KeyValue> after = db.readAsync(tr ->
              tr.getRange(dirs.segmentsIndexSubspace().range()).asList())
          .get(5, TimeUnit.SECONDS);
      List<Integer> reg = after.stream()
          .map(kv -> Math.toIntExact(
              dirs.segmentsIndexSubspace().unpack(kv.getKey()).getLong(0)))
          .toList();
      assertThat(reg).doesNotContain(0, 1);
      int newSeg = reg.stream().mapToInt(Integer::intValue).max().orElseThrow();

      // Query and ensure gids resolve to the new segment; at least one original gid appears
      var hits = index.query(new float[] {1, 0, 0, 0}, 10).get(5, TimeUnit.SECONDS);
      List<Long> returned = hits.stream().map(SearchResult::gid).toList();
      // at least one original gid should show up verbatim
      assertThat(returned).anySatisfy(g -> assertThat(gids).contains(g));

      // resolveIds(gid) should now map to the new segment id
      long[] gArr = returned.stream().mapToLong(Long::longValue).toArray();
      int[][] pairs = index.resolveIds(gArr).get(5, TimeUnit.SECONDS);
      for (int[] p : pairs) assertThat(p[0]).isEqualTo(newSeg);
      // And all returned gids resolve to the new segment as well
      int[][] rPairs = index.resolveIds(
              returned.stream().mapToLong(Long::longValue).toArray())
          .get(5, TimeUnit.SECONDS);
      for (int[] p : rPairs) assertThat(p[0]).isEqualTo(newSeg);
    } finally {
      index.close();
    }
  }
}
