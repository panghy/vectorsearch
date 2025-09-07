package io.github.panghy.vectorsearch.api;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.tasks.MaintenanceService;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Integration test for delete API and vacuum maintenance. */
public class VectorIndexDeleteAndVacuumTest {

  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-deltest", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .pqM(4)
        .pqK(16)
        .graphDegree(8)
        .maxSegmentSize(20)
        .localWorkerThreads(1)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
  }

  @AfterEach
  public void teardown() {
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
  public void deleteAndVacuumFlow() throws Exception {
    // Insert a handful of vectors.
    int n = 30;
    int[][] ids = new int[n][2];
    for (int i = 0; i < n; i++) {
      float[] v = new float[8];
      for (int d = 0; d < 8; d++) v[d] = (float) Math.sin(0.01 * i + d);
      int[] id = index.add(v, null).get(5, TimeUnit.SECONDS);
      ids[i] = id;
    }

    // Let builder seal at least one segment
    Thread.sleep(1000);

    // Delete a few vectors across segments
    index.delete(ids[0][0], ids[0][1]).get(5, TimeUnit.SECONDS);
    index.delete(ids[5][0], ids[5][1]).get(5, TimeUnit.SECONDS);

    // Ensure deleted ids are not returned
    float[] q = new float[] {1f, 0.1f, -0.2f, 0.3f, 0.0f, 0.4f, -0.1f, 0.2f};
    var results = index.query(q, 10).get(5, TimeUnit.SECONDS);
    assertThat(results.stream()
            .noneMatch(r -> (r.segmentId() == ids[0][0] && r.vectorId() == ids[0][1])
                || (r.segmentId() == ids[5][0] && r.vectorId() == ids[5][1])))
        .isTrue();

    // Run vacuum directly via service (bypasses queue; validates functionality)
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var svc = new MaintenanceService(
        VectorIndexConfig.builder(db, root)
            .dimension(8)
            .pqM(4)
            .pqK(16)
            .graphDegree(8)
            .maxSegmentSize(20)
            .build(),
        dirs);
    svc.vacuumSegment(ids[0][0], 0.0).get(5, TimeUnit.SECONDS);
  }
}
