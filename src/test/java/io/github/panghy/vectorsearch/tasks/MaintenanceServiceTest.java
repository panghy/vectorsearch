package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThatCode;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Basic coverage for MaintenanceService threshold and compaction skeleton. */
public class MaintenanceServiceTest {
  private Database db;
  private DirectorySubspace root;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-mt", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  public void teardown() {
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  public void thresholdSkipAndCompactionNoop() throws Exception {
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var svc = new MaintenanceService(
        VectorIndexConfig.builder(db, root).dimension(4).build(), dirs);
    // No index initialized; methods should be resilient
    assertThatCode(() -> svc.vacuumSegment(0, 1.0).get(5, TimeUnit.SECONDS)).doesNotThrowAnyException();
    assertThatCode(() -> svc.compactSegments(java.util.List.of(0, 1)).get(5, TimeUnit.SECONDS))
        .doesNotThrowAnyException();
  }
}
