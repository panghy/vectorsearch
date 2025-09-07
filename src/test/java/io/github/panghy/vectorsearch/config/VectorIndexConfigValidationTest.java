package io.github.panghy.vectorsearch.config;

/**
 * Validation tests for VectorIndexConfig to ensure invalid parameters are rejected.
 */
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.*;

class VectorIndexConfigValidationTest {
  static Database db;
  static com.apple.foundationdb.directory.DirectorySubspace root;

  @BeforeAll
  static void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vic-validate", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterAll
  static void teardown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void builder_validations_throw() {
    // pqM <= 0
    assertThatThrownBy(() ->
            VectorIndexConfig.builder(db, root).dimension(4).pqM(0).build())
        .isInstanceOf(IllegalArgumentException.class);
    // pqK <= 1
    assertThatThrownBy(() ->
            VectorIndexConfig.builder(db, root).dimension(4).pqK(1).build())
        .isInstanceOf(IllegalArgumentException.class);
    // graphDegree <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .graphDegree(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // oversample <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .oversample(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // estimatedWorkerCount <= 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .estimatedWorkerCount(0)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // localWorkerThreads < 0
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .localWorkerThreads(-1)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // defaultTtl non-positive
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .defaultTtl(Duration.ZERO)
            .build())
        .isInstanceOf(IllegalArgumentException.class);
    // defaultThrottle negative
    assertThatThrownBy(() -> VectorIndexConfig.builder(db, root)
            .dimension(4)
            .defaultThrottle(Duration.ofSeconds(-1))
            .build())
        .isInstanceOf(IllegalArgumentException.class);
  }
}
