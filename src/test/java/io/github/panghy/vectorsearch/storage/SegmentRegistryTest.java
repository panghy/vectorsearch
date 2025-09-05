package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SegmentRegistryTest {

  private Database db;
  private DirectorySubspace dir;
  private VectorIndexKeys keys;
  private SegmentRegistry registry;

  @BeforeEach
  void setUp() {
    db = FDB.selectAPIVersion(730).open();
    String ns = "test/vectorsearch/segments/" + UUID.randomUUID();
    db.run(tr -> {
      dir = DirectoryLayer.getDefault()
          .createOrOpen(tr, java.util.List.of(ns.split("/")))
          .join();
      return null;
    });
    keys = new VectorIndexKeys(dir);
    registry = new SegmentRegistry(keys);
  }

  @AfterEach
  void tearDown() {
    if (db != null && dir != null) {
      db.run(tr -> {
        DirectoryLayer.getDefault().removeIfExists(tr, dir.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  @DisplayName("create, seal, set active, and read segment meta")
  void testSegmentLifecycle() throws Exception {
    db.run(tr -> {
      registry.createSegment(tr, 1L, 7).join();
      return null;
    });
    db.run(tr -> {
      registry.setActiveSegment(tr, 1L).join();
      return null;
    });

    Long active = db.read(tr -> registry.getActiveSegment(tr)).get(5, TimeUnit.SECONDS);
    org.assertj.core.api.Assertions.assertThat(active).isEqualTo(1L);

    SegmentMeta meta = db.read(tr -> registry.readSegmentMeta(tr, 1L)).get(5, TimeUnit.SECONDS);
    org.assertj.core.api.Assertions.assertThat(meta).isNotNull();
    org.assertj.core.api.Assertions.assertThat(meta.getSegmentId()).isEqualTo(1L);
    org.assertj.core.api.Assertions.assertThat(meta.getCodebookVersion()).isEqualTo(7);
    org.assertj.core.api.Assertions.assertThat(meta.getStatus()).isEqualTo(SegmentMeta.Status.ACTIVE);

    db.run(tr -> {
      registry.sealSegment(tr, 1L).join();
      return null;
    });
    SegmentMeta sealed = db.read(tr -> registry.readSegmentMeta(tr, 1L)).get(5, TimeUnit.SECONDS);
    org.assertj.core.api.Assertions.assertThat(sealed.getStatus()).isEqualTo(SegmentMeta.Status.SEALED);
  }
}
