package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class NodeSegmentStorageTest {
  private Database db;
  private DirectorySubspace dir;
  private VectorIndexKeys keys;

  @BeforeEach
  void setUp() {
    db = FDB.selectAPIVersion(730).open();
    String ns = "test/vectorsearch/node_segment/" + UUID.randomUUID();
    db.run(tr -> {
      dir = DirectoryLayer.getDefault()
          .createOrOpen(tr, java.util.List.of(ns.split("/")))
          .join();
      return null;
    });
    keys = new VectorIndexKeys(dir);
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
  @DisplayName("store and get node->segment mapping")
  void testStoreGet() throws Exception {
    NodeSegmentStorage storage = new NodeSegmentStorage(keys);
    long nodeId = 123L;
    long segId = 9L;

    db.run(tr -> {
      storage.store(tr, nodeId, segId).join();
      return null;
    });

    Long got = db.read(tr -> storage.get(tr, nodeId)).get(5, TimeUnit.SECONDS);
    org.assertj.core.api.Assertions.assertThat(got).isEqualTo(segId);
  }
}
