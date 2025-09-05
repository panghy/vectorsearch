package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import java.time.Duration;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SegmentPqBlockStorageTest {
  private Database db;
  private DirectorySubspace dir;
  private VectorIndexKeys keys;

  private static final int CODES_PER_BLOCK = 64;
  private static final int PQ_SUBVECTORS = 8;

  @BeforeEach
  void setUp() {
    db = FDB.selectAPIVersion(730).open();
    String ns = "test/vectorsearch/segment_pq/" + UUID.randomUUID();
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
  @DisplayName("store and load single PQ code in segment")
  void testStoreLoadSingle() throws Exception {
    SegmentPqBlockStorage storage = new SegmentPqBlockStorage(
        db, keys, CODES_PER_BLOCK, PQ_SUBVECTORS, InstantSource.system(), 100, Duration.ofMinutes(5));
    long seg = 11L;
    long nodeId = 123L;
    byte[] code = new byte[PQ_SUBVECTORS];
    for (int i = 0; i < code.length; i++) code[i] = (byte) i;

    db.run(tr -> {
      storage.storePqCode(tr, seg, nodeId, code).join();
      return null;
    });

    byte[] loaded = storage.loadPqCode(seg, nodeId).get(5, TimeUnit.SECONDS);
    assertThat(loaded).isEqualTo(code);
  }

  @Test
  @DisplayName("batch store and load PQ codes across blocks in segment")
  void testBatchStoreLoad() throws Exception {
    SegmentPqBlockStorage storage = new SegmentPqBlockStorage(
        db, keys, CODES_PER_BLOCK, PQ_SUBVECTORS, InstantSource.system(), 100, Duration.ofMinutes(5));
    long seg = 7L;

    List<Long> nodeIds = new ArrayList<>();
    List<byte[]> codes = new ArrayList<>();
    for (int i = 0; i < 10; i++) {
      nodeIds.add((long) i);
      byte[] c = new byte[PQ_SUBVECTORS];
      for (int j = 0; j < c.length; j++) c[j] = (byte) (i + j);
      codes.add(c);
    }

    db.run(tr -> {
      storage.batchStorePqCodesInTransaction(tr, seg, nodeIds, codes).join();
      return null;
    });

    List<byte[]> loaded = storage.batchLoadPqCodes(seg, nodeIds).get(5, TimeUnit.SECONDS);
    for (int i = 0; i < nodeIds.size(); i++) {
      assertThat(loaded.get(i)).isEqualTo(codes.get(i));
    }

    storage.deleteSegment(seg).get(5, TimeUnit.SECONDS);
    List<byte[]> afterDelete = storage.batchLoadPqCodes(seg, nodeIds).get(5, TimeUnit.SECONDS);
    for (byte[] arr : afterDelete) {
      assertThat(arr).isNull();
    }
  }
}
