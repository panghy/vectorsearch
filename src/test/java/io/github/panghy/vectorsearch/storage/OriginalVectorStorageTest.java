package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.pq.VectorUtils;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletionException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertThrows;

class OriginalVectorStorageTest {

  private Database db;
  private DirectorySubspace collection;
  private VectorIndexKeys keys;

  private static final int DIM = 16;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    String ns = "test/vectorsearch/original_vectors/" + UUID.randomUUID();
    db.run(tr -> {
      DirectoryLayer dl = DirectoryLayer.getDefault();
      collection = dl.createOrOpen(tr, List.of(ns.split("/"))).join();
      return null;
    });

    keys = new VectorIndexKeys(collection);
  }

  @AfterEach
  void tearDown() {
    if (db != null && collection != null) {
      db.run(tr -> {
        DirectoryLayer dl = DirectoryLayer.getDefault();
        dl.removeIfExists(tr, collection.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  void storeAndReadVector_roundTripsFp16() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    long nodeId = 42L;
    float[] vec = randomVector(DIM);

    db.run(tr -> {
      storage.storeVector(tr, nodeId, vec).join();
      return null;
    });

    float[] read = db.read(tr -> storage.readVector(tr, nodeId)).join();
    assertThat(read).isNotNull();
    assertThat(read.length).isEqualTo(DIM);

    // fp16 is lossy; verify element-wise within tolerance
    for (int i = 0; i < DIM; i++) {
      assertThat(read[i]).isCloseTo(vec[i], withinAbs(1e-2f));
    }
  }

  @Test
  void deleteVector_removesEntry() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    long nodeId = 7L;
    float[] vec = randomVector(DIM);

    db.run(tr -> {
      storage.storeVector(tr, nodeId, vec).join();
      return null;
    });

    // Delete
    db.run(tr -> {
      storage.deleteVector(tr, nodeId).join();
      return null;
    });

    float[] read = db.read(tr -> storage.readVector(tr, nodeId)).join();
    assertThat(read).isNull();
  }

  @Test
  void storeVector_throwsOnWrongDimension() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    long nodeId = 1L;
    float[] wrong = randomVector(DIM - 1);

    CompletionException ex = assertThrows(
        CompletionException.class,
        () -> db.run(tr -> {
          storage.storeVector(tr, nodeId, wrong).join();
          return null;
        }));
    assertThat(ex.getCause()).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void readVector_nonexistentReturnsNull() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    float[] read = db.read(tr -> storage.readVector(tr, 999L)).join();
    assertThat(read).isNull();
  }

  @Test
  void constructor_throwsOnInvalidDimension() {
    assertThrows(IllegalArgumentException.class, () -> new OriginalVectorStorage(keys, 0));
  }

  @Test
  void readVector_throwsOnMismatchedStoredDimension() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    long nodeId = 555L;

    // Manually write fp16 bytes for a vector of a different dimension
    float[] smaller = randomVector(DIM - 2);
    byte[] bytes = VectorUtils.toFloat16Bytes(smaller);

    db.run(tr -> {
      tr.set(keys.vectorKey(nodeId), bytes);
      return null;
    });

    CompletionException ex = assertThrows(
        CompletionException.class,
        () -> db.read(tr -> storage.readVector(tr, nodeId)).join());
    assertThat(ex.getCause()).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void overwriteVector_updatesValue() {
    OriginalVectorStorage storage = new OriginalVectorStorage(keys, DIM);
    long nodeId = 8080L;
    float[] v1 = randomVector(DIM);
    float[] v2 = randomVector(DIM);

    db.run(tr -> {
      storage.storeVector(tr, nodeId, v1).join();
      return null;
    });
    db.run(tr -> {
      storage.storeVector(tr, nodeId, v2).join();
      return null;
    });

    float[] read = db.read(tr -> storage.readVector(tr, nodeId)).join();
    for (int i = 0; i < DIM; i++) {
      assertThat(read[i]).isCloseTo(v2[i], withinAbs(1e-2f));
    }
  }

  private static float[] randomVector(int dim) {
    // Reuse VectorUtils random to keep consistent range
    return VectorUtils.randomVector(dim, new java.util.Random());
  }

  private static org.assertj.core.data.Offset<Float> withinAbs(float tol) {
    return org.assertj.core.data.Offset.offset(tol);
  }
}
