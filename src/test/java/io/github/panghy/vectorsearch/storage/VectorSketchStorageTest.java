package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.VectorSketch;
import java.util.Arrays;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorSketchStorageTest {

  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private VectorSketchStorage storage;
  private String testCollectionName;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "vector_sketch", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace);
    storage = new VectorSketchStorage(keys);
  }

  @AfterEach
  void tearDown() {
    if (db != null && testSpace != null) {
      db.run(tr -> {
        DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
        directoryLayer.removeIfExists(tr, testSpace.getPath()).join();
        return null;
      });
      db.close();
    }
  }

  @Test
  void testStoreVectorSketch() {
    float[] vector = {1.0f, 2.0f, 3.0f, 4.0f};
    long nodeId = 123L;

    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, vector)).join();

    // Verify sketch was stored
    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] storedBytes = db.runAsync(tr -> tr.get(key)).join();

    assertThat(storedBytes).isNotNull();

    try {
      VectorSketch stored = VectorSketch.parseFrom(storedBytes);
      assertThat(stored.getNodeId()).isEqualTo(nodeId);
      assertThat(stored.getSketchType()).isEqualTo(VectorSketch.SketchType.SIMHASH_256);
      assertThat(stored.getSketchData()).isNotEmpty();
      assertThat(stored.getSketchData().size()).isEqualTo(32); // 256 bits = 32 bytes
    } catch (Exception e) {
      throw new RuntimeException("Failed to parse stored sketch", e);
    }
  }

  @Test
  void testGenerateSimHashConsistency() {
    float[] vector = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    long nodeId = 456L;

    // Store the same vector twice
    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, vector)).join();

    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] sketch1 = db.runAsync(tr -> tr.get(key)).join();

    // Store again (should be consistent)
    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, vector)).join();
    byte[] sketch2 = db.runAsync(tr -> tr.get(key)).join();

    assertThat(sketch1).isEqualTo(sketch2);
  }

  @Test
  void testDifferentVectorsDifferentSketches() {
    float[] vector1 = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] vector2 = {-1.0f, -2.0f, -3.0f, -4.0f};
    long nodeId1 = 111L;
    long nodeId2 = 222L;

    // Store different vectors
    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId1, vector1)).join();
    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId2, vector2)).join();

    // Get sketches
    byte[] sketch1 =
        db.runAsync(tr -> tr.get(keys.vectorSketchKey(nodeId1))).join();
    byte[] sketch2 =
        db.runAsync(tr -> tr.get(keys.vectorSketchKey(nodeId2))).join();

    // Parse and verify they're different
    try {
      VectorSketch proto1 = VectorSketch.parseFrom(sketch1);
      VectorSketch proto2 = VectorSketch.parseFrom(sketch2);

      assertThat(proto1.getSketchData()).isNotEqualTo(proto2.getSketchData());
      assertThat(proto1.getNodeId()).isEqualTo(nodeId1);
      assertThat(proto2.getNodeId()).isEqualTo(nodeId2);
    } catch (Exception e) {
      throw new RuntimeException("Failed to parse sketches", e);
    }
  }

  @Test
  void testLargeVectorSketch() {
    // Test with larger vector to ensure SimHash handles it properly
    float[] largeVector = new float[768]; // BERT-like dimensions
    for (int i = 0; i < largeVector.length; i++) {
      largeVector[i] = (float) Math.sin(i);
    }
    long nodeId = 999L;

    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, largeVector)).join();

    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] storedBytes = db.runAsync(tr -> tr.get(key)).join();

    assertThat(storedBytes).isNotNull();

    try {
      VectorSketch stored = VectorSketch.parseFrom(storedBytes);
      assertThat(stored.getNodeId()).isEqualTo(nodeId);
      assertThat(stored.getSketchData().size()).isEqualTo(32); // Still 256 bits
    } catch (Exception e) {
      throw new RuntimeException("Failed to parse stored sketch", e);
    }
  }

  @Test
  void testZeroVector() {
    // Test with zero vector
    float[] zeroVector = new float[10];
    // All values are 0.0f by default
    long nodeId = 888L;

    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, zeroVector)).join();

    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] storedBytes = db.runAsync(tr -> tr.get(key)).join();

    assertThat(storedBytes).isNotNull();

    try {
      VectorSketch stored = VectorSketch.parseFrom(storedBytes);
      assertThat(stored.getNodeId()).isEqualTo(nodeId);
      assertThat(stored.getSketchData()).isNotNull();
    } catch (Exception e) {
      throw new RuntimeException("Failed to parse stored sketch", e);
    }
  }

  @Test
  void testVectorWithNegativeValues() {
    // Test handling of negative values
    float[] vector = {-1.5f, -2.3f, -0.1f, -99.9f};
    long nodeId = 777L;

    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, vector)).join();

    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] storedBytes = db.runAsync(tr -> tr.get(key)).join();

    assertThat(storedBytes).isNotNull();

    try {
      VectorSketch stored = VectorSketch.parseFrom(storedBytes);
      assertThat(stored.getSketchType()).isEqualTo(VectorSketch.SketchType.SIMHASH_256);
    } catch (Exception e) {
      throw new RuntimeException("Failed to parse stored sketch", e);
    }
  }

  @Test
  void testEmptyVector() {
    // Test with empty vector (edge case)
    float[] emptyVector = new float[0];
    long nodeId = 666L;

    db.runAsync(tr -> storage.storeVectorSketch(tr, nodeId, emptyVector)).join();

    byte[] key = keys.vectorSketchKey(nodeId);
    byte[] storedBytes = db.runAsync(tr -> tr.get(key)).join();

    assertThat(storedBytes).isNotNull();
    // Should still generate a sketch even for empty vector
  }
}
