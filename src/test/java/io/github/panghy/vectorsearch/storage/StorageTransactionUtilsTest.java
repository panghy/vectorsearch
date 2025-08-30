package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.tuple.Tuple;
import com.google.protobuf.InvalidProtocolBufferException;
import io.github.panghy.vectorsearch.proto.CodebookSub;
import io.github.panghy.vectorsearch.proto.Config;
import io.github.panghy.vectorsearch.proto.PqCodesBlock;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class StorageTransactionUtilsTest {

  private Database db;
  private DirectorySubspace testSpace;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    String testName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "storage_utils", testName))
          .join();
      return null;
    });
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
  void testReadWriteProto() throws ExecutionException, InterruptedException {
    Config config = Config.newBuilder()
        .setDimension(128)
        .setMetric("L2")
        .setPqSubvectors(16)
        .setPqNbits(8)
        .setGraphDegree(32)
        .build();

    byte[] key = testSpace.pack(Tuple.from("config"));

    // Write proto
    db.run(tr -> {
      StorageTransactionUtils.writeProto(tr, key, config);
      return null;
    });

    // Read proto
    Config readConfig = db.runAsync(tr -> StorageTransactionUtils.readProto(tr, key, Config.parser()))
        .get();

    assertThat(readConfig).isNotNull();
    assertThat(readConfig.getDimension()).isEqualTo(128);
    assertThat(readConfig.getMetric()).isEqualTo("L2");
    assertThat(readConfig.getPqSubvectors()).isEqualTo(16);
  }

  @Test
  void testReadProtoNotFound() throws ExecutionException, InterruptedException {
    byte[] key = testSpace.pack(Tuple.from("nonexistent"));

    Config result = db.runAsync(tr -> StorageTransactionUtils.readProto(tr, key, Config.parser()))
        .get();

    assertThat(result).isNull();
  }

  @Test
  void testReadProtoInvalidData() {
    byte[] key = testSpace.pack(Tuple.from("invalid"));

    // Write invalid data
    db.run(tr -> {
      tr.set(key, "invalid protobuf data".getBytes());
      return null;
    });

    // Attempt to read as proto should throw
    assertThatThrownBy(() -> {
          db.runAsync(tr -> StorageTransactionUtils.readProto(tr, key, Config.parser()))
              .get();
        })
        .hasRootCauseInstanceOf(InvalidProtocolBufferException.class);
  }

  @Test
  void testAtomicUpdate() throws ExecutionException, InterruptedException {
    byte[] key = testSpace.pack(Tuple.from("config"));

    // Initial write
    Config initial = Config.newBuilder().setDimension(128).setMetric("L2").build();

    db.run(tr -> {
      StorageTransactionUtils.writeProto(tr, key, initial);
      return null;
    });

    // Atomic update
    Function<Config, Config> modifier = config -> {
      if (config == null) {
        return Config.newBuilder().setDimension(256).build();
      }
      return config.toBuilder().setDimension(256).build();
    };

    Config updated = db.runAsync(tr -> StorageTransactionUtils.atomicUpdate(tr, key, Config.parser(), modifier))
        .get();

    assertThat(updated.getDimension()).isEqualTo(256);
    assertThat(updated.getMetric()).isEqualTo("L2");

    // Verify persisted
    Config persisted = db.runAsync(tr -> StorageTransactionUtils.readProto(tr, key, Config.parser()))
        .get();
    assertThat(persisted.getDimension()).isEqualTo(256);
  }

  @Test
  void testAtomicUpdateNull() throws ExecutionException, InterruptedException {
    byte[] key = testSpace.pack(Tuple.from("new_config"));

    Function<Config, Config> modifier = config -> {
      if (config == null) {
        return Config.newBuilder().setDimension(512).setMetric("COSINE").build();
      }
      return config;
    };

    Config created = db.runAsync(tr -> StorageTransactionUtils.atomicUpdate(tr, key, Config.parser(), modifier))
        .get();

    assertThat(created.getDimension()).isEqualTo(512);
    assertThat(created.getMetric()).isEqualTo("COSINE");
  }

  @Test
  void testReadRange() throws ExecutionException, InterruptedException {
    // Write multiple key-value pairs
    db.run(tr -> {
      for (int i = 0; i < 10; i++) {
        byte[] key = testSpace.pack(Tuple.from("item", i));
        tr.set(key, String.valueOf(i).getBytes());
      }
      return null;
    });

    // Read range
    byte[] prefix = testSpace.pack(Tuple.from("item"));
    Range range = Range.startsWith(prefix);

    List<KeyValue> results =
        db.runAsync(tr -> StorageTransactionUtils.readRange(tr, range)).get();

    assertThat(results).hasSize(10);
    for (int i = 0; i < 10; i++) {
      assertThat(new String(results.get(i).getValue())).isEqualTo(String.valueOf(i));
    }
  }

  @Test
  void testReadRangeWithLimit() throws ExecutionException, InterruptedException {
    // Write multiple key-value pairs
    db.run(tr -> {
      for (int i = 0; i < 20; i++) {
        byte[] key = testSpace.pack(Tuple.from("limited", i));
        tr.set(key, String.valueOf(i).getBytes());
      }
      return null;
    });

    byte[] prefix = testSpace.pack(Tuple.from("limited"));
    Range range = Range.startsWith(prefix);

    List<KeyValue> results = db.runAsync(tr -> StorageTransactionUtils.readRange(tr, range, 5))
        .get();

    assertThat(results).hasSize(5);
  }

  @Test
  void testReadRangeInvalidLimit() {
    Range range = Range.startsWith(new byte[] {0x01});

    assertThatThrownBy(() -> db.runAsync(tr -> StorageTransactionUtils.readRange(tr, range, 0))
            .get())
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasRootCauseMessage("Limit must be between 1 and 1000");

    assertThatThrownBy(() -> db.runAsync(tr -> StorageTransactionUtils.readRange(tr, range, 1001))
            .get())
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasRootCauseMessage("Limit must be between 1 and 1000");
  }

  @Test
  void testReadProtoRange() throws ExecutionException, InterruptedException {
    // Write multiple proto messages
    db.run(tr -> {
      for (int i = 0; i < 5; i++) {
        byte[] key = testSpace.pack(Tuple.from("codebook", i));
        CodebookSub codebook = CodebookSub.newBuilder()
            .setSubspaceIndex(i)
            .setVersion(1)
            .setTrainedOnVectors(1000L + i)
            .build();
        StorageTransactionUtils.writeProto(tr, key, codebook);
      }
      return null;
    });

    byte[] prefix = testSpace.pack(Tuple.from("codebook"));
    Range range = Range.startsWith(prefix);

    List<CodebookSub> codebooks = db.runAsync(
            tr -> StorageTransactionUtils.readProtoRange(tr, range, CodebookSub.parser()))
        .get();

    assertThat(codebooks).hasSize(5);
    for (int i = 0; i < 5; i++) {
      assertThat(codebooks.get(i).getSubspaceIndex()).isEqualTo(i);
      assertThat(codebooks.get(i).getTrainedOnVectors()).isEqualTo(1000L + i);
    }
  }

  @Test
  void testBatchWrite() {
    List<KeyValue> kvPairs = Arrays.asList(
        new KeyValue(testSpace.pack(Tuple.from("batch", 1)), "value1".getBytes()),
        new KeyValue(testSpace.pack(Tuple.from("batch", 2)), "value2".getBytes()),
        new KeyValue(testSpace.pack(Tuple.from("batch", 3)), "value3".getBytes()));

    db.run(tr -> {
      StorageTransactionUtils.batchWrite(tr, kvPairs);
      return null;
    });

    // Verify all were written
    db.run(tr -> {
      for (int i = 1; i <= 3; i++) {
        byte[] value = tr.get(testSpace.pack(Tuple.from("batch", i))).join();
        assertThat(new String(value)).isEqualTo("value" + i);
      }
      return null;
    });
  }

  @Test
  void testClearRange() throws ExecutionException, InterruptedException {
    // Write multiple keys
    db.run(tr -> {
      for (int i = 0; i < 10; i++) {
        byte[] key = testSpace.pack(Tuple.from("clear", i));
        tr.set(key, String.valueOf(i).getBytes());
      }
      return null;
    });

    // Clear range
    byte[] prefix = testSpace.pack(Tuple.from("clear"));
    Range range = Range.startsWith(prefix);

    db.run(tr -> {
      StorageTransactionUtils.clearRange(tr, range);
      return null;
    });

    // Verify cleared
    List<KeyValue> remaining =
        db.runAsync(tr -> StorageTransactionUtils.readRange(tr, range)).get();

    assertThat(remaining).isEmpty();
  }

  @Test
  void testExists() throws ExecutionException, InterruptedException {
    byte[] existingKey = testSpace.pack(Tuple.from("exists"));
    byte[] missingKey = testSpace.pack(Tuple.from("missing"));

    db.run(tr -> {
      tr.set(existingKey, "value".getBytes());
      return null;
    });

    boolean exists = db.runAsync(tr -> StorageTransactionUtils.exists(tr, existingKey))
        .get();
    boolean missing = db.runAsync(tr -> StorageTransactionUtils.exists(tr, missingKey))
        .get();

    assertThat(exists).isTrue();
    assertThat(missing).isFalse();
  }

  @Test
  void testCountRange() throws ExecutionException, InterruptedException {
    // Write keys for counting
    int numKeys = 15;
    db.run(tr -> {
      for (int i = 0; i < numKeys; i++) {
        byte[] key = testSpace.pack(Tuple.from("count", i));
        tr.set(key, String.valueOf(i).getBytes());
      }
      return null;
    });

    byte[] prefix = testSpace.pack(Tuple.from("count"));
    Range range = Range.startsWith(prefix);

    Long count =
        db.runAsync(tr -> StorageTransactionUtils.countRange(tr, range)).get();

    assertThat(count).isEqualTo(numKeys);
  }

  @Test
  void testCountRangeEmpty() throws ExecutionException, InterruptedException {
    byte[] prefix = testSpace.pack(Tuple.from("empty"));
    Range range = Range.startsWith(prefix);

    Long count =
        db.runAsync(tr -> StorageTransactionUtils.countRange(tr, range)).get();

    assertThat(count).isEqualTo(0L);
  }

  @Test
  void testSetReadVersion() {
    db.run(tr -> {
      // Set a specific read version before any reads
      long version = 1000000L; // Use a fixed version
      StorageTransactionUtils.setReadVersion(tr, version);

      // This should not throw
      return null;
    });
  }

  @Test
  void testAtomicIncrement() throws ExecutionException, InterruptedException {
    byte[] counterKey = testSpace.pack(Tuple.from("counter"));

    // Initialize counter with little-endian 64-bit integer
    db.run(tr -> {
      byte[] initialValue = new byte[8]; // 0L in little-endian
      tr.set(counterKey, initialValue);
      return null;
    });

    // Increment atomically
    db.run(tr -> {
      StorageTransactionUtils.atomicIncrement(tr, counterKey, 5L);
      return null;
    });

    // Read counter value (little-endian 64-bit integer)
    long value = db.run(tr -> {
      byte[] bytes = tr.get(counterKey).join();
      long result = 0;
      for (int i = 0; i < 8; i++) {
        result |= (bytes[i] & 0xFFL) << (i * 8);
      }
      return result;
    });

    assertThat(value).isEqualTo(5L);

    // Increment again
    db.run(tr -> {
      StorageTransactionUtils.atomicIncrement(tr, counterKey, 3L);
      return null;
    });

    value = db.run(tr -> {
      byte[] bytes = tr.get(counterKey).join();
      long result = 0;
      for (int i = 0; i < 8; i++) {
        result |= (bytes[i] & 0xFFL) << (i * 8);
      }
      return result;
    });

    assertThat(value).isEqualTo(8L);
  }

  @Test
  void testReadProtoRangeWithInvalidData() {
    byte[] prefix = testSpace.pack(Tuple.from("invalid_proto"));

    // Write some invalid proto data
    db.run(tr -> {
      tr.set(testSpace.pack(Tuple.from("invalid_proto", 1)), "not a proto".getBytes());
      return null;
    });

    Range range = Range.startsWith(prefix);

    assertThatThrownBy(() -> {
          db.runAsync(tr -> StorageTransactionUtils.readProtoRange(tr, range, PqCodesBlock.parser()))
              .get();
        })
        .hasRootCauseInstanceOf(InvalidProtocolBufferException.class);
  }

  @Test
  void testAtomicUpdateReturningNull() throws ExecutionException, InterruptedException {
    byte[] key = testSpace.pack(Tuple.from("null_update"));

    // Write initial value
    Config initial = Config.newBuilder().setDimension(128).build();
    db.run(tr -> {
      StorageTransactionUtils.writeProto(tr, key, initial);
      return null;
    });

    // Update that returns null (deletes the key)
    Function<Config, Config> modifier = config -> null;

    Config result = db.runAsync(tr -> StorageTransactionUtils.atomicUpdate(tr, key, Config.parser(), modifier))
        .get();

    assertThat(result).isNull();

    // Verify key still exists (atomicUpdate doesn't delete on null)
    Config existing = db.runAsync(tr -> StorageTransactionUtils.readProto(tr, key, Config.parser()))
        .get();
    assertThat(existing).isNotNull();
    assertThat(existing.getDimension()).isEqualTo(128);
  }

  @Test
  void testLargeRangeCount() throws ExecutionException, InterruptedException {
    // Write many keys to trigger estimated count
    int numKeys = 500;
    db.run(tr -> {
      for (int i = 0; i < numKeys; i++) {
        byte[] key = testSpace.pack(Tuple.from("large", i));
        // Write larger values to exceed 100KB threshold
        byte[] value = new byte[300];
        Arrays.fill(value, (byte) i);
        tr.set(key, value);
      }
      return null;
    });

    byte[] prefix = testSpace.pack(Tuple.from("large"));
    Range range = Range.startsWith(prefix);

    Long count =
        db.runAsync(tr -> StorageTransactionUtils.countRange(tr, range)).get();

    // For large ranges, it uses estimation, so allow some variance
    // The actual size might not trigger estimation, so accept exact or estimated count
    assertThat(count).isBetween((long) (numKeys * 0.5), (long) (numKeys * 1.5));
  }
}
