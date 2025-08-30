package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.EntryList;
import java.time.Clock;
import java.time.InstantSource;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EntryPointStorageTestSimple {
  private Database db;
  private DirectorySubspace testSpace;
  private VectorIndexKeys keys;
  private EntryPointStorage storage;
  private InstantSource instantSource;
  private String testCollectionName;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(730);
    db = fdb.open();

    testCollectionName = "test_" + UUID.randomUUID().toString().substring(0, 8);
    db.run(tr -> {
      DirectoryLayer directoryLayer = DirectoryLayer.getDefault();
      testSpace = directoryLayer
          .createOrOpen(tr, Arrays.asList("test", "entry_points", testCollectionName))
          .join();
      return null;
    });

    keys = new VectorIndexKeys(testSpace, testCollectionName);
    instantSource = Clock.systemUTC();
    storage = new EntryPointStorage(testSpace, testCollectionName, instantSource);
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
  void testStoreAndLoadEntryList() {
    List<Long> primaryEntries = Arrays.asList(1L, 2L, 3L);
    List<Long> randomEntries = Arrays.asList(10L, 20L, 30L);
    List<Long> highDegreeEntries = Arrays.asList(100L, 200L, 300L);

    // Store entry list
    db.runAsync(tx -> {
          return storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries);
        })
        .join();

    // Load and verify
    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getPrimaryEntriesList()).containsExactlyElementsOf(primaryEntries);
    assertThat(loaded.getRandomEntriesList()).containsExactlyElementsOf(randomEntries);
    assertThat(loaded.getHighDegreeEntriesList()).containsExactlyElementsOf(highDegreeEntries);
    assertThat(loaded.getVersion()).isEqualTo(1);
    assertThat(loaded.hasUpdatedAt()).isTrue();
  }

  @Test
  void testGetHierarchicalEntryPoints() {
    List<Long> primaryEntries = Arrays.asList(1L, 2L);
    List<Long> randomEntries = Arrays.asList(10L, 20L, 30L);
    List<Long> highDegreeEntries = Arrays.asList(100L, 200L, 300L);

    db.runAsync(tx -> {
          return storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries);
        })
        .join();

    // Request exactly primary size
    List<Long> entries = db.runAsync(tx -> {
          return storage.getHierarchicalEntryPoints(tx, 2);
        })
        .join();
    assertThat(entries).containsExactly(1L, 2L);

    // Request more than primary, should include random
    entries = db.runAsync(tx -> {
          return storage.getHierarchicalEntryPoints(tx, 4);
        })
        .join();
    assertThat(entries).containsExactly(1L, 2L, 10L, 20L);

    // Request more than primary + random, should include high-degree
    entries = db.runAsync(tx -> {
          return storage.getHierarchicalEntryPoints(tx, 7);
        })
        .join();
    assertThat(entries).containsExactly(1L, 2L, 10L, 20L, 30L, 100L, 200L);
  }

  @Test
  void testEmptyEntryList() {
    // Load non-existent entry list
    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

    assertThat(loaded).isNull();

    // Get hierarchical entries from non-existent list
    List<Long> entries = db.runAsync(tx -> {
          return storage.getHierarchicalEntryPoints(tx, 10);
        })
        .join();

    assertThat(entries).isEmpty();
  }

  @Test
  void testClearEntryList() {
    // Store entry list
    List<Long> primaryEntries = Arrays.asList(1L, 2L, 3L);
    db.runAsync(tx -> {
          return storage.storeEntryList(tx, primaryEntries, null, null);
        })
        .join();

    // Verify it exists
    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();
    assertThat(loaded).isNotNull();

    // Clear it
    db.runAsync(tx -> {
          return storage.clearEntryList(tx);
        })
        .join();

    // Verify it's gone
    loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();
    assertThat(loaded).isNull();
  }
}
