package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.EntryList;
import java.time.Clock;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EntryPointStorageTest {
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
    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries))
        .join();

    // Load and verify
    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();

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

    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries))
        .join();

    // Request exactly primary size
    List<Long> entries =
        db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 2)).join();
    assertThat(entries).containsExactly(1L, 2L);

    // Request more than primary, should include random
    entries = db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 4)).join();
    assertThat(entries).containsExactly(1L, 2L, 10L, 20L);

    // Request more than primary + random, should include high-degree
    entries = db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 7)).join();
    assertThat(entries).containsExactly(1L, 2L, 10L, 20L, 30L, 100L, 200L);
  }

  @Test
  void testEmptyEntryList() {
    // Load non-existent entry list
    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();

    assertThat(loaded).isNull();

    // Get hierarchical entries from non-existent list
    List<Long> entries =
        db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 10)).join();

    assertThat(entries).isEmpty();
  }

  @Test
  void testClearEntryList() {
    // Store entry list
    List<Long> primaryEntries = Arrays.asList(1L, 2L, 3L);
    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, null, null))
        .join();

    // Verify it exists
    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();
    assertThat(loaded).isNotNull();

    // Clear it
    db.runAsync(tx -> storage.clearEntryList(tx)).join();

    // Verify it's gone
    loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();
    assertThat(loaded).isNull();
  }

  @Test
  void testUpdateEntryList() {
    // Initial store
    List<Long> initialPrimary = Arrays.asList(1L, 2L);
    db.runAsync(tx -> storage.storeEntryList(tx, initialPrimary, null, null))
        .join();

    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();
    assertThat(loaded.getVersion()).isEqualTo(1);

    // Update with new entries
    List<Long> updatedPrimary = Arrays.asList(3L, 4L, 5L);
    List<Long> updatedRandom = Arrays.asList(10L, 11L);

    db.runAsync(tx -> storage.storeEntryList(tx, updatedPrimary, updatedRandom, null))
        .join();

    // Verify update
    loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();

    assertThat(loaded.getPrimaryEntriesList()).containsExactlyElementsOf(updatedPrimary);
    assertThat(loaded.getRandomEntriesList()).containsExactlyElementsOf(updatedRandom);
    assertThat(loaded.getHighDegreeEntriesList()).isEmpty();
    assertThat(loaded.getVersion()).isEqualTo(2);
  }

  @Test
  void testGetAllEntryPoints() {
    List<Long> primaryEntries = Arrays.asList(1L, 2L);
    List<Long> randomEntries = Arrays.asList(10L, 20L);
    List<Long> highDegreeEntries = Arrays.asList(100L, 200L);

    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries))
        .join();

    // Get all entries (no limit)
    List<Long> allEntries =
        db.runAsync(tx -> storage.getAllEntryPoints(tx, 0)).join();

    assertThat(allEntries).containsExactly(1L, 2L, 10L, 20L, 100L, 200L);

    // Get limited entries
    List<Long> limitedEntries =
        db.runAsync(tx -> storage.getAllEntryPoints(tx, 3)).join();

    assertThat(limitedEntries).containsExactly(1L, 2L, 10L);
  }

  @Test
  void testEmptyLists() {
    // Store with empty lists
    db.runAsync(tx -> storage.storeEntryList(tx, Arrays.asList(), Arrays.asList(), Arrays.asList()))
        .join();

    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getPrimaryEntriesList()).isEmpty();
    assertThat(loaded.getRandomEntriesList()).isEmpty();
    assertThat(loaded.getHighDegreeEntriesList()).isEmpty();

    // Get entries from empty lists
    List<Long> entries =
        db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 10)).join();

    assertThat(entries).isEmpty();
  }

  @Test
  void testNullHandling() {
    // Store with some null categories
    List<Long> primaryEntries = Arrays.asList(1L, 2L);

    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, null, null))
        .join();

    EntryList loaded = db.runAsync(tx -> storage.loadEntryList(tx)).join();

    assertThat(loaded.getPrimaryEntriesList()).containsExactlyElementsOf(primaryEntries);
    assertThat(loaded.getRandomEntriesList()).isEmpty();
    assertThat(loaded.getHighDegreeEntriesList()).isEmpty();
  }

  @Test
  void testLargeEntryLists() {
    // Test with larger lists
    List<Long> primaryEntries = new ArrayList<>();
    List<Long> randomEntries = new ArrayList<>();
    List<Long> highDegreeEntries = new ArrayList<>();

    for (long i = 0; i < 100; i++) {
      primaryEntries.add(i);
      randomEntries.add(i + 1000);
      highDegreeEntries.add(i + 2000);
    }

    db.runAsync(tx -> storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries))
        .join();

    // Test hierarchical selection with various beam sizes
    List<Long> entries =
        db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 50)).join();
    assertThat(entries).hasSize(50);
    assertThat(entries).containsExactlyElementsOf(primaryEntries.subList(0, 50));

    entries = db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 150)).join();
    assertThat(entries).hasSize(150);
    // Should have all 100 primary + 50 random

    entries = db.runAsync(tx -> storage.getHierarchicalEntryPoints(tx, 250)).join();
    assertThat(entries).hasSize(250);
    // Should have all 100 primary + 100 random + 50 high-degree
  }
}
