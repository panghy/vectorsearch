package io.github.panghy.vectorsearch.storage;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.proto.EntryList;
import java.time.Clock;
import java.time.InstantSource;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class EntryPointStorageTest {
  private Database db;
  private DirectorySubspace subspace;
  private EntryPointStorage storage;
  private InstantSource instantSource;

  @BeforeEach
  void setUp() {
    FDB fdb = FDB.selectAPIVersion(710);
    db = fdb.open();

    // Create a unique test directory
    String testDir = "test_entry_points_" + System.currentTimeMillis();
    subspace = db.run(tr -> {
      return DirectorySubspace.create(tr, Collections.singletonList(testDir))
          .join();
    });

    instantSource = Clock.systemUTC();
    storage = new EntryPointStorage(subspace, "test_collection", instantSource);
  }

  @AfterEach
  void tearDown() {
    // Clean up test directory
    if (db != null && subspace != null) {
      db.run(tr -> {
        tr.clear(subspace.range());
        return null;
      });
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
  void testUpdateEntryList() {
    // Initial store
    List<Long> initialPrimary = Arrays.asList(1L, 2L);
    db.runAsync(tx -> {
          return storage.storeEntryList(tx, initialPrimary, null, null);
        })
        .join();

    // Update with new entries
    List<Long> updatedPrimary = Arrays.asList(3L, 4L, 5L);
    List<Long> updatedRandom = Arrays.asList(10L, 11L);

    db.runAsync(tx -> {
          return storage.storeEntryList(tx, updatedPrimary, updatedRandom, null);
        })
        .join();

    // Verify update
    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

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

    db.runAsync(tx -> {
          return storage.storeEntryList(tx, primaryEntries, randomEntries, highDegreeEntries);
        })
        .join();

    // Get all entries (no limit)
    List<Long> allEntries = db.runAsync(tx -> {
          return storage.getAllEntryPoints(tx, 0);
        })
        .join();

    assertThat(allEntries).containsExactly(1L, 2L, 10L, 20L, 100L, 200L);

    // Get limited entries
    List<Long> limitedEntries = db.runAsync(tx -> {
          return storage.getAllEntryPoints(tx, 3);
        })
        .join();

    assertThat(limitedEntries).containsExactly(1L, 2L, 10L);
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

    // Request more than available
    entries = db.runAsync(tx -> {
          return storage.getHierarchicalEntryPoints(tx, 20);
        })
        .join();
    assertThat(entries).hasSize(8); // All available entries
  }

  @Test
  void testEmptyEntryList() {
    // Load non-existent entry list
    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

    assertThat(loaded).isNull();

    // Get entries from non-existent list
    List<Long> entries = db.runAsync(tx -> {
          return storage.getAllEntryPoints(tx, 0);
        })
        .join();

    assertThat(entries).isEmpty();

    // Get hierarchical entries from non-existent list
    entries = db.runAsync(tx -> {
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

  @Test
  void testNullEntryHandling() {
    // Store with some null categories
    List<Long> primaryEntries = Arrays.asList(1L, 2L);

    db.runAsync(tx -> {
          return storage.storeEntryList(tx, primaryEntries, null, null);
        })
        .join();

    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

    assertThat(loaded.getPrimaryEntriesList()).containsExactlyElementsOf(primaryEntries);
    assertThat(loaded.getRandomEntriesList()).isEmpty();
    assertThat(loaded.getHighDegreeEntriesList()).isEmpty();
  }

  @Test
  void testEmptyListHandling() {
    // Store with empty lists
    db.runAsync(tx -> {
          return storage.storeEntryList(
              tx, Collections.emptyList(), Collections.emptyList(), Collections.emptyList());
        })
        .join();

    EntryList loaded = db.runAsync(tx -> {
          return storage.loadEntryList(tx);
        })
        .join();

    assertThat(loaded).isNotNull();
    assertThat(loaded.getPrimaryEntriesList()).isEmpty();
    assertThat(loaded.getRandomEntriesList()).isEmpty();
    assertThat(loaded.getHighDegreeEntriesList()).isEmpty();
  }
}
