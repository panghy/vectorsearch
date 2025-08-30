package io.github.panghy.vectorsearch.storage;

import static io.github.panghy.vectorsearch.storage.StorageTransactionUtils.readProto;

import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.Timestamp;
import io.github.panghy.vectorsearch.proto.EntryList;
import java.time.Instant;
import java.time.InstantSource;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.logging.Logger;

/**
 * Storage layer for managing search entry points in the vector index.
 * Entry points are the starting nodes for graph traversal during search operations.
 *
 * <p>This class supports multiple entry strategies:
 * <ul>
 *   <li>Primary entries: High-quality representatives (medoids or centrality-based)</li>
 *   <li>Random entries: Randomly sampled nodes for diversity</li>
 *   <li>High-degree entries: Well-connected nodes for good coverage</li>
 * </ul>
 *
 * <p>All operations use transaction parameters to allow callers to control
 * transaction boundaries and batch operations efficiently.
 */
public class EntryPointStorage {
  private static final Logger LOGGER = Logger.getLogger(EntryPointStorage.class.getName());

  private final DirectorySubspace subspace;
  private final VectorIndexKeys keys;
  private final InstantSource instantSource;

  /**
   * Creates a new EntryPointStorage instance.
   *
   * @param subspace the FDB directory subspace for this collection
   * @param collectionName the name of the collection
   * @param instantSource source for timestamps
   */
  public EntryPointStorage(DirectorySubspace subspace, String collectionName, InstantSource instantSource) {
    this.subspace = subspace;
    this.keys = new VectorIndexKeys(subspace, collectionName);
    this.instantSource = instantSource;
  }

  /**
   * Stores or updates the entry list for search initialization.
   *
   * @param tx the transaction to use
   * @param primaryEntries primary entry point node IDs (can be empty)
   * @param randomEntries random sample entry node IDs (can be empty)
   * @param highDegreeEntries high-degree entry node IDs (can be empty)
   * @return future that completes when the entry list is stored
   */
  public CompletableFuture<Void> storeEntryList(
      Transaction tx, List<Long> primaryEntries, List<Long> randomEntries, List<Long> highDegreeEntries) {

    byte[] key = keys.entryListKey();

    return readProto(tx, key, EntryList.parser()).thenApply(existing -> {
      EntryList.Builder builder = existing != null ? existing.toBuilder() : EntryList.newBuilder();

      // Update entries
      builder.clearPrimaryEntries();
      builder.clearRandomEntries();
      builder.clearHighDegreeEntries();

      if (primaryEntries != null) {
        builder.addAllPrimaryEntries(primaryEntries);
      }
      if (randomEntries != null) {
        builder.addAllRandomEntries(randomEntries);
      }
      if (highDegreeEntries != null) {
        builder.addAllHighDegreeEntries(highDegreeEntries);
      }

      // Update metadata
      builder.setUpdatedAt(currentTimestamp());
      builder.setVersion(existing != null ? existing.getVersion() + 1 : 1);

      // Calculate statistics if primary entries exist
      if (!primaryEntries.isEmpty()) {
        // These would be calculated by the caller based on graph analysis
        // For now, we just store what we're given
      }

      tx.set(key, builder.build().toByteArray());
      LOGGER.fine(String.format(
          "Stored entry list with %d primary, %d random, %d high-degree entries",
          primaryEntries != null ? primaryEntries.size() : 0,
          randomEntries != null ? randomEntries.size() : 0,
          highDegreeEntries != null ? highDegreeEntries.size() : 0));

      return null;
    });
  }

  /**
   * Loads the entry list for search initialization.
   *
   * @param tx the transaction to use
   * @return future containing the entry list, or null if not found
   */
  public CompletableFuture<EntryList> loadEntryList(Transaction tx) {
    byte[] key = keys.entryListKey();
    return readProto(tx, key, EntryList.parser());
  }

  /**
   * Gets all entry points from the entry list, combining all strategies.
   * Returns entries in priority order: primary, random, then high-degree.
   *
   * @param tx the transaction to use
   * @param maxEntries maximum number of entries to return (0 = all)
   * @return future containing list of entry node IDs
   */
  public CompletableFuture<List<Long>> getAllEntryPoints(Transaction tx, int maxEntries) {
    return loadEntryList(tx).thenApply(entryList -> {
      if (entryList == null) {
        return Collections.emptyList();
      }

      List<Long> allEntries = new ArrayList<>();

      // Add in priority order
      allEntries.addAll(entryList.getPrimaryEntriesList());
      allEntries.addAll(entryList.getRandomEntriesList());
      allEntries.addAll(entryList.getHighDegreeEntriesList());

      // Limit if requested
      if (maxEntries > 0 && allEntries.size() > maxEntries) {
        return allEntries.subList(0, maxEntries);
      }

      return allEntries;
    });
  }

  /**
   * Gets entry points using a hierarchical strategy.
   * Fills up to beamSize entries using primary entries first, then random, then high-degree.
   *
   * @param tx the transaction to use
   * @param beamSize target number of entries to return
   * @return future containing list of entry node IDs
   */
  public CompletableFuture<List<Long>> getHierarchicalEntryPoints(Transaction tx, int beamSize) {
    return loadEntryList(tx).thenApply(entryList -> {
      if (entryList == null) {
        return Collections.emptyList();
      }

      List<Long> entries = new ArrayList<>();

      // Add primary entries up to beam size
      List<Long> primary = entryList.getPrimaryEntriesList();
      int toAdd = Math.min(primary.size(), beamSize);
      entries.addAll(primary.subList(0, toAdd));

      // If we need more, add random entries
      if (entries.size() < beamSize) {
        List<Long> random = entryList.getRandomEntriesList();
        toAdd = Math.min(random.size(), beamSize - entries.size());
        entries.addAll(random.subList(0, toAdd));
      }

      // If we still need more, add high-degree entries
      if (entries.size() < beamSize) {
        List<Long> highDegree = entryList.getHighDegreeEntriesList();
        toAdd = Math.min(highDegree.size(), beamSize - entries.size());
        entries.addAll(highDegree.subList(0, toAdd));
      }

      return entries;
    });
  }

  /**
   * Clears the entry list.
   *
   * @param tx the transaction to use
   * @return future that completes when the entry list is cleared
   */
  public CompletableFuture<Void> clearEntryList(Transaction tx) {
    byte[] key = keys.entryListKey();
    tx.clear(key);
    return CompletableFuture.completedFuture(null);
  }

  private Timestamp currentTimestamp() {
    Instant now = instantSource.instant();
    return Timestamp.newBuilder()
        .setSeconds(now.getEpochSecond())
        .setNanos(now.getNano())
        .build();
  }
}
