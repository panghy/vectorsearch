package io.github.panghy.vectorsearch.storage;

import com.apple.foundationdb.KeyValue;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.Transaction;
import com.apple.foundationdb.async.AsyncIterable;
import com.apple.foundationdb.async.AsyncIterator;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.Message;
import com.google.protobuf.Parser;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;

/**
 * Utility methods for common FoundationDB transaction patterns.
 * Provides helpers for protobuf serialization, range reads, and atomic operations.
 */
public class StorageTransactionUtils {

  private static final int DEFAULT_BATCH_SIZE = 100;
  private static final int MAX_BATCH_SIZE = 1000;

  /**
   * Reads a protobuf message from FDB.
   *
   * @param tx the transaction
   * @param key the key to read
   * @param parser protobuf parser for deserialization
   * @param <T> the protobuf message type
   * @return the message or null if not found
   */
  public static <T extends Message> CompletableFuture<T> readProto(Transaction tx, byte[] key, Parser<T> parser) {
    return tx.get(key).thenApply(value -> {
      if (value == null) {
        return null;
      }
      try {
        return parser.parseFrom(value);
      } catch (InvalidProtocolBufferException e) {
        throw new CompletionException("Failed to parse protobuf", e);
      }
    });
  }

  /**
   * Writes a protobuf message to FDB.
   *
   * @param tx the transaction
   * @param key the key to write
   * @param message the protobuf message
   */
  public static void writeProto(Transaction tx, byte[] key, Message message) {
    tx.set(key, message.toByteArray());
  }

  /**
   * Performs an atomic read-modify-write operation.
   *
   * @param tx the transaction
   * @param key the key
   * @param parser protobuf parser
   * @param modifier function to modify the message (null input means create new)
   * @param <T> the protobuf message type
   * @return the modified message
   */
  public static <T extends Message> CompletableFuture<T> atomicUpdate(
      Transaction tx, byte[] key, Parser<T> parser, Function<T, T> modifier) {
    return readProto(tx, key, parser).thenApply(modifier).thenApply(modified -> {
      if (modified != null) {
        writeProto(tx, key, modified);
      }
      return modified;
    });
  }

  /**
   * Reads all key-value pairs in a range.
   *
   * @param tx the transaction
   * @param range the key range
   * @return list of key-value pairs
   */
  public static CompletableFuture<List<KeyValue>> readRange(Transaction tx, Range range) {
    return readRange(tx, range, DEFAULT_BATCH_SIZE);
  }

  /**
   * Reads all key-value pairs in a range with limit.
   *
   * @param tx the transaction
   * @param range the key range
   * @param limit maximum number of results
   * @return list of key-value pairs
   */
  public static CompletableFuture<List<KeyValue>> readRange(Transaction tx, Range range, int limit) {
    if (limit <= 0 || limit > MAX_BATCH_SIZE) {
      throw new IllegalArgumentException("Limit must be between 1 and " + MAX_BATCH_SIZE);
    }

    AsyncIterable<KeyValue> iterable = tx.getRange(range, limit);
    return collectAsync(iterable);
  }

  /**
   * Reads and deserializes protobuf messages from a range.
   *
   * @param tx the transaction
   * @param range the key range
   * @param parser protobuf parser
   * @param <T> the protobuf message type
   * @return list of messages
   */
  public static <T extends Message> CompletableFuture<List<T>> readProtoRange(
      Transaction tx, Range range, Parser<T> parser) {
    return readRange(tx, range).thenApply(kvs -> {
      List<T> messages = new ArrayList<>(kvs.size());
      for (KeyValue kv : kvs) {
        try {
          messages.add(parser.parseFrom(kv.getValue()));
        } catch (InvalidProtocolBufferException e) {
          throw new CompletionException("Failed to parse protobuf", e);
        }
      }
      return messages;
    });
  }

  /**
   * Performs a batched write of multiple key-value pairs.
   *
   * @param tx the transaction
   * @param kvPairs list of key-value pairs
   */
  public static void batchWrite(Transaction tx, List<KeyValue> kvPairs) {
    for (KeyValue kv : kvPairs) {
      tx.set(kv.getKey(), kv.getValue());
    }
  }

  /**
   * Performs a batched delete of keys in a range.
   *
   * @param tx the transaction
   * @param range the key range to clear
   */
  public static void clearRange(Transaction tx, Range range) {
    tx.clear(range);
  }

  /**
   * Checks if a key exists.
   *
   * @param tx the transaction
   * @param key the key to check
   * @return true if key exists
   */
  public static CompletableFuture<Boolean> exists(Transaction tx, byte[] key) {
    return tx.get(key).thenApply(value -> value != null);
  }

  /**
   * Performs an atomic increment operation.
   * Uses FDB's atomic operations for conflict-free increments.
   *
   * @param tx the transaction
   * @param key the counter key
   * @param delta the increment value
   */
  public static void atomicIncrement(Transaction tx, byte[] key, long delta) {
    // FDB's ADD mutation expects little-endian 64-bit integer
    byte[] deltaBytes = new byte[8];
    for (int i = 0; i < 8; i++) {
      deltaBytes[i] = (byte) (delta >>> (i * 8));
    }
    tx.mutate(com.apple.foundationdb.MutationType.ADD, key, deltaBytes);
  }

  /**
   * Gets the count of keys in a range.
   *
   * @param tx the transaction
   * @param range the key range
   * @return number of keys
   */
  public static CompletableFuture<Long> countRange(Transaction tx, Range range) {
    // Use estimated range count for large ranges
    return tx.getEstimatedRangeSizeBytes(range).thenCompose(sizeEstimate -> {
      // If range is small, do exact count
      if (sizeEstimate < 100_000) { // Less than 100KB
        return exactCount(tx, range);
      } else {
        // For large ranges, use sampling
        return estimatedCount(tx, range, sizeEstimate);
      }
    });
  }

  /**
   * Gets the exact count by scanning all keys.
   */
  private static CompletableFuture<Long> exactCount(Transaction tx, Range range) {
    AsyncIterable<KeyValue> iterable = tx.getRange(range);
    CompletableFuture<Long> countFuture = new CompletableFuture<>();

    AsyncIterator<KeyValue> iterator = iterable.iterator();
    countKeys(iterator, 0L, countFuture);

    return countFuture;
  }

  /**
   * Recursively counts keys using the async iterator.
   */
  private static void countKeys(AsyncIterator<KeyValue> iterator, Long count, CompletableFuture<Long> result) {
    iterator.onHasNext()
        .thenAccept(hasNext -> {
          if (hasNext) {
            iterator.next(); // Consume the value
            countKeys(iterator, count + 1, result);
          } else {
            result.complete(count);
          }
        })
        .exceptionally(e -> {
          result.completeExceptionally(e);
          return null;
        });
  }

  /**
   * Estimates count based on size for large ranges.
   */
  private static CompletableFuture<Long> estimatedCount(Transaction tx, Range range, long sizeBytes) {
    // Sample first 100 keys to get average size
    return readRange(tx, range, 100).thenApply(sample -> {
      if (sample.isEmpty()) {
        return 0L;
      }
      long sampleSize = sample.stream()
          .mapToLong(kv -> kv.getKey().length + kv.getValue().length)
          .sum();
      double avgSize = (double) sampleSize / sample.size();
      return Math.round(sizeBytes / avgSize);
    });
  }

  /**
   * Collects all results from an async iterable into a list.
   */
  private static <T> CompletableFuture<List<T>> collectAsync(AsyncIterable<T> iterable) {
    List<T> results = new ArrayList<>();
    CompletableFuture<List<T>> future = new CompletableFuture<>();

    AsyncIterator<T> iterator = iterable.iterator();
    collectNext(iterator, results, future);

    return future;
  }

  /**
   * Recursively collects items from async iterator.
   */
  private static <T> void collectNext(AsyncIterator<T> iterator, List<T> results, CompletableFuture<List<T>> future) {
    iterator.onHasNext()
        .thenAccept(hasNext -> {
          if (hasNext) {
            T item = iterator.next();
            results.add(item);
            collectNext(iterator, results, future);
          } else {
            future.complete(results);
          }
        })
        .exceptionally(e -> {
          future.completeExceptionally(e);
          return null;
        });
  }

  /**
   * Sets a read version for the transaction.
   * Useful for ensuring read consistency across multiple operations.
   *
   * @param tx the transaction
   * @param version the read version
   */
  public static void setReadVersion(Transaction tx, long version) {
    tx.setReadVersion(version);
  }
}
