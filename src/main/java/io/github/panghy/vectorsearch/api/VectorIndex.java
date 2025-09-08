package io.github.panghy.vectorsearch.api;

import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * Public API façade for performing vector operations against an index.
 *
 * <p>Provides async methods for index initialization, insertion, and KNN queries.
 * Queries perform graph+PQ search for SEALED segments and brute-force for ACTIVE/PENDING,
 * followed by exact re-rank.</p>
 */
public interface VectorIndex extends AutoCloseable {
  static CompletableFuture<VectorIndex> createOrOpen(VectorIndexConfig config) {
    return FdbVectorIndex.createOrOpen(config).thenApply(ix -> (VectorIndex) ix);
  }

  CompletableFuture<int[]> add(float[] embedding, byte[] payload);

  CompletableFuture<List<int[]>> addAll(float[][] embeddings, byte[][] payloads);

  CompletableFuture<List<SearchResult>> query(float[] q, int k);

  CompletableFuture<List<SearchResult>> query(float[] q, int k, SearchParams params);

  CompletableFuture<Void> delete(int segId, int vecId);

  CompletableFuture<Void> deleteAll(int[][] ids);

  CompletableFuture<Void> awaitIndexingComplete();

  long getCodebookCacheSize();

  long getAdjacencyCacheSize();

  @Override
  void close();
}
