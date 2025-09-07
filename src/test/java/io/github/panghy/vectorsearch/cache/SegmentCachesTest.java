package io.github.panghy.vectorsearch.cache;

/**
 * Unit test for SegmentCaches adjacency cache key and basic put/get behavior.
 */
import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class SegmentCachesTest {
  @Test
  void adjacency_cache_stores_and_computes_key() {
    com.github.benmanes.caffeine.cache.AsyncLoadingCache<Integer, float[][][]> dummyCodebooks =
        com.github.benmanes.caffeine.cache.Caffeine.newBuilder()
            .maximumSize(1)
            .buildAsync((Integer k1, java.util.concurrent.Executor ex) ->
                java.util.concurrent.CompletableFuture.completedFuture(null));
    com.github.benmanes.caffeine.cache.AsyncLoadingCache<Long, int[]> dummyAdj =
        com.github.benmanes.caffeine.cache.Caffeine.newBuilder()
            .maximumSize(10)
            .buildAsync((Long k2, java.util.concurrent.Executor ex) ->
                java.util.concurrent.CompletableFuture.completedFuture(new int[0]));
    SegmentCaches c = new SegmentCaches(dummyCodebooks, dummyAdj);
    long k = SegmentCaches.adjKey(3, 5);
    c.getAdjacencyCache().put(k, new int[] {1, 2, 3});
    assertThat(c.getAdjacencyCache().getIfPresent(k)).containsExactly(1, 2, 3);
  }
}
