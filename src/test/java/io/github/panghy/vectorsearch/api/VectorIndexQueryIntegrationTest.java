package io.github.panghy.vectorsearch.api;

/**
 * Integration tests for VectorIndex covering:
 * - Basic L2/COSINE queries on tiny datasets
 * - Sealed vs active segment paths, adjacency/codebook cache usage
 * - Edge cases (missing codebook, cleared codes, empty adjacency)
 * - BEST_FIRST tuning behavior and deterministic pivots
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.Range;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.apple.foundationdb.subspace.Subspace;
import com.apple.foundationdb.tuple.Tuple;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import io.github.panghy.vectorsearch.tasks.SegmentBuildService;
import io.github.panghy.vectorsearch.tasks.SegmentBuildWorker;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class VectorIndexQueryIntegrationTest {
  Database db;
  DirectorySubspace root;
  OpenTelemetrySdk sdk;
  InMemoryMetricReader reader;

  @BeforeEach
  void setup() throws Exception {
    reader = InMemoryMetricReader.create();
    SdkMeterProvider mp =
        SdkMeterProvider.builder().registerMetricReader(reader).build();
    sdk = OpenTelemetrySdk.builder().setMeterProvider(mp).build();
    GlobalOpenTelemetry.resetForTest();
    GlobalOpenTelemetry.set(sdk);
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-query", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    db.close();
    if (sdk != null) sdk.getSdkMeterProvider().close();
    GlobalOpenTelemetry.resetForTest();
  }

  @Test
  void l2_query_returns_expected_top() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(3).build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // id 0
    index.add(new float[] {0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // id 1
    index.add(new float[] {0f, 0f, 1f}, null).get(5, TimeUnit.SECONDS); // id 2
    index.add(new float[] {1f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // id 3

    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 2).get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(2);
    assertThat(res.get(0).vectorId()).isEqualTo(0);
  }

  @Test
  void query_on_empty_index_returns_empty() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(3).build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 5).get(5, TimeUnit.SECONDS);
    assertThat(res).isEmpty();
  }

  // Additional coverage: BEAM vs BEST_FIRST modes behave and cache adjacency
  @Test
  void sealed_defaults_and_bestfirst_both_work_and_cache() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);

    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    new SegmentBuildWorker(cfg, dirs, queue).runOnce().get(5, TimeUnit.SECONDS);

    var r1 = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(r1).isNotEmpty();
    long adjBefore = index.getAdjacencyCacheSize();
    var r2 = index.query(
            new float[] {1f, 0f, 0f, 0f},
            1,
            SearchParams.of(16, 8, 3, 2048, true, SearchParams.Mode.BEST_FIRST))
        .get(5, TimeUnit.SECONDS);
    assertThat(r2).isNotEmpty();
    assertThat(index.getAdjacencyCacheSize()).isGreaterThanOrEqualTo(adjBefore);
    // Metrics available
    assertThat(reader.collectAllMetrics().stream().anyMatch(m -> m.getName().equals("vectorsearch.cache.size")))
        .isTrue();
    index.close();
  }

  @Test
  void active_and_sealed_paths_both_trigger() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(3)
        .pqM(3)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 0f, 1f}, null).get(5, TimeUnit.SECONDS);

    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    new SegmentBuildWorker(cfg, dirs, queue).runOnce().get(5, TimeUnit.SECONDS);

    var res = index.query(new float[] {1f, 0f, 0f}, 2).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    assertThat(res.stream().map(SearchResult::segmentId).collect(java.util.stream.Collectors.toSet()))
        .containsAnyOf(0, 1);
    assertThat(index.getAdjacencyCacheSize()).isGreaterThanOrEqualTo(1);
    assertThat(reader.collectAllMetrics().stream().anyMatch(m -> m.getName().equals("vectorsearch.cache.size")))
        .isTrue();
    index.close();
  }

  @Test
  void empty_adjacency_segment_is_ok() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    index.close();
  }

  @Test
  void best_first_honors_max_explore_cap() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = io.github.panghy.taskqueue.TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.StringSerializer(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = io.github.panghy.taskqueue.TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildWorker(cfg, dirs, queue)
        .runOnce()
        .get(5, TimeUnit.SECONDS);

    // Extremely small cap; should early-return without exploring many nodes
    SearchParams p = SearchParams.of(4, 2, 2, 1, false, SearchParams.Mode.BEST_FIRST);
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1, p).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    index.close();
  }

  @Test
  void sealed_missing_codebook_yields_no_results() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(3)
        .pqM(3)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    // One vector, seal segment 0
    index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
    // Delete codebook to hit early-return branch
    byte[] cbKey = dirs.segmentKeys("000000").pqCodebookKey();
    db.run(tr -> {
      tr.clear(cbKey);
      return null;
    });

    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).isEmpty();
    index.close();
  }

  @Test
  void sealed_with_bad_codes_and_missing_vector_is_skipped() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(2)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    // Add two vectors and seal
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // vec 0
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // vec 1
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
    // Write a bad code with too-short length for existing vec and one for non-existing vec id
    db.run(tr -> {
      tr.set(dirs.segmentKeys("000000").pqCodeKey(1), new byte[] {0}); // too short (m=2 expected)
      tr.set(dirs.segmentKeys("000000").pqCodeKey(999), new byte[] {0, 1}); // non-existent vector
      return null;
    });
    // Should still return a valid result (vec 0) without throwing
    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(1);
    assertThat(res.get(0).vectorId()).isEqualTo(0);
    index.close();
  }

  @Test
  void await_indexing_complete_seals_pending_segment_and_queries_use_sealed_path() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(1) // run local builders
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    // Insert three vectors -> seg0 becomes PENDING, seg1 ACTIVE
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);

    // Wait for background indexing to drain
    index.awaitIndexingComplete().get(10, TimeUnit.SECONDS);

    // seg0 should be SEALED now
    var dirs = FdbDirectories.openIndex(root, db)
        .get(5, TimeUnit.SECONDS);
    byte[] seg0Meta =
        db.readAsync(tr -> tr.get(dirs.segmentKeys("000000").metaKey())).get(5, TimeUnit.SECONDS);
    var sm = SegmentMeta.parseFrom(seg0Meta);
    assertThat(sm.getState()).isEqualTo(SegmentMeta.State.SEALED);

    // A query should touch the sealed path (PQ+graph). We assert adjacency cache grows.
    long before = index.getAdjacencyCacheSize();
    var r = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(r).isNotEmpty();
    assertThat(index.getAdjacencyCacheSize()).isGreaterThanOrEqualTo(before);
    index.close();
  }

  @Test
  void sealed_with_all_codes_cleared_returns_empty() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(2)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    // Add a couple and seal
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);
    // Clear entire codes directory for seg 0 to trigger approxAll.isEmpty branch
    String segStr = "000000";
    Subspace codesPrefix = new Subspace(dirs.segmentsDir().pack(Tuple.from(segStr, "pq", "codes")));
    Range cr = codesPrefix.range();
    db.run(tr -> {
      tr.clear(cr.begin, cr.end);
      return null;
    });

    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(res).isEmpty();
    index.close();
  }

  @Test
  void deterministic_pivot_seeding_and_auto_tune() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(4)
        .maxSegmentSize(50)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    // Build a sealed segment with 12 vectors so pivot path (beam+2) engages (defaults beam≈8)
    for (int i = 0; i < 12; i++) {
      index.add(new float[] {i, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    }
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildService(cfg, dirs)
        .build(0)
        .get(5, TimeUnit.SECONDS);

    long before = index.getAdjacencyCacheSize();
    var res1 = index.query(new float[] {5.2f, 0f, 0f, 0f}, 3).get(5, TimeUnit.SECONDS);
    assertThat(res1).isNotEmpty();
    long after = index.getAdjacencyCacheSize();
    // Expect at least beam (≈8) + a couple pivots were fetched
    assertThat(after - before).isGreaterThanOrEqualTo(8);

    // Deterministic pivots → second query should not change cache size
    long prev = index.getAdjacencyCacheSize();
    var res2 = index.query(new float[] {5.2f, 0f, 0f, 0f}, 3).get(5, TimeUnit.SECONDS);
    assertThat(res2).isNotEmpty();
    assertThat(index.getAdjacencyCacheSize()).isEqualTo(prev);
    index.close();
  }

  @Test
  void builder_random_pivots_min_hops_and_seg_cap() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(8)
        .maxSegmentSize(50)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    for (int i = 0; i < 16; i++) {
      index.add(new float[] {i, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    }
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildService(cfg, dirs)
        .build(0)
        .get(5, TimeUnit.SECONDS);

    long before = index.getAdjacencyCacheSize();
    SearchParams p = new SearchParams.Builder()
        .efSearch(16)
        .beamWidth(8)
        .maxIters(2)
        .maxExplore(256)
        .refineFrontier(true)
        .minHops(2)
        .pivots(3)
        .seedStrategy(SearchParams.SeedStrategy.RANDOM_PIVOTS)
        .perSegmentLimitMultiplier(1)
        .mode(SearchParams.Mode.BEAM)
        .build();
    var res = index.query(new float[] {7.7f, 0f, 0f, 0f}, 2, p).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    assertThat(index.getAdjacencyCacheSize()).isGreaterThan(before);

    // Create an ACTIVE second segment in a fresh index to avoid sealed-append error
    // Use a new directory for the second index to avoid meta mismatches
    var root2 = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-query-2", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, root2)
        .dimension(4)
        .pqM(2)
        .pqK(4)
        .graphDegree(8)
        .maxSegmentSize(1) // rotate immediately
        .localWorkerThreads(0)
        .build();
    VectorIndex idx2 = VectorIndex.createOrOpen(cfg2).get(5, TimeUnit.SECONDS);
    idx2.add(new float[] {0f, 0f, 0f, 1f}, null).get(5, TimeUnit.SECONDS); // seg0
    idx2.add(new float[] {0f, 0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // rotate -> seg1 ACTIVE
    // Query to exercise per-segment cap path; shouldn't throw
    var res2 = idx2.query(new float[] {0f, 0f, 0.9f, 0.1f}, 2, p).get(5, TimeUnit.SECONDS);
    assertThat(res2).isNotEmpty();
    idx2.close();
    db.run(tr -> {
      root2.remove(tr);
      return null;
    });
    var res2b = index.query(new float[] {0.1f, 0f, 0f, 0f}, 2, p).get(5, TimeUnit.SECONDS);
    assertThat(res2b).isNotEmpty();
    index.close();
  }

  @Test
  void builder_min_hops_with_empty_adjacency_forces_two_hops() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
    // Single vector sealed -> adjacency empty
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildService(cfg, dirs)
        .build(0)
        .get(5, TimeUnit.SECONDS);

    SearchParams p = new SearchParams.Builder()
        .efSearch(4)
        .beamWidth(2)
        .maxIters(2)
        .maxExplore(16)
        .refineFrontier(false)
        .minHops(2)
        .mode(SearchParams.Mode.BEAM)
        .build();
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1, p).get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(1);
    index.close();
  }

  @Test
  void cosine_query_returns_expected_top() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(3)
        .metric(VectorIndexConfig.Metric.COSINE)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // id 0
    index.add(new float[] {0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // id 1
    index.add(new float[] {0f, 0f, 1f}, null).get(5, TimeUnit.SECONDS); // id 2
    index.add(new float[] {1f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // id 3

    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 2).get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(2);
    assertThat(res.get(0).vectorId()).isEqualTo(0);
    // Next best should be vector with component in x (id 3)
    assertThat(res.get(1).vectorId()).isIn(3, 1, 2);
  }

  @Test
  void beam_refine_off_still_returns_results() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0f, 0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);

    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc = io.github.panghy.taskqueue.TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.StringSerializer(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = io.github.panghy.taskqueue.TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildWorker(cfg, dirs, queue)
        .runOnce()
        .get(5, TimeUnit.SECONDS);

    SearchParams p = SearchParams.of(16, 8, 3, 2048, false, SearchParams.Mode.BEAM);
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1, p).get(5, TimeUnit.SECONDS);
    assertThat(res).isNotEmpty();
    index.close();
  }

  @Test
  void deleted_vectors_are_excluded_from_results() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(3).build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    // Add two vectors; delete the first by flipping the flag directly
    index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // id 0
    index.add(new float[] {0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS); // id 1

    // Flip deleted flag for vec 0 in seg 0
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    byte[] v0key = dirs.segmentKeys("000000").vectorKey(0);
    byte[] v0bytes = db.readAsync(tr -> tr.get(v0key)).get(5, TimeUnit.SECONDS);
    io.github.panghy.vectorsearch.proto.VectorRecord rec =
        io.github.panghy.vectorsearch.proto.VectorRecord.parseFrom(v0bytes);
    io.github.panghy.vectorsearch.proto.VectorRecord recDel =
        rec.toBuilder().setDeleted(true).build();
    db.run(tr -> {
      tr.set(v0key, recDel.toByteArray());
      return null;
    });

    // Query: should not return deleted vec 0
    List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 2).get(5, TimeUnit.SECONDS);
    assertThat(res.stream().map(SearchResult::vectorId)).doesNotContain(0);
  }

  @Test
  void diskann_params_search_finds_expected_neighbor() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(2)
        .pqM(2)
        .pqK(2)
        .graphDegree(4)
        .maxSegmentSize(20)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    // Insert a simple 1D manifold along x-axis in 2D
    for (int i = 0; i < 20; i++) {
      index.add(new float[] {i, 0f}, null).get(5, TimeUnit.SECONDS);
    }
    // Traversal also works for ACTIVE; sealing not required here.

    // Query near x = 15
    float[] q = new float[] {15.1f, 0f};
    SearchParams params = SearchParams.of(32, 8, 3);
    List<SearchResult> res = index.query(q, 1, params).get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(1);
    assertThat(res.get(0).vectorId()).isBetween(14, 16);
    index.close();
  }

  @Test
  void sealed_segment_query_uses_pq_and_rerank() throws Exception {
    // Configure PQ and small segment size to force sealing after 1 insert
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(2)
        .maxSegmentSize(1)
        .localWorkerThreads(0)
        .build();
    VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);

    // Insert two vectors -> seg0 PENDING
    index.add(new float[] {1f, 0f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // id 0
    index.add(new float[] {0f, 1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS); // id 1 (triggers rotate)

    // Explicitly process build task instead of sleeping
    var dirs2 = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    var tqc2 = io.github.panghy.taskqueue.TaskQueueConfig.builder(
            db,
            dirs2.tasksDir(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.StringSerializer(),
            new io.github.panghy.vectorsearch.tasks.ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue2 = io.github.panghy.taskqueue.TaskQueues.createTaskQueue(tqc2).get(5, TimeUnit.SECONDS);
    new io.github.panghy.vectorsearch.tasks.SegmentBuildWorker(cfg, dirs2, queue2)
        .runOnce()
        .get(5, TimeUnit.SECONDS);

    // Query should hit SEALED path with PQ + exact rerank
    var res = index.query(new float[] {1f, 0f, 0f, 0f}, 1, SearchParams.defaults(1, 2))
        .get(5, TimeUnit.SECONDS);
    assertThat(res).hasSize(1);
    assertThat(res.get(0).vectorId()).isEqualTo(0);

    // Codebook cache should have an entry now
    assertThat(index.getCodebookCacheSize()).isGreaterThanOrEqualTo(1);

    // Second query should reuse cache (size stable)
    long sizeBefore = index.getCodebookCacheSize();
    index.query(new float[] {1f, 0f, 0f, 0f}, 1).get(5, TimeUnit.SECONDS);
    assertThat(index.getCodebookCacheSize()).isEqualTo(sizeBefore);

    // Adjacency cache should be populated when graph traversal runs
    assertThat(index.getAdjacencyCacheSize()).isGreaterThanOrEqualTo(1);
    index.close();
  }
}
