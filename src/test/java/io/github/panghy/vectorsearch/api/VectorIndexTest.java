package io.github.panghy.vectorsearch.api;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.tasks.MaintenanceService;
import io.github.panghy.vectorsearch.tasks.MaintenanceWorker;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import io.github.panghy.vectorsearch.tasks.SegmentBuildService;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.LongPointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Combined VectorIndex tests to reduce file count and speed CI classpath scanning. */
class VectorIndexTest {

  // ---------- Former: VectorIndexAddBatchOrderingTest ----------
  static class AddBatchOrdering {
    Database db;
    DirectorySubspace root;

    @BeforeEach
    void setup() throws Exception {
      db = FDB.selectAPIVersion(730).open();
      root = db.runAsync(tr -> DirectoryLayer.getDefault()
              .createOrOpen(
                  tr,
                  List.of("vs-ordering", UUID.randomUUID().toString()),
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
    }

    @Test
    void addAll_preserves_order_across_rotations_and_chunks() throws Exception {
      final int dim = 4;
      final int maxSeg = 50;
      final int n = 123;
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(dim)
          .maxSegmentSize(maxSeg)
          .localWorkerThreads(0)
          .build();
      VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
      float[][] data = new float[n][dim];
      for (int i = 0; i < n; i++) for (int d = 0; d < dim; d++) data[i][d] = i * 0.001f + d;
      List<int[]> ids = index.addAll(data, null).get(30, TimeUnit.SECONDS);
      assertThat(ids).hasSize(n);
      for (int i = 0; i < n; i++) {
        assertThat(ids.get(i)[0]).isEqualTo(i / maxSeg);
        assertThat(ids.get(i)[1]).isEqualTo(i % maxSeg);
      }
      index.close();
    }
  }

  // ---------- Former: VectorIndexBatchDeleteIntegrationTest ----------
  static class BatchDeleteIntegration {
    private Database db;
    private DirectorySubspace root;
    private VectorIndex index;

    @BeforeEach
    public void setup() throws Exception {
      db = FDB.selectAPIVersion(730).open();
      root = db.runAsync(tr -> DirectoryLayer.getDefault()
              .createOrOpen(
                  tr,
                  List.of("vs-delall", UUID.randomUUID().toString()),
                  "vectorsearch".getBytes(StandardCharsets.UTF_8)))
          .get(5, TimeUnit.SECONDS);
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(8)
          .pqM(4)
          .pqK(16)
          .graphDegree(8)
          .maxSegmentSize(50)
          .estimatedWorkerCount(1)
          .defaultTtl(Duration.ofSeconds(30))
          .vacuumMinDeletedRatio(0.0)
          .localWorkerThreads(1)
          .build();
      index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    }

    @AfterEach
    public void teardown() {
      if (index != null) index.close();
      if (db != null) {
        db.run(tr -> {
          root.remove(tr);
          return null;
        });
        db.close();
      }
    }

    @Test
    public void batchDeleteRemovesAndEnqueuesVacuum() throws Exception {
      int[][] ids = new int[12][2];
      for (int i = 0; i < 12; i++) {
        float[] v = new float[8];
        for (int d = 0; d < 8; d++) v[d] = (float) Math.cos(0.01 * i + d);
        ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
      }
      int[][] toDel = new int[][] {ids[2], ids[5], ids[7]};
      index.deleteAll(toDel).get(5, TimeUnit.SECONDS);
      var res = index.query(new float[] {1, 0, 0, 0, 0, 0, 0, 0}, 10).get(5, TimeUnit.SECONDS);
      for (int[] id : toDel)
        assertThat(res.stream().noneMatch(r -> r.segmentId() == id[0] && r.vectorId() == id[1]))
            .isTrue();

      var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
      TaskQueueConfig<String, MaintenanceTask> tqc = TaskQueueConfig.builder(
              db,
              maintDir,
              new ProtoSerializers.StringSerializer(),
              new ProtoSerializers.MaintenanceTaskSerializer())
          .estimatedWorkerCount(1)
          .defaultTtl(Duration.ofSeconds(30))
          .defaultThrottle(Duration.ofMillis(25))
          .taskNameExtractor(mt ->
              mt.hasVacuum() ? ("vacuum-segment:" + mt.getVacuum().getSegId()) : "compact")
          .build();
      var mq = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
      boolean processed = new MaintenanceWorker(
              VectorIndexConfig.builder(db, root).dimension(8).build(), dirs, mq)
          .runOnce()
          .get(5, TimeUnit.SECONDS);
      assertThat(processed).isTrue();
    }
  }

  // ---------- Former: VectorIndexBestFirstRecallTest ----------
  static class BestFirstRecall {
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
                  List.of("vs-bestfirst", UUID.randomUUID().toString()),
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
    void best_first_default_has_high_self_recall() throws Exception {
      int dim = 8, n = 200, k = 10, qCount = 30;
      Random rnd = new Random(42);
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(dim)
          .pqM(4)
          .pqK(16)
          .graphDegree(16)
          .maxSegmentSize(50)
          .localWorkerThreads(1)
          .build();
      VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
      List<float[]> data = new ArrayList<>(n);
      for (int i = 0; i < n; i++) {
        float[] v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = (float) rnd.nextGaussian();
        data.add(v);
      }
      List<int[]> ids = index.addAll(data.toArray(new float[0][]), null).get(30, TimeUnit.SECONDS);
      index.awaitIndexingComplete().get(20, TimeUnit.SECONDS);
      List<Integer> picks = new ArrayList<>();
      for (int i = 0; i < qCount; i++) picks.add(rnd.nextInt(n));
      List<CompletableFuture<List<SearchResult>>> futures = new ArrayList<>();
      for (int idx : picks) futures.add(index.query(data.get(idx), k));
      CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).get(30, TimeUnit.SECONDS);
      int hits = 0;
      for (int i = 0; i < picks.size(); i++) {
        int idx = picks.get(i);
        int[] gt = ids.get(idx);
        List<SearchResult> res = futures.get(i).get(5, TimeUnit.SECONDS);
        boolean found = res.stream().anyMatch(r -> r.segmentId() == gt[0] && r.vectorId() == gt[1]);
        if (found) hits++;
      }
      double recall = hits / (double) qCount;
      assertThat(recall).isGreaterThanOrEqualTo(0.9);
      assertThat(reader.collectAllMetrics().stream()
              .anyMatch(m -> m.getName().equals("vectorsearch.cache.size")))
          .isTrue();
      index.close();
    }
  }

  // ---------- Former: VectorIndexDeleteAndVacuumTest ----------
  static class DeleteAndVacuum {
    private Database db;
    private DirectorySubspace root;
    private VectorIndex index;

    @BeforeEach
    public void setup() throws Exception {
      db = FDB.selectAPIVersion(730).open();
      root = db.runAsync(tr -> DirectoryLayer.getDefault()
              .createOrOpen(
                  tr,
                  List.of("vs-deltest", UUID.randomUUID().toString()),
                  "vectorsearch".getBytes(StandardCharsets.UTF_8)))
          .get(5, TimeUnit.SECONDS);
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(8)
          .pqM(4)
          .pqK(16)
          .graphDegree(8)
          .maxSegmentSize(20)
          .localWorkerThreads(1)
          .build();
      index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    }

    @AfterEach
    public void teardown() {
      if (index != null) index.close();
      if (db != null) {
        db.run(tr -> {
          root.remove(tr);
          return null;
        });
        db.close();
      }
    }

    @Test
    public void deleteAndVacuumFlow() throws Exception {
      int n = 30;
      int[][] ids = new int[n][2];
      for (int i = 0; i < n; i++) {
        float[] v = new float[8];
        for (int d = 0; d < 8; d++) v[d] = (float) Math.sin(0.01 * i + d);
        ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
      }
      Thread.sleep(1000);
      index.delete(ids[0][0], ids[0][1]).get(5, TimeUnit.SECONDS);
      index.delete(ids[5][0], ids[5][1]).get(5, TimeUnit.SECONDS);
      float[] q = new float[] {1f, 0.1f, -0.2f, 0.3f, 0.0f, 0.4f, -0.1f, 0.2f};
      var results = index.query(q, 10).get(5, TimeUnit.SECONDS);
      assertThat(results.stream()
              .noneMatch(r -> (r.segmentId() == ids[0][0] && r.vectorId() == ids[0][1])
                  || (r.segmentId() == ids[5][0] && r.vectorId() == ids[5][1])))
          .isTrue();
      var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      new MaintenanceService(
              VectorIndexConfig.builder(db, root)
                  .dimension(8)
                  .pqM(4)
                  .pqK(16)
                  .graphDegree(8)
                  .maxSegmentSize(20)
                  .build(),
              dirs)
          .vacuumSegment(ids[0][0], 0.0)
          .get(5, TimeUnit.SECONDS);
    }
  }

  // ---------- Former: VectorIndexHeavyRotationIntegrationTest ----------
  static class HeavyRotationIntegration {
    Database db;
    DirectorySubspace root;

    @BeforeEach
    void setup() throws Exception {
      db = FDB.selectAPIVersion(730).open();
      root = db.runAsync(tr -> DirectoryLayer.getDefault()
              .createOrOpen(
                  tr,
                  List.of("vs-heavy", UUID.randomUUID().toString()),
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
    }

    @Test
    void inserts_10k_vectors_and_parallel_queries_have_high_recall() throws Exception {
      int dimension = 8;
      int vectorCount = Integer.getInteger("VS_HEAVY_N", 1000);
      int segmentSize = 200;
      int queryCount = 100;
      int topK = 10;
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(dimension)
          .pqM(4)
          .pqK(16)
          .graphDegree(16)
          .maxSegmentSize(segmentSize)
          .localWorkerThreads(2)
          .build();
      VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
      Random rnd = new Random(1234);
      List<float[]> inserted = new ArrayList<>(vectorCount);
      List<int[]> ids = new ArrayList<>(vectorCount);
      int batchSize = Integer.getInteger("VS_HEAVY_BATCH", 200);
      for (int start = 0; start < vectorCount; start += batchSize) {
        int end = Math.min(vectorCount, start + batchSize);
        float[][] batch = new float[end - start][dimension];
        for (int i = start; i < end; i++) {
          float[] v = new float[dimension];
          for (int d = 0; d < dimension; d++) v[d] = (float) rnd.nextGaussian();
          batch[i - start] = v;
          inserted.add(v);
        }
        var got = index.addAll(batch, null).get(120, TimeUnit.SECONDS);
        ids.addAll(got);
      }
      List<Integer> picks = new ArrayList<>();
      for (int i = 0; i < queryCount; i++) picks.add(rnd.nextInt(vectorCount));
      index.awaitIndexingComplete().get(120, TimeUnit.SECONDS);
      List<CompletableFuture<List<SearchResult>>> futures = new ArrayList<>();
      for (int i : picks) futures.add(index.query(inserted.get(i), topK));
      CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).get(60, TimeUnit.SECONDS);
      int hits = 0;
      for (int qi = 0; qi < picks.size(); qi++) {
        int idx = picks.get(qi);
        int[] gt = ids.get(idx);
        List<SearchResult> res = futures.get(qi).get(5, TimeUnit.SECONDS);
        boolean found = res.stream().anyMatch(r -> r.segmentId() == gt[0] && r.vectorId() == gt[1]);
        if (found) hits++;
      }
      double recall = hits / (double) queryCount;
      assertThat(recall).isGreaterThanOrEqualTo(0.9);
      index.close();
    }
  }

  // ---------- Former: VectorIndexPrefetchSmokeTest ----------
  static class PrefetchSmoke {
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
                  List.of("vs-prefetch", UUID.randomUUID().toString()),
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
    void codebook_prefetch_warms_cache_for_sealed_segments() throws Exception {
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(4)
          .pqM(2)
          .pqK(2)
          .graphDegree(2)
          .maxSegmentSize(1)
          .localWorkerThreads(0)
          .build();
      var index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
      var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
      for (int i = 0; i < 5; i++) {
        index.add(new float[] {i, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
        new SegmentBuildService(cfg, dirs).build(i).get(5, TimeUnit.SECONDS);
      }
      long before = index.getCodebookCacheSize();
      assertThat(before).isLessThanOrEqualTo(1L);
      index.query(new float[] {0, 0, 0, 0}, 1).get(5, TimeUnit.SECONDS);
      long deadline = System.currentTimeMillis() + 2000, size;
      do {
        Thread.sleep(50);
        size = index.getCodebookCacheSize();
        if (size >= 5) break;
      } while (System.currentTimeMillis() < deadline);
      assertThat(index.getCodebookCacheSize()).isGreaterThanOrEqualTo(3L);
      long cbSize = -1;
      for (MetricData m : reader.collectAllMetrics()) {
        if (!"vectorsearch.cache.size".equals(m.getName())) continue;
        for (LongPointData p : m.getLongGaugeData().getPoints()) {
          String cache = p.getAttributes().get(AttributeKey.stringKey("cache"));
          if ("codebook".equals(cache)) cbSize = Math.max(cbSize, p.getValue());
        }
      }
      assertThat(cbSize).isGreaterThanOrEqualTo(3L);
      index.close();
    }
  }

  // ---------- Former: VectorIndexQueryIntegrationTest (selected methods) ----------
  static class QueryIntegration {
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
      VectorIndexConfig cfg =
          VectorIndexConfig.builder(db, root).dimension(3).build();
      VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
      index.add(new float[] {1f, 0f, 0f}, null).get(5, TimeUnit.SECONDS);
      index.add(new float[] {0f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);
      index.add(new float[] {0f, 0f, 1f}, null).get(5, TimeUnit.SECONDS);
      index.add(new float[] {1f, 1f, 0f}, null).get(5, TimeUnit.SECONDS);
      List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 2).get(5, TimeUnit.SECONDS);
      assertThat(res).hasSize(2);
      assertThat(res.get(0).vectorId()).isEqualTo(0);
    }

    @Test
    void query_on_empty_index_returns_empty() throws Exception {
      VectorIndexConfig cfg =
          VectorIndexConfig.builder(db, root).dimension(3).build();
      VectorIndex index = VectorIndex.createOrOpen(cfg).get(5, TimeUnit.SECONDS);
      List<SearchResult> res = index.query(new float[] {1f, 0f, 0f}, 5).get(5, TimeUnit.SECONDS);
      assertThat(res).isEmpty();
    }
  }
}
