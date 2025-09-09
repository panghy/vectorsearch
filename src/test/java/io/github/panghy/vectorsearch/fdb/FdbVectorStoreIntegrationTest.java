package io.github.panghy.vectorsearch.fdb;

/**
 * Integration tests for FdbVectorStore covering index creation, rotation, task enqueueing,
 * worker sealing, and configuration mismatch validation.
 */
import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.proto.Adjacency;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.tasks.ProtoSerializers;
import io.github.panghy.vectorsearch.tasks.SegmentBuildWorker;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class FdbVectorStoreIntegrationTest {

  Database db;
  DirectorySubspace appRoot;

  @BeforeEach
  void setUp() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    appRoot = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vectorsearch-test", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
  }

  @AfterEach
  void tearDown() {
    db.run(tr -> {
      appRoot.remove(tr);
      return null;
    });
    db.close();
  }

  @Test
  void createIndex_and_addVectors_rotatesAtThreshold() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, appRoot)
        .dimension(4)
        .maxSegmentSize(2) // rotate after 2 inserts
        .defaultTtl(Duration.ofMinutes(5))
        .build();

    var indexDirs = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqc0 = TaskQueueConfig.builder(
            db,
            indexDirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue0 = TaskQueues.createTaskQueue(tqc0).get(5, TimeUnit.SECONDS);
    FdbVectorStore store = new FdbVectorStore(cfg, indexDirs, queue0);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Sanity: index meta and currentSegment keys exist
    var dirs = store.openIndexDirs().get(5, TimeUnit.SECONDS);
    byte[] indexMeta = db.readAsync(tr -> tr.get(dirs.metaKey())).get(5, TimeUnit.SECONDS);
    byte[] curSeg = db.readAsync(tr -> tr.get(dirs.currentSegmentKey())).get(5, TimeUnit.SECONDS);
    assertThat(indexMeta).isNotNull();
    assertThat(curSeg).isNotNull();
    // Sanity: segment 0 meta exists (flattened key)
    byte[] seg0Meta =
        db.readAsync(tr -> tr.get(dirs.segmentKeys(0).metaKey())).get(5, TimeUnit.SECONDS);
    if (seg0Meta == null) {
      int segKv = db.readAsync(
              tr -> tr.getRange(dirs.segmentsDir().range(), 100).asList())
          .get(5, TimeUnit.SECONDS)
          .size();
      int idxKv = db.readAsync(
              tr -> tr.getRange(dirs.indexDir().range(), 100).asList())
          .get(5, TimeUnit.SECONDS)
          .size();
      assertThat(idxKv).as("no keys under index dir").isGreaterThanOrEqualTo(1);
      assertThat(segKv).as("no keys under segments dir").isGreaterThanOrEqualTo(1);
    }
    assertThat(seg0Meta).isNotNull();

    // Initial segment must be 0 and ACTIVE
    long cur = store.getCurrentSegment().get(5, TimeUnit.SECONDS);
    assertThat(cur).isEqualTo(0);
    SegmentMeta seg0 = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(seg0.getState()).isEqualTo(SegmentMeta.State.ACTIVE);
    assertThat(seg0.getCount()).isEqualTo(0);

    // Insert two vectors -> still segment 0, count=2
    int[] id0 = store.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS);
    int[] id1 = store.add(new float[] {4, 3, 2, 1}, new byte[] {1}).get(5, TimeUnit.SECONDS);
    assertThat(id0[0]).isEqualTo(0);
    assertThat(id0[1]).isEqualTo(0);
    assertThat(id1[0]).isEqualTo(0);
    assertThat(id1[1]).isEqualTo(1);
    SegmentMeta after2 = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(after2.getCount()).isEqualTo(2);
    assertThat(after2.getState())
        .isIn(SegmentMeta.State.ACTIVE, SegmentMeta.State.PENDING); // pending if rotated in same txn

    // Third insert -> triggers rotation BEFORE write: seg0 becomes PENDING; vector lands in seg1 as vecId=0
    int[] id2 = store.add(new float[] {0, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    assertThat(id2[0]).isEqualTo(1);
    assertThat(id2[1]).isEqualTo(0);

    long currentAfter = store.getCurrentSegment().get(5, TimeUnit.SECONDS);
    assertThat(currentAfter).isEqualTo(1);
    SegmentMeta seg0Final = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(seg0Final.getState()).isEqualTo(SegmentMeta.State.PENDING);
    SegmentMeta seg1 = store.getSegmentMeta(1).get(5, TimeUnit.SECONDS);
    assertThat(seg1.getState()).isEqualTo(SegmentMeta.State.ACTIVE);
    assertThat(seg1.getCount()).isEqualTo(1);
  }

  @Test
  void single_transaction_addAll_rotates_and_enqueues_build_tasks() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, appRoot)
        .dimension(4)
        .maxSegmentSize(2)
        .build();
    var dirs = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    FdbVectorStore store = new FdbVectorStore(cfg, dirs, queue);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    float[][] data = new float[5][4];
    for (int i = 0; i < 5; i++) data[i] = new float[] {i, 0, 0, 0};
    db.runAsync(tr -> store.addBatch(tr, data, null).thenApply(ids -> null)).get(5, TimeUnit.SECONDS);

    SegmentMeta s0 = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    SegmentMeta s1 = store.getSegmentMeta(1).get(5, TimeUnit.SECONDS);
    SegmentMeta s2 = store.getSegmentMeta(2).get(5, TimeUnit.SECONDS);
    assertThat(s0.getState()).isEqualTo(SegmentMeta.State.PENDING);
    assertThat(s0.getCount()).isEqualTo(2);
    assertThat(s1.getState()).isEqualTo(SegmentMeta.State.PENDING);
    assertThat(s1.getCount()).isEqualTo(2);
    assertThat(s2.getState()).isEqualTo(SegmentMeta.State.ACTIVE);
    assertThat(s2.getCount()).isEqualTo(1);

    // Two build tasks should be present (seg 0 and 1)
    var c1 = queue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    assertThat(c1).isNotNull();
    var c2 = queue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    assertThat(c2).isNotNull();
  }

  @Test
  void rotation_enqueues_build_task() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, appRoot)
        .dimension(4)
        .maxSegmentSize(2)
        .pqM(2)
        .pqK(2)
        .build();

    var indexDirs2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqc1 = TaskQueueConfig.builder(
            db,
            indexDirs2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue1 = TaskQueues.createTaskQueue(tqc1).get(5, TimeUnit.SECONDS);
    FdbVectorStore store = new FdbVectorStore(cfg, indexDirs2, queue1);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Insert 3 vectors -> rotation happens and a build task for seg 0 should be enqueued
    store.add(new float[] {1, 1, 1, 1}, null).get(5, TimeUnit.SECONDS);
    store.add(new float[] {2, 2, 2, 2}, null).get(5, TimeUnit.SECONDS);
    store.add(new float[] {3, 3, 3, 3}, null).get(5, TimeUnit.SECONDS);

    // Create a task queue pointing at the index's tasks directory and claim the task
    var dirs = store.openIndexDirs().get(5, TimeUnit.SECONDS);
    var tqc = TaskQueueConfig.builder(
            db,
            dirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue = TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    var claim = queue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    assertThat(claim).isNotNull();
    BuildTask bt = claim.task();
    assertThat(bt.getSegId()).isEqualTo(0);
    claim.complete().get(5, TimeUnit.SECONDS);
  }

  @Test
  void worker_seals_pending_segment() throws Exception {
    String indexName = "ix_worker";
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, appRoot)
        .dimension(4)
        .maxSegmentSize(1)
        .pqM(2)
        .pqK(2)
        .build();
    var dirs3 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqc2 = TaskQueueConfig.builder(
            db,
            dirs3.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var queue2 = TaskQueues.createTaskQueue(tqc2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store = new FdbVectorStore(cfg, dirs3, queue2);
    store.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Two inserts → rotate: seg 0 becomes PENDING and a BuildTask is enqueued
    store.add(new float[] {1, 2, 3, 4}, null).get(5, TimeUnit.SECONDS);
    store.add(new float[] {5, 6, 7, 8}, null).get(5, TimeUnit.SECONDS);
    SegmentMeta seg0 = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(seg0.getState()).isEqualTo(SegmentMeta.State.PENDING);

    // Run worker once to build artifacts and seal the segment
    var wDirs = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqcW = TaskQueueConfig.builder(
            db,
            wDirs.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qW = TaskQueues.createTaskQueue(tqcW).get(5, TimeUnit.SECONDS);
    SegmentBuildWorker worker = new SegmentBuildWorker(cfg, wDirs, qW);
    Boolean processed = worker.runOnce().get(5, TimeUnit.SECONDS);
    assertThat(processed).isTrue();

    SegmentMeta seg0After = store.getSegmentMeta(0).get(5, TimeUnit.SECONDS);
    assertThat(seg0After.getState()).isEqualTo(SegmentMeta.State.SEALED);

    // Verify PQ codebook and codes exist; with strict-cap rotation and maxSegmentSize=1
    // segment 0 has one vector (id 0) and segment 1 has one vector (id 0).
    var dirs = store.openIndexDirs().get(5, TimeUnit.SECONDS);
    byte[] codebook =
        db.readAsync(tr -> tr.get(dirs.segmentKeys(0).pqCodebookKey())).get(5, TimeUnit.SECONDS);
    assertThat(codebook).isNotNull();
    byte[] code0 =
        db.readAsync(tr -> tr.get(dirs.segmentKeys(0).pqCodeKey(0))).get(5, TimeUnit.SECONDS);
    assertThat(code0).isNotNull();
    assertThat(code0.length).isEqualTo(2); // m bytes
    byte[] adj0 =
        db.readAsync(tr -> tr.get(dirs.segmentKeys(0).graphKey(0))).get(5, TimeUnit.SECONDS);
    assertThat(adj0).isNotNull();
    var adjProto0 = Adjacency.parseFrom(adj0);
    assertThat(adjProto0.getNeighborIdsCount()).isEqualTo(0);
    // Only segment 0 was sealed by the worker; segment 1 remains ACTIVE and has no adjacency entries.
  }

  @Test
  void reopen_with_mismatched_config_throws() throws Exception {
    VectorIndexConfig cfg1 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .maxSegmentSize(2)
        .build();
    var d1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tq1 = TaskQueueConfig.builder(
            db,
            d1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var q1 = TaskQueues.createTaskQueue(tq1).get(5, TimeUnit.SECONDS);
    FdbVectorStore store1 = new FdbVectorStore(cfg1, d1, q1);
    store1.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // Mismatch dimension
    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(16) // mismatch
        .maxSegmentSize(2)
        .build();
    var d2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tq2 = TaskQueueConfig.builder(
            db,
            d2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var q2 = TaskQueues.createTaskQueue(tq2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, d2, q2);

    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Dimension mismatch");

    // Matching config (different TTL/throttle not persisted) should succeed
    VectorIndexConfig cfg3 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .maxSegmentSize(2)
        .defaultTtl(Duration.ofMinutes(10))
        .build();
    var d3 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tq3 = TaskQueueConfig.builder(
            db,
            d3.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var q3b = TaskQueues.createTaskQueue(tq3).get(5, TimeUnit.SECONDS);
    FdbVectorStore store3 = new FdbVectorStore(cfg3, d3, q3b);
    store3.createOrOpenIndex().get(5, TimeUnit.SECONDS);
  }

  @Test
  void reopen_with_metric_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 =
        VectorIndexConfig.builder(db, appRoot).dimension(8).build(); // default L2
    var dm1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqm1 = TaskQueueConfig.builder(
            db,
            dm1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qm1 = TaskQueues.createTaskQueue(tqm1).get(5, TimeUnit.SECONDS);
    FdbVectorStore store1 = new FdbVectorStore(cfg1, dm1, qm1);
    store1.createOrOpenIndex().get(5, TimeUnit.SECONDS);

    // COSINE metric mismatch
    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .metric(VectorIndexConfig.Metric.COSINE)
        .build();
    var dm2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqm2 = TaskQueueConfig.builder(
            db,
            dm2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qm2 = TaskQueues.createTaskQueue(tqm2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dm2, qm2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Metric mismatch");
  }

  @Test
  void reopen_with_pqM_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 =
        VectorIndexConfig.builder(db, appRoot).dimension(8).pqM(16).build();
    var dA1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqA1 = TaskQueueConfig.builder(
            db,
            dA1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qA1 = TaskQueues.createTaskQueue(tqA1).get(5, TimeUnit.SECONDS);
    new FdbVectorStore(cfg1, dA1, qA1).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .pqM(32) // mismatch
        .build();
    var dA2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqA2 = TaskQueueConfig.builder(
            db,
            dA2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qA2 = TaskQueues.createTaskQueue(tqA2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dA2, qA2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("pqM mismatch");
  }

  @Test
  void reopen_with_oversample_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .oversample(2)
        .build();
    var dB1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqB1 = TaskQueueConfig.builder(
            db,
            dB1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qB1 = TaskQueues.createTaskQueue(tqB1).get(5, TimeUnit.SECONDS);
    new FdbVectorStore(cfg1, dB1, qB1).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .oversample(3) // mismatch
        .build();
    var dB2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqB2 = TaskQueueConfig.builder(
            db,
            dB2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qB2 = TaskQueues.createTaskQueue(tqB2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dB2, qB2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("oversample mismatch");
  }

  @Test
  void reopen_with_pqK_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 =
        VectorIndexConfig.builder(db, appRoot).dimension(8).pqK(256).build();
    var dC1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqC1 = TaskQueueConfig.builder(
            db,
            dC1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qC1 = TaskQueues.createTaskQueue(tqC1).get(5, TimeUnit.SECONDS);
    new FdbVectorStore(cfg1, dC1, qC1).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg2 =
        VectorIndexConfig.builder(db, appRoot).dimension(8).pqK(128).build();
    var dC2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqC2 = TaskQueueConfig.builder(
            db,
            dC2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qC2 = TaskQueues.createTaskQueue(tqC2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dC2, qC2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("pqK mismatch");
  }

  @Test
  void reopen_with_graphDegree_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .graphDegree(64)
        .build();
    var dD1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqD1 = TaskQueueConfig.builder(
            db,
            dD1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qD1 = TaskQueues.createTaskQueue(tqD1).get(5, TimeUnit.SECONDS);
    new FdbVectorStore(cfg1, dD1, qD1).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .graphDegree(32)
        .build();
    var dD2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqD2 = TaskQueueConfig.builder(
            db,
            dD2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qD2 = TaskQueues.createTaskQueue(tqD2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dD2, qD2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("graphDegree mismatch");
  }

  @Test
  void reopen_with_maxSegmentSize_mismatch_throws() throws Exception {
    VectorIndexConfig cfg1 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .maxSegmentSize(10)
        .build();
    var dE1 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqE1 = TaskQueueConfig.builder(
            db,
            dE1.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qE1 = TaskQueues.createTaskQueue(tqE1).get(5, TimeUnit.SECONDS);
    new FdbVectorStore(cfg1, dE1, qE1).createOrOpenIndex().get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, appRoot)
        .dimension(8)
        .maxSegmentSize(20)
        .build();
    var dE2 = FdbDirectories.openIndex(appRoot, db).get(5, TimeUnit.SECONDS);
    var tqE2 = TaskQueueConfig.builder(
            db,
            dE2.tasksDir(),
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.BuildTaskSerializer())
        .build();
    var qE2 = TaskQueues.createTaskQueue(tqE2).get(5, TimeUnit.SECONDS);
    FdbVectorStore store2 = new FdbVectorStore(cfg2, dE2, qE2);
    Assertions.assertThatThrownBy(() -> store2.createOrOpenIndex().get(5, TimeUnit.SECONDS))
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxSegmentSize mismatch");
  }
}
