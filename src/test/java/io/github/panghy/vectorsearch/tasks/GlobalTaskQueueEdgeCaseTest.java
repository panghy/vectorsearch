package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import com.google.protobuf.ByteString;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.GlobalTaskQueueConfig;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import io.github.panghy.vectorsearch.proto.IndexMeta;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Edge case tests for the global task queue feature.
 *
 * <p>Covers adapter null checks, unsupported method exceptions, serializer round-trips,
 * {@link GlobalWorkerRunner} lifecycle (start/close/sentinel handling),
 * {@link GlobalTaskQueueConfig} validation, {@link VectorIndexConfig#isGlobalTaskQueueEnabled()},
 * config reconstruction from {@link IndexMeta}, and missing IndexMeta handling.</p>
 */
class GlobalTaskQueueEdgeCaseTest {

  private Database db;
  private DirectorySubspace root;
  private DirectorySubspace globalDir;
  private TaskQueue<String, GlobalBuildTask> globalBuildQueue;
  private TaskQueue<String, GlobalMaintenanceTask> globalMaintQueue;
  private GlobalTaskQueueConfig globalConfig;
  private GlobalWorkerRunner runner;
  private VectorIndex index;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    String uid = UUID.randomUUID().toString();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr, List.of("vs-edge-test", uid), "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);

    globalDir = db.runAsync(tr -> DirectoryLayer.getDefault().createOrOpen(tr, List.of("vs-edge-queues", uid)))
        .get(5, TimeUnit.SECONDS);

    DirectorySubspace buildDir =
        globalDir.createOrOpen(db, List.of("build")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<String, GlobalBuildTask> buildTqc = TaskQueueConfig.builder(
            db,
            buildDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.GlobalBuildTaskSerializer())
        .defaultTtl(Duration.ofSeconds(30))
        .defaultThrottle(Duration.ofMillis(50))
        .build();
    globalBuildQueue = TaskQueues.createTaskQueue(buildTqc).get(5, TimeUnit.SECONDS);

    DirectorySubspace maintDir =
        globalDir.createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<String, GlobalMaintenanceTask> maintTqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.GlobalMaintenanceTaskSerializer())
        .defaultTtl(Duration.ofSeconds(30))
        .defaultThrottle(Duration.ofMillis(50))
        .build();
    globalMaintQueue = TaskQueues.createTaskQueue(maintTqc).get(5, TimeUnit.SECONDS);

    globalConfig = new GlobalTaskQueueConfig(globalBuildQueue, globalMaintQueue);
  }

  @AfterEach
  void teardown() {
    if (runner != null) runner.close();
    if (index != null) index.close();
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        globalDir.remove(tr);
        return null;
      });
      db.close();
    }
  }

  // ---- Test 1: Build adapter unsupported consumer/inspection methods ----
  @Test
  void buildAdapter_unsupportedMethods_throwUnsupportedOperationException() {
    GlobalBuildQueueAdapter adapter = new GlobalBuildQueueAdapter(globalBuildQueue, List.of("a"));
    assertThatThrownBy(adapter::getConfig).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.awaitAndClaimTask(db)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.completeTask(null, null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.failTask(null, null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.extendTtl(null, null, Duration.ofSeconds(1)))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.isEmpty(null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.hasVisibleUnclaimedTasks(null))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.hasClaimedTasks(null)).isInstanceOf(UnsupportedOperationException.class);
  }

  // ---- Test 1b: Maintenance adapter unsupported methods ----
  @Test
  void maintAdapter_unsupportedMethods_throwUnsupportedOperationException() {
    GlobalMaintenanceQueueAdapter adapter = new GlobalMaintenanceQueueAdapter(globalMaintQueue, List.of("a"));
    assertThatThrownBy(adapter::getConfig).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.awaitAndClaimTask(db)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.completeTask(null, null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.failTask(null, null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.extendTtl(null, null, Duration.ofSeconds(1)))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.isEmpty(null)).isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.hasVisibleUnclaimedTasks(null))
        .isInstanceOf(UnsupportedOperationException.class);
    assertThatThrownBy(() -> adapter.hasClaimedTasks(null)).isInstanceOf(UnsupportedOperationException.class);
  }

  // ---- Test 2: Maintenance adapter awaitQueueEmpty throws ----
  @Test
  void maintAdapter_awaitQueueEmpty_throwsUnsupportedOperationException() {
    GlobalMaintenanceQueueAdapter adapter = new GlobalMaintenanceQueueAdapter(globalMaintQueue, List.of("a"));
    assertThatThrownBy(() -> adapter.awaitQueueEmpty(db)).isInstanceOf(UnsupportedOperationException.class);
  }

  // ---- Test 3: Build adapter awaitQueueEmpty delegates ----
  @Test
  void buildAdapter_awaitQueueEmpty_delegatesToGlobalQueue() throws Exception {
    GlobalBuildQueueAdapter adapter = new GlobalBuildQueueAdapter(globalBuildQueue, List.of("a"));
    // Queue is empty, so awaitQueueEmpty should complete immediately
    adapter.awaitQueueEmpty(db).get(5, TimeUnit.SECONDS);
  }

  // ---- Test 4: Adapter enqueue methods wrap correctly and claim verifies wrapping ----
  @Test
  void buildAdapter_enqueueAndClaim_wrapsCorrectly() throws Exception {
    GlobalBuildQueueAdapter adapter = new GlobalBuildQueueAdapter(globalBuildQueue, List.of("idx", "path"));
    BuildTask bt = BuildTask.newBuilder().setSegId(42).build();
    // Enqueue via simple overload
    db.runAsync(tr -> adapter.enqueueIfNotExists(tr, "key1", bt)).get(5, TimeUnit.SECONDS);
    // Claim from global queue and verify wrapping
    var claim = globalBuildQueue.awaitAndClaimTask(db).get(10, TimeUnit.SECONDS);
    GlobalBuildTask gbt = claim.task();
    assertThat(gbt.getIndexPathList()).containsExactly("idx", "path");
    assertThat(gbt.getTask().getSegId()).isEqualTo(42);
    claim.complete().get(5, TimeUnit.SECONDS);
  }

  // ---- Test 5: Adapter enqueue wraps correctly ----
  @Test
  void buildAdapter_enqueue_wrapsCorrectly() throws Exception {
    GlobalBuildQueueAdapter adapter = new GlobalBuildQueueAdapter(globalBuildQueue, List.of("my", "index"));
    BuildTask bt = BuildTask.newBuilder().setSegId(7).build();
    db.runAsync(tr -> adapter.enqueue(tr, "key2", bt, Duration.ofSeconds(10), Duration.ofMillis(50)))
        .get(5, TimeUnit.SECONDS);
    var claim = globalBuildQueue.awaitAndClaimTask(db).get(10, TimeUnit.SECONDS);
    assertThat(claim.task().getIndexPathList()).containsExactly("my", "index");
    assertThat(claim.task().getTask().getSegId()).isEqualTo(7);
    claim.complete().get(5, TimeUnit.SECONDS);
  }

  // ---- Test 5b: Maintenance adapter enqueue wraps correctly ----
  @Test
  void maintAdapter_enqueueIfNotExists_wrapsCorrectly() throws Exception {
    GlobalMaintenanceQueueAdapter adapter =
        new GlobalMaintenanceQueueAdapter(globalMaintQueue, List.of("m", "idx"));
    MaintenanceTask mt = MaintenanceTask.newBuilder()
        .setVacuum(MaintenanceTask.Vacuum.newBuilder().setSegId(3).build())
        .build();
    db.runAsync(tr -> adapter.enqueueIfNotExists(tr, "mkey", mt)).get(5, TimeUnit.SECONDS);
    var claim = globalMaintQueue.awaitAndClaimTask(db).get(10, TimeUnit.SECONDS);
    assertThat(claim.task().getIndexPathList()).containsExactly("m", "idx");
    assertThat(claim.task().getTask().getVacuum().getSegId()).isEqualTo(3);
    claim.complete().get(5, TimeUnit.SECONDS);
  }

  // ---- Test 6: Adapter constructors reject nulls ----
  @Test
  void buildAdapter_nullGlobalQueue_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalBuildQueueAdapter(null, List.of("a")))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void buildAdapter_nullIndexPath_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalBuildQueueAdapter(globalBuildQueue, null))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void maintAdapter_nullGlobalQueue_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalMaintenanceQueueAdapter(null, List.of("a")))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  void maintAdapter_nullIndexPath_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalMaintenanceQueueAdapter(globalMaintQueue, null))
        .isInstanceOf(NullPointerException.class);
  }

  // ---- Test 7: start(0,0) is no-op ----
  @Test
  void workerRunner_startZeroZero_isNoOp() {
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    runner.start(0, 0);
    assertThat(runner.isRunning()).isFalse();
  }

  // ---- Test 8: double start is idempotent ----
  @Test
  void workerRunner_doubleStart_isIdempotent() {
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    runner.start(1, 0);
    assertThat(runner.isRunning()).isTrue();
    // Second start should be no-op
    runner.start(2, 2);
    assertThat(runner.isRunning()).isTrue();
  }

  // ---- Test 9: close sets isRunning to false ----
  @Test
  void workerRunner_close_setsIsRunningFalse() {
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    runner.start(1, 1);
    assertThat(runner.isRunning()).isTrue();
    runner.close();
    assertThat(runner.isRunning()).isFalse();
    runner = null; // prevent double-close in teardown
  }

  // ---- Test 10: Build sentinel is claimed and completed ----
  @Test
  void workerRunner_buildSentinel_claimedAndCompleted() throws Exception {
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    // Enqueue a build sentinel
    GlobalBuildTask sentinel = GlobalBuildTask.newBuilder()
        .setTask(BuildTask.newBuilder().setSegId(-1).build())
        .build();
    globalBuildQueue.enqueueIfNotExists("sentinel-build", sentinel).get(5, TimeUnit.SECONDS);
    // runOnceBuild should claim and complete it without error
    Boolean result = runner.runOnceBuild().get(5, TimeUnit.SECONDS);
    assertThat(result).isTrue();
  }

  // ---- Test 11: Maintenance sentinel is claimed and completed ----
  @Test
  void workerRunner_maintSentinel_claimedAndCompleted() throws Exception {
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    // Enqueue a maintenance sentinel
    GlobalMaintenanceTask sentinel = GlobalMaintenanceTask.newBuilder()
        .setTask(MaintenanceTask.newBuilder()
            .setVacuum(
                MaintenanceTask.Vacuum.newBuilder().setSegId(-1).build())
            .build())
        .build();
    globalMaintQueue.enqueueIfNotExists("sentinel-maint", sentinel).get(5, TimeUnit.SECONDS);
    Boolean result = runner.runOnceMaint().get(5, TimeUnit.SECONDS);
    assertThat(result).isTrue();
  }

  // ---- Test 12: GlobalTaskQueueConfig rejects null buildQueue ----
  @Test
  void globalTaskQueueConfig_nullBuildQueue_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalTaskQueueConfig(null, globalMaintQueue))
        .isInstanceOf(NullPointerException.class);
  }

  // ---- Test 13: GlobalTaskQueueConfig rejects null maintenanceQueue ----
  @Test
  void globalTaskQueueConfig_nullMaintenanceQueue_throwsNullPointerException() {
    assertThatThrownBy(() -> new GlobalTaskQueueConfig(globalBuildQueue, null))
        .isInstanceOf(NullPointerException.class);
  }

  // ---- Test 14: isGlobalTaskQueueEnabled returns false by default ----
  @Test
  void vectorIndexConfig_isGlobalTaskQueueEnabled_falseByDefault() {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root).dimension(4).build();
    assertThat(cfg.isGlobalTaskQueueEnabled()).isFalse();
    assertThat(cfg.getGlobalTaskQueueConfig()).isNull();
  }

  // ---- Test 15: isGlobalTaskQueueEnabled returns true when set ----
  @Test
  void vectorIndexConfig_isGlobalTaskQueueEnabled_trueWhenSet() {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .globalTaskQueueConfig(globalConfig)
        .build();
    assertThat(cfg.isGlobalTaskQueueEnabled()).isTrue();
    assertThat(cfg.getGlobalTaskQueueConfig()).isSameAs(globalConfig);
  }

  // ---- Test 16: GlobalBuildTaskSerializer round-trip ----
  @Test
  void globalBuildTaskSerializer_roundTrip() {
    ProtoSerializers.GlobalBuildTaskSerializer serializer = new ProtoSerializers.GlobalBuildTaskSerializer();
    GlobalBuildTask original = GlobalBuildTask.newBuilder()
        .addAllIndexPath(List.of("test", "index"))
        .setTask(BuildTask.newBuilder().setSegId(99).build())
        .build();
    ByteString bytes = serializer.serialize(original);
    GlobalBuildTask deserialized = serializer.deserialize(bytes);
    assertThat(deserialized.getIndexPathList()).containsExactly("test", "index");
    assertThat(deserialized.getTask().getSegId()).isEqualTo(99);
  }

  // ---- Test 17: GlobalMaintenanceTaskSerializer round-trip ----
  @Test
  void globalMaintenanceTaskSerializer_roundTrip() {
    ProtoSerializers.GlobalMaintenanceTaskSerializer serializer =
        new ProtoSerializers.GlobalMaintenanceTaskSerializer();
    GlobalMaintenanceTask original = GlobalMaintenanceTask.newBuilder()
        .addAllIndexPath(List.of("maint", "path"))
        .setTask(MaintenanceTask.newBuilder()
            .setVacuum(MaintenanceTask.Vacuum.newBuilder()
                .setSegId(5)
                .setMinDeletedRatio(0.3)
                .build())
            .build())
        .build();
    ByteString bytes = serializer.serialize(original);
    GlobalMaintenanceTask deserialized = serializer.deserialize(bytes);
    assertThat(deserialized.getIndexPathList()).containsExactly("maint", "path");
    assertThat(deserialized.getTask().getVacuum().getSegId()).isEqualTo(5);
    assertThat(deserialized.getTask().getVacuum().getMinDeletedRatio()).isEqualTo(0.3);
  }

  // ---- Test 18: Serializers handle null input ----
  @Test
  void globalBuildTaskSerializer_nullInput_returnsEmpty() {
    ProtoSerializers.GlobalBuildTaskSerializer serializer = new ProtoSerializers.GlobalBuildTaskSerializer();
    ByteString bytes = serializer.serialize(null);
    assertThat(bytes).isEqualTo(ByteString.EMPTY);
  }

  @Test
  void globalMaintenanceTaskSerializer_nullInput_returnsEmpty() {
    ProtoSerializers.GlobalMaintenanceTaskSerializer serializer =
        new ProtoSerializers.GlobalMaintenanceTaskSerializer();
    ByteString bytes = serializer.serialize(null);
    assertThat(bytes).isEqualTo(ByteString.EMPTY);
  }

  @Test
  void globalBuildTaskSerializer_nullDeserialize_returnsDefault() {
    ProtoSerializers.GlobalBuildTaskSerializer serializer = new ProtoSerializers.GlobalBuildTaskSerializer();
    GlobalBuildTask result = serializer.deserialize(null);
    assertThat(result).isEqualTo(GlobalBuildTask.getDefaultInstance());
  }

  @Test
  void globalMaintenanceTaskSerializer_nullDeserialize_returnsDefault() {
    ProtoSerializers.GlobalMaintenanceTaskSerializer serializer =
        new ProtoSerializers.GlobalMaintenanceTaskSerializer();
    GlobalMaintenanceTask result = serializer.deserialize(null);
    assertThat(result).isEqualTo(GlobalMaintenanceTask.getDefaultInstance());
  }

  // ---- Test 19: Config reconstruction reads IndexMeta data params ----
  @Test
  void workerRunner_configReconstruction_usesIndexMetaDataParams() throws Exception {
    // Create an index with specific params
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(16)
        .metric(VectorIndexConfig.Metric.COSINE)
        .pqM(8)
        .pqK(32)
        .graphDegree(64)
        .maxSegmentSize(2)
        .oversample(3)
        .globalTaskQueueConfig(globalConfig)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    // Insert enough vectors to trigger rotation (maxSegmentSize=2)
    index.add(new float[] {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, null)
        .get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, null)
        .get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, null)
        .get(5, TimeUnit.SECONDS);

    // Verify IndexMeta was written with correct params
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    byte[] metaBytes = db.readAsync(tr -> tr.get(dirs.metaKey())).get(5, TimeUnit.SECONDS);
    assertThat(metaBytes).isNotNull();
    IndexMeta meta = IndexMeta.parseFrom(metaBytes);
    assertThat(meta.getDimension()).isEqualTo(16);
    assertThat(meta.getMetric()).isEqualTo(IndexMeta.Metric.METRIC_COSINE);
    assertThat(meta.getPqM()).isEqualTo(8);
    assertThat(meta.getPqK()).isEqualTo(32);
    assertThat(meta.getGraphDegree()).isEqualTo(64);

    // Create a template with different data params — worker should use IndexMeta's params
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .estimatedWorkerCount(1)
        .defaultTtl(Duration.ofSeconds(30))
        .build();
    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    // Process the build task — should succeed using IndexMeta's dimension=16
    runner.runOnceBuild().get(30, TimeUnit.SECONDS);
  }

  // ---- Test 20: Missing IndexMeta fails the task, doesn't crash ----
  @Test
  void workerRunner_missingIndexMeta_failsTaskGracefully() throws Exception {
    // Create a directory but don't write IndexMeta
    DirectorySubspace emptyDir = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-edge-test", "empty-" + UUID.randomUUID()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    try {
      // Open index dirs so the directory structure exists
      FdbDirectories.openIndex(emptyDir, db).get(5, TimeUnit.SECONDS);

      // Enqueue a build task pointing to this empty directory
      GlobalBuildTask task = GlobalBuildTask.newBuilder()
          .addAllIndexPath(emptyDir.getPath())
          .setTask(BuildTask.newBuilder().setSegId(0).build())
          .build();
      globalBuildQueue.enqueueIfNotExists("missing-meta-task", task).get(5, TimeUnit.SECONDS);

      VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
          .dimension(4)
          .pqM(2)
          .pqK(8)
          .graphDegree(4)
          .build();
      runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);

      // runOnceBuild should fail the task (not crash the runner)
      // The task will be claimed, fail due to missing IndexMeta, and be marked failed
      Boolean result = runner.runOnceBuild().get(10, TimeUnit.SECONDS);
      assertThat(result).isTrue(); // Task was processed (claimed + failed)
    } finally {
      db.run(tr -> {
        emptyDir.remove(tr);
        return null;
      });
    }
  }
}
