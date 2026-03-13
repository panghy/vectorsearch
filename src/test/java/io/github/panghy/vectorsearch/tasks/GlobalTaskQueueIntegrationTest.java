package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.taskqueue.TaskQueue;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.taskqueue.TaskQueues;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.GlobalTaskQueueConfig;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.fdb.FdbVectorIndex;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Integration tests for the global task queue feature.
 *
 * <p>Verifies end-to-end flows: inserting vectors with global queues enabled routes build/maintenance
 * tasks to shared global queues, {@link GlobalWorkerRunner} processes them, and multiple indices
 * can share the same global queues. Also verifies that local worker pools are not started when
 * global mode is enabled.</p>
 */
class GlobalTaskQueueIntegrationTest {

  private Database db;
  private DirectorySubspace root;
  private DirectorySubspace globalDir;
  private TaskQueue<String, GlobalBuildTask> globalBuildQueue;
  private TaskQueue<String, GlobalMaintenanceTask> globalMaintQueue;
  private GlobalTaskQueueConfig globalConfig;
  private VectorIndex index;
  private VectorIndex index2;
  private GlobalWorkerRunner runner;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    String uid = UUID.randomUUID().toString();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr, List.of("vs-global-test", uid), "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);

    // Create global queue directories
    globalDir = db.runAsync(tr -> DirectoryLayer.getDefault().createOrOpen(tr, List.of("vs-global-queues", uid)))
        .get(5, TimeUnit.SECONDS);

    DirectorySubspace buildDir =
        globalDir.createOrOpen(db, List.of("build")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<String, GlobalBuildTask> buildTqc = TaskQueueConfig.builder(
            db,
            buildDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.GlobalBuildTaskSerializer())
        .build();
    globalBuildQueue = TaskQueues.createTaskQueue(buildTqc).get(5, TimeUnit.SECONDS);

    DirectorySubspace maintDir =
        globalDir.createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    TaskQueueConfig<String, GlobalMaintenanceTask> maintTqc = TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.GlobalMaintenanceTaskSerializer())
        .build();
    globalMaintQueue = TaskQueues.createTaskQueue(maintTqc).get(5, TimeUnit.SECONDS);

    globalConfig = new GlobalTaskQueueConfig(globalBuildQueue, globalMaintQueue);
  }

  @AfterEach
  void teardown() {
    if (runner != null) runner.close();
    if (index != null) index.close();
    if (index2 != null) index2.close();
    if (db != null) {
      db.run(tr -> {
        root.remove(tr);
        globalDir.remove(tr);
        return null;
      });
      db.close();
    }
  }

  @Test
  void insertVectors_segmentRotation_buildTaskAppearsInGlobalQueue() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .globalTaskQueueConfig(globalConfig)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    // Insert 2 vectors to fill seg 0, then 1 more to trigger rotation
    index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS);

    // Claim the build task from the global queue and verify it has the correct index path
    var claim = globalBuildQueue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    GlobalBuildTask gbt = claim.task();
    assertThat(gbt.getTask().getSegId()).isEqualTo(0);
    assertThat(gbt.getIndexPathList()).isEqualTo(root.getPath());
    claim.complete().get(5, TimeUnit.SECONDS);
  }

  @Test
  void globalWorkerRunner_buildsAndSealsSegment() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .globalTaskQueueConfig(globalConfig)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    // Template config for the worker
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
    runner.start(1, 0);

    // Insert vectors to trigger rotation
    index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS);

    // Wait for the global build queue to drain
    index.awaitIndexingComplete().get(30, TimeUnit.SECONDS);

    // Verify segment 0 is SEALED
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    SegmentMeta seg0 = db.readAsync(tr -> dirs.segmentKeys(tr, 0)
            .thenCompose(sk -> tr.get(sk.metaKey()))
            .thenApply(b -> {
              try {
                return SegmentMeta.parseFrom(b);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(seg0.getState()).isEqualTo(SegmentMeta.State.SEALED);
  }

  @Test
  void deleteVectors_vacuumTaskInGlobalMaintenanceQueue() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(10)
        .vacuumMinDeletedRatio(0.0)
        .globalTaskQueueConfig(globalConfig)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

    // Insert vectors and seal segment 0 manually
    long g0 = index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    new SegmentBuildService(cfg, dirs).build(0).get(5, TimeUnit.SECONDS);

    // Delete a vector — should enqueue vacuum into global maintenance queue
    index.delete(g0).get(5, TimeUnit.SECONDS);

    // Claim the maintenance task and verify it's a vacuum for seg 0
    var claim = globalMaintQueue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    GlobalMaintenanceTask gmt = claim.task();
    assertThat(gmt.getTask().hasVacuum()).isTrue();
    assertThat(gmt.getTask().getVacuum().getSegId()).isEqualTo(0);
    assertThat(gmt.getIndexPathList()).isEqualTo(root.getPath());

    // Now start a GlobalWorkerRunner to process it
    VectorIndexConfig templateCfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .build();
    // Fail the claim so the worker can re-claim it
    claim.fail().get(5, TimeUnit.SECONDS);

    runner = new GlobalWorkerRunner(db, templateCfg, globalConfig);
    // Process the maintenance task via runOnceMaint
    runner.runOnceMaint().get(30, TimeUnit.SECONDS);
  }

  @Test
  void twoIndices_sharedGlobalQueues_bothServiced() throws Exception {
    // Create two separate index directories
    DirectorySubspace root2 = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-global-test", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);

    VectorIndexConfig cfg1 = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .globalTaskQueueConfig(globalConfig)
        .build();
    VectorIndexConfig cfg2 = VectorIndexConfig.builder(db, root2)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .globalTaskQueueConfig(globalConfig)
        .build();

    index = VectorIndex.createOrOpen(cfg1).get(10, TimeUnit.SECONDS);
    index2 = VectorIndex.createOrOpen(cfg2).get(10, TimeUnit.SECONDS);

    // Create a global worker (not started as a loop — we'll call runOnceBuild directly)
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

    // Trigger rotation on index 1 first, process its build task
    index.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS);
    runner.runOnceBuild().get(30, TimeUnit.SECONDS);

    // Verify index 1 segment 0 is SEALED
    var dirs1 = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    SegmentMeta seg0_1 = db.readAsync(tr -> dirs1.segmentKeys(tr, 0)
            .thenCompose(sk -> tr.get(sk.metaKey()))
            .thenApply(b -> {
              try {
                return SegmentMeta.parseFrom(b);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(seg0_1.getState()).isEqualTo(SegmentMeta.State.SEALED);

    // Now trigger rotation on index 2 and process its build task
    index2.add(new float[] {1, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    index2.add(new float[] {0, 1, 1, 0}, null).get(5, TimeUnit.SECONDS);
    index2.add(new float[] {0, 0, 1, 1}, null).get(5, TimeUnit.SECONDS);
    runner.runOnceBuild().get(30, TimeUnit.SECONDS);

    // Verify index 2 segment 0 is SEALED
    var dirs2 = FdbDirectories.openIndex(root2, db).get(5, TimeUnit.SECONDS);
    SegmentMeta seg0_2 = db.readAsync(tr -> dirs2.segmentKeys(tr, 0)
            .thenCompose(sk -> tr.get(sk.metaKey()))
            .thenApply(b -> {
              try {
                return SegmentMeta.parseFrom(b);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(seg0_2.getState()).isEqualTo(SegmentMeta.State.SEALED);

    // Clean up root2
    db.run(tr -> {
      root2.remove(tr);
      return null;
    });
  }

  @Test
  void globalModeEnabled_noLocalWorkerPoolsStarted() throws Exception {
    // Create index with global mode AND localWorkerThreads > 0
    // The global mode should prevent local pools from starting
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(2)
        .localWorkerThreads(2)
        .localMaintenanceWorkerThreads(2)
        .globalTaskQueueConfig(globalConfig)
        .build();
    FdbVectorIndex fdbIndex = FdbVectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    index = fdbIndex;

    // Insert vectors to trigger rotation — if local pools were started, they would
    // try to claim from the adapter queues and throw UnsupportedOperationException
    fdbIndex.add(new float[] {1, 0, 0, 0}, null).get(5, TimeUnit.SECONDS);
    fdbIndex.add(new float[] {0, 1, 0, 0}, null).get(5, TimeUnit.SECONDS);
    fdbIndex.add(new float[] {0, 0, 1, 0}, null).get(5, TimeUnit.SECONDS);

    // Verify the build task is in the global queue (not consumed by local workers)
    var claim = globalBuildQueue.awaitAndClaimTask(db).get(5, TimeUnit.SECONDS);
    assertThat(claim.task().getTask().getSegId()).isEqualTo(0);
    claim.complete().get(5, TimeUnit.SECONDS);

    // Close should not throw — no local pools to shut down
    fdbIndex.close();
    index = null; // prevent double-close in teardown
  }
}
