package io.github.panghy.vectorsearch.tasks;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.fdb.FdbDirectories;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class CompactionPlannerAndThrottlingTest {
  Database db;
  DirectorySubspace root;

  @BeforeEach
  void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-plan-throttle", UUID.randomUUID().toString()),
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
  void planner_applies_age_bias_on_ties() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    // Create three sealed segments with equal count but different createdAtMs
    db.run(tr -> {
      try {
        for (int sid = 0; sid < 3; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          SegmentMeta sm = SegmentMeta.newBuilder()
              .setSegmentId(sid)
              .setState(SegmentMeta.State.SEALED)
              .setCount(3)
              .setCreatedAtMs(1000L + sid) // older = smaller ms
              .build();
          tr.set(sk.metaKey(), sm.toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    // Anchor at sid=2; expect pick includes [2,0] (older 0 preferred over 1 on tie)
    List<Integer> pick = svc.findCompactionCandidates(2).get(5, TimeUnit.SECONDS);
    assertThat(pick).contains(2);
    // Ensure the next chosen (same counts) is the oldest (sid=0)
    assertThat(pick.get(1)).isEqualTo(0);
  }

  @Test
  void throttling_limits_inflight_compactions() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(10)
        .maxConcurrentCompactions(0)
        .localWorkerThreads(0)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    // Create two sealed segments
    db.run(tr -> {
      try {
        for (int sid = 0; sid < 2; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          SegmentMeta sm = SegmentMeta.newBuilder()
              .setSegmentId(sid)
              .setState(SegmentMeta.State.SEALED)
              .setCount(3)
              .setCreatedAtMs(1000L + sid)
              .build();
          tr.set(sk.metaKey(), sm.toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    // Enqueue a find-candidates task, run worker once; since maxConcurrentCompactions=0, it should no-op
    var maintDir = dirs.tasksDir().createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);
    var tqc = io.github.panghy.taskqueue.TaskQueueConfig.builder(
            db,
            maintDir,
            new ProtoSerializers.StringSerializer(),
            new ProtoSerializers.MaintenanceTaskSerializer())
        .build();
    var queue = io.github.panghy.taskqueue.TaskQueues.createTaskQueue(tqc).get(5, TimeUnit.SECONDS);
    MaintenanceTask.FindCompactionCandidates fcc = MaintenanceTask.FindCompactionCandidates.newBuilder()
        .setAnchorSegId(0)
        .build();
    MaintenanceTask mt = MaintenanceTask.newBuilder().setFindCandidates(fcc).build();
    queue.enqueue("find-candidates:0", mt).get(5, TimeUnit.SECONDS);
    boolean claimed = new MaintenanceWorker(cfg, dirs, queue).runOnce().get(5, TimeUnit.SECONDS);
    assertThat(claimed).isTrue();
    // Ensure no segments moved to COMPACTING
    int inflight =
        new MaintenanceService(cfg, dirs).countInFlightCompactions().get(5, TimeUnit.SECONDS);
    assertThat(inflight).isEqualTo(0);
  }
}
