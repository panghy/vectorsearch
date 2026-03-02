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

/**
 * Tests for the compaction planner's weighted scoring heuristic and throttling behavior.
 *
 * <p>Verifies that {@link MaintenanceService#findCompactionCandidates(int)} uses a composite
 * score (age, size, fragmentation) to rank candidates, respects configurable min/max segment
 * limits, and honours the minimum fragmentation threshold.</p>
 */
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
    // Disable fragmentation threshold so zero-deleted segments are still eligible
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
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
    // Anchor at sid=2; with age bias, older segments score higher
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
        .compactionMinFragmentation(0.0)
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

  @Test
  void weighted_scoring_prefers_fragmented_small_old_segments() throws Exception {
    // Weight fragmentation heavily to verify it influences selection
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(100)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
        .compactionAgeBiasWeight(0.1)
        .compactionSizeBiasWeight(0.1)
        .compactionFragBiasWeight(0.8)
        .compactionMaxSegments(3)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        // seg0: large, new, no fragmentation
        var sk0 = dirs.segmentKeys(tr, 0).get();
        tr.set(
            sk0.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(0)
                .setState(SegmentMeta.State.SEALED)
                .setCount(50)
                .setDeletedCount(0)
                .setCreatedAtMs(3000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(0), new byte[0]);
        // seg1: small, old, high fragmentation (best candidate)
        var sk1 = dirs.segmentKeys(tr, 1).get();
        tr.set(
            sk1.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(1)
                .setState(SegmentMeta.State.SEALED)
                .setCount(5)
                .setDeletedCount(15)
                .setCreatedAtMs(1000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(1), new byte[0]);
        // seg2: medium, medium age, some fragmentation
        var sk2 = dirs.segmentKeys(tr, 2).get();
        tr.set(
            sk2.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(2)
                .setState(SegmentMeta.State.SEALED)
                .setCount(20)
                .setDeletedCount(5)
                .setCreatedAtMs(2000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(2), new byte[0]);
        // seg3: small, new, no fragmentation
        var sk3 = dirs.segmentKeys(tr, 3).get();
        tr.set(
            sk3.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(3)
                .setState(SegmentMeta.State.SEALED)
                .setCount(5)
                .setDeletedCount(0)
                .setCreatedAtMs(3000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(3), new byte[0]);
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    // Anchor at seg0; with frag bias 0.8, seg1 (75% frag) should be picked first
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    assertThat(pick).contains(0); // anchor always present
    assertThat(pick).contains(1); // highest fragmentation
    // seg1 should be second (first after anchor) due to highest composite score
    assertThat(pick.get(1)).isEqualTo(1);
  }

  @Test
  void min_fragmentation_threshold_filters_clean_segments() throws Exception {
    // Set high min fragmentation; segments with no deletes should be rejected
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(100)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.5)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        for (int sid = 0; sid < 3; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          tr.set(
              sk.metaKey(),
              SegmentMeta.newBuilder()
                  .setSegmentId(sid)
                  .setState(SegmentMeta.State.SEALED)
                  .setCount(10)
                  .setDeletedCount(0)
                  .setCreatedAtMs(1000L + sid)
                  .build()
                  .toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    assertThat(pick).isEmpty();
  }

  @Test
  void configurable_max_segments_limits_candidates() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(1000)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
        .compactionMaxSegments(3)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        for (int sid = 0; sid < 6; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          tr.set(
              sk.metaKey(),
              SegmentMeta.newBuilder()
                  .setSegmentId(sid)
                  .setState(SegmentMeta.State.SEALED)
                  .setCount(10)
                  .setCreatedAtMs(1000L + sid)
                  .build()
                  .toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    assertThat(pick).hasSizeLessThanOrEqualTo(3);
  }

  @Test
  void min_segments_threshold_prevents_single_segment_compaction() throws Exception {
    // With compactionMinSegments=3, having only 2 sealed segments should return empty
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(100)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
        .compactionMinSegments(3)
        .compactionMaxSegments(8)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        for (int sid = 0; sid < 2; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          tr.set(
              sk.metaKey(),
              SegmentMeta.newBuilder()
                  .setSegmentId(sid)
                  .setState(SegmentMeta.State.SEALED)
                  .setCount(5)
                  .setCreatedAtMs(1000L + sid)
                  .build()
                  .toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    assertThat(pick).isEmpty();
  }

  @Test
  void size_bias_prefers_smaller_segments() throws Exception {
    // Weight size heavily; smaller segments should be picked first
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(1000)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
        .compactionAgeBiasWeight(0.0)
        .compactionSizeBiasWeight(1.0)
        .compactionFragBiasWeight(0.0)
        .compactionMaxSegments(3)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        // seg0: large (anchor)
        var sk0 = dirs.segmentKeys(tr, 0).get();
        tr.set(
            sk0.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(0)
                .setState(SegmentMeta.State.SEALED)
                .setCount(100)
                .setCreatedAtMs(1000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(0), new byte[0]);
        // seg1: medium
        var sk1 = dirs.segmentKeys(tr, 1).get();
        tr.set(
            sk1.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(1)
                .setState(SegmentMeta.State.SEALED)
                .setCount(50)
                .setCreatedAtMs(1000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(1), new byte[0]);
        // seg2: smallest
        var sk2 = dirs.segmentKeys(tr, 2).get();
        tr.set(
            sk2.metaKey(),
            SegmentMeta.newBuilder()
                .setSegmentId(2)
                .setState(SegmentMeta.State.SEALED)
                .setCount(10)
                .setCreatedAtMs(1000L)
                .build()
                .toByteArray());
        tr.set(dirs.segmentsIndexKey(2), new byte[0]);
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    // Anchor at seg0; with pure size bias, seg2 (smallest) should be picked next
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    assertThat(pick.get(0)).isEqualTo(0); // anchor
    assertThat(pick.get(1)).isEqualTo(2); // smallest
  }

  @Test
  void budget_threshold_stops_at_80_percent() throws Exception {
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(1)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
        .compactionMinFragmentation(0.0)
        .compactionMaxSegments(8)
        .build();
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    db.run(tr -> {
      try {
        // Create 5 segments with count=3 each; budget = 0.8*10 = 8
        // After picking 3 segments (count=9 >= 8), should stop
        for (int sid = 0; sid < 5; sid++) {
          var sk = dirs.segmentKeys(tr, sid).get();
          tr.set(
              sk.metaKey(),
              SegmentMeta.newBuilder()
                  .setSegmentId(sid)
                  .setState(SegmentMeta.State.SEALED)
                  .setCount(3)
                  .setCreatedAtMs(1000L + sid)
                  .build()
                  .toByteArray());
          tr.set(dirs.segmentsIndexKey(sid), new byte[0]);
        }
        return null;
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    });
    MaintenanceService svc = new MaintenanceService(cfg, dirs);
    List<Integer> pick = svc.findCompactionCandidates(0).get(5, TimeUnit.SECONDS);
    // Budget is 8; each segment has count=3. After 3 segments (sum=9>=8), should stop
    assertThat(pick).hasSize(3);
  }
}
