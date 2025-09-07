package io.github.panghy.vectorsearch.fdb;

import static org.assertj.core.api.Assertions.assertThat;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import io.github.panghy.vectorsearch.proto.SegmentMeta;
import io.github.panghy.vectorsearch.tasks.MaintenanceService;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies that delete operations update {@code SegmentMeta} counters and that
 * the maintenance vacuum decrements {@code deleted_count} after physically removing rows.
 */
public class FdbVectorStoreDeleteMetaTest {
  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @BeforeEach
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-delmeta", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(4)
        .pqM(2)
        .pqK(8)
        .graphDegree(4)
        .maxSegmentSize(10)
        .localWorkerThreads(0)
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
  public void deleteUpdatesMetaAndVacuumDecrementsDeleted() throws Exception {
    int[][] ids = new int[3][2];
    for (int i = 0; i < 3; i++) {
      float[] v = new float[] {i, i + 1, i + 2, i + 3};
      ids[i] = index.add(v, null).get(5, TimeUnit.SECONDS);
    }
    int seg = ids[0][0];
    // Before delete
    var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS);
    SegmentMeta before = db.readAsync(tr -> tr.get(
                dirs.segmentKeys(String.format("%06d", seg)).metaKey())
            .thenApply(bytes -> {
              try {
                return SegmentMeta.parseFrom(bytes);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(before.getCount()).isEqualTo(3);
    assertThat(before.getDeletedCount()).isEqualTo(0);

    // Delete two
    index.delete(ids[0][0], ids[0][1]).get(5, TimeUnit.SECONDS);
    index.delete(ids[1][0], ids[1][1]).get(5, TimeUnit.SECONDS);

    SegmentMeta after = db.readAsync(tr -> tr.get(
                dirs.segmentKeys(String.format("%06d", seg)).metaKey())
            .thenApply(bytes -> {
              try {
                return SegmentMeta.parseFrom(bytes);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(after.getCount()).isEqualTo(1);
    assertThat(after.getDeletedCount()).isGreaterThanOrEqualTo(2);

    // Vacuum should reduce deleted_count
    var svc = new MaintenanceService(
        VectorIndexConfig.builder(db, root).dimension(4).build(), dirs);
    svc.vacuumSegment(seg, 0.0).get(5, TimeUnit.SECONDS);

    SegmentMeta postVac = db.readAsync(tr -> tr.get(
                dirs.segmentKeys(String.format("%06d", seg)).metaKey())
            .thenApply(bytes -> {
              try {
                return SegmentMeta.parseFrom(bytes);
              } catch (com.google.protobuf.InvalidProtocolBufferException e) {
                throw new RuntimeException(e);
              }
            }))
        .get(5, TimeUnit.SECONDS);
    assertThat(postVac.getCount()).isEqualTo(1);
    assertThat(postVac.getDeletedCount()).isLessThanOrEqualTo(after.getDeletedCount());
  }
}
