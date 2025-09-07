package io.github.panghy.vectorsearch.bench;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.api.SearchParams;
import io.github.panghy.vectorsearch.api.SearchResult;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

@State(Scope.Benchmark)
@Warmup(iterations = 3)
@Measurement(iterations = 5)
@Fork(value = 1)
public class VectorIndexSearchBenchmark {
  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;

  @Param({"BEAM", "BEST_FIRST"})
  public String mode;

  @Param({"10"})
  public int k;

  private float[] query;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    db = FDB.selectAPIVersion(730).open();
    root = db.runAsync(tr -> DirectoryLayer.getDefault()
            .createOrOpen(
                tr,
                List.of("vs-bench", UUID.randomUUID().toString()),
                "vectorsearch".getBytes(StandardCharsets.UTF_8)))
        .get(5, TimeUnit.SECONDS);
    VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
        .dimension(8)
        .pqM(4)
        .pqK(16)
        .graphDegree(32)
        .maxSegmentSize(500)
        .localWorkerThreads(1)
        .build();
    index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
    for (int i = 0; i < 1000; i++) {
      float[] v = new float[8];
      for (int d = 0; d < 8; d++) v[d] = (float) Math.sin(0.01 * i + d);
      index.add(v, null).get(5, TimeUnit.SECONDS);
    }
    // allow sealing
    Thread.sleep(1500);
    query = new float[] {1f, 0.5f, -0.5f, 0.2f, -0.1f, 0.3f, 0.0f, 0.8f};
  }

  @TearDown(Level.Trial)
  public void teardown() {
    db.run(tr -> {
      root.remove(tr);
      return null;
    });
    index.close();
    db.close();
  }

  @Benchmark
  public List<SearchResult> search() throws Exception {
    SearchParams.Mode m = SearchParams.Mode.valueOf(mode);
    SearchParams params = SearchParams.of(64, 32, 4, m);
    return index.query(query, k, params).get(5, TimeUnit.SECONDS);
  }
}
