package io.github.panghy.vectorsearch.bench;

import com.apple.foundationdb.Database;
import com.apple.foundationdb.FDB;
import com.apple.foundationdb.directory.DirectoryLayer;
import com.apple.foundationdb.directory.DirectorySubspace;
import io.github.panghy.vectorsearch.api.SearchParams;
import io.github.panghy.vectorsearch.api.SearchResult;
import io.github.panghy.vectorsearch.api.VectorIndex;
import io.github.panghy.vectorsearch.config.VectorIndexConfig;
import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

/**
 * JMH benchmark for {@link VectorIndex#query} latency across search modes and k values.
 *
 * <p>Starts an FDB 7.3 Docker container in {@code @Setup(Level.Trial)}, inserts 1000+ vectors
 * with {@code maxSegmentSize=500} to produce sealed segments, waits for indexing to complete,
 * then measures search latency.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 3)
@Fork(value = 1)
public class VectorIndexSearchBenchmark {

  private static final String CONTAINER_NAME = "vs-bench-fdb";
  private static final String FDB_IMAGE = "foundationdb/foundationdb:7.3.27";
  private static final int FDB_PORT = 4500;

  private Database db;
  private DirectorySubspace root;
  private VectorIndex index;
  private Path clusterFilePath;
  private volatile boolean setupFailed = false;

  @Param({"BEST_FIRST", "BEAM"})
  public String mode;

  @Param({"1", "10", "50"})
  public int k;

  private float[] query;

  @Setup(Level.Trial)
  public void setup() throws Exception {
    try {
      startFdbContainer();
      db = FDB.selectAPIVersion(730).open(clusterFilePath.toString());
      root = db.runAsync(tr -> DirectoryLayer.getDefault()
              .createOrOpen(
                  tr,
                  List.of("vs-bench", UUID.randomUUID().toString()),
                  "vectorsearch".getBytes(StandardCharsets.UTF_8)))
          .get(10, TimeUnit.SECONDS);
      VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
          .dimension(8)
          .pqM(4)
          .pqK(16)
          .graphDegree(32)
          .maxSegmentSize(500)
          .localWorkerThreads(1)
          .build();
      index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

      // Insert 1000+ vectors so we get at least 2 segments (500 each).
      for (int i = 0; i < 1100; i++) {
        float[] v = new float[8];
        for (int d = 0; d < 8; d++) {
          v[d] = (float) Math.sin(0.01 * i + d);
        }
        index.add(v, null).get(5, TimeUnit.SECONDS);
      }

      // Wait for background builders to seal segments.
      index.awaitIndexingComplete().get(120, TimeUnit.SECONDS);

      query = new float[] {1f, 0.5f, -0.5f, 0.2f, -0.1f, 0.3f, 0.0f, 0.8f};
    } catch (Exception e) {
      setupFailed = true;
      throw e;
    }
  }

  @TearDown(Level.Trial)
  public void teardown() {
    try {
      if (root != null && db != null) {
        db.run(tr -> {
          root.remove(tr);
          return null;
        });
      }
    } catch (Exception e) {
      System.err.println("Warning: failed to clean up directory: " + e.getMessage());
    }
    if (index != null) {
      index.close();
    }
    if (db != null) {
      db.close();
    }
    stopFdbContainer();
    if (clusterFilePath != null) {
      try {
        Files.deleteIfExists(clusterFilePath);
      } catch (Exception ignored) {
        // best-effort cleanup
      }
    }
  }

  @Benchmark
  public List<SearchResult> search() throws Exception {
    if (setupFailed) {
      throw new RuntimeException("Setup failed — aborting benchmark");
    }
    SearchParams.Mode m = SearchParams.Mode.valueOf(mode);
    SearchParams params = SearchParams.of(64, 32, 4, m);
    return index.query(query, k, params).get(5, TimeUnit.SECONDS);
  }

  // --------------- Docker helpers ---------------

  private void startFdbContainer() throws Exception {
    // Stop any leftover container from a previous run.
    exec("docker", "rm", "-f", CONTAINER_NAME);

    // Start a fresh FDB 7.3 container.
    exec("docker", "run", "-d", "--name", CONTAINER_NAME, "-p", FDB_PORT + ":4500", FDB_IMAGE);

    // Initialize the database — required for FDB 7.3 containers.
    // Wait a few seconds for fdbmonitor to start the fdbserver process.
    Thread.sleep(3000);
    exec("docker", "exec", CONTAINER_NAME, "fdbcli", "--exec", "configure new single memory");

    // Wait for FDB to become ready (up to 60 seconds).
    waitForFdbReady(60);

    // Copy the cluster file from the container and rewrite the internal IP to 127.0.0.1.
    clusterFilePath = Files.createTempFile("fdb-bench-", ".cluster");
    exec("docker", "cp", CONTAINER_NAME + ":/var/fdb/fdb.cluster", clusterFilePath.toString());
    String clusterContent = Files.readString(clusterFilePath, StandardCharsets.UTF_8);
    // Replace the container-internal IP (e.g. 172.x.x.x) with 127.0.0.1 for host access.
    clusterContent = clusterContent.replaceAll("@[0-9.]+:", "@127.0.0.1:");
    Files.writeString(clusterFilePath, clusterContent, StandardCharsets.UTF_8);
  }

  private void stopFdbContainer() {
    try {
      exec("docker", "rm", "-f", CONTAINER_NAME);
    } catch (Exception e) {
      System.err.println("Warning: failed to stop FDB container: " + e.getMessage());
    }
  }

  private void waitForFdbReady(int timeoutSeconds) throws Exception {
    long deadline = System.currentTimeMillis() + timeoutSeconds * 1000L;
    while (System.currentTimeMillis() < deadline) {
      try {
        ProcessBuilder pb =
            new ProcessBuilder("docker", "exec", CONTAINER_NAME, "fdbcli", "--exec", "status minimal");
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String output;
        try (BufferedReader br =
            new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
          output = br.lines().reduce("", (a, b) -> a + "\n" + b);
        }
        int exit = p.waitFor();
        if (exit == 0 && output.contains("healthy")) {
          return;
        }
      } catch (Exception ignored) {
        // container not ready yet
      }
      Thread.sleep(1000);
    }
    throw new IllegalStateException("FDB container did not become ready within " + timeoutSeconds + " seconds");
  }

  private static void exec(String... cmd) throws Exception {
    ProcessBuilder pb = new ProcessBuilder(cmd);
    pb.redirectErrorStream(true);
    pb.directory(new File(System.getProperty("user.dir")));
    Process p = pb.start();
    // Drain output to prevent blocking.
    try (BufferedReader br =
        new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
      while (br.readLine() != null) {
        // discard
      }
    }
    p.waitFor(30, TimeUnit.SECONDS);
  }
}
