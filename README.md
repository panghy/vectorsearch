# VectorSearch

A Java vector search library built on FoundationDB with per-segment DiskANN-style graphs and Product Quantization (PQ). VectorSearch supports online inserts, background sealing (ACTIVE → PENDING → SEALED), and low-latency KNN queries with exact re-rank. All storage lives under FoundationDB’s DirectoryLayer; APIs are fully async (CompletableFuture).

## Features

- Segmented index with strict-cap rotation (ACTIVE → PENDING → SEALED)
- Background builders for PQ codebooks/codes and L2 neighbor graphs
- Search modes: BEST_FIRST (default) and BEAM (deprecated with WARN-once)
- Exact re-rank after PQ/graph candidate selection
- Async Caffeine caches for PQ codebooks and adjacency; optional prefetch
- OpenTelemetry metrics for caches and segment states
- Maintenance: tombstone delete APIs and cooldown-aware vacuuming
- Global task queue support for centralized background workers across multiple indices
- 90%+ line / 75%+ branch coverage (JaCoCo gates)

Implementation notes

- Segment registry: the index maintains a compact `segmentsIndex` range under the index root to list existing segments efficiently. Queries list segments from this registry; there is no legacy per‑meta scan fallback.
- Builder invariants: only PENDING and WRITING segments are sealed by the builder; ACTIVE segments are never sealed directly.

## Quick Start

```java
// Open FDB and DirectoryLayer root
Database db = FDB.selectAPIVersion(730).open();
DirectorySubspace root = db.runAsync(tr -> DirectoryLayer.getDefault()
    .createOrOpen(tr, List.of("myapp", "vectorsearch"), "vectorsearch".getBytes(StandardCharsets.UTF_8)))
  .get(5, TimeUnit.SECONDS);

// Configure the index
VectorIndexConfig cfg = VectorIndexConfig.builder(db, root)
    .dimension(8)           // fixed vector dimensionality
    .pqM(4).pqK(16)         // PQ configuration
    .graphDegree(32)        // disk graph degree
    .maxSegmentSize(50_000) // strict cap per segment
    .localWorkerThreads(1)  // start local background builders
    .vacuumMinDeletedRatio(0.25) // threshold to enqueue vacuum after deletes
    .vacuumCooldown(Duration.ofMinutes(15)) // coalesce repeated vacuums
    .build();

// Create/open
VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);

// Insert: returns a 64-bit gid (stable across compaction)
long id = index.add(new float[]{1,0,0,0,0,0,0,0}, null).get();

// Query (top-K, default BEST_FIRST)
List<SearchResult> hits = index.query(new float[]{1,0,0,0,0,0,0,0}, 10).get();

// Delete (tombstone) and schedule cooldown-aware vacuum if threshold satisfied
index.delete(id).get();

// Optionally wait for builders to drain (uses hasVisibleUnclaimedTasks + hasClaimedTasks)
// Waits for build queue emptiness (no polling/reflection)
index.awaitIndexingComplete().get(10, TimeUnit.SECONDS);

index.close();
```

## Concepts

- Segments: New vectors land in the ACTIVE segment until `maxSegmentSize` is reached; the next insert rotates, marking the previous segment PENDING. Builders seal PENDING → SEALED by writing PQ/graph artifacts. Compaction uses a hidden WRITING destination segment that is not visible to search, then seals it to SEALED.
- Search: SEALED (and COMPACTING sources) use PQ/graph traversal + exact re-rank; ACTIVE/PENDING use brute-force. WRITING segments are ignored by search.
- Maintenance: Deletes set a tombstone flag and updates counters. If the deleted ratio exceeds a threshold, the index enqueues a vacuum maintenance task (with a configurable cooldown). Vacuum physically removes vector, PQ code, and adjacency, and decrements `deleted_count` while stamping `last_vacuum_at_ms`.

## Telemetry (OpenTelemetry)

This library emits OpenTelemetry metrics and spans. It uses the global OpenTelemetry SDK (GlobalOpenTelemetry), so you can plug any exporter (OTLP/Prometheus/etc.) without code changes.

Metrics (histograms are in milliseconds):

- vectorsearch.query.duration_ms (histogram)
- vectorsearch.query.count (counter)
- vectorsearch.build.duration_ms (histogram)
- vectorsearch.build.count (counter)
- vectorsearch.vacuum.duration_ms (histogram)
- vectorsearch.vacuum.run (counter)
- vectorsearch.vacuum.removed (counter)
- vectorsearch.compaction.duration_ms (histogram)
- vectorsearch.compaction.run (counter)

Common attributes on spans and metrics:

- index.path: user‑readable DirectorySubspace path of the index (e.g. indexes/myIndex)
- metric, dim, k (queries)
- segId (vacuum/build)
- anchorSegId (compaction)

Spans:

- vectorsearch.query, vectorsearch.build, vectorsearch.vacuum, vectorsearch.compaction

Enablement:

- Provide a GlobalOpenTelemetry instance in your app initialization with your preferred exporter. If none is provided, no‑op SDK is used and metrics/spans are dropped.

## Build, Test, and Coverage

- Build: `./gradlew build`
- Run tests: `./gradlew test`
- Code format (Palantir Java Format via Spotless): `./gradlew spotlessApply`
- Coverage report: `./gradlew jacocoTestReport`
- Coverage gates: `./gradlew jacocoTestCoverageVerification`

## Benchmarks

JMH microbenchmarks live under `src/jmh/`. Two benchmark suites are available:

### Search Latency (`VectorIndexSearchBenchmark`)

End-to-end kNN query latency against a real FDB instance. Requires Docker.

The benchmark automatically starts/stops an FDB 7.3 Docker container (`foundationdb/foundationdb:7.3.69`), inserts 1100 vectors (dim=8, maxSegmentSize=500), waits for segment sealing, then measures query latency across search modes (BEST_FIRST, BEAM) and k values (1, 10, 50).

```bash
# Run search benchmarks (Docker must be available)
./gradlew jmh -PjmhInclude="VectorIndexSearchBenchmark"
```

### Distance & PQ Microbenchmarks (`DistanceAndPqBenchmark`)

Pure-CPU benchmarks for hot-path operations. No FDB or Docker required.

- L2 distance (dim=128, 768)
- Cosine similarity (dim=128, 768)
- PQ encode (M=16, K=256, dim=128)
- PQ lookup-table distance (simulated inner loop)

```bash
# Run CPU-only benchmarks
./gradlew jmh -PjmhInclude="DistanceAndPqBenchmark"
```

### Running All Benchmarks

```bash
./gradlew jmh
```

## Configuration (high-level)

VectorIndexConfig includes:

- `dimension`, `metric` (L2 or COSINE)
- `maxSegmentSize`, `graphDegree`, `pqM`, `pqK`
- `oversample` (merge fan-in), `localWorkerThreads` (auto-start builders)
- `vacuumMinDeletedRatio`, `vacuumCooldown(Duration)` for maintenance behavior
- `codebookBatchLoadSize`, `adjacencyBatchLoadSize` (async cache loaders)
- `prefetchCodebooksEnabled` (default true) and `prefetchCodebooksSync` (test‑only; block query until codebook prefetch completes for sealed segments)
- `globalTaskQueueConfig(GlobalTaskQueueConfig)` for routing build/maintenance tasks to shared global queues

See `src/main/java/.../VectorIndexConfig.java` for the full builder.

## Maintenance

### Compaction Throttling

Background compactions are planned from small SEALED segments and throttled by `maxConcurrentCompactions` in `VectorIndexConfig`. Set it to `0` to disable compactions. The maintenance worker only transitions segments to COMPACTING when the in‑flight count is below the limit.

- Deletes: `index.delete(...)` marks tombstones and updates counters. If the deleted ratio exceeds the configured threshold and cooldown allows, a vacuum task is enqueued automatically.
- Programmatic vacuum:
  - Use `MaintenanceService.vacuumSegment(segId, minDeletedRatio)` to run a targeted vacuum now.
  - Or enqueue and process via the maintenance queue using `MaintenanceWorker`.
  - Example:`var dirs = FdbDirectories.openIndex(root, db).get(5, TimeUnit.SECONDS); var svc = new MaintenanceService(cfg, dirs); svc.vacuumSegment(0 /* segId */, 0.0 /* minDeletedRatio */).get(10, TimeUnit.SECONDS);`
- Background processing:
  - Set `localMaintenanceWorkerThreads > 0` to auto‑start a maintenance worker pool, or use `new MaintenanceWorker(cfg, dirs, queue).runOnce()` against the `tasks/maint` queue.

## Global Task Queue

The global task queue allows background workers (build and maintenance) to run in a separate process, servicing multiple vector indices from a single shared pool. When enabled on an index, local task queues and workers are not used. This is useful for multi-index deployments or when you want to run worker processes separately from query-serving processes.

### Setup

Create global queue directories and configure an index to use them:

```java
// Create global queue directories
DirectorySubspace globalDir = db.runAsync(tr -> DirectoryLayer.getDefault()
    .createOrOpen(tr, List.of("myapp", "global-queues")))
    .get(5, TimeUnit.SECONDS);

DirectorySubspace buildDir = globalDir.createOrOpen(db, List.of("build")).get(5, TimeUnit.SECONDS);
DirectorySubspace maintDir = globalDir.createOrOpen(db, List.of("maint")).get(5, TimeUnit.SECONDS);

// Create task queues
TaskQueue<String, GlobalBuildTask> globalBuildQueue = TaskQueues.createTaskQueue(
    TaskQueueConfig.builder(db, buildDir,
        new ProtoSerializers.StringSerializer(),
        new ProtoSerializers.GlobalBuildTaskSerializer())
    .build()).get(5, TimeUnit.SECONDS);

TaskQueue<String, GlobalMaintenanceTask> globalMaintQueue = TaskQueues.createTaskQueue(
    TaskQueueConfig.builder(db, maintDir,
        new ProtoSerializers.StringSerializer(),
        new ProtoSerializers.GlobalMaintenanceTaskSerializer())
    .build()).get(5, TimeUnit.SECONDS);

// Configure index with global queues (no local workers)
GlobalTaskQueueConfig globalCfg = new GlobalTaskQueueConfig(globalBuildQueue, globalMaintQueue);
VectorIndexConfig cfg = VectorIndexConfig.builder(db, indexDir)
    .dimension(8)
    .globalTaskQueueConfig(globalCfg)
    .build();
VectorIndex index = VectorIndex.createOrOpen(cfg).get(10, TimeUnit.SECONDS);
```

### Running Global Workers

Use `GlobalWorkerRunner` to start shared worker threads that process tasks from the global queues:

```java
// Operational settings for all indices processed by this runner
WorkerConfig workerConfig = WorkerConfig.builder()
    .buildTxnLimitBytes(5_000_000)
    .maxConcurrentCompactions(2)
    .build();

// Start workers: 4 build threads + 2 maintenance threads
GlobalWorkerRunner runner = new GlobalWorkerRunner(db, workerConfig, globalCfg);
runner.start(4, 2);

// ... later
runner.close(); // graceful shutdown
```

### Key Points

- Separate queues for build and maintenance tasks
- Multiple indices can share the same global queues
- Data-format params (dimension, metric, pqM, pqK, graphDegree, etc.) are read from each index's persisted `IndexMeta`
- Operational params (compaction knobs, vacuum thresholds, txn limits) come from a uniform template config
- Backward compatible — existing behavior is unchanged when `globalTaskQueueConfig` is not set
- `awaitIndexingComplete()` waits for ALL global build tasks (not just the calling index's)

## Design Highlights

- DirectoryLayer layout per index: `meta`, `currentSegment`, `segmentsIndex/<segId>`, `segments/<segId>/{meta,vectors/,pq/{codebook,codes/},graph/}`, `tasks/`, and `gid/{map,rev}` for global id mapping.
- Async batched I/O with transaction size guards (FDB 10MB/5s)
- Caffeine AsyncLoadingCache with bulk loaders, optional prefetch for codebooks
- Deterministic seeding and auto‑tuning for BEST_FIRST traversal

### IDs

- Public APIs return a single 64‑bit global id (gid). Use `SearchResult.gid()` in query results.
- For testing/admin tooling, `VectorIndex.resolveIds(long[] gids)` returns `(segmentId, vectorId)` pairs.

## Contributing

- Keep PRs green: `./gradlew check` runs tests + coverage gates.
- Prefer async composition (`thenCompose/thenApply`) in storage paths; tests may `.get()`/`.join()`.
- Avoid blocking calls inside FDB transactions; batch reads/writes and respect approximate size checks.

## License

Apache 2.0