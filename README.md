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
- 90%+ line / 75%+ branch coverage (JaCoCo gates)

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

// Insert
int[] id = index.add(new float[]{1,0,0,0,0,0,0,0}, null).get();

// Query (top-K, default BEST_FIRST)
List<SearchResult> hits = index.query(new float[]{1,0,0,0,0,0,0,0}, 10).get();

// Delete (tombstone) and schedule cooldown-aware vacuum if threshold satisfied
index.delete(id[0], id[1]).get();

// Optionally wait for builders to drain (uses hasVisibleUnclaimedTasks + hasClaimedTasks)
index.awaitIndexingComplete().get(10, TimeUnit.SECONDS);

index.close();
```

## Concepts

- Segments: New vectors land in the ACTIVE segment until `maxSegmentSize` is reached; the next insert rotates, marking the previous segment PENDING. Builders seal PENDING → SEALED by writing PQ/graph artifacts.
- Search: SEALED segments use PQ-seeded traversal + exact re-rank; ACTIVE/PENDING use brute-force.
- Maintenance: Deletes set a tombstone flag and updates counters. If the deleted ratio exceeds a threshold, the index enqueues a vacuum maintenance task (with a configurable cooldown). Vacuum physically removes vector, PQ code, and adjacency, and decrements `deleted_count` while stamping `last_vacuum_at_ms`.

## Metrics

- `vectorsearch.cache.size{cache=codebook|adjacency}` (gauge)
- `vectorsearch.cache.*` hit/miss/load counters (gauges of Caffeine stats)
- `vectorsearch.segments.state_count{state=ACTIVE|PENDING|SEALED}` (gauge)
- `vectorsearch.maintenance.vacuum.scheduled|skipped` (counters)

To use OTel in tests:
```java
InMemoryMetricReader reader = InMemoryMetricReader.create();
SdkMeterProvider mp = SdkMeterProvider.builder().registerMetricReader(reader).build();
OpenTelemetrySdk sdk = OpenTelemetrySdk.builder().setMeterProvider(mp).build();
GlobalOpenTelemetry.resetForTest();
GlobalOpenTelemetry.set(sdk);
```

## Build, Test, and Coverage

- Build: `./gradlew build`
- Run tests: `./gradlew test`
- Code format (Palantir Java Format via Spotless): `./gradlew spotlessApply`
- Coverage report: `./gradlew jacocoTestReport`
- Coverage gates: `./gradlew jacocoTestCoverageVerification`

## Configuration (high-level)

VectorIndexConfig includes:
- `dimension`, `metric` (L2 or COSINE)
- `maxSegmentSize`, `graphDegree`, `pqM`, `pqK`
- `oversample` (merge fan-in), `localWorkerThreads` (auto-start builders)
- `vacuumMinDeletedRatio`, `vacuumCooldown(Duration)` for maintenance behavior
- `codebookBatchLoadSize`, `adjacencyBatchLoadSize` (async cache loaders)

See `src/main/java/.../VectorIndexConfig.java` for the full builder.

## Maintenance

- Programmatic vacuum:
  - `index.vacuumSegment(segId)` to force a vacuum.
  - `index.vacuumIfNeeded(segId, minDeletedRatio)` to enqueue a threshold-aware task.
- Background processing:
  - Set `localMaintenanceWorkerThreads > 0` to auto‑start a maintenance worker pool, or run `MaintenanceWorker.runOnce()` against the `tasks/maint` queue.

## Design Highlights

- DirectoryLayer layout per index: `meta`, `currentSegment`, `segments/<segId>/{meta,vectors,pq,graph}`, `tasks/`
- Async batched I/O with transaction size guards (FDB 10MB/5s)
- Caffeine AsyncLoadingCache with bulk loaders, optional prefetch for codebooks
- Deterministic seeding and auto‑tuning for BEST_FIRST traversal

## Contributing

- Keep PRs green: `./gradlew check` runs tests + coverage gates.
- Prefer async composition (`thenCompose/thenApply`) in storage paths; tests may `.get()`/`.join()`.
- Avoid blocking calls inside FDB transactions; batch reads/writes and respect approximate size checks.

## License

Apache 2.0
