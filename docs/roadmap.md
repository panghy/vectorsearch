# Roadmap: DiskANN + PQ Vector Index on FoundationDB (DirectoryLayer)

This roadmap restates the implementation plan to explicitly use FoundationDB’s DirectoryLayer for namespacing and key management. Every index, segment, and subsystem (vectors, graph, PQ, tasks) is rooted in a DirectorySubspace to avoid manual prefix handling and to enable clean organization and tooling.

## Goals and Non‑Goals

- Goals:
  - Directory‑backed layout for all data (no ad‑hoc tuple roots). Per‑segment data is keyed under a single `segmentsDir` using integer `segId` as the first tuple field.
  - Segment lifecycle: ACTIVE → PENDING → SEALED with background build of graph + PQ. Compaction uses COMPACTING (sources) and a WRITING destination that remains hidden from queries until sealed.
  - Strict-cap rotation: when an ACTIVE segment reaches maxSegmentSize, the NEXT insert rotates
    immediately (before writing) so the new vector becomes vecId=0 in the new ACTIVE segment.
  - Async, non‑blocking storage APIs (`CompletableFuture`), respecting FDB 5s and 10MB limits.
  - Top‑K KNN queries with ≤30ms target latency (K=10) and exact re‑rank.
  - Real FDB integration tests; coverage ≥90% lines / ≥75% branches.
- Non‑Goals (initially):
  - Cross‑cluster sharding, advanced multi‑tenancy, or alternate ANN algorithms.

## Directory Layout

Assume a caller‑provided application root directory (e.g. `/myapp/vectorsearch`). All paths below are relative to that root. Use `DirectoryLayer` or a parent `DirectorySubspace` for creation/open.

Tree (conceptual):

/<app>/vectorsearch/
  indexes/
    <indexName>/                      (indexDir)
      meta                            (key in indexDir)
      currentSegment                  (key in indexDir)
      segments/                       (segmentsDir)
        (segId, "meta")               SegmentMeta (segId is an integer tuple field)
        (segId, "vectors", vecId)     VectorRecord
        (segId, "pq", "codebook")     PQCodebook
        (segId, "pq", "codes", vecId) PQ code bytes
        (segId, "graph", vecId)       Adjacency list bytes
      segmentsIndex/                  (indexDir subspace; one key per existing segId)
        <segId>                       empty value (presence = exists)
      tasks/                          (taskQueueBaseDir for TaskQueue)

Notes:
- `segId` is now encoded as an integer in tuple keys (breaking change, no migration/back‑compat maintained).
- All key building is via `DirectorySubspace.pack(Tuple.from(...))`. Per‑segment data is not its own DirectorySubspace; it lives under `segmentsDir` keyed by `(segId, ...)` for compact listing and scanning.

## Milestones Overview

1) Protos + Directory scaffolding
2) Directory‑backed storage (ACTIVE segment CRUD)
3) Segment lifecycle + TaskQueue directories
4) Background builders (PQ + graph) with chunked writes
5) Search pipeline (SEALED graph + ACTIVE brute‑force) and re‑rank
6) Caching & tuning
7) Maintenance tasks (vacuum, compaction merge MVP)
8) Observability & docs
9) Tests, benchmarks, hardening

---

## Status Snapshot (as of 2025-09-10)

- Completed:
  - M1 Protos + Directory scaffolding (vectorsearch.proto, FdbDirectories).
  - M2 ACTIVE CRUD and rotation (FdbVectorStore + tests) honoring strict-cap rotation semantics.
  - M3 Segment lifecycle with TaskQueue under `tasks/` (enqueue on rotate; worker processes one build at a time).
  - M4 Background builders writing PQ codebook/codes and L2-based adjacency; segment sealing.
  - M5 Search: Graph-guided traversal for SEALED segments seeded by PQ with exact re‑rank; BEAM and BEST_FIRST modes implemented; default is BEST_FIRST. Adjacency I/O batched; explored caps and early‑exit honored. WRITING segments are ignored by search; COMPACTING sources are treated as sealed.
  - M6 Caching: Async Caffeine caches for PQ codebooks and adjacency with asyncLoadAll; code paths use `getAll` for batch warms; optional query-time codebook prefetch for SEALED segments.
  - M8 Observability: OpenTelemetry gauges registered for cache size and stats; configurable metric attributes; tests use InMemoryMetricReader.
  - M9 Hardening: consolidated tests, added edge-case coverage; JaCoCo gates (≥90% lines / ≥75% branches) passing.
- Not Started / Next:
  - M7 Compaction: improve candidate selection and throttling (planner heuristics, backpressure).
  - M8 Observability (index-level): add per-query latency histograms, SLOs.
  - M9 Benchmarks (micro + macro latency harness): JMH + macro harness.

Recent additions:
- Vacuum + delete API with cooldown-aware scheduling (runtime-configured) and
  `last_vacuum_at_ms` tracked in `SegmentMeta`.
- Index-level metrics: `vectorsearch.segments.state_count{state=*}` and vacuum
  `vectorsearch.maintenance.vacuum.{scheduled|skipped}` counters.

## 1) Protos + Directory Scaffolding

- Protobuf messages (augment `src/main/proto/vectorsearch.proto`):
  - `IndexMeta { string name; int32 dimension; enum Metric { L2, COSINE } metric; int32 max_segment_size; int32 pq_m; int32 pq_k; int32 graph_degree; int32 oversample; }`
  - `SegmentMeta { int32 segment_id; enum State { ACTIVE=0; PENDING=1; SEALED=2; } state; int32 count; int64 created_at_ms; int64 deleted_count; }`
  - `VectorRecord { int32 seg_id; int32 vec_id; bytes embedding; bool deleted; bytes payload; }`
  - `PQCodebook { int32 m; int32 k; repeated bytes centroids; }`
  - `Adjacency { repeated int32 neighbor_ids; }`
  - Optional: `BuildTask { string index_name; int32 seg_id; }`
- Directory helpers:
  - `FdbDirectories` utility to open/create:
    - `indexDir = root.createOrOpen(ctx, List.of("indexes", indexName))`
    - `segmentsDir = indexDir.createOrOpen(ctx, List.of("segments"))`
    - `segmentDir = segmentsDir.createOrOpen(ctx, List.of(segIdStr))`
    - `vectorsDir = segmentDir.createOrOpen(ctx, List.of("vectors"))`, `pqDir`, `codesDir`, `graphDir`
    - `tasksBaseDir = indexDir.createOrOpen(ctx, List.of("tasks"))` (for TaskQueue)

Deliverables:
- Compilable protos; `FdbDirectories` with open/create methods; constants for component names.

## 2) Directory‑Backed Storage (ACTIVE CRUD)

- Keys within directories:
  - `indexDir.pack(Tuple.from("meta"))` → `IndexMeta`
  - `indexDir.pack(Tuple.from("currentSegment"))` → current segId (int serialized)
  - `segmentDir.pack(Tuple.from("meta"))` → `SegmentMeta`
  - `vectorsDir.pack(Tuple.from(vecId))` → `VectorRecord`
  - `pqDir.pack(Tuple.from("codebook"))` → `PQCodebook`
  - `codesDir.pack(Tuple.from(vecId))` → PQ code
  - `graphDir.pack(Tuple.from(vecId))` → adjacency
- API skeleton (all async):
  - `VectorIndex`:
    - `CompletableFuture<Void> createIndex(IndexMeta)` → sets `meta`, `currentSegment`, creates directories
    - `CompletableFuture<int[]> add(float[] embedding, byte[] payload)` → returns `[segId, vecId]`
    - `CompletableFuture<Void> delete(int segId, int vecId)`
    - `CompletableFuture<List<Result>> query(float[] q, int k)`
    - `CompletableFuture<Void> sealActiveIfNeeded()`
    - `CompletableFuture<List<SegmentMeta>> listSegments()`
- ACTIVE insert transaction:
  - Read `currentSegment`; fetch its `SegmentMeta` from `segmentDir/meta`.
  - Allocate next `vecId` (from `SegmentMeta.count`), write `VectorRecord` at `vectorsDir/(vecId)`.
  - Update `SegmentMeta.count`. If threshold reached, set `PENDING`, create next ACTIVE segment directory, set `currentSegment`.

Deliverables:
- `FdbVectorStore` reading/writing via DirectorySubspaces; rotation to PENDING; tests for CRUD and rotation.

## 3) Segment Lifecycle + TaskQueue Directories

- When a segment transitions to PENDING, enqueue a build task in a TaskQueue rooted at `indexDir/tasks`.
- TaskQueue config:
  - `TaskQueueConfig<String, BuildTask>` with `directory = indexDir.createOrOpen(ctx, List.of("tasks"))`
  - The KeyedTaskQueue will create its own internal directories under `tasks/` (unclaimed, claimed, task_keys, watch).
- Task key: `build-segment:<indexName>:<segId>` (string serializer).
- Worker:
  - `awaitAndClaimTask(db)` → parse `BuildTask` → re‑check `segmentDir/meta` state; if SEALED, `complete()`.
  - Otherwise run `SegmentBuildService.build(indexDir, segId)`; use `claim.extend()` for long steps.

Deliverables:
- `SegmentBuildWorker` using `tasks/` directory; tests for dedupe, retries, TTL extension.

## 4) Background Builders (PQ + Graph) with Chunked Writes

- Build flow:
  1) Snapshot range read all `VectorRecord` from `vectorsDir.range()`.
  2) Train PQ codebook; encode PQ codes.
  3) Build DiskANN‑style graph (degree R, `L_build`).
  4) Persist in chunks (e.g., 500–1,000 keys/txn) using directory‑scoped keys:
     - `pqDir/("codebook")`, `codesDir/(vecId)`, `graphDir/(vecId)`.
  5) Update `segmentDir/("meta")` to SEALED.
- Idempotency: Upserts are safe; seal only after all writes complete.

Deliverables:
- `PqTrainer`, `PqEncoder`, `GraphBuilder`, `SegmentBuildService` with chunked commits; tests on small segments.

## 5) Search Pipeline (SEALED + ACTIVE) and Re‑rank

- SEALED: graph traversal + PQ distance tables; return top‑N (N = k * oversample).
- ACTIVE/PENDING: brute‑force on `vectorsDir`; exact distances.
- Merge, filter deleted, batch‑fetch full vectors (range over `vectorsDir`), exact re‑rank; return top‑K.
- Parallelize across `segments/` with `CompletableFuture` composition.

Deliverables:
- `GraphSearcher` (implemented inside `VectorIndex`) + orchestrated multi‑segment query; correctness tests vs brute‑force baseline.

Status:
- Implemented. SEALED search performs PQ seeding, then graph traversal (`BEAM` and `BEST_FIRST` modes) with batched adjacency prefetch and optional frontier refinement. Traversal halts on `efSearch`/`maxExplore` or early‑exit when no improvement. Exact re‑rank finalizes results.

Per‑query tuning API (shipped):
```java
SearchParams p = SearchParams.defaults(k, oversample);
// or full control
SearchParams tuned = SearchParams.of(64, 32, 4, 2048, true, SearchParams.Mode.BEST_FIRST);
var res = index.query(q, k, tuned).get();
```

## 6) Caching & Tuning

- Caffeine caches bound to directories:
  - `PQCodebook` per `segmentDir`.
  - Optionally small LRU for `graphDir` adjacency blocks and hot vectors.
- Configurables (from `IndexMeta`): `oversample`, `graph_degree`, `L_build`, `max_segment_size`, `pq_m`, `pq_k`, sample size.

Deliverables:
- Cache wiring and configuration; tests for cache behavior and correctness.

Status:
- Implemented in search path. `SegmentCaches` provides async caches + bulk loaders; adjacency/codebooks can be batch loaded via `getAll`. Prefetch for codebooks is optional per config. Sizes/eviction follow Caffeine defaults with 10-minute idle expiry.

## 7) Maintenance Tasks

- Tombstone vacuum: scan `vectorsDir` for deleted records; schedule in `tasks/` as `vacuum:<segId>`.
- Compaction/merge (MVP implemented): read multiple small SEALED `segments/`, produce a consolidated WRITING segment, build PQ/graph, seal, and atomically swap into `segmentsIndex` while removing sources; scheduled via TaskQueue.

Deliverables:
- Vacuum job; compaction skeleton (optional for v1), both using DirectorySubspaces.

Status:
- Vacuum job shipped (tombstoned vectors physically removed; `deleted_count` decremented; `last_vacuum_at_ms` stamped). Cooldown-aware enqueueing after deletes based on configurable threshold.
- Compaction/merge MVP shipped: `MaintenanceService.compactSegments(...)` writes a destination WRITING segment, invokes `SegmentBuildService.build(...)`, then swaps registry entries and cleans up sources. A simple heuristic (`findCompactionCandidates`) enqueues compaction when a segment falls below 50% capacity post‑vacuum. Future work: richer planner, throttling, and backpressure.

## 10) Path to 1B+ vectors (prioritized)

This section outlines the additional work needed to comfortably scale to billions of vectors while meeting low-latency targets and keeping operational complexity in check.

1) Segment catalog and listing at scale (shipped)
- Implemented: a compact `segmentsIndex/<segId>` range under the index root. All queries list segments from this registry; no legacy fallback scans. A monotonic `maxSegmentId` is still maintained for rotation bookkeeping.

2) Compaction/merge planner (sealed→sealed)
- MVP implemented: read several SEALED segments, write WRITING destination, build PQ/graph, seal, and atomically swap via `segmentsIndex`; sources cleared.
- Next: add planner heuristics (size/age aware), throttle concurrent compactions, and improve backpressure. Keep adjacency/codebooks block‑aligned.

3) Graph construction quality (DiskANN/Vamana)
- Add alpha/pruning and better candidate pool management to the builder; tune `graphDegree` and build breadth.
- Optionally adopt medoid/centroid starts and reordering to improve locality.
- Evaluate storing adjacency in block-compressed form; consider varint/indexed blocks.

4) PQ improvements
- Support OPQ and/or IVF-PQ hybrid for high-d with better recall/latency tradeoffs.
- Sampling strategy for training (reservoir, stratified per segment); persist training samples for reproducibility.

5) Search merge and per-segment budgeting
- Dynamic per-segment fan-in (cap) based on segment age/size; early termination when improvement stalls.
- Learn/tune `ef`, `beam`, `perSegmentLimitMultiplier` online from latency budgets.

6) Memory & I/O efficiency
- Float16/byte quantization options for raw vectors (ACTIVE/PENDING) to cut reread costs at rerank.
- Cache adjacency/codebooks with size/TTL caps and hot-key protection; consider Bloom filters to avoid cold misses.

7) Filtered / hybrid queries (future)
- Sparse (BM25) + dense hybrid scoring hooks; optional attribute filters via side indexes.

8) Operability
- Backpressure: limit concurrent builders and queue depth; auto-shed enqueue when FDB latency rises.
- Fine-grained metrics: p50/p95/p99 query latency, builder durations, compaction metrics.
- Admin accessors: list segments, deleted ratios, last vacuum time; on-demand maintenance.

9) Benchmarks & continuous tuning
- Add JMH microbenchmarks (distance, PQ encode/decode, traversal) and macro harnesses (end-to-end p50/p99 with varying K, d, segment counts).

10) Multi-index / tenancy (non-goal for v1, design for v2)
- Keep index directories isolated; ensure queues and metrics are labeled. Explore horizontal partitioning at service layer when needed.

## 8) Observability & Docs

- OpenTelemetry gauges for caches; hooks ready for spans in main flows.
- Logging: per-query and sealed-search details at DEBUG; BEAM mode emits a one-time WARN.
- Docs updated with Directory usage and worker setup.

Deliverables:
- Spans/metrics wired; README/CLAUDE updated with Directory examples.

Status:
- Metrics done (gauges); tests in place using OTel SDK testing. Codebook prefetch can be enabled/disabled; a test‑only `prefetchCodebooksSync` option makes prefetch blocking to simplify deterministic tests.

## 9) Tests, Benchmarks, Hardening

- Real FDB integration tests exercising Directory creation/open, CRUD, lifecycle, search, and tasks.
- Benchmarks to validate query latency and build throughput.
- Verify chunked commits stay within 5s/10MB limits.

Deliverables:
- Coverage thresholds met; basic latency numbers captured.

Status:
- Benchmarks pending. Tests in place for CRUD/rotation/build/query and search (including edge cases); JaCoCo thresholds enforced and passing.

---

## TaskQueue Integration (Directory‑first)

- Directory: `tasksBaseDir = indexDir.createOrOpen(ctx, List.of("tasks"))` passed to `TaskQueueConfig`.
- The queue manages its own subdirectories: `unclaimed_tasks/`, `claimed_tasks/`, `task_keys/`, `watch/`.
- Worker pattern:
  - `awaitAndClaimTask(db)` → parse `BuildTask` (protobuf via serializer) → `SegmentBuildService`.
  - Use `extendTtl` during PQ/train/graph steps; `complete()` on success; `fail()` on error.
- Enqueue policy: on PENDING transition, `enqueueIfNotExists(taskKey, buildTask, delay, ttl, false)`.

---

## API Sketch (Directory‑based)

- `VectorIndexFactory.open(Database db, DirectorySubspace appRoot, String indexName, IndexMeta)` → creates/opens `indexDir` and subdirs, sets `meta`, `currentSegment` if missing.
- `VectorIndex` provides transaction‑scoped and convenience overloads, all async.

---

## Suggested Packages/Classes

- `io/github/panghy/vectorsearch/api/VectorIndex.java`
- `io/github/panghy/vectorsearch/api/VectorIndexFactory.java`
- `io/github/panghy/vectorsearch/fdb/FdbDirectories.java` (open/create helpers)
- `io/github/panghy/vectorsearch/fdb/FdbVectorStore.java` (Directory‑backed CRUD, rotation)
- `io/github/panghy/vectorsearch/pq/PqTrainer.java`, `PqEncoder.java`
- `io/github/panghy/vectorsearch/graph/GraphBuilder.java`, `GraphSearcher.java`
- `io/github/panghy/vectorsearch/tasks/SegmentBuildService.java`, `SegmentBuildWorker.java`, `BuildTasks.java`
- `io/github/panghy/vectorsearch/util/Distances.java`, `FloatPacker.java`

---

## Acceptance Criteria per Milestone

- M1: Protos + `FdbDirectories` compile; basic Directory open/create tests pass.
- M2: ACTIVE CRUD works via DirectorySubspaces; rotation to PENDING verified.
- M3: PENDING → enqueue to `tasks/`; worker claims and noops.
- M4: PQ + graph built; batch writes complete; segment SEALED.
- M5: Multi‑segment query with rerank matches brute‑force on small sets.
- M6: Caches wired; tunables passed from `IndexMeta`.
- M7: Vacuum/compaction tasks runnable via `tasks/`.
- M8: OTel spans/metrics present; docs updated.
- M9: Coverage thresholds met; basic latency benchmarks captured.

---

## Next Steps (Next 3–5 Days)

- Graph‑based SEALED search:
  - Add `GraphSearcher` that performs best‑first/beam search over adjacency, seeded by multiple entry points.
  - Integrate with existing PQ LUT scoring to prioritize frontier; fall back to small PQ scan when needed.
  - Update `VectorIndex.query(...)` to use graph traversal for SEALED segments; keep exact rerank.
  - Tests: verify parity vs brute‑force on small data and reduced IO on larger data.

- Transaction‑scoped APIs:
  - Provide storage/query overloads that accept `Transaction`/`ReadTransaction` in addition to convenience methods, per project standards.
  - Refactor internal helpers to compose futures rather than open transactions internally where possible.

- Delete API and vacuum skeleton:
  - Add `delete(segId, vecId)` to set tombstone and increment `deleted_count`.
  - Create `VacuumJob` queued under `tasks/` that erases tombstoned vectors and associated PQ/graph keys in chunked commits.
  - Tests for deletion visibility in queries and vacuum idempotency.

- Observability:
  - Wire OpenTelemetry spans and metrics around add/rotate/build/query and queue operations.
  - Basic counters: inserts, rotations, sealed segments, build duration, query latency, PQ scan vs graph traversals.

- Benchmarks & coverage:
  - Add a simple benchmark harness (e.g., JMH or integration microbench) to capture latency numbers for K=10.
  - Raise/verify JaCoCo thresholds; extend tests to cover SEALED query path and error cases.

- Docs:
  - Update `design.md` with the DirectoryLayer layout now used and the SEALED search plan.
  - Add examples for transaction‑scoped usage patterns in CLAUDE.md.
