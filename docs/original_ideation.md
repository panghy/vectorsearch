# FDB-backed PQ + DiskANN — Design Document

---

## 1. Goals & Non-goals

### Goals
- Millisecond-latency ANN over large collections with DiskANN-style graph walk and PQ in-RAM scoring.
- Online upsert/delete; new points become searchable after link tasks complete.
- All data in one FDB cluster (Redwood); no external object store.
- Small, retry-friendly transactions that respect FDB limits (≤100 KB value, ~10 MB affected data/txn, ~5 s txn time).

### Non-goals
- Returning original float vectors (we store sketches instead for codebook rotation).
- Distributed query fan-out across multiple collections (single collection per index here).

---

## 2. Defaults (Milvus-aligned)

We mirror Milvus where it's defined and follow upstream DiskANN where Milvus is silent.

### PQ
- **nbits = 8** (per-subvector code width). Milvus IVF_PQ default.
- **m** (number of subvectors): Milvus does not define a default; common guidance picks m = D/2. We adopt m=D/2 if not specified.

### DiskANN Search
- **search_list default 16**; if topk > 16, Milvus auto-sets search_list = topk. We do the same.
- **Graph degree (R)** not exposed by Milvus; upstream DiskANN default R=64. We set degree = 64.

### Engine / Placement
- **Storage engine:** ssd-redwood-v1 (production-ready since 7.1). Size Redwood cache appropriately (e.g., several GB).
- **Optional:** steer hot ranges with rangeconfig (FDB 7.3+) if needed.

### DiskANN Background
- DiskANN uses a Vamana proximity graph with bounded degree; Milvus couples it with PQ codes in memory and SSD pages.

---

## 3. System Architecture

### Components

1. **Query service**
   - Loads PQ codebooks; maintains PQ-block & adjacency LRUs.
   - Executes beam search (beam ties to search_list) using snapshot reads.

2. **Link worker(s)**
   - Consumes upsert tasks from the transactional queue.
   - Persists PQ codes into blocks (RMW) and (re)links nodes with robust pruning + back-links (small write txns).

3. **Unlink worker(s)**
   - Consumes delete tasks; removes back-links, clears node adjacency.

4. **Task queue**
   - Provided. Enqueue/ack are part of the same FDB cluster/txn.

5. **Admin**
   - Codebook rotation (two-phase), entry-list refresh, GC, metrics.

### Transactions
- Writes: tiny, conflict-resilient, ≪10 MB affected data and ≪5 s. Reads on the hot path are snapshot to avoid conflicts.

---

## 4. Keyspace (tuple-encoded)

All keys are under `/C/{coll}/...`. Keep keys compact (≲32 B ideal).

```
/C/{coll}/meta/config                     -> pb.Config
/C/{coll}/meta/cbv_active                 -> uint32              # active PQ codebook version
/C/{coll}/entry                           -> pb.EntryList        # hierarchical entry points for search

# PQ
/C/{coll}/pq/codebook/{cbv}/{sub}         -> pb.CodebookSub      # sub ∈ [0..m-1]
/C/{coll}/pq/block/{cbv}/{blockNo}        -> pb.PqCodesBlock     # ~24–64KB values; RMW on upsert

# Graph (per-node adjacency)
/C/{coll}/graph/node/{NID}                -> pb.NodeAdj          # neighbors for one node
/C/{coll}/graph/meta/connectivity         -> pb.GraphMeta       # connectivity stats & repair state

# Vector sketches (for codebook retraining)
/C/{coll}/sketch/{NID}                    -> pb.VectorSketch    # compressed vector representation

# Tasks (your queue owns actual keys; shown for context)
# /queue/{topic}/...                      -> (opaque)
```

### Mapping
- blockNo = NID / codes_per_block.
- Value sizes: keep under 100 KB; prefer ≤10 KB for very hot/random values. PQ blocks at ~24–64 KB are fine (few RMWs, high cache hit-rate).

---

## 5. On-disk Protobufs

```protobuf
syntax = "proto3";
package ann;

message Config {
    uint32 D = 1;
    string metric = 2;                  // "L2" | "IP" | "COSINE"
    // PQ (Milvus-aligned)
    uint32 m = 3;                       // default: D/2 if unset
    uint32 nbits = 4;                   // default: 8
    // Graph
    uint32 degree = 5;                  // default: 64 (DiskANN R)
    // Storage tuning
    uint32 codes_per_block = 6;         // choose 256–512 => ~24–64KB per block
}

message CodebookSub {                   // 256 codewords × (D/m) dims
    bytes codewords_fp16 = 1;           // fp16 packed to save space
}

message PqCodesBlock {
    uint64 block_first_nid = 1;         // inclusive
    uint32 codes_in_block = 2;          // <= codes_per_block
    bytes  codes = 3;                   // row-major [codes_in_block × m], uint8
    uint32 ver = 4;                     // optional debug/version
}

message NodeAdj {
    uint64 nid = 1;
    repeated uint64 neighbors = 2;      // sorted, unique, cap at Config.degree
    uint32 ver = 3;
}

message EntryList { 
    repeated uint64 nid = 1;           // primary entry points
    repeated uint64 random_samples = 2; // random sample fallbacks
    repeated uint64 high_degree = 3;    // high-connectivity nodes
}

message VectorSketch {
    bytes sketch = 1;                   // 256-bit SimHash or PCA projection
    uint32 version = 2;                 // sketch algorithm version
}

message GraphMeta {
    uint64 last_repair_ts = 1;
    uint32 connected_components = 2;
    repeated uint64 orphaned_nodes = 3;
```

---

## 6. Algorithms

### 6.1 Upsert (batch of N)

**Input:** `[(NID, float[D])]`

#### Phase A — Enqueue link work & ensure adjacency record
- For each NID in chunks (≤512 items/txn):
  - If `/graph/node/{NID}` missing, set `NodeAdj{nid, neighbors:[]}`.
  - Encode PQ client-side using cbv_active → code : `uint8[m]`.
  - `queue.enqueue(tr, topic="link", payload={coll,nid,cbv,code})`.
  - Commit (small txn, idempotent).

#### Phase B — Link worker
1. **Persist PQ code into its block:**
   - blockNo = nid / codes_per_block.
   - Read `/pq/block/{cbv}/{blockNo}` (or create empty); RMW: write code at offset `(nid - block_first_nid) * m`, bump codes_in_block if needed; commit.

2. **Neighbor discovery (PQ-only beam; snapshot reads):**
   - Load EntryList, build per-subspace LUT for the query vector.
   - Seed heap with entries scored by PQ (see §6.4).
   - Pop/beams until maxVisits, pushing neighbors via NodeAdj reads.
   - Robust prune frontier to ≤degree.

3. **Write adjacency & back-links in tiny batches (e.g., 16 neighbors per txn):**
   - Update `/graph/node/{nid}` with pruned list.
   - For each neighbor v in the batch, read `/graph/node/{v}` and insert nid (cap at degree). Commit; retry on conflicts.

**Why this shape?** RMW of one PQ block (single value, ~24–64 KB) + a handful of per-node adjacency rewrites => low contention & well below FDB txn limits.

### 6.2 Delete (batch of N)

**Input:** `[NID]`

- Enqueue unlink task for each NID; worker:
  - Read `NodeAdj(nid)` (snapshot).
  - For neighbors in small batches, remove nid from `/graph/node/{v}`.
  - Clear `/graph/node/{nid}`.
  - (Optional) zero out PQ slot by RMW on its block.

### 6.3 Entry list maintenance (hierarchical)

Maintain multiple entry strategies for robustness:
- **Primary**: 32 k-means medoids or high-degree nodes
- **Random**: 16 randomly sampled nodes (for diversity)
- **High-degree**: 16 nodes with highest connectivity

Refresh periodically (hourly) outside transactions. This multi-strategy approach prevents search degradation from poor entry selection.

### 6.4 Query (top-k)

**Input:** `q[D]`, `topk`

- search_list = max(16, topk) (Milvus behavior); set beam = search_list.
- Load codebooks for cbv_active; build LUT (size m×256).
- Seed heap with hierarchical entry points:
  - Score all primary entry nodes
  - If heap size < beam/2, add random samples
  - If still sparse, add high-degree nodes
- Loop until heap empty or maxVisits:
  - Pop best u; if unseen: add to results; read `/graph/node/{u}` (snapshot).
  - Score up to beam neighbors and push.
- Return top-k `{nid, distance}`.

**DiskANN note:** We mirror the graph walk style (Vamana) and keep per-node degree bounded (default 64).

---

## 7. Concurrency & FDB Specifics

- **Value/txn/time limits:** Never exceed 100 KB per value, ~10 MB affected data per txn, or ~5 s per txn. Split work accordingly.
- **Snapshot reads** on the search path avoid read conflict ranges; writes use normal (non-snapshot) reads.
- **RMW hotspots:** If many concurrent upserts touch the same PQ block, stripe the block key: `/pq/block/{cbv}/{blockNo}/{stripe}`, `stripe = hash(nid)%S`. Readers select the correct stripe; blocks remain ~24–64 KB.
- **Retries:** On commit_unknown_result or conflicts, exponential backoff and retry; batch size keeps retries cheap.
- **Range placement (optional):** With FDB 7.3+ rangeconfig, assign hot ranges to distinct teams and split explicit shard boundaries.

---

## 8. Caching & Memory

- **Codebooks:** always in RAM.
- **PQ blocks:** LRU sized to hundreds of MB (or a few GB) to minimize RMW/read pressure.
- **Adjacency:** small LRU (each NodeAdj is tiny).
- **Redwood cache:** configure cache_memory generously on storage processes (mind process memory sizing).

---

## 9. Parameter Tuning

| Knob | Default | Effect |
|------|---------|--------|
| m | D/2 | RAM for LUT ~ m×256 floats; larger m → higher recall (lower quantization error) but bigger PQ blocks. |
| nbits | 8 | One byte/subvector; compatible with Milvus/IVF_PQ default. |
| degree | 64 | DiskANN default R; larger means more edges, better recall, more write IO. |
| codes_per_block | 256–512 | Controls PQ block (RMW) value size (~24–64 KB). |
| search_list | max(16, topk) | Larger → better recall, more node expansions. Milvus behavior. |
| maxVisits | 1000–2000 | Safety cap on expansions/latency. |

---

## 10. Codebook Rotation (two-phase)

1. Stage new codebooks under `/pq/codebook/{cbv'}/...`.
2. Re-encode nodes into `/pq/block/{cbv'}/...` via a sweeper (N at a time).
3. Flip `cbv_active := cbv'` in one tiny txn; query service watches/refreshes.
4. GC old `/pq/block/{old}` and codebooks when safe.

This avoids mixed-cbv reads on the hot path (queries use a single active cbv).

---

## 11. Capacity & Sizing (rules of thumb)

- **PQ RAM:** N × m bytes (since nbits=8 → 1 B/subvector). E.g., N=200M, m=96 → ~19.2 GB if fully cached; in practice keep LRU of hot blocks.
- **Adjacency storage:** N × degree × 8 B (u64 neighbors) plus overhead; e.g., N=200M, degree=64 → ~102 GB logical before compression/encoding.
- **FDB overhead:** plan for replication & metadata; keep values compact to reduce Redwood IO. (FDB splits keyspace into shards and spreads across storage teams automatically.)

---

## 12. Monitoring & Ops

- **Metrics:** QPS, p50/p95 latency; cache hit-rates (PQ, adjacency); RMW retries; txn conflicts; bytes read/written per request; queue lag.
- **Alarms:** queue backlog, storage server queue, Redwood page cache pressure, read bandwidth spikes.
- **Backups:** use FDB backup tooling; our payload is all KVs (no external blobs). (FDB's backup uses a mutation log per range under the hood.)
- **Wiggle/maintenance:** standard Redwood best practices apply.

---

## 13. Testing Plan

1. **Unit**
   - PQ encode/decode, LUT distances, protobuf codecs.
   - robust_prune, adjacency merge/prune idempotence.

2. **Integration**
   - Upsert-link-search pipeline on ~1–10 M points; verify recall vs ground-truth on a sampled subset.
   - Delete correctness (no dangling edges).

3. **Faults**
   - Kill link workers mid-batch (idempotence).
   - Inject commit conflicts; ensure retry logic succeeds.

4. **Perf**
   - Sweep search_list (16 → 256); plot recall/latency.
   - Measure RMW costs vs codes_per_block choices.

5. **Ops**
   - Codebook rotation rehearsal (two-phase).
   - Range hot-spot drill; (optional) add rangeconfig and observe rebalancing.

---

## 14. Implementation Sketch (modules)

- `storage/keys.ts` — tuple helpers for key construction.
- `proto/*.proto` — compiled to JVM/Go/TS as needed.
- `pq/encode.ts` — train(), encode(), LUT builder.
- `graph/link_worker.ts` — consumes link tasks, does B1/B2/B3.
- `graph/unlink_worker.ts` — consumes unlink tasks.
- `graph/repair_worker.ts` — periodic connectivity checks and repair.
- `query/service.ts` — search endpoint; LRUs; watch cbv_active.
- `admin/entry_maint.ts` — recompute hierarchical /entry.
- `admin/cbv_rotate.ts` — two-phase codebook rotation.
- `sketch/manager.ts` — vector sketch storage and retrieval.

---

## 15. Risk Register & Mitigations

- **PQ block contention during bursts** → stripe block keys; throttle enqueue per block.
- **Large transactions (e.g., writing too many neighbors)** → batch writes in ≤16-neighbor slices; validate affected-data budget.
- **Cache under-provisioning** → Redwood/read IOPS spikes → increase PQ LRU & Redwood cache_memory.
- **Mixed codebooks** → enforce two-phase rotation; query reads only cbv_active.
- **Graph disconnection** → implement periodic connectivity checks via graph repair worker; maintain backup entry points.
- **Codebook rotation quality loss** → use vector sketches for approximate reconstruction; maintain sketch versioning.

---

## 16. Milvus Parity Quick-map

| Milvus concept | This design |
|----------------|-------------|
| DISKANN.search_list default 16; auto = topk if larger | search_list = max(16, topk) in query path. |
| PQ defaults (nbits=8, m not defaulted) | nbits=8, m=D/2 if unset, codebooks in /pq/codebook. |
| DiskANN graph (Vamana) | Per-node adjacency with degree=64 (DiskANN default R). |

---

## 17. Graph Connectivity & Repair

### 17.1 Connectivity Monitoring

Periodic background job (every 6 hours) to detect graph partitions:
- Sample random nodes and attempt to reach all others via BFS
- Track connected component sizes in `/graph/meta/connectivity`
- Alert if components < 95% of total nodes

### 17.2 Repair Mechanism

When disconnection detected:
1. Identify orphaned nodes (unreachable from main component)
2. For each orphan, find k-nearest neighbors in main component using PQ
3. Add bidirectional edges to reconnect
4. Update entry points to include bridges between components

### 17.3 Future Scaling Considerations

For future scaling beyond single-cluster limits:
- **Bulk index building**: Construct graph offline in memory, then batch-write to FDB
- **Admission control**: Rate limiting and backpressure when queue depth exceeds thresholds  
- **Hybrid storage**: Consider keeping hot adjacency lists in Redis/memory tier for reduced FDB load

---

## 18. End-to-End Implementation Pseudocode

### 18.0 Constants & Defaults (Milvus-aligned)

```pseudocode
// PQ
DEFAULT_NBITS        = 8                // 1 byte / subvector
m                    = D / 2            // if not specified

// Graph (DiskANN-like)
degree               = 64               // max neighbors per node

// Search behavior (Milvus DISKANN)
function default_search_list(topk): return max(16, topk)

// Storage tuning
codes_per_block      = 512              // ≈ 512*m bytes per block (e.g., m=96 → ~49KB)
pq_block_target_size = 24..64 KB        // keep protobuf value around this size

// Query limits
beam                 = default_search_list(topk)
maxVisits            = 1500
```

### 18.1 Key Builders (tuple-encoded)

```pseudocode
function k_cfg(coll)                       -> tuple("/C", coll, "meta", "config")
function k_cbv_active(coll)               -> tuple("/C", coll, "meta", "cbv_active")
function k_entry(coll)                    -> tuple("/C", coll, "entry")

function k_codebook(coll, cbv, sub)       -> tuple("/C", coll, "pq", "codebook", cbv, sub)
function k_pq_block(coll, cbv, blockNo)   -> tuple("/C", coll, "pq", "block", cbv, blockNo)

function k_node(coll, nid)                -> tuple("/C", coll, "graph", "node", nid)
function k_sketch(coll, nid)              -> tuple("/C", coll, "sketch", nid)
```

### 18.2 FDB Helpers (txn, protobuf I/O, retries)

```pseudocode
// Generic transactional helper with automatic retry on conflicts / unknown result.
function with_txn(fn):
  backoff = 5ms
  loop:
    tr = db.beginTransaction()
    try:
      result = fn(tr)
      tr.commit()
      return result
    catch (commit_unknown_result or conflict):
      sleep(backoff); backoff = min(backoff*2, 200ms); continue

function get_pb(tr, key, snapshot=false):
  bytes = tr.get(key, snapshot)          // snapshot to avoid conflict ranges for reads-only
  if bytes == null: return null
  return protobuf_decode(bytes)

function set_pb(tr, key, message):
  tr.set(key, protobuf_encode(message))
```

### 18.3 PQ Routines (codebooks, LUT, scoring)

```pseudocode
// Cache codebooks for active cbv in memory per process.
global CODEBOOKS: Map<(coll, cbv) -> Codebooks>

// Load or return cached codebooks (pb.CodebookSub for sub=0..m-1).
function load_codebooks(coll, cbv):
  if CODEBOOKS.contains((coll, cbv)): return CODEBOOKS[(coll, cbv)]
  subs = []
  with_txn(tr -> {
    for sub in 0..m-1:
      subs[sub] = get_pb(tr, k_codebook(coll, cbv, sub), snapshot=true)
  })
  CODEBOOKS[(coll, cbv)] = subs
  return subs

// Encode a float vector to PQ code (m bytes).
function pq_encode(vec[D], codebooks):
  code[m] = uint8[ m ]
  for s in 0..m-1:
    qsub = slice(vec, s)                            // length D/m
    code[s] = argmin_c( distance(qsub, codebooks[s].codeword[c]) for c in 0..255 )
  return code

// Build LUT (m x 256) for a query vector.
function build_LUT_from_query(vec[D], codebooks):
  LUT = float[m][256]
  for s in 0..m-1:
    qsub = slice(vec, s)
    for c in 0..255:
      LUT[s][c] = distance(qsub, codebooks[s].codeword[c])
  return LUT

// PQ blocks cache (LRU by (coll,cbv,blockNo)).
global PQ_BLOCK_CACHE: LruCache<(coll, cbv, blockNo) -> PqCodesBlock>

// Read PQ code for nid, using LRU of blocks; snapshot read.
function load_pq_code(coll, cbv, nid):
  blockNo = floor(nid / codes_per_block)
  key = (coll, cbv, blockNo)
  block = PQ_BLOCK_CACHE.get(key)
  if block == null:
    block = with_txn(tr -> get_pb(tr, k_pq_block(coll, cbv, blockNo), snapshot=true))
    if block == null:
      // create a virtual empty block aligned to block_first_nid = blockNo * codes_per_block
      block = PqCodesBlock{ block_first_nid: blockNo*codes_per_block, codes_in_block: 0, codes: [] }
    PQ_BLOCK_CACHE.put(key, block)
  offset = nid - block.block_first_nid
  // Guard: if offset outside size, treat as not yet written → return zeros or sentinel
  return slice(block.codes, offset*m, (offset+1)*m)  // returns uint8[m]

// Compute PQ distance of a node given LUT.
function score_by_pq(coll, cbv, nid, LUT):
  code = load_pq_code(coll, cbv, nid)
  sum = 0
  for s in 0..m-1: sum += LUT[s][ code[s] ]
  return sum
```

### 18.4 Graph Utilities (robust prune, merge/prune)

```pseudocode
// Robust prune: keep <=degree neighbors; discard candidate c if it's too close to any kept neighbor.
// dist_q(x) must be the distance of x to the query; dist(x,y) is symmetric distance between nodes x,y.
// Here we approximate pairwise test by dist_q if you don't store pairwise distances.
function robust_prune(candidates_sorted_by_dist_to_q, degree, dist_q):
  R = []
  tau = 1.0                                // tweak 0.95..1.1 for aggression
  for c in candidates_sorted_by_dist_to_q:
    dominated = false
    for r in R:
      if dist_q(c) <= tau * dist_q(r):     // cheap dominance test proxy
        dominated = true; break
    if not dominated:
      R.append(c)
      if len(R) == degree: break
  return R

// Keep list sorted unique, cap at 'degree'.
function merge_prune(current: List<u64>, add: List<u64>, degree):
  merged = sort_unique( current ∪ add )
  if len(merged) <= degree: return merged
  // Trim by heuristic (e.g., keep lowest-degree IDs first or random trim)
  return merged[0:degree]
```

### 18.5 Upsert (batch of N) — Enqueue Linking Work

```pseudocode
// INPUT: items = List of (nid: u64, vec: float[D])
function upsert_batch(coll, items):
  cbv = with_txn(tr -> tr.get(k_cbv_active(coll), snapshot=true) or 0)
  codebooks = load_codebooks(coll, cbv)

  // Chunk to keep transactions small.
  for chunk in chunked(items, 512):
    with_txn(tr -> {
      for (nid, vec) in chunk:
        // Ensure NodeAdj exists
        node = tr.get(k_node(coll, nid), snapshot=false)
        if node == null:
          set_pb(tr, k_node(coll, nid), NodeAdj{ nid: nid, neighbors: [], ver: 1 })

        // PQ-encode client-side (sends M bytes to queue, not D floats)
        code = pq_encode(vec, codebooks)
        
        // Store vector sketch for future codebook retraining
        sketch = compute_vector_sketch(vec)  // 256-bit SimHash or PCA projection
        set_pb(tr, k_sketch(coll, nid), VectorSketch{sketch: sketch, version: 1})

        // Enqueue work on your transactional queue
        queue.enqueue(tr, topic="link", payload={
          "coll": coll, "nid": nid, "cbv": cbv, "code": code
        })
    })
```

### 18.6 Link Worker — Persist PQ Code + Link Neighbors

```pseudocode
worker link_worker():
  loop:
    task = queue.next(topic="link")        // blocks or long-polls
    if task == null: continue
    coll = task["coll"]; nid = task["nid"]; cbv = task["cbv"]; code = task["code"]

    // B1. Persist PQ code (RMW single block value)
    blockNo = floor(nid / codes_per_block)
    with_txn(tr -> {
      pb = get_pb(tr, k_pq_block(coll, cbv, blockNo), snapshot=false)
      if pb == null:
        pb = PqCodesBlock{
          block_first_nid: blockNo*codes_per_block,
          codes_in_block:  0,
          codes:           new_bytes(codes_per_block*m) // zeroed
        }
      offset = nid - pb.block_first_nid
      ensure_capacity(pb.codes, (offset+1)*m)          // grow if sparse
      write_bytes(pb.codes, offset*m, code)            // overwrite M bytes
      pb.codes_in_block = max(pb.codes_in_block, offset+1)
      pb.ver += 1
      set_pb(tr, k_pq_block(coll, cbv, blockNo), pb)
    })

    // B2. Neighbor discovery via PQ-only beam (snapshot reads)
    entries = with_txn(tr -> get_pb(tr, k_entry(coll), snapshot=true) or EntryList{nid:[]}).nid
    codebooks = load_codebooks(coll, cbv)
    // If you ALSO have the original vector, use build_LUT_from_query(vec, ..).
    // If not, approximate LUT from 'code' by mapping each codeword to its centroid.
    LUT = build_LUT_from_query( reconstruct_vec_from_code(code, codebooks), codebooks )

    cand = MinHeap()                       // (distance, nid)
    visited = Bitset()                     // or HashSet for sparse
    results = []                           // (distance, nid)

    for e in entries:
      cand.push( (score_by_pq(coll, cbv, e, LUT), e) )

    visits = 0
    local beam = default_search_list(topk=10)   // or pass down a desired topk; here use 16 by default
    while !cand.empty() and visits < maxVisits:
      (du, u) = cand.pop()
      if visited.contains(u): continue
      visited.add(u)
      results.append((du, u))
      visits += 1

      adj = with_txn(tr -> get_pb(tr, k_node(coll, u), snapshot=true) or NodeAdj{nid:u,neighbors:[]})
      expanded = 0
      for v in adj.neighbors:
        if visited.contains(v): continue
        dv = score_by_pq(coll, cbv, v, LUT)
        cand.push((dv, v))
        expanded += 1
        if expanded >= beam: break

    // B3. Robust prune to degree and write adjacency + back-links in small batches
    // Sort 'results' by distance (ascending) and strip self if present.
    sorted_frontier = sort_by(results, key=distance)
    neigh = robust_prune( map(nid, sorted_frontier), degree, dist_q = lambda x -> score_by_pq(coll, cbv, x, LUT) )

    // Update this node plus back-links 16 at a time
    for batch in chunked(neigh, 16):
      with_txn(tr -> {
        u = get_pb(tr, k_node(coll, nid), snapshot=false) or NodeAdj{nid:nid,neighbors:[]}
        u.neighbors = merge_prune(u.neighbors, batch, degree)
        u.ver += 1
        set_pb(tr, k_node(coll, nid), u)

        for v in batch:
          nv = get_pb(tr, k_node(coll, v), snapshot=false) or NodeAdj{nid:v,neighbors:[]}
          nv.neighbors = merge_prune(nv.neighbors, [nid], degree)
          nv.ver += 1
          set_pb(tr, k_node(coll, v), nv)
      })

    queue.ack(task)
```

### 18.7 Delete (batch of N) — Enqueue Unlink

```pseudocode
// INPUT: nids = List<u64>
function delete_batch(coll, nids):
  for chunk in chunked(nids, 512):
    with_txn(tr -> {
      for nid in chunk:
        queue.enqueue(tr, topic="unlink", payload={ "coll": coll, "nid": nid })
    })
```

### 18.8 Unlink Worker — Remove Back-links and Node

```pseudocode
worker unlink_worker():
  loop:
    task = queue.next(topic="unlink")
    if task == null: continue
    coll = task["coll"]; nid = task["nid"]

    // Read current neighbors (snapshot ok)
    adj = with_txn(tr -> get_pb(tr, k_node(coll, nid), snapshot=true))
    neigh = (adj == null) ? [] : adj.neighbors

    // Remove back-links in small batches
    for batch in chunked(neigh, 16):
      with_txn(tr -> {
        for v in batch:
          nv = get_pb(tr, k_node(coll, v), snapshot=false)
          if nv != null:
            nv.neighbors = remove(nv.neighbors, nid)
            nv.ver += 1
            set_pb(tr, k_node(coll, v), nv)
      })

    // Clear this node's adjacency
    with_txn(tr -> tr.clear( k_node(coll, nid) ))

    // (Optional) zero PQ slot by RMW of its block (same as link B1 but write zeros)
    // blockNo = floor(nid / codes_per_block); with_txn(tr -> zero_slot(tr, coll, cbv_active, blockNo, nid))

    queue.ack(task)
```

### 18.9 Search (top-k via PQ-only beam over per-node adjacency)

```pseudocode
// INPUT: q[D], topk
function search_topk(coll, q, topk):
  cbv = with_txn(tr -> tr.get(k_cbv_active(coll), snapshot=true))
  codebooks = load_codebooks(coll, cbv)
  LUT = build_LUT_from_query(q, codebooks)

  entries = with_txn(tr -> get_pb(tr, k_entry(coll), snapshot=true) or EntryList{nid:[]}).nid
  local_search_list = default_search_list(topk)   // Milvus behavior
  beam = local_search_list

  cand = MinHeap()         // (distance, nid)
  visited = Bitset()
  results = TopK(k=topk)   // max-heap of size k

  for e in entries:
    cand.push( (score_by_pq(coll, cbv, e, LUT), e) )

  visits = 0
  while !cand.empty() and visits < maxVisits:
    (du, u) = cand.pop()
    if visited.contains(u): continue
    visited.add(u); visits += 1
    results.consider( (du, u) )

    adj = with_txn(tr -> get_pb(tr, k_node(coll, u), snapshot=true) or NodeAdj{nid:u,neighbors:[]})
    expanded = 0
    for v in adj.neighbors:
      if visited.contains(v): continue
      dv = score_by_pq(coll, cbv, v, LUT)
      cand.push((dv, v))
      expanded += 1
      if expanded >= beam: break

  // Return sorted ascending by distance
  ans = results.items()           // list of (d, nid)
  return sort_by(ans, key=distance)
```

### 18.10 Entry List Maintenance (hierarchical seed set)

```pseudocode
// Recompute periodically (e.g., hourly) with hierarchical strategy.
function refresh_entry_list(coll):
  // Primary: k-means medoids or high-degree nodes
  medoids = compute_medoids_from_sketches(coll, k=32)
  
  // Random samples for diversity
  random = sample_random_nodes(coll, k=16)
  
  // High-degree nodes for connectivity
  high_deg = scan_highest_degree_nodes(coll, k=16)
  
  with_txn(tr -> set_pb(tr, k_entry(coll), EntryList{ 
    nid: medoids, 
    random_samples: random,
    high_degree: high_deg 
  }))
```

### 18.11 Codebook Rotation (two-phase)

```pseudocode
// Phase 1: stage new cbv' codebooks
function stage_codebooks(coll, cbv_prime, codebooks_prime):
  with_txn(tr -> {
    for sub in 0..m-1:
      set_pb(tr, k_codebook(coll, cbv_prime, sub), codebooks_prime[sub])
  })

// Phase 2: re-encode NIDs in batches, writing /pq/block/{cbv'} via link-like RMW
function reencode_all(coll, cbv_prime):
  codebooks = load_codebooks(coll, cbv_prime)
  for batch in iterate_all_nids_in_blocks(coll, codes_per_block):
    // Use stored sketches to approximate original vectors
    for nid in batch:
      sketch = with_txn(tr -> get_pb(tr, k_sketch(coll, nid), snapshot=true))
      if sketch != null:
        approx_vec = reconstruct_from_sketch(sketch)  // Better than PQ reconstruction
        new_code = pq_encode(approx_vec, codebooks)
        // Write new code to cbv_prime blocks
        persist_pq_code(coll, cbv_prime, nid, new_code)

// Phase 3: atomically flip active cbv
function activate_cbv(coll, cbv_prime):
  with_txn(tr -> tr.set(k_cbv_active(coll), cbv_prime))

// Phase 4: GC old blocks when safe
function gc_old_cbv(coll, cbv_old):
  delete_range_prefix("/C", coll, "pq", "block", cbv_old)
  delete_range_prefix("/C", coll, "pq", "codebook", cbv_old)
```

### 18.12 Notes on Sizing & Contention (practical)

```pseudocode
// Keep PQ block updates cheap:
codes_per_block  in [256..512]
value size       in [~24KB .. ~64KB]          // protobuf encoded
neighbors batch  size 16 per write txn        // << 10 MB affected data

// If a single block becomes a hotspot:
stripe_count = 4
k_pq_block(coll, cbv, blockNo, stripe = hash(nid)%stripe_count)
// Readers pick stripe by same hash; writers RMW only that stripe value.
```

---

## Appendix A — Why FDB Fits Here

- **Small objects & switches:** FDB excels at compact KVs and fast transactional flips (e.g., cbv_active).
- **Limits are compatible:** our value sizes (24–64 KB) and write batches fit within 100 KB/value and ~10 MB/txn; search is snapshot-read-heavy (≪5 s/txn).

## Appendix B — Useful References

- [Milvus DISKANN overview and enabling docs.](https://milvus.io/docs/disk_index.md)
- [Milvus IVF_PQ defaults & PQ param notes.](https://milvus.io/docs/ivf-pq.md)
- [Milvus DISKANN search_list default/behavior.](https://github.com/milvus-io/milvus/discussions/35302)
- [DiskANN default R=64 (max degree).](https://github.com/microsoft/DiskANN/blob/main/workflows/filtered_in_memory.md)
- [FDB known limits & value size guidance.](https://apple.github.io/foundationdb/known-limitations.html)
- [Redwood readiness / cache memory guidance.](https://forums.foundationdb.org/t/redwood-storage-engine-documentation-for-7-1-is-missing/3575)
- [Optional per-range placement via rangeconfig (FDB 7.3+).](https://forums.foundationdb.org/t/can-i-force-key-prefixes-into-different-shards/4440)