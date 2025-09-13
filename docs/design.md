Design of DiskANN-PQ Vector Index on FoundationDB

Overview and Goals

This document describes a Java-based vector indexing system that integrates DiskANN (a disk-based approximate nearest neighbor graph) with Product Quantization (PQ) compression on top of FoundationDB. The design supports online indexing from an initially empty index, handling insertions, deletions, and queries in real-time. Key goals include: fixed vector dimensionality (set at index creation), configurable distance metrics (L2 or cosine), efficient segment-based indexing with Active/Pending/Sealed states, and low-latency top-10 KNN queries in ≤30 ms. All vector data, index structures, and metadata are stored in FoundationDB as key-value pairs (serialized via Protocol Buffers), and all updates occur within ACID transactions. We outline how segments are managed, how DiskANN graphs and PQ codebooks are built per segment, and how queries traverse these structures efficiently. We also detail a FoundationDB key-value schema optimized for minimal round-trips and guidance on implementation (caching, concurrency, parameter tuning, etc.).

Implementation snapshot (2025-09-07)

- Rotation semantics are strict-cap: when an ACTIVE segment reaches `maxSegmentSize`, the next insert rotates and is assigned `vecId=0` in the new ACTIVE segment within the same transaction. We also advance a `maxSegmentKey` sentinel to ensure queries discover all segments.
- Compaction lifecycle: source segments are marked COMPACTING; a destination segment is created in WRITING state (hidden from queries) and populated, then sealed to SEALED before a registry swap.
- Search defaults use BEST_FIRST traversal with elevated `ef` and a conservative `maxExplore`. BEAM traversal is supported but deprecated; using it emits a one-time WARN. Query semantics by state: SEALED and COMPACTING are searched via PQ/graph; ACTIVE and PENDING use brute-force; WRITING is not searched.
- Query prefetch (optional) fires an async warmup for PQ codebooks across SEALED segments to reduce cold-start latency.
- SegmentCaches use Caffeine AsyncLoadingCache for both PQ codebooks and adjacency, with `asyncLoadAll` and internal chunking to bound transaction size; query paths call `getAll` to leverage bulk loads.
- Observability exposes OpenTelemetry gauges for cache sizes and stats; metric attributes can be added via configuration.
- Logging levels are tuned for quiet operation: per-query details are DEBUG; worker lifecycle is DEBUG; BEAM usage logs WARN once.

Data Model and Key-Value Schema

All data is stored in FoundationDB using a structured key design to group related information together. The index is divided into segments (each up to a fixed size, e.g. 100,000 vectors). Each segment has its own vector data, graph, and PQ compression metadata. The `maxSegmentSize` is a hard cap: once an ACTIVE segment's count reaches this threshold, the next insert rotates to a new ACTIVE segment before writing. The vector that triggers rotation is stored as `vecId=0` in the new segment, while the previous segment flips to PENDING in the same transaction and is later SEALED by the background builder.

Key-Value Schema (DirectoryLayer-based): We use DirectorySubspaces to avoid manual tuple prefixing. Each index has an `indexDir` with child subspaces for `segments/`, `tasks/`, a registry `segmentsIndex/`, and a `gid/` subtree for global ids. Each segment is its own child DirectorySubspace `segments/<segId>/` with subspaces for `vectors/`, `pq/{codebook,codes/}`, and `graph/`.

Global Ids (gid): Public APIs return a single 64‑bit gid per vector. Internally we keep two mappings under `gid/`:
- `gid/map`: `pack(gid) -> pack(segId, vecId)`
- `gid/rev`: `pack(segId, vecId) -> pack(gid)`

Compaction preserves gids by rewriting both mappings when records move to the WRITING destination segment, ensuring ids remain stable even as `(segId, vecId)` changes.
• Index Metadata – Key: indexDir.pack(("meta")) → Global index settings (serialized), e.g. {dimension, metric, maxSegmentSize, pq_m, pq_k, graph_degree, oversample}.
• Current Segment Pointer – Key: indexDir.pack(("currentSegment")) → Integer segId of the ACTIVE segment.
• Segment Metadata – Key: segments/<segId>/.pack(("meta")) → SegmentMeta for segId (state, count, timestamps, deleted_count).
• Vector Record – Key: segments/<segId>/vectors/.pack((vecId)) → Serialized VectorRecord with embedding/payload/deleted.
• PQ Codebook – Key: segments/<segId>/pq/.pack(("codebook")) → Serialized PQCodebook for the segment.
• PQ Compressed Codes – Key: segments/<segId>/pq/codes/.pack((vecId)) → PQ code bytes for vecId.
• Graph Adjacency – Key: segments/<segId>/graph/.pack((vecId)) → Encoded neighbor IDs for vecId in the segment graph (about R neighbors).

Tombstones: Deletions are handled via tombstone markers. Instead of immediately removing a vector’s data, we mark it as deleted:
•	The VectorRecord contains a boolean deleted flag that is set to true in a deletion transaction. (Alternatively, a separate key (indexName, "segment", <segId>, "deleted", <vecId>) -> true could mark deletions, but using an inline flag avoids extra keys.)
•	Adjacency lists are not updated immediately on deletion (removing edges on the fly is complex in a graph index). Instead, search operations will check the tombstone flag and skip over deleted vectors in results ￼ ￼. A background merge/compaction process will eventually rebuild segments to purge tombstoned entries (discussed later).

Figure: FoundationDB Key-Value Schema (for one segment):

segmentsIndex/(segId)                            -> empty (presence indicates known segment)
segments/<segId>/meta                            -> { state: ACTIVE/PENDING/SEALED/WRITING/COMPACTING, count, ... }
segments/<segId>/vectors/(vecId)                 -> { VectorRecord proto (embedding, metadata, deleted_flag) }
segments/<segId>/pq/codebook                     -> { PQCodebook proto (centroids for each subspace) }
segments/<segId>/pq/codes/(vecId)                -> (binary PQ code for vector)
segments/<segId>/graph/(vecId)                   -> [ neighbor_vecId1, neighbor_vecId2, ... neighbor_vecIdR ]

All keys under `segmentsDir` are lexicographically ordered by `segId` then sub‑keys, so data for a segment is clustered (enabling range reads over a segment’s data). Listing segments is done by scanning `segmentsIndex/<segId>` under the index root.

Segment Lifecycle and Online Indexing

To support dynamic updates, the index is organized into segments that transition through three states: ACTIVE, PENDING, and SEALED ￼. This approach is inspired by LSM-tree style batching and is used in vector databases like Milvus/Manu for handling streaming data efficiently ￼ ￼.
•	ACTIVE Segment: At any time, new vectors are inserted into a single active segment. This segment is in memory (for fast writes) and accepts new entities until it reaches a predefined size (e.g. 100,000 vectors) or a certain time interval passes without new inserts ￼. The vector dimensionality and metric are fixed for the index, so all segments share those properties. Insertions are performed in a single FoundationDB transaction:
•	The current active segment ID is read from (indexName, "currentSegment").
•	A new vecId is assigned (e.g. by reading and incrementing a counter or using the segment’s count).
•	The new vector’s data is serialized into a VectorRecord proto and written to (indexName, "segment", segId, "vector", vecId).
•	The segment’s metadata (count) is incremented. If the segment’s count will reach the max segment size after this insert, the transaction also flips the segment state to PENDING and creates a new segment metadata entry for a fresh Active segment:
•	e.g. set (indexName, "segment", segId, "meta").state = PENDING and (indexName, "segment", newSegId, "meta") = {state: ACTIVE, count:0} in the same commit.
•	Update (indexName, "currentSegment") = newSegId to point to the new active segment for subsequent inserts.
•	This transactional approach ensures that segment state changes and vector writes are atomic. If the transaction fails (e.g. conflict), it will be retried to avoid partial updates.
•	PENDING Segment: Once an active segment is full, it becomes PENDING, meaning no new writes occur to it. It’s essentially a read-only frozen segment that has not yet been indexed into a graph. In the PENDING state:
•	The segment’s vectors are still searchable, but since the DiskANN graph and PQ codebook are not built yet, any queries covering this segment will use brute-force scanning (linear search) over its vectors ￼. This is identical to how active segments are queried (discussed in Query Processing).
•	A background indexing task is triggered for the pending segment. This task will build the DiskANN graph and PQ compression for that segment (described in the next section). Multiple segments could be pending if ingestion outpaces indexing, but typically the system tries to index them one at a time.
•	PENDING segments remain queryable (via brute force) until they are fully indexed and sealed. This ensures no downtime for search on recent data.
•	SEALED Segment: After the background task finishes building the graph index and PQ codebook, the segment is marked SEALED. SEALED segments are immutable: they have a complete DiskANN graph built and a PQ codebook trained for their vectors ￼. In a FoundationDB transaction at the end of indexing:
•	All PQ codebook and graph data for the segment (neighbor lists, PQ codes, etc.) will have been written to FoundationDB (possibly in batches, see Index Construction). The segment’s meta state is updated to SEALED.
•	Queries can now use DiskANN graph traversal with PQ for this segment instead of brute force.
•	The in-memory representation of this segment’s full vectors can be released if desired (since the data is on disk and queries will mostly use PQ codes), to save memory.

This segmented approach allows online indexing: new data lands in active segments for fast ingestion, while older segments are indexed in the background. It also inherently supports concurrency – inserts always go to the active segment (which changes infrequently), and queries search a mix of sealed (graph) and active/pending (brute-force) segments.

Additionally, because segments are capped in size and sealed segments are read-only, the system can scale to billions of vectors by chaining segments, without any single graph growing unbounded. Each segment’s DiskANN graph is independent and can be stored on disk (in FoundationDB) efficiently.

DiskANN Graph Construction and PQ Compression

DiskANN is a graph-based ANN algorithm that uses a Vamana graph structure for efficient search on SSD/disk ￼ ￼. It builds a navigable small-world graph where each vector is a node connected to a set of nearest neighbors. DiskANN is optimized for disk by ensuring a greedy search converges in relatively few hops, reducing random reads ￼ ￼. In our design, each sealed segment has its own DiskANN graph.

Index Building (Background Task): When a segment enters PENDING state, a background indexer process (or thread) will:
1.	Load Segment Data: Retrieve all vectors of that segment (either from an in-memory buffer if the active segment was cached, or by reading from FoundationDB keys). Having the full precision vectors is needed to build the graph and train PQ.
2.	Build Graph (DiskANN/Vamana): Construct the ANN graph for the segment:
•	Use the Vamana algorithm (as in DiskANN) to insert vectors one by one into a graph. This starts with an initial random or centroid-based graph and iteratively refines neighbor links using greedy search and a pruning strategy ￼ ￼. Each vector ends up with up to R neighbors (configurable).
•	Ensure the graph satisfies the searchable neighborhood properties so that a greedy search from any entry point will find near neighbors quickly ￼ ￼. Typically DiskANN picks a medoid (center vector) as a starting node for searches ￼ – we will store such an entry point for the segment if needed (could be stored in segment meta).
•	The graph building is computationally heavy (O(N log N) or more), but acceptable for N=100k offline. We may parallelize some steps (e.g. using multiple threads to compute distances during neighbor search).
•	Graph storage: Once built, for each vector we prepare its adjacency list of neighbor IDs (each neighbor ID refers to another vector in the same segment). These lists will be written to FoundationDB as the graph keys described in the schema. Each adjacency list is usually small (tens of IDs), easily fitting under FDB’s value size limit. Writing 100k keys (one per vector) might exceed a single transaction’s limits, so the adjacency lists will be stored in batches (e.g. 1000 vectors per transaction) to avoid hitting transaction size limits. After writing all adjacency entries, a final transaction can mark the segment as sealed.
3.	Train PQ Codebook: Using the full-precision vectors of the segment, perform Product Quantization training:
•	Partition the vector’s dimensions into M equal subspaces (as configured). For example, a 1000-dimensional vector with M=20 subspaces yields 20 groups of 50 dimensions each.
•	Run a clustering (typically k-means) on each subspace across all vectors in the segment to learn K centroid vectors per subspace (e.g. K=256). This yields the PQ codebook for that segment ￼ ￼.
•	Each vector can then be encoded by finding, for each subspace, the index of the nearest centroid. The concatenation of these centroid indices (one byte each if K<=256) forms the PQ code for the vector ￼ ￼. For example, a vector might be represented as 20 bytes if M=20 and each index is 1 byte.
•	Storing PQ data: The codebook (all centroids) is stored as one key (segId, "pq", "codebook"). Then, PQ codes for each vector are written to (segId, "pq", "code", vecId) keys. Like adjacency, writing 100k codes might be done in batches of transactions. Each code value is small (tens of bytes).
4.	Finalize Segment (Seal): After graph and PQ data are written, a final transaction sets the segment’s state to SEALED in its metadata. At this point, the segment is fully indexed:
•	The full vectors remain in FoundationDB (they may be needed for exact distance re-ranking or future rebuilds), but searches can largely rely on the compressed codes and graph neighbors for speed.
•	Optionally, the system may drop any in-memory copy of the segment’s vectors, to free RAM, relying on disk (FDB) + cache for retrieval.

Notably, DiskANN’s design often stores full vectors on disk and keeps compressed vectors (PQ codes) in memory to balance accuracy and speed ￼ ￼. Our system mirrors that approach: by storing full vectors in FoundationDB (backed by SSD) and keeping PQ codes + graph links in memory cache (or quickly accessible), we reduce memory usage while still enabling fast distance computations. Each query will use the PQ codes for most distance calculations, and only fetch full vectors for a small number of candidates to refine the results (see Query section).

All parameters for index building are configurable:
•	Graph parameters: e.g. R (max neighbors per node), L (search beam width / candidate list size), and possibly the alpha parameter from Vamana for pruning. These affect graph quality and search speed and can be tuned in the index metadata.
•	PQ parameters: M (number of sub-vectors) and K (centroids per sub-vector). These impact compression ratio and accuracy. For instance, using more sub-vectors or a larger K improves accuracy but increases memory and CPU usage for search.
•	Distance Metric: L2 vs Cosine. For Cosine similarity, note that if all vectors are normalized, cosine distance is monotonic with L2 distance, so the same graph can serve both; or we could use inner product as metric. The distance metric is stored in index meta and used both in graph building (d(p,q) calculation) and search. If cosine is used, we might normalize vectors on insertion or adjust distance computations accordingly (e.g. use dot product for similarity).
•	Segment size: (perhaps 100k by default, but configurable). Larger segments mean fewer graphs (less overhead per data size) but slower indexing and possibly slower search within a segment (more nodes to traverse). 100k is a balance between manageability and performance.

Query Processing and Search Path

At query time, the system needs to find the top-K nearest neighbors to a given query vector across all segments, with low latency. The query flow is designed to leverage the strengths of each segment type:
1.	Query Fan-out to Segments: The query is dispatched to all segments in parallel (or in batches if thread pool is limited). Each segment (active, pending, or sealed) will produce its list of candidate nearest vectors (with distances):
•	SEALED Segments: Use the DiskANN graph with PQ acceleration to perform approximate search.
•	ACTIVE/PENDING Segments: Use brute-force linear search, since these segments have no graph yet ￼. (Active and pending are essentially the same for querying: both hold full vectors and require scanning.)
2.	Search in Sealed Segment (DiskANN + PQ): When searching a sealed segment, the process is:
•	Initialization: Start from an entry point in the graph. Typically, DiskANN stores a representative starting node (e.g. the dataset medoid or a random node) ￼. This could be stored in the segment metadata (e.g. meta.entryNode). Initialize a candidate list (min-heap or list) with this entry node, calculating its distance to the query.
•	Greedy Graph Traversal: Perform a best-first search (like HNSW/DiskANN):
•	Pop the closest not-yet-visited candidate from the list.
•	Fetch that node’s neighbor list from FoundationDB (key: graph, nodeId). Because of key locality, this is a fast lookup (and often the data may be cached from a previous visit).
•	For each neighbor, compute the approximate distance to the query using PQ codes:
•	The query vector is first encoded with the segment’s PQ codebook: i.e. compute the query’s sub-vector residuals or distances to each centroid in each subspace. This yields lookup tables of size K for each subspace (precomputed once per query) ￼ ￼.
•	Then for a neighbor’s PQ code (which is available either via cache or via a quick lookup of the pq, code, neighborId key), compute the distance by summing the distance of each codebook centroid indicated by the code to the query’s corresponding sub-vector ￼ ￼. This gives an approximate L2 distance (or inner product) very quickly without fetching the full vector.
•	Optimization: If neighbor PQ codes are frequently needed, they may be cached in memory. Alternatively, we can store the PQ code along with the neighbor list to save an extra lookup. However, storing codes in two places would double-write; instead, we rely on either a quick fetch from the pq, code key or a prior caching of all codes in memory for that segment (since 100k codes * ~M bytes might be only a few MB).
•	Add each neighbor to the candidate list (skip if already visited). The DiskANN search will keep track of a visited set to avoid cycles.
•	Continue this graph exploration until the candidate list size reaches the search parameter limit L or the next closest candidate is further than the current best K found.
•	This graph search finds a set of nearest neighbor candidates quickly by exploring only a small fraction of nodes (much smaller than the full segment). Batching & Parallelism: To minimize FoundationDB round-trips, the implementation can batch-read multiple neighbor lists at once if the candidate list adds many neighbors. For example, if we pop one node and get 64 neighbors, instead of 64 sequential gets, the system can issue asynchronous reads for all those neighbor keys in parallel (FoundationDB’s asynchronous API allows pipelined reads). Similarly, reading multiple PQ codes can be batched. The keys being lexicographically ordered (neighbors and codes grouped by segment) means a range read can potentially fetch a contiguous set of codes if neighbor IDs are near each other. In practice, an async fan-out of reads and relying on FDB’s high IOPS will achieve low latency.
•	Segment Top-K: Once the search completes, we have the top-K (or more, if oversampling) approximate nearest neighbors within that segment, along with their approximate distances.
3.	Search in Active/Pending Segment (Brute Force): For each active or pending segment (likely only one of each at a time):
•	The query is compared with every vector in the segment by computing the exact distance (e.g. L2 or cosine) on the full vector. This is obviously O(N) per segment, but N is at most 100k. With optimized math libraries or parallelization, 100k 1000-d dot products can be done quite fast (e.g. using SIMD or GPU). Moreover, the active segment is often much smaller than max size in typical operation.
•	To minimize overhead, the active segment’s vectors are likely already in memory (since recent inserts have been buffered). We can maintain an in-memory array of vectors for the active segment. The same for a pending segment if it was just frozen – the background thread can leave the data available for query until indexing is done. This way, brute-force search doesn’t require reading 100k keys from FDB on each query – it simply iterates an array in RAM.
•	Calculate distance for each vector and keep track of the top K. This yields an exact top-K for that segment (with exact distances).
•	If tombstones are present, those vectors are skipped during this scan ￼ ￼.
4.	Merging Segment Results: The query coordinator (which could be the thread orchestrating the fan-out) collects the partial results from each segment. Now we have, say, several lists of candidates (one per segment). To get the global top-10, we merge these lists:
•	A simple way: put all candidates together and take the top 10 by distance. Since each segment’s results were top-K for that segment, and distances are comparable across segments (same metric), this merge is efficient.
•	Re-ranking with Exact Distances: Because sealed segments used PQ (approximate distances), the top results need to be refined. We oversample the candidates from sealed segments to improve accuracy ￼ ￼. For example, if the user wants top-10, each sealed segment might return top-15 or top-20 candidates (the oversampling factor, e.g. 1.5x or 2x) ￼. After merging all segments’ candidates, we might have 50 or 100 total candidates.
•	For these merged candidates, fetch their full vectors from FoundationDB (this is a small number of vectors total). We can batch-get all needed (segment, vector) keys in one go. Then compute exact distances between the query and each of these candidates using the original vector values.
•	Sort by the true distance and take the top 10 as the final result set. This re-ranking corrects any small errors introduced by PQ compression ￼, ensuring the returned neighbors are the true nearest by the chosen metric. (In practice, DiskANN with PQ has >95% accuracy, and refining top-10 from top-15/20 candidates yields final results that are nearly identical to exact search ￼ ￼.)
5.	Return Results: The final top-10 neighbors (often including their original IDs or metadata from the VectorRecord) are returned. The query latency is kept low by:
•	Parallelizing segment searches (possibly using a thread per segment, or an async event loop that issues all FDB reads in parallel).
•	Using PQ to avoid heavy computation on full 1000-d vectors for most candidates (distances are mostly computed via lookup tables and byte codes, which is very fast in CPU).
•	Minimizing FDB calls: reading neighbor lists and PQ codes in batches, and caching frequently accessed data (see Caching below).
•	Limiting brute-force work to small segments (active ones) that can be scanned quickly in memory.

This approach is essentially performing a distributed search across segments followed by a merge, which is analogous to approaches seen in vector databases like Mani/Manu ￼ ￼. Each segment yields its top-k, and then a global merge gives the final top-k ￼. Deleted vectors (tombstones) are filtered out either during segment search or at least before final ranking ￼.

Note: The search algorithm parameters (L - candidate list size, ef etc.) can be tuned. A higher L means exploring more nodes in the graph, increasing recall at cost of more FDB reads. We need to choose these such that 10-NN recall is high while keeping latency under 30ms. The 30ms budget might involve exploring on the order of a few hundred graph nodes per sealed segment (which in practice is enough with DiskANN). Also, because each segment is at most 100k, the search per segment is reasonably bounded.

Deletions and Segment Merging

Deletions: When a vector needs to be deleted, we issue a transaction that:
•	Marks the vector’s deleted flag in its VectorRecord to true (tombstone). This is a small in-place update to the proto or a separate marker key.
•	Optionally, updates a deletion count in the segment’s metadata (for stats).
•	No immediate change is made to the graph or PQ data. The vector’s entry still exists in the index structures but is now inert. Queries will identify it as deleted and skip it in results ￼. If a deleted vector is encountered during graph traversal (e.g. as a neighbor), the search can ignore it and continue (potentially exploring an extra neighbor to compensate).

If deletions accumulate, the effectiveness of the index could degrade (e.g. many dead-ends in the graph). To mitigate this, the system performs periodic segment merges/compactions:
•	Small sealed segments (far below capacity) or segments with a high fraction of tombstoned vectors are selected for merge ￼.
•	The merge job will read all live vectors from one or more source segments and combine them into a new segment. For example, if two sealed segments each have only 50k live vectors after deletions, they could be merged into a new segment of 100k.
•	The merge process resembles initial index building: it creates a new segment (Active state), bulk-inserts all the collected vectors, then builds a DiskANN graph and PQ for them (or directly builds and marks sealed, since this is effectively an offline rebuild).
•	During the merge, the old segments remain available for queries. After the new merged segment is sealed and ready, a transaction can swap them:
•	Mark the old segments as “retired” (or delete their metadata so they are no longer searched).
•	Add the new segment to the index metadata (state SEALED).
•	Optionally, write a mapping so that any query that still touches old segment IDs will know they are superseded (though if the coordinator is careful, it won’t query retired segments).
•	Delete the old segments’ keys from FoundationDB to reclaim space (this can be done gradually or in the background to avoid a huge transaction – e.g. range clear in chunks).
•	This compaction ensures that the number of segments doesn’t grow without bound and that each sealed segment is densely packed with live vectors (maximizing search efficiency). The system could also merge very small segments (like if an index was created and only got a few vectors, rather than leaving a tiny sealed segment, it might merge it with another).
•	Merging is a heavy operation, so it might be scheduled during low load times or triggered when certain thresholds are exceeded (e.g. >50% vectors in a segment deleted, or >N segments exist). It trades off some background cost to keep query performance high.

Because FoundationDB transactions are all-or-nothing, we ensure that any metadata updates that swap segments or mark deletions are transactional. For example, marking a vector deleted is one transaction; marking segments retired and a new one active is one transaction. This guarantees the index’s logical view is always consistent (no partial deletion or half-merged state visible to queries). Even though the physical removal of data (deleting many keys) might be done in batches, we would first update metadata so that queries ignore those segments/keys, then asynchronously clean up the storage.

Caching and Performance Optimizations

To meet the ~30 ms query latency target, careful use of caching and batched access is required, given that FoundationDB is a distributed KV store (accessing it has network + I/O latency). We employ the following optimizations:
•	Caffeine Cache for Codebooks: PQ codebooks (segId: pq: codebook) are relatively small (~hundreds KB) and are needed for every query on a sealed segment. We will cache each segment’s codebook in memory (using Caffeine, an efficient Java in-memory cache) ￼ ￼. When a query for segment <segId> starts, the codebook is retrieved from cache (loading from FDB only on a cache miss). This allows immediate computation of the query’s PQ centroid lookup tables without an FDB read.
•	Cached Graph Nodes / Neighbor Lists: We also use a cache for frequently accessed graph data. This could be at the granularity of whole adjacency lists or even entire node records. For example, if certain nodes are often visited as entry points or part of many searches, their neighbor list (and possibly their PQ code) can be kept hot. We can cache entries like “node <segId>:<vecId> -> {neighbors[], PQ code}”. Caffeine can be configured with a maximum size (in number of nodes or memory footprint) to avoid overuse. Empirically, DiskANN search tends to visit neighbors that are spatially close to many queries (if query distribution is not uniform), so a cache can improve performance by avoiding redundant FDB reads for those nodes.
•	Batch Reads and Async Pipelines: The FoundationDB Java client supports asynchronous transactions and batch reads. Our query implementation will take advantage of that:
•	For example, when expanding a graph node, instead of synchronously reading each neighbor’s adjacency one by one, we collect all needed keys and issue one getBatch or perform a getRange if the keys are contiguous. Since our keys are tuple-ordered, reading a range from (segId, "graph", minNeighborId) to (segId, "graph", maxNeighborId) will pull possibly all neighbor lists in that range in one call. We can also pipeline further neighbor reads while processing current ones.
•	Similarly, reading multiple PQ codes or multiple full vectors for reranking is done with a single batch call. For instance, to fetch 50 vectors for final re-ranking, we can issue a single transaction.getRange over those 50 specific keys (if they share a common prefix) or 50 individual async gets and then join them. FoundationDB will handle these requests efficiently, often in parallel on the storage servers.
•	Locality and Prefetch: The key schema ensures segment data is clustered. If the DiskANN graph algorithm tends to visit neighbors with nearby IDs, many adjacency lists might reside on the same FDB storage shard, benefiting from locality. We can detect access patterns and prefetch subsequent neighbor lists. E.g., if we visit node 42 and its neighbors include 43, 45, 47, there’s a chance those adjacencies reside near each other in the key space; a single range read could fetch 42’s neighbors and some of 43,45,47 in one go.
•	Parallel Segment Search: We distribute query work across segments, potentially on multiple threads or async tasks. Because each segment search is largely independent (except using the same query vector), this parallelism can reduce overall latency – the slowest segment search determines the response time. If one sealed segment search and one brute-force scan can happen concurrently, it maximizes utilization of CPU and I/O. In practice, a thread pool could handle sealed segment searches concurrently, or if running on multiple query nodes, segments might be partitioned among nodes ￼.
•	In-Memory Active Segment: As noted, keep the active (and maybe recent pending) segments in memory for quick brute-force search. This is a form of caching the newest data by default. It also accelerates the common case where recent vectors (active segment) are frequently queried (temporal locality).
•	Distance Computation Optimizations: Use efficient libraries or algorithms for computing distances:
•	We may utilize Java’s vectorized operations (e.g. using the JVM’s vector API or offloading to BLAS). For instance, computing 100k dot products of 1000-d vectors can be sped up with SIMD instructions.
•	Cosine similarity queries can reuse precomputed vector norms (if needed) or assume normalized vectors so that it reduces to dot product.
•	PQ distance computation can be optimized by computing the query’s subspace centroid distances once (array of length K for each subspace) and storing them in an array. Then each code lookup is just M array accesses and additions, which is very fast in Java (especially if data is in L1 cache). This is essentially how product quantization enables fast distance estimates ￼ ￼.

With these measures, we aim to keep the search path lean. The most expensive part – accessing disk for graph neighbors – is mitigated by caching and limited hops. DiskANN’s property of logarithmic search hops (it tends to visit O(log n) nodes for n points) ￼ ￼ means even a 100k segment might be traversed via, say, 50–100 nodes on average to find top-10. That translates to perhaps 50–100 FDB reads of neighbor lists (which can be batched/pipelined), and the rest are in-memory operations. Combined with parallel segment processing, this can comfortably fit in tens of milliseconds.

Configurability and Tuning

All major parameters are exposed as configuration, to allow tuning for different use cases:
•	Vector Dimension & Metric: Set at index creation. The dimension (e.g. 1000) and metric (L2 or COSINE) are stored in the index metadata. The metric influences distance computations and can also be used to decide if vectors should be normalized (for cosine).
•	Segment Size: The maximum number of vectors per segment (e.g. 100k) can be configured based on memory/disk trade-offs. Larger segments mean less overhead per vector but slower indexing. Smaller segments mean faster background indexing jobs but more segments to search.
•	PQ Parameters:
•	M (subvectors) and K (centroids per subvector). For example, M=16, K=256 is a common choice yielding 16-byte codes per vector ￼. These affect the compression ratio. The system could auto-set these based on dimension (e.g. Azure’s guidance uses ~96 dims compressed for 1536 original ￼), but we allow manual override.
•	PQ Training samples: How many vectors to sample for k-means training (could use the whole segment or a subset). A configurable pqSampleSize can cap the training set for speed vs accuracy trade-off ￼.
•	DiskANN Graph Parameters:
•	R (max neighbors per node): e.g. 64 or 32. Higher R gives better connectivity at cost of index size and search time.
•	L_search (search breadth): how many neighbors to explore (like HNSW’s efSearch). The user could specify an ef or similar that controls the quality vs latency. We might also expose L_build (the graph construction search parameter) if advanced tuning is needed.
•	Alpha (pruning parameter): in Vamana, controls graph sparseness. Possibly fixed default (like 1.2) or configurable for experts.
•	Parallelism and Caching:
•	Number of threads for background indexing (can index multiple segments in parallel if CPU allows).
•	Query parallelism (how many threads for searching segments, or how many segments one thread handles).
•	Cache sizes for Caffeine (max entries or memory). These should be tunable to accommodate different deployment sizes.
•	Merge Policies:
•	Thresholds for triggering segment merge (e.g. if a segment’s live vectors fall below X% of capacity due to deletions, or if more than Y small sealed segments exist, etc.).
•	Option to enable/disable automatic merges, or run them at scheduled times.

All these parameters would be part of a configuration object and persisted in FoundationDB (under the index meta or a separate config key). They can be loaded at service startup and adjusted if needed (some changes might only apply to new segments or require reindexing).

Transactional Integrity and Implementation Notes

Building on FoundationDB ensures strong consistency for updates:
•	Every insertion, deletion, or segment state change is done in a single FDB transaction, which commits or fails as a whole. This guarantees, for example, that we never mark a segment as sealed without its graph actually present, or never partially insert a vector without metadata.
•	The FoundationDB ACID semantics relieve us from worrying about partial failures. However, we must mind the 10 MB transaction size and ~5 sec transaction time limits. That’s why large writes (like building an index for 100k vectors) are chunked into multiple transactions.
•	The Java FoundationDB client should be used with retry loops for transactions. We will use FDB’s Directory layer or raw tuple keys as above for clarity.

Concurrency: The system has distinct components that work concurrently:
•	Ingestor thread(s): handle incoming insert requests, appending to the active segment. Likely single-threaded per index to avoid ordering issues (or using a global FDB transaction ordering anyways).
•	Index Builder thread: picks up pending segments and performs graph/PQ building. This can be decoupled to not stall inserts (which go to a new active). The builder will use snapshot reads on FDB (or better, the in-memory copy) to get segment data, and then multiple transactions to write the results.
•	Query threads: can run at the same time, reading the current state. Because all updates are transactional, a query might see a segment in PENDING or just sealed state consistently. We ensure to check segment meta states at query start. If a segment happens to be sealing while a query is in progress, there’s no harm – the query might either treat it as PENDING (brute force) or SEALED depending on when it read the meta. Either approach yields a correct (approximate) result. The next query will see the updated state and use the graph. The slightly inconsistent view window is acceptable in this eventually consistent search context (similar to Manu’s delta consistency where minor staleness is allowed) ￼ ￼.
•	Locking: Because each segment is mostly independent, we can avoid coarse locking. The main point of contention is the currentSegment for inserts – which can be handled with one transaction at a time (FDB will serialize conflicting transactions). The background builder writes to different keys (the pending segment’s graph keys), which don’t conflict with ongoing inserts to the new active segment. Deletions touch individual vector keys and segment meta counters, which is fine. We will, however, coordinate if a segment is being merged: the merge operation will likely lock those segments from being queried or updated, perhaps by a meta flag state like “MERGING”. During merge, inserts don’t go to those segments anyway (they’re sealed), and queries can either still read them (if we allow) or be directed to the new merged segment once ready.

Memory considerations: The design uses memory for caching codebooks, some graph nodes, and active segment vectors. None of these caches need to grow without bound:
•	Active segment is at most 100k vectors * 1000 dims * 2 bytes ≈ 200 MB of raw data, which is acceptable in memory on a modern server.

Search Defaults and Caching (Updated)

•	Default traversal is BEST_FIRST (priority-queue) with a larger ef (Milvus DISKANN style). BEAM mode is retained for debugging and A/B testing but is not the default.
•	Prefetch: VectorIndex.query pre-warms PQ codebooks for all SEALED segments via the codebook async cache (fire-and-forget) to reduce cold starts.
•	Async caches: Both PQ codebooks and adjacency are backed by AsyncLoadingCache with asyncLoadAll implementations. Bulk loads are batched to reduce the number of FDB transactions.
•	Batch sizes are configurable per deployment via VectorIndexConfig:
   – codebookBatchLoadSize (default 10,000)
   – adjacencyBatchLoadSize (default 10,000)
•	Codebooks: each maybe ~0.5 MB, if we had 50 segments that’s 25 MB.
•	Graph neighbors: 100k * R (say 64) * 4 bytes ≈ 25 MB per segment if fully cached. Caching all segments fully isn’t necessary – we rely on partial caching and fast I/O.

Traversal Parameters

•	Mode
   – BEST_FIRST (default): priority queue by approximate (PQ LUT) distance.
   – BEAM (debug): fixed-width frontier for experiments and parity checks.

•	Entry seeds (pivots)
   – PQ-seeded candidates: top-N by PQ distance as initial seeds.
   – Optional deterministic random pivots (SearchParams.SeedStrategy) to broaden coverage.

•	ef and maxExplore
   – ef controls breadth before exact rerank; higher ef → better recall, more I/O/CPU.
   – maxExplore caps total expansions to bound worst-case latency.

•	Per-segment fan-in
   – Limit candidates per segment (≥ k, typically k×oversample) to keep tail latency bounded when many segments exist.
   – Future: dynamic budgeting by segment size/age and early-stopping when improvement stalls.

•	Cosine normalization
   – For COSINE, exact rerank can normalize on read if requested via SearchParams.

•	Prefetch and caches
   – PQ codebooks: optional prefetch; test-only flag can block until warm.
   – Adjacency: async bulk loaders with getAll; batch cache fetches for frontier nodes.

Guidance for Implementation: Below we outline how some operations can be implemented in pseudocode to tie the design together:
•	Insertion (Single Vector): (simplified pseudocode)

void insertVector(float[] vector, Metadata meta) {
db.run(tx -> {
  int segId = decodeInt(tx.get(indexDir.pack(("currentSegment"))));
  // Resolve per-segment subspaces
  Dir segDir = segmentsDir.open(tx, List.of(String.valueOf(segId))); // conceptual
  byte[] metaBytes = tx.get(segDir.pack(("meta")));
  SegmentMeta segMeta = SegmentMeta.parse(metaBytes);
  int newVecId = segMeta.getCount();
  // Prepare VectorRecord proto and write
  VectorRecord rec = new VectorRecord(vector, meta, false);
  tx.set(segDir.sub("vectors").pack((newVecId)), rec.serialize());
  segMeta = segMeta.toBuilder().setCount(newVecId + 1).build();

  // Check if segment is now full
  if (segMeta.getCount() >= MAX_SEGMENT_SIZE) {
    // Mark current as PENDING
    tx.set(segDir.pack(("meta")), segMeta.toBuilder().setState(PENDING).build().toByteArray());
    // Create next ACTIVE segment (pre-create subspaces and meta)
    int newSegId = segId + 1; // or allocate via maxSegmentKey
    Dir next = segmentsDir.createOrOpen(tx, List.of(String.valueOf(newSegId)));
    SegmentMeta nextMeta = SegmentMeta.newBuilder().setSegmentId(newSegId).setState(ACTIVE).setCount(0).build();
    tx.set(next.pack(("meta")), nextMeta.toByteArray());
    tx.set(indexDir.pack(("currentSegment")), encodeInt(newSegId));
  } else {
    // Just update count on current segment
    tx.set(segDir.pack(("meta")), segMeta.toByteArray());
  }
}); // automatic retry on conflict
// If segment became PENDING, trigger background index build for segId (e.g. add to a queue)
}

This transaction reads and writes a few keys (currentSegment pointer, the segment meta, the new vector key). All done within limits.

	•	Background Index Build (Pseudo):

void buildIndexForSegment(int segId) {
// 1. Load data
List<VectorRecord> vectors = fetchAllVectors(segId);  // possibly from memory or FDB range read
// 2. Build graph
Graph graph = DiskAnnBuilder.buildGraph(vectors, R, L_build, metric);
// 3. Train PQ
PQCodebook codebook = PQ.train(vectors, M, K, sampleSize);
List<byte[]> codes = new ArrayList<>();
for (VectorRecord v : vectors) {
codes.add(codebook.encode(v.embedding));
}
// 4. Write graph & PQ in batches
List<FDBKeyValue> kvs = new ArrayList<>();
kvs.add(new FDBKeyValue((indexName, "segment", segId, "pq", "codebook"), codebook.serialize()));
for (int i = 0; i < vectors.size(); i++) {
int vecId = i;
kvs.add(new FDBKeyValue((indexName, "segment", segId, "pq", "code", vecId), codes.get(i)));
// neighbors list
List<Integer> neighbors = graph.getNeighbors(vecId);
kvs.add(new FDBKeyValue((indexName, "segment", segId, "graph", vecId), serializeNeighbors(neighbors)));
// commit in chunks to avoid big transaction
if (kvs.size() >= 1000) {
db.run(tx -> {
for (FDBKeyValue kv : kvs) tx.set(kv.key, kv.value);
});
kvs.clear();
}
}
// commit remaining
db.run(tx -> { for (FDBKeyValue kv : kvs) tx.set(kv.key, kv.value); });
// 5. Seal segment
db.run(tx -> {
SegmentMeta segMeta = tx.get((indexName, "segment", segId, "meta")).deserialize();
segMeta.state = SEALED;
tx.set((indexName, "segment", segId, "meta"), segMeta.serialize());
});
}

In practice, error handling and corner cases (e.g. if segment was deleted before finishing, etc.) should be handled. Also, one might compress or quantize data further if needed.

	•	Query (Pseudo):

List<Result> knnQuery(float[] query, int K) {
// Compute normalized query or PQ lookup tables up-front
Map<Integer, List<Result>> segmentResults = new ConcurrentHashMap<>();
// Parallel search each segment
for each segment seg:
if (seg.state == SEALED) {
segmentResults.put(seg.id, searchSealedSegment(seg, query, K * oversample));
} else { // ACTIVE or PENDING
segmentResults.put(seg.id, bruteForceSearchSegment(seg, query, K));
}
// Merge results
List<Result> allCandidates = mergeAll(segmentResults.values());
// Remove any deleted results
allCandidates.removeIf(res -> res.vectorRecord.deleted);
// Re-rank top K by exact distance
List<Result> topK = allCandidates.stream().sorted(byExactDistance(query)).limit(K).collect();
return topK;
}

The searchSealedSegment would implement the graph traversal with PQ as discussed, and bruteForceSearchSegment does a linear scan. Both return a list of candidates with distances (approx for sealed, exact for active). The final exact distance computation requires fetching full vector data for those candidates (which we can get from the VectorRecord either from an in-memory cache or a batch FDB read).

Diagram of Query Path: (Pseudo-visual)

          Query Vector
               |
     --------------------------
     |    |     |     |   (parallel to each segment)
    Seg0 Seg1  Seg2  ... 
     |     |     |
    (sealed?)   (active?) ...
     |           |
Graph-PQ search   Brute-force scan
|           |
Top-K0       Top-K1      ... (per segment)
\     |_____/
\____|____/    (merge)
|
Global candidates (oversampled)
|
Fetch full vectors for candidates
|
Compute exact distances
|
Final Top-K results

The design balances read performance (through PQ and graph search) with write/update simplicity (by segmenting and avoiding in-place graph modifications). Each segment is an independent index that can be built or merged without affecting others, which aligns with building blocks used in modern vector databases ￼ ￼.

In conclusion, this Java-based DiskANN+PQ implementation on FoundationDB provides a scalable, dynamic vector search index. It leverages FoundationDB’s transactional KV store to reliably manage data and metadata, and uses DiskANN’s disk-optimized graph for efficient search on large datasets ￼, combined with product quantization to compress vectors for speed ￼. By carefully designing the key schema and using caching, parallelism, and batching, the solution can achieve real-time performance (top-10 in ~30ms) even as the dataset grows to billions of vectors. It supports continuous inserts and deletes, with background processes to maintain index freshness (sealing segments, merging, etc.), all while ensuring correctness through ACID transactions.

Metrics and Testing (current)

- Metrics: `SegmentCaches.registerMetrics` publishes the following gauges via OpenTelemetry:
  - `vectorsearch.cache.size{cache=codebook|adjacency}`
  - `vectorsearch.cache.hit_count{cache=...}` and `vectorsearch.cache.miss_count{cache=...}`
  - `vectorsearch.cache.load_success_count{cache=...}` and `vectorsearch.cache.load_failure_count{cache=...}`
  Additional attributes can be injected at construction time via `VectorIndexConfig.metricAttribute(key, value)`.
- Tests: Metrics are asserted using the SDK’s `InMemoryMetricReader`. Each test installs a temporary `SdkMeterProvider` with the reader and resets `GlobalOpenTelemetry` in setup/teardown.

Sources:
•	Microsoft DiskANN paper: graph-based ANN index with PQ compression (stores full vectors on disk, compressed in memory) ￼ ￼.
•	Azure Cosmos DB: use of Product Quantization and oversampling to refine top-K results ￼ ￼.
•	Manu (Zilliz Cloud) architecture: segment lifecycle (growing→sealed), brute-force on growing segments, merging segments, and distributed search merge ￼ ￼.

## Compaction Planning & Throttling

- Planning selects SEALED segments with the smallest `count` first and breaks ties by older `created_at_ms`.
- The anchor segment is included when sealed; candidates are added until roughly 80% of `maxSegmentSize` is reached (up to 4 segments).
- Throttling is enforced via `maxConcurrentCompactions` in `VectorIndexConfig`:
  - `0` disables compactions (planner may run but the worker will not transition any segment to COMPACTING).
  - When `> 0`, the maintenance worker counts segments currently in COMPACTING; if the count is at the limit, the task is a no‑op.
- The worker atomically marks selected candidates as COMPACTING before calling the compaction request hook, preventing overlap.
