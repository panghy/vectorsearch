# Segment-Based Codebooks and Original Vector Storage

This document proposes replacing the single global codebook design with a scalable, segment-based architecture and introducing original vector storage for accurate reranking and future retraining.

## Problems with the Single Codebook
- Training lock-in: Replacing the global codebook requires re-encoding all vectors and risks downtime.
- Query coupling: One LUT per query assumes a single codebook; multiple versions are awkward.
- Drift over time: A single codebook degrades as distribution shifts; retraining forces disruptive migration.
- Sketches don’t help: SimHash/PCA sketches are too lossy for high-quality retraining or exact reranking.

## Goals
- Online retraining without re-encoding existing data.
- Zero-downtime codebook rotation by isolating new writes.
- Accurate results via exact reranking using original vectors.
- Bounded memory and simple caching on the query path.

## High-Level Design

### 1) Immutable Segments (per-codebook)
- Introduce segments as write-once partitions, each pinned to a single codebook.
- New writes go to the active ingest segment; when retraining occurs, create a new segment with the new codebook and switch ingest to it.
- Existing segments remain sealed (read-only) and continue to use their original codebooks.

### 2) Multi-Codebook Query
- Maintain multiple codebooks in RAM (LRU) keyed by `segment_id`.
- For each query, build a per-codebook LUT on demand and cache it within the query context: `Map<segmentId, LUT>`.
- When scoring a candidate, use the LUT for that candidate’s segment.
- Final reranking uses original vectors, removing inter-codebook approximation bias.

### 3) Store Original Vectors (drop sketches)
- Replace sketches with original vectors, stored as fp16 to balance accuracy and size.
- Use exact distances for final reranking of top-K results.
- Retraining can sample original vectors directly; sketches become unnecessary.

## Keyspace Changes (additions)

Current (abbrev):
- `/C/{coll}/pq/codebook/{cbv}/{sub}` -> `pb.CodebookSub`
- `/C/{coll}/pq/block/{cbv}/{blockNo}` -> `pb.PqCodesBlock`

Proposed:
- `/C/{coll}/segments/meta/{segId}` -> `pb.SegmentMeta{ seg_id, codebook_version, status, stats... }`
- `/C/{coll}/segments/active` -> `uint64 segId` (active ingest segment)
- `/C/{coll}/segment/{segId}/pq/codebook/{sub}` -> `pb.CodebookSub`
- `/C/{coll}/segment/{segId}/pq/block/{blockNo}` -> `pb.PqCodesBlock`
- `/C/{coll}/vector/{nid}` -> raw or fp16-packed vector bytes (original vector)

Notes:
- Keep global graph keys; add `segment_id` to `NodeAdj` (or store `/node_segment/{nid} -> segId`) so the query can select the correct LUT.
- Per-vector keys are preferred over block storage for vectors to respect the 100KB/value limit (e.g., 768 dims × 2B ≈ 1.5KB per vector).

## Query Path
- Load entry points (possibly across segments).
- For each segment encountered, compute/query-cached LUT: `pq.buildLookupTable(queryVec)`.
- Traverse graph with PQ distances using the appropriate segment’s LUT.
- Accumulate candidates; fetch their original vectors (`/vector/{nid}`) and compute exact distances for final reranking.
- Return exact distances to users.

## Upsert Path
1. Encode PQ codes using active ingest segment’s codebook.
2. Write codes to `/segment/{segId}/pq/block/{blockNo}` (RMW, small txn).
3. Write original vector to `/vector/{nid}` (fp16-packed, small value).
4. Link in graph (as today), recording `segId` for the node.

## Codebook Rotation Workflow
- Train new codebook from sampled original vectors (optionally biased to recent data).
- Create new segment with this codebook; atomically set `/segments/active` to the new `segId`.
- No re-encoding of existing segments is required.
- Optional, background: selective migration of hot shards to the new segment.

## Caching
- Codebooks: `AsyncLoadingCache<segmentId, ProductQuantizer>` (bounded, expire-after-access).
- Per-query LUTs: lightweight `Map<segmentId, float[][]>` built on first use per query; typically only a handful of segments touched.
- PQ blocks: cache unchanged, now keyed by `(segmentId, blockNo)`.

## Protobuf Additions (sketch; not binding)
- `SegmentMeta { uint64 seg_id; uint32 codebook_version; enum Status { ACTIVE, SEALED }; uint64 vector_count; }`
- Add `uint64 segment_id` to `NodeAdj` or introduce `/node_segment/{nid} -> segId`.

## API Changes (proposed)

New:
- `SegmentRegistry` (new): create/list/seal segments, set active ingest segment, read `SegmentMeta`.
- `OriginalVectorStorage` (new): get/put vectors; fp16 pack/unpack helpers.

Refactor:
- Replace `CodebookStorage` with `CodebookRegistry`: methods accept `segmentId` and load codebooks under `/segment/{segId}/...`.
- Update `BeamSearchEngine`: build LUT per segment from query vector; rerank top-K with exact distances using `OriginalVectorStorage`.

## Migration Plan
Clean cutover only (no backward-compat required):
1. Add `OriginalVectorStorage` and enable reranking.
2. Introduce `SegmentRegistry` and per-segment codebook storage.
3. Update upsert path to write original vectors and track `segId`.
4. Switch query to multi-codebook LUTs and exact reranking.
5. Remove `VectorSketchStorage`.

## Rationale & Trade-offs
- Multiple codebooks increase query-time LUTs: mitigated by usually few active segments and per-query caching.
- Per-vector keys add read cost for reranking: bounded to top-K×rerank_factor; acceptable for accuracy gains.
- No mandatory re-encoding on rotation: avoids heavy background jobs and downtime.

## Open Questions
- Cross-segment graph edges: keep global graph with `segment_id` per node or use light cross-segment entry links.
- Exact distance metric choice for reranking under COSINE/IP: prefer consistent normalization strategy.

## Summary
Segmented codebooks plus original vector storage deliver scalable retraining and higher accuracy without disruptive re-encoding. The design keeps hot paths simple, respects FDB constraints, and provides a clean migration path from the current single-codebook model.
