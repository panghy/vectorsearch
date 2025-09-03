# Segment-Based PQ Architecture Design

## Overview

This document describes the transition from a global Product Quantization (PQ) codebook to a segment-based architecture, similar to Milvus's approach. This fundamental change will improve scalability, eliminate the need for complex codebook rotation, and enable more efficient incremental indexing.

## Current Architecture (Global Codebook)

### Problems with Current Approach
1. **Single Global Codebook** - One codebook for entire index
   - Becomes stale as data distribution changes
   - Requires retraining on entire dataset
   - Re-encoding all vectors is prohibitively expensive

2. **Complex Rotation Required** - Two-phase rotation mechanism needed
   - Phase 1: Train new codebook, store as new version
   - Phase 2: Re-encode ALL vectors with new codebook
   - Atomic switch once complete
   - Very expensive and complex to coordinate

3. **Scalability Issues**
   - Training time grows with dataset size
   - Re-encoding time grows linearly with data
   - Memory requirements for training increase

## Proposed Architecture (Segment-Based)

### Core Concepts

#### 1. Data Segments
- **Definition**: Self-contained units of vectors with their own PQ codebook
- **Size**: Target 512MB per segment (configurable)
- **Structure**: Each segment contains:
  - Local PQ codebook trained on segment's data
  - PQ-encoded vectors using local codebook
  - Graph adjacency for nodes in segment
  - Metadata (vector count, creation time, etc.)

#### 2. Growing vs Sealed Segments
- **Growing Segment**: 
  - Active segment accepting new vectors
  - Stored in memory (not yet PQ-encoded)
  - Searchable using original vectors
  - Once reaches size threshold → sealed

- **Sealed Segment**:
  - Immutable segment with PQ encoding
  - Has trained codebook specific to its data
  - Stored in FDB with efficient block structure
  - Never modified (only soft-deleted)

### Implementation Design

#### Key Changes to Storage Layout

```
Current (Global):
/C/{coll}/pq/codebook/{version}/{sub}     -> Single global codebook
/C/{coll}/pq/block/{cbv}/{blockNo}        -> All blocks use same codebook

Proposed (Segment-Based):
/C/{coll}/segment/{segmentId}/meta        -> SegmentMetadata
/C/{coll}/segment/{segmentId}/codebook/{sub} -> Segment-local codebook
/C/{coll}/segment/{segmentId}/pq/{blockNo}   -> PQ blocks for this segment
/C/{coll}/segment/{segmentId}/graph/{nodeId} -> Graph adjacency
/C/{coll}/segment/active                  -> Current growing segment ID
```

#### New Protobuf Definitions

```protobuf
message SegmentMetadata {
  int64 segment_id = 1;
  enum State {
    GROWING = 0;
    SEALING = 1;
    SEALED = 2;
    COMPACTING = 3;
  }
  State state = 2;
  int32 vector_count = 3;
  int64 size_bytes = 4;
  Timestamp created_at = 5;
  Timestamp sealed_at = 6;
  repeated int64 node_id_range = 7;  // [min_id, max_id]
  int32 codebook_version = 8;        // Local version within segment
}

message GrowingSegmentData {
  int64 segment_id = 1;
  map<int64, RawVector> vectors = 2;  // nodeId -> vector
  int32 vector_count = 3;
  int64 memory_bytes = 4;
}
```

### Core Workflows

#### 1. Vector Insertion
```java
public CompletableFuture<Void> upsert(long nodeId, float[] vector) {
  return db.runAsync(tx -> {
    // 1. Get or create growing segment
    GrowingSegment growing = getOrCreateGrowingSegment(tx);
    
    // 2. Add vector to growing segment
    growing.addVector(nodeId, vector);
    
    // 3. Check if segment should be sealed
    if (growing.getSize() >= SEGMENT_SIZE_THRESHOLD) {
      return sealSegment(tx, growing);
    }
    
    return CompletableFuture.completedFuture(null);
  });
}
```

#### 2. Segment Sealing Process
```java
private CompletableFuture<Void> sealSegment(Transaction tx, GrowingSegment growing) {
  // 1. Mark segment as SEALING
  updateSegmentState(tx, growing.getId(), State.SEALING);
  
  // 2. Train PQ codebook on segment's vectors
  List<float[]> vectors = growing.getAllVectors();
  float[][][] codebook = ProductQuantizer.train(numSubvectors, metric, vectors);
  
  // 3. Store codebook for this segment
  storeSegmentCodebook(tx, growing.getId(), codebook);
  
  // 4. Encode all vectors with segment's codebook
  ProductQuantizer pq = new ProductQuantizer(codebook);
  for (Entry<Long, float[]> entry : growing.getVectors()) {
    byte[] pqCode = pq.encode(entry.getValue());
    storeSegmentPqCode(tx, growing.getId(), entry.getKey(), pqCode);
  }
  
  // 5. Build graph connections within segment
  buildSegmentGraph(tx, growing.getId(), vectors);
  
  // 6. Mark segment as SEALED
  updateSegmentState(tx, growing.getId(), State.SEALED);
  
  // 7. Create new growing segment
  createNewGrowingSegment(tx);
  
  return CompletableFuture.completedFuture(null);
}
```

#### 3. Search Across Segments
```java
public CompletableFuture<List<SearchResult>> search(float[] query, int k) {
  return db.runAsync(tx -> {
    // 1. Get all searchable segments (SEALED + current GROWING)
    List<SegmentInfo> segments = getSearchableSegments(tx);
    
    // 2. Search each segment in parallel
    List<CompletableFuture<List<Candidate>>> segmentSearches = 
      segments.stream()
        .map(segment -> searchSegment(tx, segment, query, k * 2))
        .collect(Collectors.toList());
    
    // 3. Merge results from all segments
    return CompletableFuture.allOf(segmentSearches.toArray(new CompletableFuture[0]))
      .thenApply(v -> {
        PriorityQueue<Candidate> topK = new PriorityQueue<>(k);
        for (CompletableFuture<List<Candidate>> future : segmentSearches) {
          topK.addAll(future.join());
        }
        return extractTopK(topK, k);
      });
  });
}

private CompletableFuture<List<Candidate>> searchSegment(
    Transaction tx, SegmentInfo segment, float[] query, int k) {
  
  if (segment.getState() == State.GROWING) {
    // Search using exact distances on original vectors
    return searchGrowingSegment(tx, segment, query, k);
  } else {
    // Load segment's PQ and search using PQ distances
    return loadSegmentPQ(tx, segment)
      .thenCompose(pq -> {
        // Encode query with segment's codebook
        byte[] queryCode = pq.encode(query);
        return beamSearchInSegment(tx, segment, queryCode, k);
      });
  }
}
```

#### 4. Background Compaction
```java
// Periodically merge small segments and remove deleted vectors
private CompletableFuture<Void> compactSegments() {
  return db.runAsync(tx -> {
    // 1. Find segments eligible for compaction
    List<SegmentInfo> candidates = findCompactionCandidates(tx);
    
    if (candidates.size() < 2) {
      return CompletableFuture.completedFuture(null);
    }
    
    // 2. Create new merged segment
    long mergedId = createMergedSegment(tx);
    
    // 3. Collect all valid vectors from candidate segments
    List<float[]> validVectors = new ArrayList<>();
    for (SegmentInfo segment : candidates) {
      validVectors.addAll(getValidVectors(tx, segment)); // Skip soft-deleted
    }
    
    // 4. Train new codebook on merged data
    float[][][] mergedCodebook = ProductQuantizer.train(
      numSubvectors, metric, validVectors);
    
    // 5. Store merged segment with new codebook
    storeMergedSegment(tx, mergedId, mergedCodebook, validVectors);
    
    // 6. Atomically replace old segments with merged
    replaceSegments(tx, candidates, mergedId);
    
    return CompletableFuture.completedFuture(null);
  });
}
```

### Migration Strategy

#### Phase 1: Dual-Mode Operation
1. Keep existing global codebook for current data
2. New insertions go to segment-based structure
3. Search queries check both systems

#### Phase 2: Background Migration
1. Read existing PQ codes in batches
2. Create segments of appropriate size
3. Re-train codebooks per segment
4. Migrate graph structure

#### Phase 3: Cutover
1. Atomic switch to segment-only mode
2. Remove global codebook code paths
3. Clean up old data

### Benefits of Segment-Based Architecture

1. **No Global Retraining Needed**
   - Each segment has optimal codebook for its data
   - New data doesn't affect existing segments

2. **Incremental Indexing**
   - Only seal and encode when segment is full
   - Cost is O(segment_size), not O(dataset_size)

3. **Better Locality**
   - Codebooks match local data distribution
   - Can result in better quantization quality

4. **Simpler Operations**
   - No complex two-phase rotation
   - Deletes handled via compaction
   - Updates create new segments

5. **Scalability**
   - Training time bounded by segment size
   - Can process segments in parallel
   - Memory requirements are constant

### Configuration Parameters

```java
public class SegmentConfig {
  // Target size for sealed segments
  private long segmentSizeBytes = 512 * 1024 * 1024; // 512MB
  
  // Minimum vectors before sealing (even if size not reached)
  private int minVectorsPerSegment = 10000;
  
  // Maximum vectors in a segment (even if size not reached)  
  private int maxVectorsPerSegment = 1000000;
  
  // Threshold for compaction (segments below this size are candidates)
  private long compactionThresholdBytes = 128 * 1024 * 1024; // 128MB
  
  // Maximum segments to merge in one compaction
  private int maxCompactionMerge = 4;
  
  // How often to run compaction
  private Duration compactionInterval = Duration.ofHours(1);
}
```

### Performance Considerations

#### Memory Management
- Only one growing segment in memory at a time
- Sealed segments load PQ codes on-demand with caching
- Can set memory limits for growing segment

#### Write Performance
- Writes go to in-memory growing segment (fast)
- Sealing happens asynchronously in background
- No blocking on PQ training

#### Query Performance  
- Growing segment uses exact distances (higher quality)
- Sealed segments use PQ distances (faster)
- Parallel search across segments
- Result merging overhead is minimal

### Testing Strategy

1. **Unit Tests**
   - Test segment lifecycle (growing → sealing → sealed)
   - Test codebook training per segment
   - Test search across multiple segments

2. **Integration Tests**
   - Test migration from global to segment-based
   - Test compaction with soft-deleted vectors
   - Test query consistency during sealing

3. **Performance Tests**
   - Measure sealing time vs segment size
   - Compare search latency: global vs segmented
   - Test memory usage with multiple segments

## Implementation Steps

1. **Define New Storage Schema** - Create protobuf definitions and key structure
2. **Implement Segment Manager** - Handle segment lifecycle and state transitions  
3. **Adapt PQ Storage** - Make PQ storage segment-aware
4. **Modify Insert Path** - Route inserts to growing segment
5. **Implement Sealing** - Background task to seal full segments
6. **Update Search** - Search across all segments and merge
7. **Add Compaction** - Merge small segments and clean deleted data
8. **Migration Tools** - Convert existing global index to segments
9. **Testing** - Comprehensive test coverage
10. **Performance Tuning** - Optimize segment size and parameters

## Conclusion

Moving to a segment-based architecture fundamentally solves the codebook staleness problem while improving scalability and operational simplicity. This change should be implemented before adding original vector storage, as it affects the core structure of the index.