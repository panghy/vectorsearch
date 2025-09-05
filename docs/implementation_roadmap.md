# FDB-backed Vector Search Implementation Roadmap

This roadmap tracks the implementation progress of the FDB-backed PQ + DiskANN vector search system described in `original_ideation.md`.

## 🎯 Project Overview

Building a millisecond-latency Approximate Nearest Neighbor (ANN) search system using:
- **FoundationDB** as the distributed key-value store
- **Product Quantization (PQ)** for vector compression and fast distance computation
- **DiskANN-style graph traversal** for high-recall search
- **Transactional queue system** for online upsert/delete operations

---

## 📊 Implementation Status

### ✅ **Phase 1: Foundation & Storage Layer** (COMPLETED)

#### Core Infrastructure
- [x] **Protocol Buffers** - Define all data structures (`Config`, `CodebookSub`, `PqCodesBlock`, `NodeAdj`, etc.)
- [x] **FDB Key Management** - Tuple-encoded keyspace structure (`VectorIndexKeys`)
- [x] **Storage Transaction Utils** - Common FDB transaction patterns and protobuf I/O
- [x] **Comprehensive Test Coverage** - 93% line coverage, 91% instruction coverage, 81% branch coverage

#### PQ (Product Quantization) System  
- [x] **PQ Core Algorithm** - K-means clustering, vector encoding/decoding
- [x] **Codebook Management** - Versioned storage with fp16 compression and caching
- [x] **PQ Block Storage** - Efficient blocked storage of quantized codes with batch operations
- [x] **Distance Computation** - LUT-based fast distance calculations

#### Vector Index Framework
- [x] **FdbVectorSearchIndex** - Main index interface with configuration management
- [x] **Vector Utilities** - Float16 conversion, distance metrics, vector operations

---

## 🚧 **Phase 2: Graph & Search System** (IN PROGRESS)

### Graph Management
- [x] **Node Adjacency Storage** - Per-node neighbor lists with degree bounds (COMPLETED)
- [x] **Robust Pruning Algorithm** - DiskANN-style neighbor selection with diversity (COMPLETED)
- [x] **Back-link Maintenance** - Bidirectional edge consistency (COMPLETED)
- [x] **Graph Connectivity Monitoring** - Detect and repair disconnected components (COMPLETED)

### Search Engine
- [x] **Beam Search Implementation** - Graph traversal with PQ-based scoring (COMPLETED ✅)
- [x] **Entry Point Management** - Hierarchical entry strategies (medoids, random, high-degree) (COMPLETED)
- [x] **Query Processing Pipeline** - End-to-end search with configurable parameters (COMPLETED ✅)
- [x] **Search Result Ranking** - Top-k result selection and scoring (COMPLETED)

---

## 📋 **Phase 3: Online Operations** (IN PROGRESS)

### Upsert System
- [x] **Transactional Queue Integration** - Link worker task management (COMPLETED)
- [x] **Link Worker Implementation** - Neighbor discovery and graph linking (COMPLETED)
- [x] **PQ Code Persistence** - Block-level RMW operations (COMPLETED)
- [x] **Batch Processing** - Efficient handling of bulk upserts with adaptive sizing (COMPLETED)

### Delete System  
- [ ] **Unlink Worker** - Remove nodes and back-links
- [ ] **Orphan Detection** - Identify disconnected nodes
- [ ] **Graph Repair** - Reconnect components after deletions

### Background Workers
- [x] **Entry List Maintenance** - Periodic refresh of search entry points (COMPLETED)
- [x] **Connectivity Monitor** - Graph health checking and repair (COMPLETED)
- [ ] **Statistics Collection** - Index health and performance metrics

---

## 🔄 **Phase 4: Advanced Features** (FUTURE)

### Codebook Management
- [ ] **Two-Phase Rotation** - Safe codebook updates without downtime
- [x] **Vector Sketches** - SimHash/PCA for codebook retraining (STORAGE ONLY - NO RETRAINING IMPL)
- [ ] **Quality Monitoring** - Track quantization error and recall

### Performance & Scaling
- [x] **Cache Management** - LRU caches for PQ blocks and adjacency data (IMPLEMENTED)
- [ ] **Hotspot Mitigation** - Block striping for high-contention scenarios
- [ ] **Range Configuration** - FDB placement optimization
- [ ] **Admission Control** - Rate limiting and backpressure

### Operational Features
- [ ] **Monitoring & Metrics** - Comprehensive observability
- [ ] **Configuration Management** - Runtime parameter tuning
- [ ] **Backup & Recovery** - Data protection strategies
- [ ] **Load Testing** - Performance validation and tuning

---

## 🏗️ **Architecture Components**

### Storage Layer ✅ 
```
/C/{collection}/meta/config              -> pb.Config (DONE)
/C/{collection}/meta/cbv_active           -> uint32 (DONE)  
/C/{collection}/pq/codebook/{cbv}/{sub}   -> pb.CodebookSub (DONE)
/C/{collection}/pq/block/{cbv}/{blockNo}  -> pb.PqCodesBlock (DONE)
```

### Graph Layer ✅
```
/C/{collection}/graph/node/{NID}          -> pb.NodeAdj (DONE)
/C/{collection}/graph/meta/connectivity   -> pb.GraphMeta (DONE)
/C/{collection}/entry                     -> pb.EntryList (DONE)
```

### Sketch Layer ⚠️
```
/C/{collection}/sketch/{NID}              -> pb.VectorSketch (STORING BUT NOT USED FOR RETRAINING)
```

---

## 📐 **Key Implementation Decisions Made**

### **Storage Architecture**
- **Block Size**: 256-512 codes per block (~24-64KB) for optimal FDB performance
- **Compression**: Float16 for codebooks, uint8 for PQ codes
- **Caching**: LRU caches with configurable size limits
- **Transactions**: Small, conflict-resilient writes (<10MB affected data)

### **PQ Configuration** 
- **Default nbits**: 8 (Milvus-compatible)
- **Default subvectors**: D/2 if not specified
- **Codebook size**: 256 codewords (2^8)
- **Distance metrics**: L2, IP, COSINE supported

### **Test Coverage Standards**
- **Instruction coverage**: 91% achieved (target: >90%) ✅
- **Line coverage**: 90% achieved (meets 90% target) ✅
- **Branch coverage**: 81% achieved (exceeds 75% target) ✅
- **Integration**: Real FDB connections, no mocks per project standards
- **Patterns**: Follow existing test structure
- **Large-scale tests**: FdbVectorSearchIntegrationTest with 500+ vectors

### **Recent Implementation Patterns**
- **Maintenance Methods**: Made package-private for better testability while keeping public APIs clean
- **Async Maintenance**: All maintenance operations return `CompletableFuture<Void>` for proper testing
- **Cache Testing**: Added `getPqBlockCache()` accessor for testing cache behavior and eviction
- **Background Services**: Integrated scheduled maintenance tasks for entry point refresh and connectivity repair
- **Test Isolation**: Each test uses unique collections to avoid interference

---

## ⚠️ **Critical Integration Gaps**

### Components Built But Not Fully Utilized:
1. **VectorSketchStorage** - Storing sketches but no codebook retraining implementation uses them
2. **Statistics Collection** - getStats() returns hardcoded zeros, no actual metrics gathering

### Missing Components:
1. **UnlinkWorker** - Does not exist, delete operations enqueue tasks with no processor
2. **Codebook Rotation** - Two-phase rotation not implemented
3. **Vector Count** - getVectorCount() returns 0, not implemented

## 🎯 **Next Milestone: Accuracy Improvement (DiskANN-style)**

**Priority 1 - Accuracy Improvement (DiskANN-style):**
1. **Store Original Vectors** - Store full-precision vectors alongside graph for exact distance computation
   - Consider float16 compression for storage efficiency
2. **Implement Reranking Pipeline** - Use PQ for candidate selection, original vectors for final ranking
   - Overquery strategy: Fetch 3x candidates using PQ distance, rerank with original vectors
   - Two-stage process: PQ for graph traversal, original vectors for top-k results
3. **Hybrid Distance Computation** - PQ distances for graph traversal, exact distances for results
   - BeamSearchEngine continues using PQ for navigation
   - Load original vectors only for final candidate set
   - Return exact distances to users (fixing the ~6-7 distance for exact matches issue)

**Priority 2 - Critical (System Functional):**
1. **Create UnlinkWorker** - Process delete operations (or handle via compaction)
2. **Implement getStats()** - Gather actual metrics  
3. **Implement getVectorCount()** - Track actual vector count

**Success Criteria for Original Vector Storage:**
- Original vector storage enables exact distance computation (distance ~0 for exact matches)
- Reranking with original vectors improves accuracy to >90% on random data
- System maintains high performance with hybrid PQ/original vector approach
- Search returns actual results with accurate distances

---

*This roadmap will be updated as implementation progresses. Current focus: **Segment-based PQ architecture to eliminate global codebook limitations.***