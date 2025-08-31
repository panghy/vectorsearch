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
- [x] **Beam Search Implementation** - Graph traversal with PQ-based scoring (IMPLEMENTED BUT NOT INTEGRATED ⚠️)
- [x] **Entry Point Management** - Hierarchical entry strategies (medoids, random, high-degree) (COMPLETED)
- [ ] **Query Processing Pipeline** - End-to-end search with configurable parameters (NOT WIRED UP)
- [x] **Search Result Ranking** - Top-k result selection and scoring (IMPLEMENTED BUT UNUSED)

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
- **Instruction coverage**: 91% achieved (target: >90%)
- **Line coverage**: 93% achieved (exceeds 90% target)
- **Branch coverage**: 81% achieved (exceeds 75% target) 
- **Integration**: Real FDB connections, no mocks per project standards
- **Patterns**: Follow existing test structure

### **Recent Implementation Patterns**
- **Maintenance Methods**: Made package-private for better testability while keeping public APIs clean
- **Async Maintenance**: All maintenance operations return `CompletableFuture<Void>` for proper testing
- **Cache Testing**: Added `getPqBlockCache()` accessor for testing cache behavior and eviction
- **Background Services**: Integrated scheduled maintenance tasks for entry point refresh and connectivity repair
- **Test Isolation**: Each test uses unique collections to avoid interference

---

## ⚠️ **Critical Integration Gaps**

### Components Built But Not Connected:
1. **BeamSearchEngine** - Fully implemented search engine returning empty results (search() in FdbVectorSearch has TODO)
2. **ProductQuantizer** - Not initialized from codebook storage (line 426-428 TODO)
3. **VectorSketchStorage** - Storing sketches but no codebook retraining implementation uses them
4. **Statistics Collection** - getStats() returns hardcoded zeros, no actual metrics gathering

### Missing Components:
1. **UnlinkWorker** - Does not exist, delete operations enqueue tasks with no processor
2. **Codebook Rotation** - Two-phase rotation not implemented
3. **Vector Count** - getVectorCount() returns 0, not implemented

## 🎯 **Next Milestone: Wire Up Existing Components**

**Priority 1 - Critical (System Non-Functional):**
1. **Wire BeamSearchEngine to search()** - Make search actually work
2. **Create UnlinkWorker** - Process delete operations
3. **Initialize ProductQuantizer** - Load from codebook storage on startup

**Priority 2 - Enhanced Functionality:**
4. **Implement getStats()** - Gather actual metrics
5. **Implement getVectorCount()** - Track actual vector count
6. **Codebook retraining** - Use stored vector sketches

**Success Criteria:**
- Search returns actual results using BeamSearchEngine
- Delete operations are processed by UnlinkWorker
- ProductQuantizer properly initialized from stored codebooks
- Statistics reflect actual system state

---

*This roadmap will be updated as implementation progresses. Current focus: **CRITICAL - Wiring up existing components to make search functional.***