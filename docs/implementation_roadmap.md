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
- [x] **Comprehensive Test Coverage** - 93% line coverage, 89% branch coverage

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
- [ ] **Robust Pruning Algorithm** - DiskANN-style neighbor selection with diversity
- [ ] **Back-link Maintenance** - Bidirectional edge consistency
- [ ] **Graph Connectivity Monitoring** - Detect and repair disconnected components

### Search Engine
- [ ] **Beam Search Implementation** - Graph traversal with PQ-based scoring
- [ ] **Entry Point Management** - Hierarchical entry strategies (medoids, random, high-degree)
- [ ] **Query Processing Pipeline** - End-to-end search with configurable parameters
- [ ] **Search Result Ranking** - Top-k result selection and scoring

---

## 📋 **Phase 3: Online Operations** (PLANNED)

### Upsert System
- [ ] **Transactional Queue Integration** - Link worker task management
- [ ] **Link Worker Implementation** - Neighbor discovery and graph linking
- [ ] **PQ Code Persistence** - Block-level RMW operations
- [ ] **Batch Processing** - Efficient handling of bulk upserts

### Delete System  
- [ ] **Unlink Worker** - Remove nodes and back-links
- [ ] **Orphan Detection** - Identify disconnected nodes
- [ ] **Graph Repair** - Reconnect components after deletions

### Background Workers
- [ ] **Entry List Maintenance** - Periodic refresh of search entry points
- [ ] **Connectivity Monitor** - Graph health checking and repair
- [ ] **Statistics Collection** - Index health and performance metrics

---

## 🔄 **Phase 4: Advanced Features** (FUTURE)

### Codebook Management
- [ ] **Two-Phase Rotation** - Safe codebook updates without downtime
- [ ] **Vector Sketches** - SimHash/PCA for codebook retraining
- [ ] **Quality Monitoring** - Track quantization error and recall

### Performance & Scaling
- [ ] **Cache Management** - LRU caches for PQ blocks and adjacency data
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

### Graph Layer 🚧
```
/C/{collection}/graph/node/{NID}          -> pb.NodeAdj (DONE)
/C/{collection}/graph/meta/connectivity   -> pb.GraphMeta (PLANNED)
/C/{collection}/entry                     -> pb.EntryList (PLANNED)
```

### Sketch Layer 📋
```
/C/{collection}/sketch/{NID}              -> pb.VectorSketch (PLANNED)
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
- **Line coverage**: 93% achieved (target: >90%)
- **Branch coverage**: 89% achieved (target: >75%) 
- **Integration**: Real FDB connections, no mocks
- **Patterns**: Follow existing test structure

---

## 🎯 **Next Milestone: Graph & Search**

**Priority Items:**
1. ~~**NodeAdj protobuf storage**~~ - ✅ Implemented with transaction-based API
2. **Graph traversal core** - Beam search with PQ distance scoring  
3. **Entry point system** - Bootstrap search with good starting nodes
4. **Basic search API** - End-to-end query processing

**Success Criteria:**
- Search latency <10ms for 95th percentile
- Recall >90% compared to brute force on test datasets
- Graph connectivity >95% of nodes reachable
- Concurrent read/write operations without conflicts

---

*This roadmap will be updated as implementation progresses. Current focus: completing the graph layer and basic search functionality.*