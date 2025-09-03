package io.github.panghy.vectorsearch.search;

import static java.util.concurrent.CompletableFuture.completedFuture;

import com.apple.foundationdb.ReadTransaction;
import com.apple.foundationdb.Transaction;
import io.github.panghy.vectorsearch.pq.ProductQuantizer;
import io.github.panghy.vectorsearch.storage.CodebookStorage;
import io.github.panghy.vectorsearch.storage.EntryPointStorage;
import io.github.panghy.vectorsearch.storage.NodeAdjacencyStorage;
import io.github.panghy.vectorsearch.storage.PqBlockStorage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implements beam search algorithm for approximate nearest neighbor search
 * using DiskANN-style graph traversal with Product Quantization distance scoring.
 *
 * <p>The search algorithm:
 * <ol>
 *   <li>Builds a PQ lookup table for the query vector</li>
 *   <li>Seeds the search with entry points (hierarchical strategy)</li>
 *   <li>Maintains a beam of best candidates</li>
 *   <li>Expands unvisited nodes, scoring neighbors via PQ codes</li>
 *   <li>Returns top-k results after visiting limit or beam exhaustion</li>
 * </ol>
 *
 * <p>This implementation uses snapshot reads to avoid conflicts during search
 * and supports configurable beam size and visit limits for latency control.
 */
public class BeamSearchEngine {
  private static final Logger LOGGER = LoggerFactory.getLogger(BeamSearchEngine.class);

  private final NodeAdjacencyStorage adjacencyStorage;
  private final PqBlockStorage pqBlockStorage;
  private final EntryPointStorage entryPointStorage;
  private final CodebookStorage codebookStorage;

  /**
   * Creates a new BeamSearchEngine.
   *
   * @param adjacencyStorage  storage for graph adjacency lists
   * @param pqBlockStorage    storage for PQ codes
   * @param entryPointStorage storage for entry points
   * @param codebookStorage   storage for codebooks and ProductQuantizers
   */
  public BeamSearchEngine(
      NodeAdjacencyStorage adjacencyStorage,
      PqBlockStorage pqBlockStorage,
      EntryPointStorage entryPointStorage,
      CodebookStorage codebookStorage) {
    this.adjacencyStorage = adjacencyStorage;
    this.pqBlockStorage = pqBlockStorage;
    this.entryPointStorage = entryPointStorage;
    this.codebookStorage = codebookStorage;
  }

  /**
   * Performs beam search to find approximate nearest neighbors.
   *
   * @param tx              the read transaction (should be snapshot for consistency)
   * @param queryVector     the query vector
   * @param topK            number of results to return
   * @param searchListSize  beam size (defaults to max(16, topK) if 0)
   * @param maxVisits       maximum nodes to visit (safety limit)
   * @param pq              the ProductQuantizer to use for distance computation
   * @return future containing top-k search results
   */
  public CompletableFuture<List<SearchResult>> search(
      Transaction tx, float[] queryVector, int topK, int searchListSize, int maxVisits, ProductQuantizer pq) {

    // Apply Milvus behavior: search_list = max(default, topK)
    int beamSize = searchListSize > 0 ? Math.max(searchListSize, topK) : Math.max(16, topK);

    LOGGER.debug("Starting beam search: topK={}, beam={}, maxVisits={}", topK, beamSize, maxVisits);

    // Build PQ lookup table for query
    if (pq == null) {
      LOGGER.warn("No ProductQuantizer available for search");
      return CompletableFuture.completedFuture(List.of());
    }

    LOGGER.debug("ProductQuantizer available, codebook version: {}", pq.getCodebookVersion());
    int codebookVersion = pq.getCodebookVersion();
    float[][] lookupTable = pq.buildLookupTable(queryVector);

    // Get more entry points than beam size for better coverage
    int entryPointCount = Math.min(beamSize * 2, 32); // Get more entry points for diversity

    // Get entry points using hierarchical strategy
    return entryPointStorage.getHierarchicalEntryPoints(tx, entryPointCount).thenCompose(entryPoints -> {
      if (entryPoints.isEmpty()) {
        LOGGER.warn("No entry points found for search, returning empty results");
        return completedFuture(Collections.emptyList());
      }

      LOGGER.debug(
          "Starting search with {} entry points (beamSize={}, topK={}, maxVisits={})",
          entryPoints.size(),
          beamSize,
          topK,
          maxVisits);

      // Score and seed initial candidates
      return scoreNodes(entryPoints, lookupTable, codebookVersion).thenCompose(initialCandidates -> {
        // Execute beam search
        return executeBeamSearch(
            tx, initialCandidates, lookupTable, codebookVersion, beamSize, topK, maxVisits);
      });
    });
  }

  /**
   * Searches for neighbors of a specific node using its PQ codes.
   * This is used during link operations to find neighbors for a new node.
   *
   * @param tx              the transaction
   * @param nodeId          the node being linked
   * @param pqCode          the PQ codes of the node
   * @param codebookVersion the codebook version
   * @param entryPoints     entry points to start search from
   * @param candidateSize   number of candidates to find
   * @param maxVisits       maximum nodes to visit
   * @return future containing candidate neighbor results with distances
   */
  public CompletableFuture<List<SearchResult>> searchForNeighbors(
      Transaction tx,
      long nodeId,
      byte[] pqCode,
      int codebookVersion,
      List<Long> entryPoints,
      int candidateSize,
      int maxVisits) {

    // Get the ProductQuantizer for this codebook version
    return codebookStorage.getProductQuantizer(codebookVersion).thenCompose(pq -> {
      if (pq == null) {
        LOGGER.warn("No ProductQuantizer available for codebook version {}", codebookVersion);
        return CompletableFuture.completedFuture(List.of());
      }

      // Build lookup table from PQ codes
      // This is an approximation - ideally we'd have the original vector
      float[][] lookupTable = pq.buildLookupTableFromPqCode(pqCode);

      // Score entry points
      return scoreNodes(entryPoints, lookupTable, codebookVersion).thenCompose(initialCandidates -> {
        // Execute beam search to find candidates
        return executeBeamSearch(
                tx,
                initialCandidates,
                lookupTable,
                codebookVersion,
                candidateSize,
                candidateSize,
                maxVisits)
            .thenApply(results -> {
              // Filter out self from results
              return results.stream()
                  .filter(result -> result.getNodeId() != nodeId)
                  .collect(Collectors.toList());
            });
      });
    });
  }

  /**
   * Executes the core beam search algorithm.
   */
  private CompletableFuture<List<SearchResult>> executeBeamSearch(
      ReadTransaction tx,
      List<SearchCandidate> initialCandidates,
      float[][] lookupTable,
      int codebookVersion,
      int beamSize,
      int topK,
      int maxVisits) {

    // Priority queue for beam (min-heap by distance)
    PriorityQueue<SearchCandidate> beam = new PriorityQueue<>();
    beam.addAll(initialCandidates);

    // Track visited nodes to avoid cycles
    Set<Long> visited = new HashSet<>();

    // Track nodes we've seen (for deduplication)
    Set<Long> seen = new HashSet<>();
    for (SearchCandidate candidate : initialCandidates) {
      seen.add(candidate.getNodeId());
    }

    // Result accumulator (best nodes found so far)
    PriorityQueue<SearchResult> results =
        new PriorityQueue<>(Collections.reverseOrder()); // Max-heap to easily drop worst

    int visitCount = 0;

    // Beam search loop
    return beamSearchLoop(
        tx, beam, visited, seen, results, lookupTable, codebookVersion, beamSize, topK, maxVisits, visitCount);
  }

  /**
   * Recursive beam search loop implementation using CompletableFuture composition.
   */
  private CompletableFuture<List<SearchResult>> beamSearchLoop(
      ReadTransaction tx,
      PriorityQueue<SearchCandidate> beam,
      Set<Long> visited,
      Set<Long> seen,
      PriorityQueue<SearchResult> results,
      float[][] lookupTable,
      int codebookVersion,
      int beamSize,
      int topK,
      int maxVisits,
      int visitCount) {

    // Check termination conditions
    if (beam.isEmpty() || visitCount >= maxVisits) {
      // Convert results to sorted list
      List<SearchResult> finalResults = new ArrayList<>(results);
      Collections.sort(finalResults);
      if (finalResults.size() > topK) {
        finalResults = finalResults.subList(0, topK);
      }

      LOGGER.info(
          "Search complete: visited={} nodes, returning {} results (requested topK={})",
          visitCount,
          finalResults.size(),
          topK);
      if (!finalResults.isEmpty()) {
        LOGGER.debug(
            "Result node IDs (first 10): {}",
            finalResults.stream()
                .limit(10)
                .map(SearchResult::getNodeId)
                .toList());
        LOGGER.debug(
            "Result distances (first 10): {}",
            finalResults.stream()
                .limit(10)
                .map(r -> String.format("%.3f", r.getDistance()))
                .toList());
      }

      return completedFuture(finalResults);
    }

    // Get best unvisited candidate
    SearchCandidate current = null;
    while (!beam.isEmpty()) {
      SearchCandidate candidate = beam.poll();
      if (!candidate.isVisited()) {
        current = candidate;
        break;
      }
    }

    if (current == null) {
      // All candidates visited, terminate
      List<SearchResult> finalResults = new ArrayList<>(results);
      Collections.sort(finalResults);
      if (finalResults.size() > topK) {
        finalResults = finalResults.subList(0, topK);
      }
      return completedFuture(finalResults);
    }

    // Mark as visited
    current.markVisited();
    visited.add(current.getNodeId());
    final int newVisitCount = visitCount + 1;

    // Add to results
    results.offer(current.toResult());
    if (results.size() > topK * 2) { // Keep some buffer
      // Remove worst results to limit memory
      while (results.size() > topK) {
        results.poll();
      }
    }

    // Re-add current to beam (as visited) to maintain beam structure
    beam.offer(current);

    // Load adjacency list for current node
    final SearchCandidate finalCurrent = current;
    return adjacencyStorage
        .loadAdjacency((Transaction) tx, current.getNodeId())
        .thenCompose(adjacency -> {
          if (adjacency == null || adjacency.getNeighborsList().isEmpty()) {
            // No neighbors, continue with remaining beam
            return beamSearchLoop(
                tx,
                beam,
                visited,
                seen,
                results,
                lookupTable,
                codebookVersion,
                beamSize,
                topK,
                maxVisits,
                newVisitCount);
          }

          // Filter neighbors to unvisited and unseen
          List<Long> newNeighbors = new ArrayList<>();
          for (long neighbor : adjacency.getNeighborsList()) {
            if (!seen.contains(neighbor)) {
              newNeighbors.add(neighbor);
              seen.add(neighbor);
            }
          }

          // Log graph traversal progress for debugging
          if (newVisitCount <= 5 || newVisitCount % 50 == 0) {
            LOGGER.debug(
                "Visit {}: node={} has {} neighbors, {} new to explore, beam size={}, seen={}",
                newVisitCount,
                finalCurrent.getNodeId(),
                adjacency.getNeighborsList().size(),
                newNeighbors.size(),
                beam.size(),
                seen.size());
          }

          if (newNeighbors.isEmpty()) {
            // No new neighbors, continue
            return beamSearchLoop(
                tx,
                beam,
                visited,
                seen,
                results,
                lookupTable,
                codebookVersion,
                beamSize,
                topK,
                maxVisits,
                newVisitCount);
          }

          // Score new neighbors
          return scoreNodes(newNeighbors, lookupTable, codebookVersion)
              .thenCompose(scoredNeighbors -> {
                // Add scored neighbors to beam
                beam.addAll(scoredNeighbors);

                // Prune beam to maintain size limit
                while (beam.size() > beamSize * 2) { // Allow some overflow
                  // Remove worst candidates from end
                  List<SearchCandidate> temp = new ArrayList<>(beam);
                  Collections.sort(temp, Collections.reverseOrder());
                  beam.clear();
                  beam.addAll(temp.subList(0, beamSize));
                }

                // Continue search
                return beamSearchLoop(
                    tx,
                    beam,
                    visited,
                    seen,
                    results,
                    lookupTable,
                    codebookVersion,
                    beamSize,
                    topK,
                    maxVisits,
                    newVisitCount);
              });
        });
  }

  /**
   * Scores a batch of nodes using PQ distance computation.
   */
  private CompletableFuture<List<SearchCandidate>> scoreNodes(
      List<Long> nodeIds, float[][] lookupTable, int codebookVersion) {

    if (nodeIds.isEmpty()) {
      return completedFuture(Collections.emptyList());
    }

    // Get the ProductQuantizer for distance computation
    return codebookStorage.getProductQuantizer(codebookVersion).thenCompose(pq -> {
      if (pq == null) {
        LOGGER.warn("No ProductQuantizer available for codebook version {}", codebookVersion);
        // Return max distance for all nodes
        List<SearchCandidate> candidates = new ArrayList<>();
        for (Long nodeId : nodeIds) {
          candidates.add(SearchCandidate.unvisited(nodeId, Float.MAX_VALUE));
        }
        return completedFuture(candidates);
      }

      CompletableFuture<List<byte[]>> listCompletableFuture =
          pqBlockStorage.batchLoadPqCodes(nodeIds, codebookVersion);
      return listCompletableFuture.thenApply(pqCodes -> {
        List<SearchCandidate> candidates = new ArrayList<>(pqCodes.size());

        for (int i = 0; i < nodeIds.size(); i++) {
          byte[] pqCode = pqCodes.get(i);
          if (pqCode == null) {
            candidates.add(SearchCandidate.unvisited(nodeIds.get(i), Float.MAX_VALUE));
          } else {
            float distance = pq.computeDistance(pqCode, lookupTable);
            candidates.add(SearchCandidate.unvisited(nodeIds.get(i), distance));
          }
        }
        return candidates;
      });
    });
  }
}
