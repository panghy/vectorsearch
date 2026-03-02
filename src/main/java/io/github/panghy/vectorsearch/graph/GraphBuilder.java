package io.github.panghy.vectorsearch.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Builds k-nearest neighbor adjacency graphs for vector sets using L2 distance.
 *
 * <p>Provides three construction strategies:
 * <ul>
 *   <li>{@link #buildL2Neighbors} — O(n²) brute-force baseline</li>
 *   <li>{@link #buildPrunedNeighbors} — brute-force with alpha-based pruning</li>
 *   <li>{@link #buildVamanaGraph} — incremental Vamana/DiskANN construction with greedy search,
 *       robust pruning, and reverse edge updates for high-quality graphs</li>
 * </ul>
 */
public final class GraphBuilder {
  private GraphBuilder() {}

  /**
   * Computes per-vector neighbor lists.
   *
   * @param vectors array of vectors [n][d]
   * @param degree  desired out-degree (neighbors per vector)
   * @return neighbors[n][] where each entry contains up to {@code degree} distinct indices (not including self)
   */
  public static int[][] buildL2Neighbors(float[][] vectors, int degree) {
    int n = vectors.length;
    int[][] neigh = new int[n][];
    for (int i = 0; i < n; i++) {
      // compute distances to all j != i
      Integer[] idx = new Integer[n - 1];
      int p = 0;
      for (int j = 0; j < n; j++) if (j != i) idx[p++] = j;
      final int ii = i;
      Arrays.sort(idx, Comparator.comparingDouble(j -> l2(vectors[ii], vectors[j])));
      int take = Math.min(degree, n - 1);
      neigh[i] = new int[take];
      for (int k = 0; k < take; k++) neigh[i][k] = idx[k];
    }
    return neigh;
  }

  /**
   * Builds neighbors with simple Vamana-style pruning.
   *
   * <p>Algorithm:
   * 1) For each node i, compute distances to all j != i and take the top L_build by distance.
   * 2) Greedily add candidates in order, pruning a candidate u if there exists a kept neighbor p
   *    such that dist(u, p) <= alpha * dist(u, i). Set alpha <= 1 to disable pruning.
   */
  public static int[][] buildPrunedNeighbors(float[][] vectors, int degree, int lBuild, double alpha) {
    int n = vectors.length;
    int[][] neigh = new int[n][];
    boolean prune = alpha > 1.0;
    for (int i = 0; i < n; i++) {
      Integer[] idx = new Integer[n - 1];
      int p = 0;
      for (int j = 0; j < n; j++) if (j != i) idx[p++] = j;
      final int ii = i;
      // Precompute distances to i for sorting and reuse in pruning
      final double[] distToI = new double[n];
      Arrays.sort(idx, Comparator.comparingDouble(j -> {
        double d = l2(vectors[ii], vectors[j]);
        distToI[j] = d;
        return d;
      }));
      int limit = Math.max(0, Math.min(lBuild, n - 1));
      int[] selected = new int[Math.min(degree, limit)];
      int s = 0;
      for (int k = 0; k < limit && s < selected.length; k++) {
        int u = idx[k];
        boolean keep = true;
        if (prune) {
          double diu = distToI[u];
          for (int t = 0; t < s; t++) {
            int pnb = selected[t];
            double dup = l2(vectors[u], vectors[pnb]);
            if (dup <= alpha * diu) {
              keep = false;
              break;
            }
          }
        }
        if (keep) selected[s++] = u;
      }
      neigh[i] = (s == selected.length) ? selected : java.util.Arrays.copyOf(selected, s);
    }
    return neigh;
  }

  /**
   * Builds a high-quality graph using the Vamana/DiskANN incremental insertion algorithm.
   *
   * <p>Algorithm overview:
   * <ol>
   *   <li>Compute the medoid (vector closest to the centroid) as the entry point</li>
   *   <li>Insert vectors one-by-one using greedy search on the partial graph to find candidates</li>
   *   <li>Apply robust pruning (alpha-based) to select neighbors</li>
   *   <li>Update reverse edges: when u→v is added, also consider adding v→u</li>
   * </ol>
   *
   * @param vectors array of vectors [n][d]
   * @param degree  target out-degree (R) per node
   * @param lBuild  search list size during construction (L_build); larger = better quality, slower
   * @param alpha   pruning parameter (>1.0); higher keeps more diverse neighbors
   * @return neighbors[n][] where each entry contains up to {@code degree} neighbor indices
   */
  public static int[][] buildVamanaGraph(float[][] vectors, int degree, int lBuild, double alpha) {
    int n = vectors.length;
    if (n == 0) return new int[0][];
    if (n == 1) return new int[][] {new int[0]};

    // Initialize adjacency lists
    @SuppressWarnings("unchecked")
    List<Integer>[] adj = new List[n];
    for (int i = 0; i < n; i++) {
      adj[i] = new ArrayList<>();
    }

    // Compute medoid: vector closest to centroid
    int medoid = findMedoid(vectors);

    // Determine insertion order: start with medoid, then all others
    int[] insertionOrder = new int[n];
    insertionOrder[0] = medoid;
    int pos = 1;
    for (int i = 0; i < n; i++) {
      if (i != medoid) insertionOrder[pos++] = i;
    }

    // Insert first node (no neighbors yet)
    boolean[] inserted = new boolean[n];
    inserted[medoid] = true;

    // Insert remaining vectors one by one
    for (int idx = 1; idx < n; idx++) {
      int node = insertionOrder[idx];
      inserted[node] = true;

      // Greedy search from medoid to find nearest candidates in the partial graph
      List<int[]> candidates = greedySearch(vectors, adj, inserted, medoid, vectors[node], lBuild);

      // Robust prune to select neighbors for node
      List<Integer> pruned = robustPrune(vectors, node, candidates, degree, alpha);
      adj[node] = new ArrayList<>(pruned);

      // Reverse edge updates: for each neighbor v of node, consider adding node as neighbor of v
      for (int v : pruned) {
        if (!adj[v].contains(node)) {
          adj[v].add(node);
          // If v exceeds degree, prune v's neighbor list
          if (adj[v].size() > degree) {
            List<int[]> vCandidates = new ArrayList<>();
            for (int nb : adj[v]) {
              vCandidates.add(new int[] {nb, 0}); // placeholder distance
            }
            // Recompute distances for v's candidates
            vCandidates.clear();
            for (int nb : adj[v]) {
              vCandidates.add(new int[] {nb});
            }
            List<int[]> vCandidatesFull = new ArrayList<>();
            for (int nb : adj[v]) {
              vCandidatesFull.add(new int[] {nb});
            }
            adj[v] = robustPrune(vectors, v, toCandidatesWithDist(vectors, v, adj[v]), degree, alpha);
          }
        }
      }
    }

    // Convert adjacency lists to arrays
    int[][] result = new int[n][];
    for (int i = 0; i < n; i++) {
      result[i] = adj[i].stream().mapToInt(Integer::intValue).toArray();
    }
    return result;
  }

  /**
   * Finds the medoid: the vector closest to the centroid of the dataset.
   */
  static int findMedoid(float[][] vectors) {
    int n = vectors.length;
    int d = vectors[0].length;

    // Compute centroid
    double[] centroid = new double[d];
    for (float[] v : vectors) {
      for (int j = 0; j < d; j++) centroid[j] += v[j];
    }
    for (int j = 0; j < d; j++) centroid[j] /= n;

    // Find nearest vector to centroid
    int best = 0;
    double bestDist = Double.MAX_VALUE;
    for (int i = 0; i < n; i++) {
      double dist = 0;
      for (int j = 0; j < d; j++) {
        double diff = vectors[i][j] - centroid[j];
        dist += diff * diff;
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best;
  }

  /**
   * Greedy search on the partial graph to find the L nearest neighbors of the query.
   *
   * @return list of [nodeId] pairs sorted by distance to query (closest first), up to lBuild entries
   */
  private static List<int[]> greedySearch(
      float[][] vectors, List<Integer>[] adj, boolean[] inserted, int startNode, float[] query, int lBuild) {

    // Priority queue: min-heap by distance
    PriorityQueue<double[]> candidates = new PriorityQueue<>(Comparator.comparingDouble(a -> a[1]));
    boolean[] visited = new boolean[vectors.length];

    double startDist = l2(vectors[startNode], query);
    candidates.add(new double[] {startNode, startDist});
    visited[startNode] = true;

    // Result list: keep best L candidates seen
    List<double[]> bestL = new ArrayList<>();
    bestL.add(new double[] {startNode, startDist});

    while (!candidates.isEmpty()) {
      double[] current = candidates.poll();
      int curNode = (int) current[0];
      double curDist = current[1];

      // If current is farther than the worst in bestL and bestL is full, stop
      if (bestL.size() >= lBuild && curDist > bestL.get(bestL.size() - 1)[1]) {
        break;
      }

      // Expand neighbors
      for (int nb : adj[curNode]) {
        if (!visited[nb] && inserted[nb]) {
          visited[nb] = true;
          double dist = l2(vectors[nb], query);
          candidates.add(new double[] {nb, dist});

          // Insert into bestL maintaining sorted order
          insertSorted(bestL, new double[] {nb, dist}, lBuild);
        }
      }
    }

    // Convert to int[] list
    List<int[]> result = new ArrayList<>();
    for (double[] entry : bestL) {
      result.add(new int[] {(int) entry[0]});
    }
    return result;
  }

  /** Insert entry into sorted list, keeping at most maxSize entries. */
  private static void insertSorted(List<double[]> list, double[] entry, int maxSize) {
    int pos = 0;
    while (pos < list.size() && list.get(pos)[1] <= entry[1]) pos++;
    list.add(pos, entry);
    if (list.size() > maxSize) list.remove(list.size() - 1);
  }

  /**
   * Robust pruning (RobustPrune from DiskANN paper).
   *
   * <p>From candidates sorted by distance to node, greedily select neighbors:
   * keep candidate p if no already-selected neighbor n satisfies
   * {@code dist(p, n) * alpha < dist(p, node)}.
   *
   * @param vectors  all vectors
   * @param node     the node being pruned
   * @param candidates list of [candidateId] sorted by distance to node (closest first)
   * @param degree   max neighbors to keep
   * @param alpha    pruning threshold (>1.0)
   * @return selected neighbor indices
   */
  private static List<Integer> robustPrune(
      float[][] vectors, int node, List<int[]> candidates, int degree, double alpha) {
    List<Integer> selected = new ArrayList<>();
    for (int[] cand : candidates) {
      int p = cand[0];
      if (p == node) continue;
      double distToNode = l2(vectors[p], vectors[node]);
      boolean keep = true;
      for (int n : selected) {
        double distToNeighbor = l2(vectors[p], vectors[n]);
        if (distToNeighbor * alpha < distToNode) {
          keep = false;
          break;
        }
      }
      if (keep) {
        selected.add(p);
        if (selected.size() >= degree) break;
      }
    }
    return selected;
  }

  /** Convert a neighbor list to candidate format with distances for robust pruning. */
  private static List<int[]> toCandidatesWithDist(float[][] vectors, int node, List<Integer> neighbors) {
    List<double[]> withDist = new ArrayList<>();
    for (int nb : neighbors) {
      withDist.add(new double[] {nb, l2(vectors[nb], vectors[node])});
    }
    withDist.sort(Comparator.comparingDouble(a -> a[1]));
    List<int[]> result = new ArrayList<>();
    for (double[] entry : withDist) {
      result.add(new int[] {(int) entry[0]});
    }
    return result;
  }

  private static double l2(float[] a, float[] b) {
    double s = 0.0;
    for (int i = 0; i < a.length; i++) {
      double d = (double) a[i] - b[i];
      s += d * d;
    }
    return s;
  }
}
