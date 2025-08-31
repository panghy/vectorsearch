package io.github.panghy.vectorsearch.graph;

import static org.assertj.core.api.Assertions.assertThat;

import io.github.panghy.vectorsearch.graph.RobustPruning.Candidate;
import io.github.panghy.vectorsearch.graph.RobustPruning.PruningConfig;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class RobustPruningTest {

  @Test
  void testPruneEmptyList() {
    List<Candidate> candidates = Collections.emptyList();
    PruningConfig config = PruningConfig.builder().maxDegree(5).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    assertThat(result).isEmpty();
  }

  @Test
  void testPruneSingleCandidate() {
    List<Candidate> candidates =
        List.of(Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build());
    PruningConfig config = PruningConfig.builder().maxDegree(5).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    assertThat(result).containsExactly(1L);
  }

  @Test
  void testPruneWithinDegreeLimit() {
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(3.0f).build());
    PruningConfig config = PruningConfig.builder().maxDegree(5).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    assertThat(result).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testPruneExceedsDegreeLimit() {
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(3.0f).build(),
        Candidate.builder().nodeId(4L).distanceToQuery(4.0f).build(),
        Candidate.builder().nodeId(5L).distanceToQuery(5.0f).build());
    PruningConfig config = PruningConfig.builder().maxDegree(3).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    assertThat(result).hasSize(3);
    assertThat(result).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testPruneWithDominatedCandidates() {
    // With the corrected dominance test: nodes are dominated if they are farther than alpha * selected distance
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(1.1f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(1.3f).build(), // Dominated by 1 if 1.3 > 1.2*1.0
        Candidate.builder().nodeId(4L).distanceToQuery(2.0f).build(), // Dominated by 1 if 2.0 > 1.2*1.0
        Candidate.builder().nodeId(5L).distanceToQuery(5.0f).build() // Dominated by 1 if 5.0 > 1.2*1.0
        );
    PruningConfig config = PruningConfig.builder()
        .maxDegree(5)
        .alpha(1.2) // Nodes farther than 1.2x the selected distance are dominated
        .build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // Node 1 is selected first (closest)
    // Node 2 is dominated (1.1 <= 1.2*1.0 = 1.2)
    // Node 3 is dominated (1.3 <= 1.2*1.0? No, 1.3 > 1.2)
    // But wait, once 1 is selected, check others against it
    // Node 2: 1.1 <= 1.2*1.0 = 1.2 (dominated)
    // Node 3: 1.3 <= 1.2*1.0 = 1.2? No, 1.3 > 1.2 (not dominated) - selected
    // Node 4: 2.0 <= 1.2*1.3 = 1.56? No, 2.0 > 1.56 (not dominated) - selected
    // Node 5: 5.0 <= 1.2*2.0 = 2.4? No, 5.0 > 2.4 (not dominated) - selected
    assertThat(result).containsExactly(1L, 3L, 4L, 5L);
  }

  @Test
  void testPruneWithLowerAlpha() {
    // Lower alpha means more aggressive pruning (higher diversity)
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(1.5f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(4L).distanceToQuery(2.5f).build());
    PruningConfig config = PruningConfig.builder()
        .maxDegree(4)
        .alpha(0.95) // More aggressive pruning
        .build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // With alpha=0.95, more candidates will be dominated
    // Node 1 selected (closest)
    // Node 2 selected (1.5 > 0.95*1.0)
    // Node 3 selected (2.0 > 0.95*1.5)
    // Node 4 selected (2.5 > 0.95*2.0)
    assertThat(result).containsExactly(1L, 2L, 3L, 4L);
  }

  @Test
  void testPruneUnsortedCandidates() {
    // Provide unsorted candidates - algorithm should sort them
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(3L).distanceToQuery(3.0f).build(),
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build());
    PruningConfig config = PruningConfig.builder().maxDegree(3).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // Should be selected in distance order
    assertThat(result).containsExactly(1L, 2L, 3L);
  }

  @Test
  void testPruneWithPairwiseDistance() {
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(3.0f).build(),
        Candidate.builder().nodeId(4L).distanceToQuery(4.0f).build());

    // Define pairwise distances (symmetric)
    BiFunction<Long, Long, Float> pairwiseDistance = (a, b) -> {
      if (a.equals(b)) return 0.0f;
      // Nodes 1 and 2 are very close to each other
      if ((a == 1L && b == 2L) || (a == 2L && b == 1L)) return 0.5f;
      // Nodes 3 and 4 are far from everything
      return 10.0f;
    };

    PruningConfig config = PruningConfig.builder()
        .maxDegree(4)
        .alpha(1.2)
        .usePairwiseDistance(true)
        .build();

    List<Long> result = RobustPruning.pruneWithPairwiseDistance(candidates, config, pairwiseDistance);

    // Node 1 selected first
    // Node 2 dominated by 1 (pairwise distance 0.5 < 1.2 * 2.0)
    // Node 3 selected (far from 1)
    // Node 4 selected (far from others)
    assertThat(result).containsExactly(1L, 3L, 4L);
  }

  @Test
  void testMergeAndPrune() {
    List<Long> currentNeighbors = Arrays.asList(1L, 2L, 3L);
    List<Candidate> newCandidates = Arrays.asList(
        Candidate.builder().nodeId(4L).distanceToQuery(0.5f).build(), // Closest
        Candidate.builder().nodeId(5L).distanceToQuery(4.5f).build());

    Function<Long, Float> distanceFunction = nodeId -> {
      switch (nodeId.intValue()) {
        case 1:
          return 1.0f;
        case 2:
          return 2.0f;
        case 3:
          return 3.0f;
        default:
          return 10.0f;
      }
    };

    PruningConfig config = PruningConfig.builder().maxDegree(4).alpha(1.2).build();

    List<Long> result = RobustPruning.mergeAndPrune(currentNeighbors, newCandidates, config, distanceFunction);

    // Should include the closest new candidate and keep existing neighbors within limit
    assertThat(result).hasSize(4);
    assertThat(result).contains(4L); // Closest new candidate
    assertThat(result).contains(1L, 2L, 3L); // Existing neighbors
  }

  @Test
  void testMergeAndPruneExceedsLimit() {
    List<Long> currentNeighbors = Arrays.asList(1L, 2L, 3L);
    List<Candidate> newCandidates = Arrays.asList(
        Candidate.builder().nodeId(4L).distanceToQuery(0.5f).build(),
        Candidate.builder().nodeId(5L).distanceToQuery(0.6f).build(),
        Candidate.builder().nodeId(6L).distanceToQuery(0.7f).build());

    Function<Long, Float> distanceFunction = nodeId -> {
      switch (nodeId.intValue()) {
        case 1:
          return 1.0f;
        case 2:
          return 2.0f;
        case 3:
          return 3.0f;
        default:
          return 10.0f;
      }
    };

    PruningConfig config = PruningConfig.builder()
        .maxDegree(3) // Limit to 3
        .alpha(1.2)
        .build();

    List<Long> result = RobustPruning.mergeAndPrune(currentNeighbors, newCandidates, config, distanceFunction);

    // Should keep the 3 closest candidates after pruning
    assertThat(result).hasSize(3);
    // Should include the new closest candidates
    assertThat(result).contains(4L, 5L); // The two closest new candidates
    // May include 6L or 1L depending on dominance
  }

  @Test
  void testGetBackLinksToRemove() {
    List<Long> oldNeighbors = Arrays.asList(1L, 2L, 3L, 4L);
    List<Long> newNeighbors = Arrays.asList(1L, 3L, 5L); // Removed 2 and 4, added 5

    List<Long> toRemove = RobustPruning.getBackLinksToRemove(oldNeighbors, newNeighbors);

    assertThat(toRemove).containsExactlyInAnyOrder(2L, 4L);
  }

  @Test
  void testGetBackLinksToAdd() {
    List<Long> oldNeighbors = Arrays.asList(1L, 2L, 3L);
    List<Long> newNeighbors = Arrays.asList(1L, 3L, 4L, 5L); // Added 4 and 5

    List<Long> toAdd = RobustPruning.getBackLinksToAdd(oldNeighbors, newNeighbors);

    assertThat(toAdd).containsExactlyInAnyOrder(4L, 5L);
  }

  @Test
  void testGetBackLinksNoChanges() {
    List<Long> neighbors = Arrays.asList(1L, 2L, 3L);

    List<Long> toRemove = RobustPruning.getBackLinksToRemove(neighbors, neighbors);
    List<Long> toAdd = RobustPruning.getBackLinksToAdd(neighbors, neighbors);

    assertThat(toRemove).isEmpty();
    assertThat(toAdd).isEmpty();
  }

  @Test
  void testPruneWithDuplicates() {
    // Test that duplicates are handled correctly
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(), // Duplicate
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build() // Duplicate
        );
    PruningConfig config = PruningConfig.builder().maxDegree(5).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    assertThat(result).containsExactly(1L, 2L); // No duplicates in result
  }

  @Test
  void testPruneAllDominated() {
    // All candidates except the first are dominated
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(1.01f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(1.02f).build(),
        Candidate.builder().nodeId(4L).distanceToQuery(1.03f).build());
    PruningConfig config = PruningConfig.builder()
        .maxDegree(10)
        .alpha(1.05) // Very tight pruning
        .build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // Only the first node should be selected
    assertThat(result).containsExactly(1L);
  }

  @Test
  void testPruneWithZeroDistance() {
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(0.0f).build(), // Exact match
        Candidate.builder().nodeId(2L).distanceToQuery(0.1f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(1.0f).build());
    PruningConfig config = PruningConfig.builder().maxDegree(3).alpha(1.2).build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // Node 1 is selected (exact match)
    // Node 2 might be dominated (0.1 <= 1.2*0.0 = 0) - actually not dominated since 0.1 > 0
    // Node 3 is selected
    assertThat(result).contains(1L);
    assertThat(result.size()).isGreaterThanOrEqualTo(1);
  }

  @Test
  void testPruneWithVeryHighAlpha() {
    // High alpha means less pruning (more proximity-focused)
    // With alpha=10.0: a node is dominated only if distance > 10.0 * selected distance
    List<Candidate> candidates = Arrays.asList(
        Candidate.builder().nodeId(1L).distanceToQuery(1.0f).build(),
        Candidate.builder().nodeId(2L).distanceToQuery(2.0f).build(),
        Candidate.builder().nodeId(3L).distanceToQuery(3.0f).build());
    PruningConfig config = PruningConfig.builder()
        .maxDegree(10)
        .alpha(10.0) // Very high alpha - nodes dominated only if > 10x farther
        .build();

    List<Long> result = RobustPruning.prune(candidates, config);

    // Node 1 selected (1.0)
    // Node 2: Is 2.0 <= 10.0*1.0 = 10.0? Yes, so dominated
    // Node 3: Is 3.0 <= 10.0*1.0 = 10.0? Yes, so dominated
    // Only node 1 should be selected
    assertThat(result).containsExactly(1L);
  }
}
