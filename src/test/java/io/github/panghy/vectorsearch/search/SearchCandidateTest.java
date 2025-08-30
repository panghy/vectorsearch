package io.github.panghy.vectorsearch.search;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

class SearchCandidateTest {

  @Test
  void testConstructorAndGetters() {
    SearchCandidate candidate = new SearchCandidate(456L, 0.75f, false);

    assertThat(candidate.getNodeId()).isEqualTo(456L);
    assertThat(candidate.getDistance()).isEqualTo(0.75f);
    assertThat(candidate.isVisited()).isFalse();
  }

  @Test
  void testUnvisitedFactory() {
    SearchCandidate candidate = SearchCandidate.unvisited(789L, 1.23f);

    assertThat(candidate.getNodeId()).isEqualTo(789L);
    assertThat(candidate.getDistance()).isEqualTo(1.23f);
    assertThat(candidate.isVisited()).isFalse();
  }

  @Test
  void testMarkVisited() {
    SearchCandidate candidate = SearchCandidate.unvisited(100L, 0.5f);

    assertThat(candidate.isVisited()).isFalse();

    candidate.markVisited();

    assertThat(candidate.isVisited()).isTrue();
    assertThat(candidate.getNodeId()).isEqualTo(100L); // Should not change
    assertThat(candidate.getDistance()).isEqualTo(0.5f); // Should not change
  }

  @Test
  void testCompareTo() {
    SearchCandidate candidate1 = SearchCandidate.unvisited(1L, 0.2f);
    SearchCandidate candidate2 = SearchCandidate.unvisited(2L, 0.8f);
    SearchCandidate candidate3 = new SearchCandidate(3L, 0.2f, true);

    // Lower distance should come first
    assertThat(candidate1.compareTo(candidate2)).isLessThan(0);
    assertThat(candidate2.compareTo(candidate1)).isGreaterThan(0);

    // Visited status should not affect comparison
    assertThat(candidate1.compareTo(candidate3)).isEqualTo(0);
  }

  @Test
  void testSorting() {
    List<SearchCandidate> candidates = new ArrayList<>();
    candidates.add(SearchCandidate.unvisited(1L, 0.7f));
    candidates.add(new SearchCandidate(2L, 0.2f, true));
    candidates.add(SearchCandidate.unvisited(3L, 0.5f));
    candidates.add(new SearchCandidate(4L, 0.9f, false));

    Collections.sort(candidates);

    // Should be sorted by distance ascending, regardless of visited status
    assertThat(candidates.get(0).getNodeId()).isEqualTo(2L); // distance 0.2, visited
    assertThat(candidates.get(1).getNodeId()).isEqualTo(3L); // distance 0.5, unvisited
    assertThat(candidates.get(2).getNodeId()).isEqualTo(1L); // distance 0.7, unvisited
    assertThat(candidates.get(3).getNodeId()).isEqualTo(4L); // distance 0.9, unvisited
  }

  @Test
  void testToResult() {
    SearchCandidate candidate = new SearchCandidate(123L, 0.456f, true);
    SearchResult result = candidate.toResult();

    assertThat(result.getNodeId()).isEqualTo(123L);
    assertThat(result.getDistance()).isEqualTo(0.456f);

    // Visited status should not be in the result
    SearchCandidate unvisitedCandidate = SearchCandidate.unvisited(123L, 0.456f);
    SearchResult unvisitedResult = unvisitedCandidate.toResult();

    assertThat(result).isEqualTo(unvisitedResult);
  }

  @Test
  void testToString() {
    SearchCandidate candidate = new SearchCandidate(42L, 2.3456f, true);
    String str = candidate.toString();

    assertThat(str).contains("42");
    assertThat(str).contains("2.345"); // At least 3 decimal places
    assertThat(str).contains("true"); // visited status
    assertThat(str).contains("SearchCandidate");
  }

  @Test
  void testEqualsAndHashCode() {
    SearchCandidate candidate1 = new SearchCandidate(100L, 0.5f, false);
    SearchCandidate candidate2 = new SearchCandidate(100L, 0.5f, false);
    SearchCandidate candidate3 = new SearchCandidate(100L, 0.5f, true); // Different visited
    SearchCandidate candidate4 = new SearchCandidate(101L, 0.5f, false); // Different ID
    SearchCandidate candidate5 = new SearchCandidate(100L, 0.6f, false); // Different distance

    // Lombok @Data generates equals/hashCode
    assertThat(candidate1).isEqualTo(candidate2);
    assertThat(candidate1.hashCode()).isEqualTo(candidate2.hashCode());

    // All fields should be considered in equals
    assertThat(candidate1).isNotEqualTo(candidate3);
    assertThat(candidate1).isNotEqualTo(candidate4);
    assertThat(candidate1).isNotEqualTo(candidate5);
  }

  @Test
  void testSetters() {
    SearchCandidate candidate = new SearchCandidate(1L, 0.1f, false);

    // Test setVisited (Lombok generated)
    candidate.setVisited(true);
    assertThat(candidate.isVisited()).isTrue();

    candidate.setVisited(false);
    assertThat(candidate.isVisited()).isFalse();
  }

  @Test
  void testWithSpecialValues() {
    // Test with special float values
    SearchCandidate candidate1 = SearchCandidate.unvisited(1L, Float.POSITIVE_INFINITY);
    SearchCandidate candidate2 = SearchCandidate.unvisited(2L, Float.NaN);
    SearchCandidate candidate3 = SearchCandidate.unvisited(3L, -0.0f);

    assertThat(candidate1.getDistance()).isEqualTo(Float.POSITIVE_INFINITY);
    assertThat(candidate2.getDistance()).isNaN();
    assertThat(candidate3.getDistance()).isEqualTo(-0.0f);

    // Convert to results should preserve values
    assertThat(candidate1.toResult().getDistance()).isEqualTo(Float.POSITIVE_INFINITY);
    assertThat(candidate2.toResult().getDistance()).isNaN();
  }
}
