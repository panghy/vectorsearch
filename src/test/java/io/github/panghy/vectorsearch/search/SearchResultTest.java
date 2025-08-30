package io.github.panghy.vectorsearch.search;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

class SearchResultTest {

  @Test
  void testConstructorAndGetters() {
    SearchResult result = new SearchResult(123L, 0.5f);

    assertThat(result.getNodeId()).isEqualTo(123L);
    assertThat(result.getDistance()).isEqualTo(0.5f);
  }

  @Test
  void testCompareTo() {
    SearchResult result1 = new SearchResult(1L, 0.1f);
    SearchResult result2 = new SearchResult(2L, 0.5f);
    SearchResult result3 = new SearchResult(3L, 0.3f);

    // Lower distance should come first
    assertThat(result1.compareTo(result2)).isLessThan(0);
    assertThat(result2.compareTo(result1)).isGreaterThan(0);
    assertThat(result3.compareTo(result2)).isLessThan(0);

    // Equal distances
    SearchResult result4 = new SearchResult(4L, 0.1f);
    assertThat(result1.compareTo(result4)).isEqualTo(0);
  }

  @Test
  void testSorting() {
    List<SearchResult> results = new ArrayList<>();
    results.add(new SearchResult(1L, 0.5f));
    results.add(new SearchResult(2L, 0.1f));
    results.add(new SearchResult(3L, 0.9f));
    results.add(new SearchResult(4L, 0.3f));

    Collections.sort(results);

    // Should be sorted by distance ascending
    assertThat(results.get(0).getNodeId()).isEqualTo(2L); // distance 0.1
    assertThat(results.get(1).getNodeId()).isEqualTo(4L); // distance 0.3
    assertThat(results.get(2).getNodeId()).isEqualTo(1L); // distance 0.5
    assertThat(results.get(3).getNodeId()).isEqualTo(3L); // distance 0.9
  }

  @Test
  void testToString() {
    SearchResult result = new SearchResult(42L, 1.2345f);
    String str = result.toString();

    assertThat(str).contains("42");
    assertThat(str).contains("1.234"); // At least 3 decimal places
    assertThat(str).contains("SearchResult");
  }

  @Test
  void testEqualsAndHashCode() {
    SearchResult result1 = new SearchResult(123L, 0.5f);
    SearchResult result2 = new SearchResult(123L, 0.5f);
    SearchResult result3 = new SearchResult(123L, 0.6f);
    SearchResult result4 = new SearchResult(124L, 0.5f);

    // Lombok @Data generates equals/hashCode
    assertThat(result1).isEqualTo(result2);
    assertThat(result1.hashCode()).isEqualTo(result2.hashCode());

    assertThat(result1).isNotEqualTo(result3);
    assertThat(result1).isNotEqualTo(result4);
  }

  @Test
  void testWithSpecialValues() {
    // Test with special float values
    SearchResult result1 = new SearchResult(1L, Float.POSITIVE_INFINITY);
    SearchResult result2 = new SearchResult(2L, Float.MAX_VALUE);
    SearchResult result3 = new SearchResult(3L, 0.0f);
    SearchResult result4 = new SearchResult(4L, Float.MIN_VALUE);

    assertThat(result1.getDistance()).isEqualTo(Float.POSITIVE_INFINITY);
    assertThat(result2.getDistance()).isEqualTo(Float.MAX_VALUE);
    assertThat(result3.getDistance()).isEqualTo(0.0f);
    assertThat(result4.getDistance()).isEqualTo(Float.MIN_VALUE);

    // Infinity should be greater than any finite value
    assertThat(result1.compareTo(result2)).isGreaterThan(0);
    assertThat(result3.compareTo(result4)).isLessThan(0);
  }
}
