package io.github.panghy.vectorsearch.api;

/**
 * Unit tests for SearchParams factories and validation, including defaults
 * (now BEST_FIRST), invalid parameter handling, and short factory with mode.
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SearchParamsTest {
  @Test
  void defaults_are_reasonable() {
    SearchParams p = SearchParams.defaults(10, 2);
    assertThat(p.efSearch()).isGreaterThan(0);
    assertThat(p.beamWidth()).isGreaterThan(0);
    assertThat(p.maxIters()).isGreaterThan(0);
    assertThat(p.maxExplore()).isGreaterThan(p.efSearch());
    assertThat(p.refineFrontier()).isTrue();
    assertThat(p.mode()).isEqualTo(SearchParams.Mode.BEST_FIRST);
  }

  @Test
  void invalid_params_throw() {
    assertThatThrownBy(() -> SearchParams.of(0, 8, 2)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> SearchParams.of(8, 0, 2)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> SearchParams.of(8, 2, 0)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> SearchParams.of(8, 2, 1, null)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> SearchParams.of(8, 2, 1, 0, false, SearchParams.Mode.BEAM))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void custom_factory_includes_cap_and_refine() {
    SearchParams p = SearchParams.of(32, 16, 3, 1000, true, SearchParams.Mode.BEST_FIRST);
    assertThat(p.efSearch()).isEqualTo(32);
    assertThat(p.beamWidth()).isEqualTo(16);
    assertThat(p.maxIters()).isEqualTo(3);
    assertThat(p.maxExplore()).isEqualTo(1000);
    assertThat(p.refineFrontier()).isTrue();
    assertThat(p.mode()).isEqualTo(SearchParams.Mode.BEST_FIRST);
  }

  @Test
  void short_factory_with_mode_builds() {
    SearchParams p = SearchParams.of(8, 4, 2, SearchParams.Mode.BEAM);
    assertThat(p.mode()).isEqualTo(SearchParams.Mode.BEAM);
    assertThat(p.maxExplore()).isGreaterThan(0);
  }
}
