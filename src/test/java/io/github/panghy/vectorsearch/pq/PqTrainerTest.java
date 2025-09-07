package io.github.panghy.vectorsearch.pq;

/**
 * Unit tests for PqTrainer: subspace extraction and k-means training sanity checks.
 */
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Test;

class PqTrainerTest {
  @Test
  void trains_centroids_and_validates_params() {
    List<float[]> vecs =
        List.of(new float[] {0f, 0f, 0f, 0f}, new float[] {1f, 1f, 1f, 1f}, new float[] {2f, 2f, 2f, 2f});
    float[][][] c = PqTrainer.train(vecs, 4, 2, 2, 2, 123L);
    assertThat(c.length).isEqualTo(2);
    assertThat(c[0].length).isEqualTo(2);
    assertThat(c[0][0].length).isEqualTo(2);

    assertThatThrownBy(() -> PqTrainer.train(vecs, 3, 2, 2, 1, 1L)).isInstanceOf(IllegalArgumentException.class);
  }
}
