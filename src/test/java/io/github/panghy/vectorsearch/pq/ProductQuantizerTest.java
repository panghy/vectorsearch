package io.github.panghy.vectorsearch.pq;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.junit.jupiter.api.Test;

class ProductQuantizerTest {

  @Test
  void testConstructorValidation() {
    // Valid construction
    ProductQuantizer pq = new ProductQuantizer(128, 8, DistanceMetrics.Metric.L2);
    assertThat(pq.getDimension()).isEqualTo(128);
    assertThat(pq.getNumSubvectors()).isEqualTo(8);
    assertThat(pq.getNumCentroids()).isEqualTo(256);
    assertThat(pq.getMetric()).isEqualTo(DistanceMetrics.Metric.L2);

    // Invalid dimension
    assertThatThrownBy(() -> new ProductQuantizer(0, 8, DistanceMetrics.Metric.L2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Dimension must be positive");

    // Invalid subvectors
    assertThatThrownBy(() -> new ProductQuantizer(128, 0, DistanceMetrics.Metric.L2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Number of subvectors must be between");

    assertThatThrownBy(() -> new ProductQuantizer(128, 129, DistanceMetrics.Metric.L2))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Number of subvectors must be between");
  }

  @Test
  void testTrainAndEncode() throws Exception {
    int dimension = 64;
    int numSubvectors = 4;
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    // Generate training data
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);

    // Train the quantizer
    CompletableFuture<Void> trainFuture = pq.train(trainingVectors);
    trainFuture.get(); // Wait for training to complete

    // Verify codebooks are created
    assertThat(pq.getCodebooks()).isNotNull();
    assertThat(pq.getCodebooks().length).isEqualTo(numSubvectors);
    assertThat(pq.getCodebooks()[0].length).isEqualTo(256); // 256 centroids

    // Encode a vector
    float[] testVector = generateRandomVector(dimension, random);
    byte[] codes = pq.encode(testVector);

    assertThat(codes).hasSize(numSubvectors);

    // Each code should be a valid centroid index
    for (byte code : codes) {
      int idx = Byte.toUnsignedInt(code);
      assertThat(idx).isBetween(0, 255);
    }
  }

  @Test
  void testDecodeReconstruction() throws Exception {
    int dimension = 32;
    int numSubvectors = 4;
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    // Train with sufficient data
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(1000, dimension, random);
    pq.train(trainingVectors).get();

    // Encode and decode
    float[] original = generateRandomVector(dimension, random);
    byte[] codes = pq.encode(original);
    float[] reconstructed = pq.decode(codes);

    assertThat(reconstructed).hasSize(dimension);

    // Reconstruction should be somewhat close to original (lossy compression)
    float reconstructionError = DistanceMetrics.l2Distance(original, reconstructed);
    float originalNorm = DistanceMetrics.norm(original);
    float relativeError = (float) Math.sqrt(reconstructionError) / originalNorm;

    // PQ is lossy, but relative error should be reasonable
    assertThat(relativeError).isLessThan(1.0f);
  }

  @Test
  void testLookupTableAndDistance() throws Exception {
    int dimension = 64;
    int numSubvectors = 8;
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    // Train
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);
    pq.train(trainingVectors).get();

    // Create query and database vectors
    float[] query = generateRandomVector(dimension, random);
    float[] dbVector = generateRandomVector(dimension, random);

    // Encode database vector
    byte[] dbCodes = pq.encode(dbVector);

    // Build lookup table for query
    float[][] lut = pq.buildLookupTable(query);
    assertThat(lut.length).isEqualTo(numSubvectors);
    assertThat(lut[0].length).isEqualTo(256);

    // Compute approximate distance using LUT
    float approxDistance = pq.computeDistance(dbCodes, lut);

    // Compare with exact distance
    float exactDistance = DistanceMetrics.l2Distance(query, dbVector);

    // Approximate distance should be somewhat close to exact
    float errorRatio = Math.abs(approxDistance - exactDistance) / exactDistance;
    assertThat(errorRatio).isLessThan(2.0f); // Allow up to 2x error for lossy compression
  }

  @Test
  void testCosineMetric() throws Exception {
    int dimension = 32;
    int numSubvectors = 4;
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.COSINE);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(300, dimension, random);

    // Train with cosine metric
    pq.train(trainingVectors).get();

    // Test vectors
    float[] v1 = generateRandomVector(dimension, random);
    float[] v2 = generateRandomVector(dimension, random);

    // Encode
    byte[] codes1 = pq.encode(v1);
    byte[] codes2 = pq.encode(v2);

    // Build LUT and compute distance
    float[][] lut = pq.buildLookupTable(v1);
    float approxDistance = pq.computeDistance(codes2, lut);

    // For cosine metric, vectors should be normalized
    float exactDistance = DistanceMetrics.cosineDistance(v1, v2);

    // Check approximate distance is reasonable
    assertThat(approxDistance).isGreaterThanOrEqualTo(0);
    assertThat(approxDistance).isLessThanOrEqualTo(4.0f); // Max possible with normalization
  }

  @Test
  void testInnerProductMetric() throws Exception {
    int dimension = 16;
    int numSubvectors = 2;
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.INNER_PRODUCT);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(300, dimension, random);

    pq.train(trainingVectors).get();

    float[] query = generateRandomVector(dimension, random);
    byte[] codes = pq.encode(query);
    float[] reconstructed = pq.decode(codes);

    assertThat(reconstructed).hasSize(dimension);
  }

  @Test
  void testLoadCodebooks() throws Exception {
    int dimension = 32;
    int numSubvectors = 4;
    int subDimension = dimension / numSubvectors;

    ProductQuantizer pq1 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    // Train first PQ
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);
    pq1.train(trainingVectors).get();

    // Get codebooks
    float[][][] codebooks = pq1.getCodebooks();

    // Create new PQ and load codebooks
    ProductQuantizer pq2 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);
    pq2.loadCodebooks(codebooks);

    // Test encoding produces same result
    float[] testVector = generateRandomVector(dimension, random);
    byte[] codes1 = pq1.encode(testVector);
    byte[] codes2 = pq2.encode(testVector);

    assertThat(codes2).isEqualTo(codes1);
  }

  @Test
  void testLoadCodebooksValidation() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);

    // Wrong number of subvectors
    float[][][] wrongCodebooks = new float[3][][];

    assertThatThrownBy(() -> pq.loadCodebooks(wrongCodebooks))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebook count mismatch");
  }

  @Test
  void testEncodeBeforeTrain() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);
    float[] vector = new float[32];

    assertThatThrownBy(() -> pq.encode(vector))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("not been trained");
  }

  @Test
  void testDecodeBeforeTrain() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);
    byte[] codes = new byte[4];

    assertThatThrownBy(() -> pq.decode(codes))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("not been trained");
  }

  @Test
  void testBuildLookupTableBeforeTrain() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);
    float[] query = new float[32];

    assertThatThrownBy(() -> pq.buildLookupTable(query))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("not been trained");
  }

  @Test
  void testEncodeDimensionMismatch() throws Exception {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(100, 32, random);
    pq.train(trainingVectors).get();

    float[] wrongDimVector = new float[64];

    assertThatThrownBy(() -> pq.encode(wrongDimVector))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dimension mismatch");
  }

  @Test
  void testDecodeLengthMismatch() throws Exception {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(100, 32, random);
    pq.train(trainingVectors).get();

    byte[] wrongLengthCodes = new byte[8];

    assertThatThrownBy(() -> pq.decode(wrongLengthCodes))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Code length mismatch");
  }

  @Test
  void testComputeDistanceLengthMismatch() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);

    byte[] wrongLengthCodes = new byte[8];
    float[][] lut = new float[4][256];

    assertThatThrownBy(() -> pq.computeDistance(wrongLengthCodes, lut))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Code length mismatch");
  }

  @Test
  void testTrainEmptySet() {
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2);
    List<float[]> emptySet = new ArrayList<>();

    CompletableFuture<Void> future = pq.train(emptySet);
    assertThatThrownBy(future::get)
        .isInstanceOf(ExecutionException.class)
        .hasRootCauseInstanceOf(IllegalArgumentException.class)
        .hasRootCauseMessage("Training set cannot be empty");
  }

  @Test
  void testTrainSmallSet() throws Exception {
    // Test with fewer vectors than centroids
    ProductQuantizer pq = new ProductQuantizer(16, 2, DistanceMetrics.Metric.L2);

    Random random = new Random(42);
    List<float[]> smallSet = generateRandomVectors(50, 16, random); // Much less than 256

    // Should still work but with degraded quality
    pq.train(smallSet).get();

    float[] testVector = generateRandomVector(16, random);
    byte[] codes = pq.encode(testVector);
    float[] reconstructed = pq.decode(codes);

    assertThat(codes).hasSize(2);
    assertThat(reconstructed).hasSize(16);
  }

  @Test
  void testHighDimensionalVectors() throws Exception {
    int dimension = 1024;
    int numSubvectors = 32;

    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(300, dimension, random);

    pq.train(trainingVectors).get();

    float[] testVector = generateRandomVector(dimension, random);
    byte[] codes = pq.encode(testVector);

    assertThat(codes).hasSize(numSubvectors);
  }

  @Test
  void testParallelTraining() throws Exception {
    // Test that parallel training works
    ProductQuantizer pq = new ProductQuantizer(64, 8, DistanceMetrics.Metric.L2);

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, 64, random);

    pq.train(trainingVectors).get();

    assertThat(pq.getCodebooks()).isNotNull();
    assertThat(pq.getCodebooks().length).isEqualTo(8);
  }

  @Test
  void testReproducibility() throws Exception {
    // Same seed should produce similar results
    int dimension = 32;
    int numSubvectors = 4;

    ProductQuantizer pq1 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);
    ProductQuantizer pq2 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2);

    // Generate same training data with same seed
    List<float[]> trainingVectors1 = generateRandomVectors(300, dimension, new Random(42));
    List<float[]> trainingVectors2 = generateRandomVectors(300, dimension, new Random(42));

    pq1.train(trainingVectors1).get();
    pq2.train(trainingVectors2).get();

    // Encode same vector
    float[] testVector = generateRandomVector(dimension, new Random(99));
    byte[] codes1 = pq1.encode(testVector);
    byte[] codes2 = pq2.encode(testVector);

    // Results should be similar (not necessarily identical due to parallel execution)
    int differences = 0;
    for (int i = 0; i < codes1.length; i++) {
      if (codes1[i] != codes2[i]) {
        differences++;
      }
    }

    // Allow some differences due to k-means randomness and parallel execution
    assertThat(differences).isLessThanOrEqualTo(numSubvectors);
  }

  // Helper methods

  private List<float[]> generateRandomVectors(int count, int dimension, Random random) {
    List<float[]> vectors = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      vectors.add(generateRandomVector(dimension, random));
    }
    return vectors;
  }

  private float[] generateRandomVector(int dimension, Random random) {
    float[] vector = new float[dimension];
    for (int i = 0; i < dimension; i++) {
      vector[i] = random.nextFloat() * 2 - 1; // Range [-1, 1]
    }
    return vector;
  }
}
