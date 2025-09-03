package io.github.panghy.vectorsearch.pq;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

class ProductQuantizerTest {

  @Test
  void testConstructorValidation() {
    // First train to get codebooks
    int dimension = 128;
    int numSubvectors = 8;
    List<float[]> trainingVectors = generateRandomVectors(100, dimension, new Random(42));

    try {
      float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors)
          .get();

      // Valid construction with codebooks
      ProductQuantizer pq =
          new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);
      assertThat(pq.getDimension()).isEqualTo(dimension);
      assertThat(pq.getNumSubvectors()).isEqualTo(numSubvectors);
      assertThat(pq.getNumCentroids()).isEqualTo(256);
      assertThat(pq.getMetric()).isEqualTo(DistanceMetrics.Metric.L2);
      assertThat(pq.getCodebookVersion()).isEqualTo(1);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Invalid dimension
    assertThatThrownBy(() -> new ProductQuantizer(0, 8, DistanceMetrics.Metric.L2, 1, new float[8][][]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Dimension must be positive");

    // Invalid subvectors
    assertThatThrownBy(() -> new ProductQuantizer(128, 0, DistanceMetrics.Metric.L2, 1, new float[0][][]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Number of subvectors must be between");

    assertThatThrownBy(() -> new ProductQuantizer(128, 129, DistanceMetrics.Metric.L2, 1, new float[129][][]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Number of subvectors must be between");
  }

  @Test
  void testTrainAndEncode() throws Exception {
    int dimension = 64;
    int numSubvectors = 4;

    // Generate training data
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);

    // Train the quantizer - static method returns codebooks
    CompletableFuture<float[][][]> trainFuture =
        ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors);
    float[][][] codebooks = trainFuture.get(); // Wait for training to complete

    // Verify codebooks are created
    assertThat(codebooks).isNotNull();
    assertThat(codebooks.length).isEqualTo(numSubvectors);
    assertThat(codebooks[0].length).isEqualTo(256); // 256 centroids

    // Create ProductQuantizer with trained codebooks
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);

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

    // Train with sufficient data
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(1000, dimension, random);
    float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors)
        .get();

    // Create ProductQuantizer with trained codebooks
    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);

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

    // Train
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);
    float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors)
        .get();

    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);

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

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(300, dimension, random);

    // Train with cosine metric
    float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.COSINE, trainingVectors)
        .get();

    ProductQuantizer pq =
        new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.COSINE, 1, codebooks);

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

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(300, dimension, random);

    float[][][] codebooks = ProductQuantizer.train(
            numSubvectors, DistanceMetrics.Metric.INNER_PRODUCT, trainingVectors)
        .get();

    ProductQuantizer pq =
        new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.INNER_PRODUCT, 1, codebooks);

    float[] query = generateRandomVector(dimension, random);
    byte[] codes = pq.encode(query);
    float[] reconstructed = pq.decode(codes);

    assertThat(reconstructed).hasSize(dimension);
  }

  @Test
  void testWithPretrainedCodebooks() throws Exception {
    int dimension = 32;
    int numSubvectors = 4;
    int subDimension = dimension / numSubvectors;

    // Train first PQ to get codebooks
    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);
    float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors)
        .get();

    // Create new PQ with same codebooks but different version
    ProductQuantizer pq1 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);
    ProductQuantizer pq2 = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 2, codebooks);

    // Both should produce same encoding
    float[] testVector = generateRandomVector(dimension, random);
    byte[] codes1 = pq1.encode(testVector);
    byte[] codes2 = pq2.encode(testVector);

    assertThat(codes1).isEqualTo(codes2);
    assertThat(pq1.getCodebookVersion()).isEqualTo(1);
    assertThat(pq2.getCodebookVersion()).isEqualTo(2);
  }

  @Test
  void testBuildLookupTableFromPqCode() throws Exception {
    int dimension = 32;
    int numSubvectors = 4;

    Random random = new Random(42);
    List<float[]> trainingVectors = generateRandomVectors(500, dimension, random);
    float[][][] codebooks = ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, trainingVectors)
        .get();

    ProductQuantizer pq = new ProductQuantizer(dimension, numSubvectors, DistanceMetrics.Metric.L2, 1, codebooks);

    // Encode a vector
    float[] vector = generateRandomVector(dimension, random);
    byte[] pqCode = pq.encode(vector);

    // Build lookup table from PQ code
    float[][] lookupTable = pq.buildLookupTableFromPqCode(pqCode);

    assertThat(lookupTable).isNotNull();
    assertThat(lookupTable.length).isEqualTo(numSubvectors);
    assertThat(lookupTable[0].length).isEqualTo(256);

    // The lookup table should give zero distance to itself
    float selfDistance = pq.computeDistance(pqCode, lookupTable);
    assertThat(selfDistance).isLessThan(0.01f); // Should be near zero
  }

  @Test
  void testEmptyTrainingSet() {
    int dimension = 32;
    int numSubvectors = 4;
    List<float[]> emptyVectors = new ArrayList<>();

    assertThatThrownBy(() -> ProductQuantizer.train(numSubvectors, DistanceMetrics.Metric.L2, emptyVectors)
            .get())
        .hasCauseInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Training set cannot be empty");
  }

  @Test
  void testNullCodebooks() {
    assertThatThrownBy(() -> new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2, 1, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebooks cannot be null");
  }

  @Test
  void testCodebookMismatch() {
    // Wrong number of codebooks for subvectors
    float[][][] wrongCodebooks = new float[2][][]; // Only 2 instead of 4

    assertThatThrownBy(() -> new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2, 1, wrongCodebooks))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebook count mismatch");
  }

  @Test
  void testProductQuantizerErrorConditions() {
    // Test with invalid dimensions
    assertThatThrownBy(() -> new ProductQuantizer(-1, 4, DistanceMetrics.Metric.L2, 1, new float[4][256][16]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Dimension must be positive");

    // Test with null codebooks
    assertThatThrownBy(() -> new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebooks cannot be null");

    // Test with mismatched codebook count
    assertThatThrownBy(() -> new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, new float[2][256][16]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Codebook count mismatch");
  }

  @Test
  void testProductQuantizerTrainingWithInsufficientData() {
    List<float[]> trainingData = new ArrayList<>();
    // Add only 5 vectors (less than 256 centroids)
    Random random = new Random(42);
    for (int i = 0; i < 5; i++) {
      float[] vec = new float[64];
      for (int j = 0; j < 64; j++) {
        vec[j] = random.nextFloat();
      }
      trainingData.add(vec);
    }

    // This should still work, just with fewer effective centroids
    CompletableFuture<float[][][]> future = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData);

    assertThat(future).isCompletedWithValueMatching(codebooks -> {
      assertThat(codebooks.length).isEqualTo(4);
      assertThat(codebooks[0].length).isEqualTo(256);
      return true;
    });
  }

  @Test
  void testProductQuantizerWithDifferentMetrics() throws Exception {
    List<float[]> trainingData = generateRandomVectors(100, 64, new Random(42));

    // Test with COSINE metric
    float[][][] cosineCodebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.COSINE, trainingData)
        .get();
    ProductQuantizer cosinePq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.COSINE, 1, cosineCodebooks);

    float[] vector = trainingData.get(0);
    byte[] encoded = cosinePq.encode(vector);
    float[] decoded = cosinePq.decode(encoded);
    assertThat(decoded.length).isEqualTo(64);

    // Test with INNER_PRODUCT metric
    float[][][] ipCodebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.INNER_PRODUCT, trainingData)
        .get();
    ProductQuantizer ipPq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.INNER_PRODUCT, 1, ipCodebooks);

    encoded = ipPq.encode(vector);
    decoded = ipPq.decode(encoded);
    assertThat(decoded.length).isEqualTo(64);

    // Test lookup table computation
    float[][] lookupTable = ipPq.buildLookupTable(vector);
    assertThat(lookupTable.length).isEqualTo(4);
    assertThat(lookupTable[0].length).isEqualTo(256);

    // Test distance computation (INNER_PRODUCT can be negative)
    float distance = ipPq.computeDistance(encoded, lookupTable);
    assertThat(distance).isNotNull();
  }

  @Test
  void testProductQuantizerDecodeWithInvalidInput() {
    List<float[]> trainingData = generateRandomVectors(100, 64, new Random(42));
    float[][][] codebooks;
    try {
      codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData)
          .get();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    ProductQuantizer pq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, codebooks);

    // Test decode with wrong size
    assertThatThrownBy(() -> pq.decode(new byte[10]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Code length mismatch");
  }

  @Test
  void testProductQuantizerDistanceComputationEdgeCases() throws Exception {
    List<float[]> trainingData = generateRandomVectors(100, 32, new Random(42));
    float[][][] codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData)
        .get();
    ProductQuantizer pq = new ProductQuantizer(32, 4, DistanceMetrics.Metric.L2, 1, codebooks);

    float[] query = trainingData.get(0);
    float[][] lookupTable = pq.buildLookupTable(query);

    // Test with invalid code length
    assertThatThrownBy(() -> pq.computeDistance(new byte[10], lookupTable))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Code length mismatch");

    // Test with invalid lookup table - should throw ArrayIndexOutOfBoundsException
    assertThatThrownBy(() -> pq.computeDistance(new byte[4], new float[2][256]))
        .isInstanceOf(ArrayIndexOutOfBoundsException.class);
  }

  @Test
  void testProductQuantizerEncodeWithWrongDimension() throws Exception {
    List<float[]> trainingData = generateRandomVectors(100, 64, new Random(42));
    float[][][] codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData)
        .get();
    ProductQuantizer pq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, codebooks);

    // Test encode with wrong dimension
    float[] wrongSize = new float[32];
    assertThatThrownBy(() -> pq.encode(wrongSize))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Vector dimension mismatch");
  }

  @Test
  void testProductQuantizerBuildLookupTableFromPqCodeWithWrongSize() throws Exception {
    List<float[]> trainingData = generateRandomVectors(100, 64, new Random(42));
    float[][][] codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData)
        .get();
    ProductQuantizer pq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, codebooks);

    // Test buildLookupTableFromPqCode with wrong size
    assertThatThrownBy(() -> pq.buildLookupTableFromPqCode(new byte[10]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Code length mismatch");
  }

  @Test
  void testProductQuantizerTrainingWithLogging() throws Exception {
    // Enable detailed logging to cover logging paths
    List<float[]> trainingData = generateRandomVectors(300, 64, new Random(42));

    // This should trigger the logging path where we have more than 256 vectors
    float[][][] codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.L2, trainingData)
        .get();

    assertThat(codebooks).isNotNull();
    assertThat(codebooks.length).isEqualTo(4);

    // Create ProductQuantizer and test methods
    ProductQuantizer pq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.L2, 1, codebooks);

    // Test encode/decode round trip
    float[] vector = trainingData.get(0);
    byte[] encoded = pq.encode(vector);
    float[] decoded = pq.decode(encoded);
    assertThat(decoded.length).isEqualTo(64);

    // Test buildLookupTableFromPqCode
    float[][] lookupTable = pq.buildLookupTableFromPqCode(encoded);
    assertThat(lookupTable.length).isEqualTo(4);
    assertThat(lookupTable[0].length).isEqualTo(256);
  }

  @Test
  void testProductQuantizerAccessors() throws Exception {
    List<float[]> trainingData = generateRandomVectors(100, 64, new Random(42));
    float[][][] codebooks = ProductQuantizer.train(4, DistanceMetrics.Metric.COSINE, trainingData)
        .get();
    ProductQuantizer pq = new ProductQuantizer(64, 4, DistanceMetrics.Metric.COSINE, 2, codebooks);

    assertThat(pq.getDimension()).isEqualTo(64);
    assertThat(pq.getNumSubvectors()).isEqualTo(4);
    assertThat(pq.getNumCentroids()).isEqualTo(256);
    assertThat(pq.getMetric()).isEqualTo(DistanceMetrics.Metric.COSINE);
    assertThat(pq.getCodebookVersion()).isEqualTo(2);
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
