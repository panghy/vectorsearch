package io.github.panghy.vectorsearch.pq;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Product Quantizer for vector compression using codebook-based quantization.
 * Splits high-dimensional vectors into subspaces and quantizes each independently.
 */
public class ProductQuantizer {
  private static final Logger LOGGER = LoggerFactory.getLogger(ProductQuantizer.class);

  // Constants
  private static final int NUM_CENTROIDS = 256; // 2^8 for 8-bit quantization

  // K-means training parameters
  private static final int KMEANS_MAX_ITERATIONS = 25;
  private static final float KMEANS_CONVERGENCE_THRESHOLD = 1e-4f;
  private static final int KMEANS_INIT_ATTEMPTS = 3;

  @Getter
  private final int dimension;

  @Getter
  private final int numSubvectors;

  @Getter
  private final int numCentroids; // Always 256 for 8-bit PQ

  @Getter
  private final DistanceMetrics.Metric metric;

  @Getter
  private final int codebookVersion;

  // Codebooks: [subvector_index][centroid_index][dimension]
  @Getter
  private final float[][][] codebooks;

  private final AtomicBoolean debugDistanceOnce = new AtomicBoolean(false);
  private final AtomicBoolean debugLutOnce = new AtomicBoolean(false);

  /**
   * Creates a Product Quantizer with trained codebooks.
   *
   * @param dimension        the dimension of input vectors
   * @param numSubvectors    the number of subvectors (m in PQ literature)
   * @param metric           the distance metric to use
   * @param codebookVersion  the version of the codebooks
   * @param codebooks        the trained codebooks [subvector_index][centroid_index][dimension]
   */
  public ProductQuantizer(
      int dimension,
      int numSubvectors,
      DistanceMetrics.Metric metric,
      int codebookVersion,
      float[][][] codebooks) {

    if (dimension <= 0) {
      throw new IllegalArgumentException("Dimension must be positive: " + dimension);
    }
    if (numSubvectors <= 0 || numSubvectors > dimension) {
      throw new IllegalArgumentException(
          "Number of subvectors must be between 1 and " + dimension + ": " + numSubvectors);
    }
    if (codebooks == null) {
      throw new IllegalArgumentException("Codebooks cannot be null");
    }
    if (codebooks.length != numSubvectors) {
      throw new IllegalArgumentException(
          "Codebook count mismatch: expected " + numSubvectors + ", got " + codebooks.length);
    }

    this.dimension = dimension;
    this.numSubvectors = numSubvectors;
    this.numCentroids = NUM_CENTROIDS;
    this.metric = metric;
    this.codebookVersion = codebookVersion;
    this.codebooks = codebooks;
  }

  /**
   * Trains PQ codebooks using k-means clustering on training vectors.
   * This is a static method that returns the trained codebooks.
   *
   * @param numSubvectors   the number of subvectors
   * @param metric          the distance metric to use
   * @param trainingVectors the training data
   * @return CompletableFuture containing the trained codebooks
   */
  public static CompletableFuture<float[][][]> train(
      int numSubvectors, DistanceMetrics.Metric metric, List<float[]> trainingVectors) {

    if (trainingVectors.isEmpty()) {
      return CompletableFuture.failedFuture(new IllegalArgumentException("Training set cannot be empty"));
    }

    return CompletableFuture.supplyAsync(
        () -> {
          LOGGER.info(
              "Training PQ with {} vectors, {} subvectors, metric={}",
              trainingVectors.size(),
              numSubvectors,
              metric);

          // Pre-process vectors for cosine similarity
          List<float[]> processedVectors = preprocessVectors(trainingVectors, metric);

          // Initialize codebooks
          float[][][] newCodebooks = new float[numSubvectors][][];

          // Train each subspace in parallel
          List<CompletableFuture<float[][]>> futures = new ArrayList<>();

          for (int s = 0; s < numSubvectors; s++) {
            final int subvectorIndex = s;
            CompletableFuture<float[][]> future = CompletableFuture.supplyAsync(
                () -> {
                  LOGGER.debug("Training subvector {}/{}", subvectorIndex + 1, numSubvectors);

                  // Extract subvectors for this subspace
                  float[][] subvectors = new float[processedVectors.size()][];
                  for (int i = 0; i < processedVectors.size(); i++) {
                    subvectors[i] = VectorUtils.getSubvector(
                        processedVectors.get(i), subvectorIndex, numSubvectors);
                  }

                  // Run k-means clustering
                  return kmeansCluster(subvectors, NUM_CENTROIDS, metric);
                },
                ForkJoinPool.commonPool());

            futures.add(future);
          }

          // Wait for all subspaces to complete
          for (int s = 0; s < numSubvectors; s++) {
            try {
              newCodebooks[s] = futures.get(s).join();
            } catch (Exception e) {
              throw new RuntimeException("Failed to train subvector " + s, e);
            }
          }

          LOGGER.info("PQ training completed");
          return newCodebooks;
        },
        ForkJoinPool.commonPool());
  }

  /**
   * Encodes a vector into PQ codes (one byte per subvector).
   *
   * @param vector the input vector
   * @return byte array of PQ codes
   */
  public byte[] encode(float[] vector) {
    if (vector.length != dimension) {
      throw new IllegalArgumentException(
          "Vector dimension mismatch: expected " + dimension + ", got " + vector.length);
    }

    // Pre-process vector for cosine similarity
    float[] processedVector = preprocessVector(vector);

    byte[] codes = new byte[numSubvectors];

    for (int s = 0; s < numSubvectors; s++) {
      float[] subvector = VectorUtils.getSubvector(processedVector, s, numSubvectors);

      // Find nearest centroid
      int nearestCentroid = 0;
      float minDistance = Float.POSITIVE_INFINITY;

      for (int c = 0; c < numCentroids; c++) {
        float distance = computeSubvectorDistance(subvector, codebooks[s][c]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestCentroid = c;
        }
      }

      codes[s] = (byte) nearestCentroid;
    }

    // Debug: Check if all codes are the same
    boolean allSame = true;
    for (int i = 1; i < codes.length; i++) {
      if (codes[i] != codes[0]) {
        allSame = false;
        break;
      }
    }
    if (allSame) {
      LOGGER.warn("All PQ codes are the same: {}", Byte.toUnsignedInt(codes[0]));
    }

    return codes;
  }

  /**
   * Decodes PQ codes back to an approximate vector.
   *
   * @param codes the PQ codes
   * @return reconstructed vector
   */
  public float[] decode(byte[] codes) {
    if (codes.length != numSubvectors) {
      throw new IllegalArgumentException(
          "Code length mismatch: expected " + numSubvectors + ", got " + codes.length);
    }

    float[] reconstructed = new float[dimension];
    int offset = 0;

    for (int s = 0; s < numSubvectors; s++) {
      int centroidIdx = Byte.toUnsignedInt(codes[s]);
      float[] centroid = codebooks[s][centroidIdx];

      System.arraycopy(centroid, 0, reconstructed, offset, centroid.length);
      offset += centroid.length;
    }

    return reconstructed;
  }

  /**
   * Builds a lookup table for efficient distance computation with PQ codes.
   * The LUT contains precomputed distances between the query and all centroids.
   *
   * @param query the query vector
   * @return lookup table [subvector_index][centroid_index] -> distance
   */
  public float[][] buildLookupTable(float[] query) {
    if (query.length != dimension) {
      throw new IllegalArgumentException(
          "Query dimension mismatch: expected " + dimension + ", got " + query.length);
    }

    // Pre-process query for cosine similarity
    float[] processedQuery = preprocessVector(query);

    float[][] lut = new float[numSubvectors][numCentroids];

    boolean debugOnce = debugLutOnce.compareAndSet(false, true);

    for (int s = 0; s < numSubvectors; s++) {
      float[] querySubvector = VectorUtils.getSubvector(processedQuery, s, numSubvectors);

      for (int c = 0; c < numCentroids; c++) {
        lut[s][c] = computeSubvectorDistance(querySubvector, codebooks[s][c]);
      }

      if (debugOnce && s == 0) {
        // Check if all values in first subvector are the same
        float firstVal = lut[0][0];
        boolean allSame = true;
        for (int c = 1; c < Math.min(10, numCentroids); c++) {
          if (Math.abs(lut[0][c] - firstVal) > 0.001) {
            allSame = false;
            break;
          }
        }
        if (allSame) {
          LOGGER.warn("LUT[0] has all same values: {}", firstVal);
        } else {
          LOGGER.debug(
              "LUT[0] sample: [{}, {}, {}, {}, {}]",
              lut[0][0],
              lut[0][1],
              lut[0][2],
              lut[0][3],
              lut[0][4]);
        }
      }
    }

    return lut;
  }

  /**
   * Computes the approximate distance between a query and a PQ-encoded vector using a lookup table.
   *
   * @param codes       the PQ codes of the database vector
   * @param lookupTable precomputed lookup table from buildLookupTable()
   * @return approximate distance
   */
  public float computeDistance(byte[] codes, float[][] lookupTable) {
    if (codes.length != numSubvectors) {
      throw new IllegalArgumentException(
          "Code length mismatch: expected " + numSubvectors + ", got " + codes.length);
    }

    float distance = 0;
    boolean firstDebug = debugDistanceOnce.compareAndSet(false, true);

    for (int s = 0; s < numSubvectors; s++) {
      int centroidIdx = Byte.toUnsignedInt(codes[s]);
      float subDist = lookupTable[s][centroidIdx];
      distance += subDist;

      if (firstDebug && s < 3) {
        LOGGER.debug("Subvector {}: code={}, distance={}", s, centroidIdx, subDist);
      }
    }

    if (firstDebug) {
      LOGGER.debug("Total distance: {}", distance);
    }

    return distance;
  }

  /**
   * Builds a lookup table from PQ codes for neighbor search.
   * This approximates the lookup table by using the reconstructed vector.
   *
   * @param pqCode the PQ codes of a vector
   * @return lookup table [subvector_index][centroid_index] -> distance
   */
  public float[][] buildLookupTableFromPqCode(byte[] pqCode) {
    if (pqCode.length != numSubvectors) {
      throw new IllegalArgumentException(
          "Code length mismatch: expected " + numSubvectors + ", got " + pqCode.length);
    }

    // Reconstruct the vector from PQ codes
    float[] reconstructed = decode(pqCode);

    // Build lookup table using the reconstructed vector
    return buildLookupTable(reconstructed);
  }

  /**
   * Pre-processes vectors based on the distance metric.
   * For cosine similarity, normalizes vectors to unit length.
   */
  private static List<float[]> preprocessVectors(List<float[]> vectors, DistanceMetrics.Metric metric) {
    if (metric == DistanceMetrics.Metric.COSINE) {
      List<float[]> normalized = new ArrayList<>(vectors.size());
      for (float[] v : vectors) {
        normalized.add(DistanceMetrics.normalize(v));
      }
      return normalized;
    }
    return vectors;
  }

  /**
   * Pre-processes a single vector based on the distance metric.
   */
  private float[] preprocessVector(float[] vector) {
    if (metric == DistanceMetrics.Metric.COSINE) {
      return DistanceMetrics.normalize(vector);
    }
    return vector;
  }

  /**
   * Computes distance between subvectors based on the configured metric.
   */
  private float computeSubvectorDistance(float[] a, float[] b) {
    // For cosine, vectors are already normalized, so use L2
    if (metric == DistanceMetrics.Metric.COSINE) {
      return DistanceMetrics.l2Distance(a, b);
    }
    return DistanceMetrics.distance(a, b, metric);
  }

  /**
   * Static version for use in training.
   */
  private static float computeSubvectorDistance(float[] a, float[] b, DistanceMetrics.Metric metric) {
    // For cosine, vectors are already normalized, so use L2
    if (metric == DistanceMetrics.Metric.COSINE) {
      return DistanceMetrics.l2Distance(a, b);
    }
    return DistanceMetrics.distance(a, b, metric);
  }

  /**
   * Performs k-means clustering on a set of vectors.
   *
   * @param vectors the input vectors
   * @param k       the number of clusters
   * @param metric  the distance metric to use
   * @return cluster centroids
   */
  private static float[][] kmeansCluster(float[][] vectors, int k, DistanceMetrics.Metric metric) {
    if (vectors.length < k) {
      LOGGER.warn("Training set size {} is less than k={}, duplicating vectors as centroids", vectors.length, k);
      float[][] centroids = new float[k][];
      for (int i = 0; i < k; i++) {
        centroids[i] = vectors[i % vectors.length].clone();
      }
      return centroids;
    }

    int dimension = vectors[0].length;
    Random random = ThreadLocalRandom.current();

    float[][] bestCentroids = null;
    float bestInertia = Float.POSITIVE_INFINITY;

    // Multiple initialization attempts for better results
    for (int attempt = 0; attempt < KMEANS_INIT_ATTEMPTS; attempt++) {
      // Initialize centroids using k-means++ method
      float[][] centroids = initializeCentroidsKMeansPlusPlus(vectors, k, random, metric);

      int[] assignments = new int[vectors.length];
      float prevInertia = Float.POSITIVE_INFINITY;

      for (int iter = 0; iter < KMEANS_MAX_ITERATIONS; iter++) {
        // Assign vectors to nearest centroids
        float inertia = 0;
        for (int i = 0; i < vectors.length; i++) {
          int nearest = 0;
          float minDist = Float.POSITIVE_INFINITY;

          for (int c = 0; c < k; c++) {
            float dist = computeSubvectorDistance(vectors[i], centroids[c], metric);
            if (dist < minDist) {
              minDist = dist;
              nearest = c;
            }
          }

          assignments[i] = nearest;
          inertia += minDist;
        }

        // Check for convergence
        if (Math.abs(prevInertia - inertia) < KMEANS_CONVERGENCE_THRESHOLD) {
          LOGGER.debug("K-means converged at iteration {} (attempt {})", iter, attempt);
          break;
        }
        prevInertia = inertia;

        // Update centroids
        int[] counts = new int[k];
        float[][] newCentroids = new float[k][dimension];

        for (int i = 0; i < vectors.length; i++) {
          int cluster = assignments[i];
          counts[cluster]++;
          for (int d = 0; d < dimension; d++) {
            newCentroids[cluster][d] += vectors[i][d];
          }
        }

        for (int c = 0; c < k; c++) {
          if (counts[c] > 0) {
            for (int d = 0; d < dimension; d++) {
              centroids[c][d] = newCentroids[c][d] / counts[c];
            }
          } else {
            // Reinitialize empty clusters
            centroids[c] = vectors[random.nextInt(vectors.length)].clone();
          }
        }

        // Keep best result across attempts
        if (inertia < bestInertia) {
          bestInertia = inertia;
          bestCentroids = centroids;
        }
      }
    }

    return bestCentroids;
  }

  /**
   * Initializes k-means centroids using the k-means++ method for better convergence.
   */
  private static float[][] initializeCentroidsKMeansPlusPlus(
      float[][] vectors, int k, Random random, DistanceMetrics.Metric metric) {
    float[][] centroids = new float[k][];

    // Choose first centroid randomly
    centroids[0] = vectors[random.nextInt(vectors.length)].clone();

    // Choose remaining centroids with probability proportional to squared distance
    for (int c = 1; c < k; c++) {
      float[] distances = new float[vectors.length];
      float totalDistance = 0;

      for (int i = 0; i < vectors.length; i++) {
        float minDist = Float.POSITIVE_INFINITY;
        for (int j = 0; j < c; j++) {
          float dist = computeSubvectorDistance(vectors[i], centroids[j], metric);
          minDist = Math.min(minDist, dist);
        }
        distances[i] = minDist * minDist; // Square for k-means++
        totalDistance += distances[i];
      }

      // Select next centroid with weighted probability
      float threshold = random.nextFloat() * totalDistance;
      float cumulative = 0;
      for (int i = 0; i < vectors.length; i++) {
        cumulative += distances[i];
        if (cumulative >= threshold) {
          centroids[c] = vectors[i].clone();
          break;
        }
      }
    }

    return centroids;
  }
}
