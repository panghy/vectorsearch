package io.github.panghy.vectorsearch.bench;

import io.github.panghy.vectorsearch.pq.PqEncoder;
import io.github.panghy.vectorsearch.util.Distances;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

/**
 * Pure-CPU microbenchmarks for distance computations and PQ operations. No FDB required.
 *
 * <p>Benchmarks:
 *
 * <ul>
 *   <li>L2 distance — parameterized by dimension (128, 768)
 *   <li>Cosine similarity — parameterized by dimension (128, 768)
 *   <li>PQ encode — M=16, K=256, dim=128 (subDim=8)
 *   <li>PQ LUT distance — simulate the hot inner loop: M table lookups + sum
 * </ul>
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1)
public class DistanceAndPqBenchmark {

  // ---- Distance benchmarks are parameterized by dimension ----

  @Param({"128", "768"})
  public int dimension;

  private float[] vecA;
  private float[] vecB;

  // ---- PQ encode fields (fixed: M=16, K=256, dim=128) ----
  private static final int PQ_M = 16;
  private static final int PQ_K = 256;
  private static final int PQ_DIM = 128;
  private static final int PQ_SUB_DIM = PQ_DIM / PQ_M; // 8

  private float[][][] pqCentroids;
  private float[] pqVector;

  // ---- PQ LUT distance fields ----
  private float[][] lut; // lut[M][K] — precomputed distance tables
  private byte[] pqCodes; // M byte codes

  @Setup(Level.Trial)
  public void setup() {
    Random rng = new Random(42);

    // Distance benchmark vectors
    vecA = randomFloats(rng, dimension);
    vecB = randomFloats(rng, dimension);

    // PQ encode data
    pqCentroids = new float[PQ_M][PQ_K][PQ_SUB_DIM];
    for (int m = 0; m < PQ_M; m++) {
      for (int k = 0; k < PQ_K; k++) {
        for (int d = 0; d < PQ_SUB_DIM; d++) {
          pqCentroids[m][k][d] = rng.nextFloat();
        }
      }
    }
    pqVector = randomFloats(rng, PQ_DIM);

    // PQ LUT distance data
    lut = new float[PQ_M][PQ_K];
    for (int m = 0; m < PQ_M; m++) {
      for (int k = 0; k < PQ_K; k++) {
        lut[m][k] = rng.nextFloat() * 10f;
      }
    }
    pqCodes = new byte[PQ_M];
    for (int m = 0; m < PQ_M; m++) {
      pqCodes[m] = (byte) rng.nextInt(PQ_K);
    }
  }

  // ---- Benchmarks ----

  @Benchmark
  public double l2Distance() {
    return Distances.l2(vecA, vecB);
  }

  @Benchmark
  public double cosineDistance() {
    return Distances.cosine(vecA, vecB);
  }

  @Benchmark
  public byte[] pqEncode() {
    return PqEncoder.encode(pqCentroids, pqVector);
  }

  /**
   * Simulates the PQ LUT distance hot path used in sealed segment search: given M byte codes and
   * M×K lookup tables, sum M table lookups.
   */
  @Benchmark
  public float pqLutDistance() {
    float dist = 0f;
    for (int m = 0; m < PQ_M; m++) {
      dist += lut[m][pqCodes[m] & 0xFF];
    }
    return dist;
  }

  // ---- Helpers ----

  private static float[] randomFloats(Random rng, int len) {
    float[] v = new float[len];
    for (int i = 0; i < len; i++) {
      v[i] = rng.nextFloat() * 2f - 1f;
    }
    return v;
  }
}
