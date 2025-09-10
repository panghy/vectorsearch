package io.github.panghy.vectorsearch.api;

import lombok.Builder;

/**
 * Result of a nearest-neighbor query.
 *
 * <p>Score semantics:
 * - L2: score = -distance (higher is better)
 * - COSINE: score = similarity in [-1, 1] (higher is better)
 *
 * @param segmentVectorId Packed id (segmentId << 32 | vectorId & 0xffffffff).
 * @param score     Ranking score (see class Javadoc for semantics).
 * @param distance  Convenience distance value for display or tie-breaking.
 * @param payload   Optional payload stored with the vector (may be empty).
 */
@Builder
public record SearchResult(long segmentVectorId, double score, double distance, byte[] payload) {}
