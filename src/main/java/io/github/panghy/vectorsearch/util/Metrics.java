package io.github.panghy.vectorsearch.util;

import com.apple.foundationdb.directory.DirectorySubspace;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.metrics.DoubleHistogram;
import io.opentelemetry.api.metrics.LongCounter;
import io.opentelemetry.api.metrics.Meter;
import io.opentelemetry.api.trace.Tracer;
import java.util.stream.Collectors;

/**
 * Centralizes OpenTelemetry instruments and helpers.
 */
public final class Metrics {
  private static final String INSTRUMENTATION_NAME = "io.github.panghy.vectorsearch";
  private static final Meter METER = GlobalOpenTelemetry.getMeter(INSTRUMENTATION_NAME);
  private static final Tracer TRACER = GlobalOpenTelemetry.getTracer(INSTRUMENTATION_NAME);

  // Histograms (ms)
  public static final DoubleHistogram QUERY_DURATION_MS = METER.histogramBuilder("vectorsearch.query.duration_ms")
      .setUnit("ms")
      .build();
  public static final DoubleHistogram BUILD_DURATION_MS = METER.histogramBuilder("vectorsearch.build.duration_ms")
      .setUnit("ms")
      .build();
  public static final DoubleHistogram VACUUM_DURATION_MS = METER.histogramBuilder("vectorsearch.vacuum.duration_ms")
      .setUnit("ms")
      .build();
  public static final DoubleHistogram COMPACTION_DURATION_MS = METER.histogramBuilder(
          "vectorsearch.compaction.duration_ms")
      .setUnit("ms")
      .build();

  // Counters
  public static final LongCounter QUERY_COUNT =
      METER.counterBuilder("vectorsearch.query.count").build();
  public static final LongCounter BUILD_COUNT =
      METER.counterBuilder("vectorsearch.build.count").build();
  public static final LongCounter VACUUM_RUN_COUNT =
      METER.counterBuilder("vectorsearch.vacuum.run").build();
  public static final LongCounter VACUUM_REMOVED =
      METER.counterBuilder("vectorsearch.vacuum.removed").build();
  public static final LongCounter COMPACTION_RUN_COUNT =
      METER.counterBuilder("vectorsearch.compaction.run").build();

  private Metrics() {}

  public static Tracer tracer() {
    return TRACER;
  }

  public static Attributes attrs(String key, String value) {
    return Attributes.of(io.opentelemetry.api.common.AttributeKey.stringKey(key), value);
  }

  public static Attributes attrs(String k1, String v1, String k2, String v2) {
    return Attributes.of(
        io.opentelemetry.api.common.AttributeKey.stringKey(k1), v1,
        io.opentelemetry.api.common.AttributeKey.stringKey(k2), v2);
  }

  public static String dirPath(DirectorySubspace dir) {
    try {
      return String.join("/", dir.getPath().stream().map(Object::toString).collect(Collectors.toList()));
    } catch (Exception e) {
      return "";
    }
  }
}
