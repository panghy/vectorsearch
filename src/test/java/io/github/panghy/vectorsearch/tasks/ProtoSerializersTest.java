package io.github.panghy.vectorsearch.tasks;

import static com.google.protobuf.ByteString.EMPTY;
import static org.assertj.core.api.Assertions.assertThat;

import com.google.protobuf.ByteString;
import io.github.panghy.vectorsearch.proto.BuildTask;
import org.junit.jupiter.api.Test;

class ProtoSerializersTest {
  @Test
  void string_serializer_round_trip() {
    ProtoSerializers.StringSerializer s = new ProtoSerializers.StringSerializer();
    ByteString bs = s.serialize("hello");
    assertThat(bs).isNotNull();
    assertThat(s.deserialize(bs)).isEqualTo("hello");
  }

  @Test
  void build_task_serializer_round_trip() {
    ProtoSerializers.BuildTaskSerializer s = new ProtoSerializers.BuildTaskSerializer();
    BuildTask t = BuildTask.newBuilder().setSegId(42).build();
    ByteString bs = s.serialize(t);
    BuildTask back = s.deserialize(bs);
    assertThat(back.getSegId()).isEqualTo(42);
  }

  @Test
  void serializers_handle_nulls() {
    ProtoSerializers.StringSerializer ss = new ProtoSerializers.StringSerializer();
    // null -> empty
    assertThat(ss.serialize(null)).isEqualTo(EMPTY);
    // null -> null
    assertThat(ss.deserialize(null)).isNull();

    ProtoSerializers.BuildTaskSerializer bs = new ProtoSerializers.BuildTaskSerializer();
    // null -> empty
    assertThat(bs.serialize(null)).isEqualTo(EMPTY);
    // null -> default instance
    assertThat(bs.deserialize(null)).isEqualTo(BuildTask.getDefaultInstance());
  }
}
