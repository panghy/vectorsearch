package io.github.panghy.vectorsearch.tasks;

import com.google.protobuf.ByteString;
import io.github.panghy.taskqueue.TaskQueueConfig;
import io.github.panghy.vectorsearch.proto.BuildTask;
import io.github.panghy.vectorsearch.proto.GlobalBuildTask;
import io.github.panghy.vectorsearch.proto.GlobalMaintenanceTask;
import io.github.panghy.vectorsearch.proto.MaintenanceTask;
import java.nio.charset.StandardCharsets;

/**
 * Utility TaskQueue serializers for {@link BuildTask}, {@link MaintenanceTask},
 * {@link GlobalBuildTask}, {@link GlobalMaintenanceTask}, and String keys.
 */
public final class ProtoSerializers {
  private ProtoSerializers() {}

  /**
   * Simple UTF-8 serializer for String task keys.
   */
  public static final class StringSerializer implements TaskQueueConfig.TaskSerializer<String> {
    /** Serializes a string to UTF-8 bytes. */
    @Override
    public ByteString serialize(String value) {
      if (value == null) return ByteString.EMPTY;
      return ByteString.copyFrom(value.getBytes(StandardCharsets.UTF_8));
    }

    /** Deserializes UTF-8 bytes back to a string. */
    @Override
    public String deserialize(ByteString bytes) {
      if (bytes == null) return null;
      return bytes.toStringUtf8();
    }
  }

  /**
   * Protobuf serializer for {@link BuildTask} payloads.
   */
  public static final class BuildTaskSerializer implements TaskQueueConfig.TaskSerializer<BuildTask> {
    /** Serializes a {@link BuildTask} into a {@link ByteString}. */
    @Override
    public ByteString serialize(BuildTask value) {
      if (value == null) return ByteString.EMPTY;
      return value.toByteString();
    }

    /** Deserializes a {@link BuildTask} from binary form. */
    @Override
    public BuildTask deserialize(ByteString bytes) {
      try {
        return bytes == null ? BuildTask.getDefaultInstance() : BuildTask.parseFrom(bytes);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw new IllegalArgumentException("Failed to parse BuildTask", e);
      }
    }
  }

  /** Protobuf serializer for {@link MaintenanceTask} payloads. */
  public static final class MaintenanceTaskSerializer implements TaskQueueConfig.TaskSerializer<MaintenanceTask> {
    @Override
    public ByteString serialize(MaintenanceTask value) {
      if (value == null) return ByteString.EMPTY;
      return value.toByteString();
    }

    @Override
    public MaintenanceTask deserialize(ByteString bytes) {
      try {
        return bytes == null ? MaintenanceTask.getDefaultInstance() : MaintenanceTask.parseFrom(bytes);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw new IllegalArgumentException("Failed to parse MaintenanceTask", e);
      }
    }
  }

  /** Protobuf serializer for {@link GlobalBuildTask} payloads. */
  public static final class GlobalBuildTaskSerializer implements TaskQueueConfig.TaskSerializer<GlobalBuildTask> {
    /** Serializes a {@link GlobalBuildTask} into a {@link ByteString}. */
    @Override
    public ByteString serialize(GlobalBuildTask value) {
      if (value == null) return ByteString.EMPTY;
      return value.toByteString();
    }

    /** Deserializes a {@link GlobalBuildTask} from binary form. */
    @Override
    public GlobalBuildTask deserialize(ByteString bytes) {
      try {
        return bytes == null ? GlobalBuildTask.getDefaultInstance() : GlobalBuildTask.parseFrom(bytes);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw new IllegalArgumentException("Failed to parse GlobalBuildTask", e);
      }
    }
  }

  /** Protobuf serializer for {@link GlobalMaintenanceTask} payloads. */
  public static final class GlobalMaintenanceTaskSerializer
      implements TaskQueueConfig.TaskSerializer<GlobalMaintenanceTask> {
    /** Serializes a {@link GlobalMaintenanceTask} into a {@link ByteString}. */
    @Override
    public ByteString serialize(GlobalMaintenanceTask value) {
      if (value == null) return ByteString.EMPTY;
      return value.toByteString();
    }

    /** Deserializes a {@link GlobalMaintenanceTask} from binary form. */
    @Override
    public GlobalMaintenanceTask deserialize(ByteString bytes) {
      try {
        return bytes == null
            ? GlobalMaintenanceTask.getDefaultInstance()
            : GlobalMaintenanceTask.parseFrom(bytes);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        throw new IllegalArgumentException("Failed to parse GlobalMaintenanceTask", e);
      }
    }
  }
}
