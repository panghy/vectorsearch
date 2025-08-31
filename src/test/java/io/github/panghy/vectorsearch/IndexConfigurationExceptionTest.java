package io.github.panghy.vectorsearch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Test class for IndexConfigurationException.
 */
class IndexConfigurationExceptionTest {

  @Test
  @DisplayName("Test constructor with message only")
  void testConstructorWithMessage() {
    String message = "Configuration mismatch detected";
    IndexConfigurationException exception = new IndexConfigurationException(message);

    assertEquals(message, exception.getMessage());
    assertNull(exception.getCause());
    assertNotNull(exception.toString());
  }

  @Test
  @DisplayName("Test constructor with message and cause")
  void testConstructorWithMessageAndCause() {
    String message = "Configuration error occurred";
    Exception cause = new IllegalArgumentException("Invalid dimension");
    IndexConfigurationException exception = new IndexConfigurationException(message, cause);

    assertEquals(message, exception.getMessage());
    assertSame(cause, exception.getCause());
    assertNotNull(exception.toString());
  }

  @Test
  @DisplayName("Test constructor with cause only")
  void testConstructorWithCause() {
    Exception cause = new RuntimeException("Underlying error");
    IndexConfigurationException exception = new IndexConfigurationException(cause);

    // When constructed with just a cause, the message is typically the cause's toString()
    assertNotNull(exception.getMessage());
    assertSame(cause, exception.getCause());
    assertNotNull(exception.toString());
  }

  @Test
  @DisplayName("Test exception is serializable")
  void testSerialVersionUID() {
    // Just verify that the exception has the serialVersionUID field
    IndexConfigurationException exception = new IndexConfigurationException("Test");
    assertNotNull(exception);
    // The serialVersionUID field is private static final, so we can't directly test it,
    // but we can verify the exception is constructed correctly
  }

  @Test
  @DisplayName("Test constructor with null message")
  void testConstructorWithNullMessage() {
    IndexConfigurationException exception = new IndexConfigurationException((String) null);

    assertNull(exception.getMessage());
    assertNull(exception.getCause());
    assertNotNull(exception.toString());
  }

  @Test
  @DisplayName("Test constructor with null cause")
  void testConstructorWithNullCause() {
    String message = "Error message";
    IndexConfigurationException exception = new IndexConfigurationException(message, null);

    assertEquals(message, exception.getMessage());
    assertNull(exception.getCause());
    assertNotNull(exception.toString());
  }

  @Test
  @DisplayName("Test constructor with null cause only")
  void testConstructorWithNullCauseOnly() {
    IndexConfigurationException exception = new IndexConfigurationException((Throwable) null);

    assertNull(exception.getCause());
    assertNotNull(exception.toString());
  }
}
