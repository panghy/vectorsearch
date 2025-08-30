package io.github.panghy.vectorsearch;

/**
 * Exception thrown when there is a configuration mismatch or error when opening an existing index.
 * This is a checked exception to ensure callers handle configuration mismatches appropriately.
 */
public class IndexConfigurationException extends Exception {

  private static final long serialVersionUID = 1L;

  /**
   * Creates a new IndexConfigurationException with the specified message.
   *
   * @param message the detail message
   */
  public IndexConfigurationException(String message) {
    super(message);
  }

  /**
   * Creates a new IndexConfigurationException with the specified message and cause.
   *
   * @param message the detail message
   * @param cause the cause of the exception
   */
  public IndexConfigurationException(String message, Throwable cause) {
    super(message, cause);
  }

  /**
   * Creates a new IndexConfigurationException with the specified cause.
   *
   * @param cause the cause of the exception
   */
  public IndexConfigurationException(Throwable cause) {
    super(cause);
  }
}
