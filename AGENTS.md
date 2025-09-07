# AGENTS.md

This project uses a terminal-first workflow. Below is a concise guide for contributors and AI agents working on the codebase.

## Commands

### Build and Test
- Build: `./gradlew build`
- Clean build: `./gradlew clean build`
- Run tests: `./gradlew test`
- Run a single test: `./gradlew test --tests "io.github.panghy.vectorsearch.SomeTest"`

### Code Quality
- Apply code formatting: `./gradlew spotlessApply`
- Check code formatting: `./gradlew spotlessCheck`
- Generate coverage report: `./gradlew jacocoTestReport`
- Check coverage thresholds: `./gradlew jacocoTestCoverageVerification`

### Publishing
- Publish snapshot: `./gradlew publishToSonatype`
- Publish release: `./gradlew publishAndReleaseToMavenCentral`

## Release Process

### Prerequisites
- Ensure all tests pass: `./gradlew test`
- Ensure code coverage meets requirements: `./gradlew jacocoTestCoverageVerification`
- Ensure code is properly formatted: `./gradlew spotlessCheck`

### Steps to Release

1. **Create Release Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b release/X.Y.Z
   ```

2. **Update Version**
   - Edit `build.gradle` and change version from `X.Y.Z-SNAPSHOT` to `X.Y.Z`
   ```gradle
   version = 'X.Y.Z'
   ```

3. **Commit and Push Release Branch**
   ```bash
   git add build.gradle
   git commit -m "chore: release version X.Y.Z"
   git push -u origin release/X.Y.Z
   ```

4. **Create GitHub Release**
   ```bash
   gh release create vX.Y.Z \
     --target release/X.Y.Z \
     --title "vX.Y.Z" \
     --notes "Release notes here..."
   ```
   
   This will automatically trigger the publish workflow to deploy to Maven Central.

5. **Update Version for Next Development Cycle**
   ```bash
   git checkout -b chore/bump-version-X.Y+1.0
   git checkout main build.gradle  # Get latest from main
   # Edit build.gradle to set version = 'X.Y+1.0-SNAPSHOT'
   git add build.gradle
   git commit -m "chore: bump version to X.Y+1.0-SNAPSHOT for next development cycle"
   git push -u origin chore/bump-version-X.Y+1.0
   ```

6. **Create PR for Version Bump**
   ```bash
   gh pr create \
     --title "chore: bump version to X.Y+1.0-SNAPSHOT" \
     --body "Bump version for next development cycle after X.Y.Z release" \
     --base main
   ```

7. **Auto-merge the Version Bump PR**
   ```bash
   # Enable auto-merge for the PR (requires admin or write permissions)
   gh pr merge --auto --rebase
   ```

### Automated Publishing
The `.github/workflows/publish.yml` workflow automatically:
- Triggers on GitHub release creation
- Builds the project
- Runs all tests
- Publishes to Maven Central via `publishAndReleaseToMavenCentral`
- Generates and submits dependency graph

### Version Numbering
- Production releases: `X.Y.Z`
- Development snapshots: `X.Y.Z-SNAPSHOT`
- Follow semantic versioning:
  - MAJOR (X): Breaking API changes
  - MINOR (Y): New features, backward compatible
  - PATCH (Z): Bug fixes, backward compatible

## Code Style
- Formatting: Palantir Java Format via Spotless (use `spotlessApply`).
- Imports:
  - Prefer static imports for assertions (AssertJ/JUnit) and Mockito.
  - Do not use fully qualified references (always use imports, static or otherwise)
- Comments: Add Javadoc for public classes and methods (purpose, params, returns). Add class-level Javadocs to tests describing intent.
- Overloads: Provide both transaction-scoped APIs (accepting `Transaction`/`ReadTransaction`) and convenience overloads that run their own transaction so callers can choose boundaries.
- Coverage: Maintain line coverage >= 90% and branch coverage >= 75% (JaCoCo verification). Protobuf-generated classes are excluded.

## Architecture Snapshot (2025-09-07)

- Segments and rotation: ACTIVE → PENDING → SEALED. Strict-cap rotation is enforced; when `count >= maxSegmentSize`, the next insert rotates and becomes `vecId=0` in the new ACTIVE segment.
- Search defaults: `SearchParams.defaults(...)` now use BEST_FIRST with higher `ef`, conservative `maxExplore`, and optional refinement. BEAM is retained but marked deprecated, with a one-time WARN if used.
- Caches: `SegmentCaches` provides AsyncLoadingCache for PQ codebooks and adjacency with `asyncLoadAll` (batched by configurable sizes). Query code uses `getAll` to trigger async bulk loads; adjacency accesses are batched per frontier.
- Prefetch: Query fire-and-forgets codebook prefetch for all SEALED segments; toggle via `VectorIndexConfig.prefetchCodebooksEnabled`.
- Observability: OpenTelemetry gauges exposed for caches:
  - `vectorsearch.cache.size{cache=codebook|adjacency}`
  - `vectorsearch.cache.hit_count{cache=...}`
  - `vectorsearch.cache.miss_count{cache=...}`
  - `vectorsearch.cache.load_success_count{cache=...}`
  - `vectorsearch.cache.load_failure_count{cache=...}`
  Additional attributes may be injected via `VectorIndexConfig.metricAttribute(...)`.
- Logging: Per-query logs are DEBUG. Worker lifecycle logs are DEBUG. BEAM usage emits WARN once. Keep INFO quiet by default.
- Segment discovery: queries enumerate segments using a `maxSegmentKey` sentinel and meta probes to avoid missing sealed segments.

## Key Config Knobs

- `VectorIndexConfig` builder:
  - `codebookBatchLoadSize`, `adjacencyBatchLoadSize` (defaults 10_000)
  - `prefetchCodebooksEnabled` (default true)
  - `metricAttribute(key, value)` to annotate OTel metrics
  - `oversample`, `graphDegree`, `pqM`, `pqK`, `maxSegmentSize`, `dimension`, `metric`

## Testing Patterns

### Async Operations
- **Never use `.join()` or blocking calls** within storage layer methods
- All storage operations should return `CompletableFuture<T>`
- Use proper future composition with `.thenCompose()`, `.thenApply()`, `.thenAccept()`
- Tests can use `.join()` at the top level, but storage methods cannot

### Transaction Management
- All storage layer methods take `Transaction` as a parameter
- Callers control transaction boundaries via `db.runAsync(tx -> ...)`
- This allows for efficient batching and conflict resolution at the application layer
- Example pattern:
  ```java
  // Storage method
  public CompletableFuture<Void> storeData(Transaction tx, long id, Data data) {
      return readProto(tx, key, Parser.parser()).thenApply(existing -> {
          // Update logic
          tx.set(key, updated.toByteArray());
          return null;
      });
  }
  
  // Caller usage
  db.runAsync(tx -> {
      return storage.storeData(tx, id1, data1)
          .thenCompose(v -> storage.storeData(tx, id2, data2));
  }).join();
  ```

### Thread Safety
- Use synchronization for shared mutable state in batch operations
- Caffeine cache handles its own thread safety
- FoundationDB transactions provide isolation between concurrent operations

### FoundationDB Constraints
- **5-second transaction limit** - All operations must complete within 5 seconds
- **10MB transaction size limit** - Cannot read/write more than 10MB in a single transaction
- **Key strategies for large-scale operations:**
  - Use sampling instead of full scans
  - Batch operations into multiple transactions
  - Implement cursor-based pagination for large result sets
  - Use range-splitting for distributed sampling

## Testing Patterns
- Use `db.runAsync()` for write operations and `db.read()` for read-only tests to exercise both overload styles.
- Chain futures properly to ensure operations complete in order
- Always verify test coverage meets requirements before committing
- Integration tests should use unique test collections to avoid conflicts
- **Use real FDB instances** instead of mocks for storage layer tests (project standard)
- Test with various graph sizes to ensure algorithms handle edge cases
- **Maintenance Method Testing**: Use package-private methods with `CompletableFuture<Void>` returns for testability
- **Cache Testing**: Use accessor methods like `getPqBlockCache()` to verify cache behavior
- **Async Testing**: All maintenance operations should be properly tested with future composition

### OpenTelemetry Testing

- Use the official SDK testing utilities. In each test class that asserts metrics:
  - Create an `InMemoryMetricReader` and `SdkMeterProvider`, set it via `OpenTelemetrySdk` and `GlobalOpenTelemetry.set(...)` in `@BeforeEach`.
  - After exercising the code, call `reader.collectAllMetrics()` and assert on `MetricData` (e.g., presence of `vectorsearch.cache.size`).
  - Close the meter provider and `GlobalOpenTelemetry.resetForTest()` in `@AfterEach`.

Example snippet:

InMemoryMetricReader reader = InMemoryMetricReader.create();
SdkMeterProvider mp = SdkMeterProvider.builder().registerMetricReader(reader).build();
OpenTelemetrySdk sdk = OpenTelemetrySdk.builder().setMeterProvider(mp).build();
GlobalOpenTelemetry.resetForTest();
GlobalOpenTelemetry.set(sdk);
// ... run code ...
boolean present = reader.collectAllMetrics().stream()
    .anyMatch(m -> m.getName().equals("vectorsearch.cache.size"));
assertThat(present).isTrue();
sdk.getSdkMeterProvider().close();
GlobalOpenTelemetry.resetForTest();

## Practical Tips
- Prefer static imports; avoid fully qualified references in code/tests.
- Provide transactional and non-transactional overloads for storage methods to give callers control over boundaries.
- Keep build green and coverage above thresholds: run `./gradlew build` frequently.

## Debugging Tips
- When tests fail, use `./gradlew test -i` to see detailed output
- For flaky tests, check for proper future composition and transaction boundaries
- Common pitfalls:
  - Forgetting to handle empty results from sampling operations
  - Not checking for division by zero in statistical calculations
  - Missing synchronization in batch operations with shared state
- Prefer DEBUG logs; keep INFO sparse. Per-query and per-segment search details are DEBUG-only.
