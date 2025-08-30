# CLAUDE.md

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
- Uses Palantir Java Format via Spotless
- 90% line coverage requirement
- 75% branch coverage requirement
- Protobuf generated classes are excluded from coverage