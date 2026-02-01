# PyPI Publishing Guide

## Prerequisites

Before publishing to PyPI, you need to set up **Trusted Publishing** (recommended by PyPI, no API tokens needed).

## Setup Steps

### 1. Configure PyPI Trusted Publishing

1. Go to [PyPI](https://pypi.org) and create an account if you don't have one
2. Navigate to your account settings → Publishing
3. Add a new "pending publisher":
   - **PyPI Project Name**: `kcuda-validate`
   - **Owner**: `kenhia` (your GitHub username)
   - **Repository name**: `kcuda`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi`

4. Repeat for [TestPyPI](https://test.pypi.org):
   - Use environment name: `testpypi`

### 2. Create GitHub Environments

1. Go to your GitHub repository → Settings → Environments
2. Create two environments:
   - **pypi**: For production releases
   - **testpypi**: For testing releases

3. For the `pypi` environment, add protection rules (optional but recommended):
   - Required reviewers: Add yourself or team members
   - Deployment branches: Only `main` or `release/*`

### 3. Test the Build Locally

```bash
# Build the package
uv build

# Check the distribution
uv run pip install twine
uv run twine check dist/*

# Inspect the contents
tar -tzf dist/kcuda_validate-*.tar.gz
unzip -l dist/kcuda_validate-*.whl
```

### 4. Create a Release

#### Option A: Using GitHub Web Interface (Recommended)

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Click "Choose a tag" and type a new version tag (e.g., `v0.1.0`)
4. Set the release title (e.g., "v0.1.0 - Initial Release")
5. Add release notes describing what's new
6. Click "Publish release"

The GitHub Action will automatically build and publish to PyPI and TestPyPI.

#### Option B: Using Command Line

```bash
# Update version in pyproject.toml first
# Then create and push a tag
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# Create release on GitHub (requires gh CLI)
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes "First stable release of kcuda-validate"
```

### 5. Verify Publication

After the release:

1. Check the Actions tab on GitHub to see the workflow run
2. Once complete, verify on PyPI: https://pypi.org/project/kcuda-validate/
3. Test installation:
   ```bash
   # In a new directory
   uv pip install kcuda-validate
   kcuda-validate --version
   ```

## Publishing Workflow

### For Production Release (PyPI)

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG or release notes
# 3. Commit changes
git add pyproject.toml
git commit -m "chore: bump version to 0.1.0"
git push

# 4. Create release (triggers publish workflow)
gh release create v0.1.0 \
  --title "v0.1.0" \
  --notes-file RELEASE_NOTES.md
```

### For Testing (TestPyPI)

TestPyPI is published automatically alongside PyPI when you create a release. To test before production:

```bash
# Install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  kcuda-validate
```

Note: `--extra-index-url` is needed because TestPyPI won't have your dependencies.

## Version Management

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

Update version in `pyproject.toml`:
```toml
version = "0.1.0"  # Update this line
```

## Troubleshooting

### "Publishing to PyPI is not enabled"
- Ensure you've configured the pending publisher on PyPI with exact repository details
- Check that the workflow environment name matches PyPI configuration

### "Failed to publish distribution"
- Verify the package name isn't already taken on PyPI
- Check build artifacts were created successfully in the build step

### "Token/OIDC authentication failed"
- Ensure `id-token: write` permission is set in workflow
- Verify GitHub environments are configured correctly

### Testing locally before release
```bash
# Build and check
uv build
uv run twine check dist/*

# Test installation from local build
uv pip install dist/kcuda_validate-*.whl
```

## Resources

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
- [uv Build Documentation](https://docs.astral.sh/uv/guides/publish/)
