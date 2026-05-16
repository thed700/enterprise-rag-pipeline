#!/usr/bin/env bash
# scripts/release.sh — AuraRAG release helper
# Usage: ./scripts/release.sh [tag]   e.g. ./scripts/release.sh v3.2.0
set -euo pipefail

TAG="${1:-v3.2.0}"

echo "═══════════════════════════════════════════"
echo "  AuraRAG — Preparing release ${TAG}"
echo "═══════════════════════════════════════════"

# 1. Verify working tree is clean
if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌  Working tree is dirty. Commit or stash changes first."
  git status --short
  exit 1
fi

# 2. Run tests
echo ""
echo "▶ Running test suite..."
pytest tests/ -v --tb=short
echo "✅  All tests passed."

# 3. Confirm version strings match the tag
APP_VERSION=$(python -c "from app.utils import APP_VERSION; print(APP_VERSION)")
TAG_VERSION="${TAG#v}"   # strip leading 'v'

if [[ "$APP_VERSION" != "$TAG_VERSION" ]]; then
  echo "❌  Version mismatch: app/utils.py says ${APP_VERSION}, tag is ${TAG}."
  echo "    Update APP_VERSION in app/utils.py before releasing."
  exit 1
fi
echo "✅  Version ${APP_VERSION} confirmed."

# 4. Create and push the annotated tag
echo ""
echo "▶ Tagging ${TAG}..."
git tag -a "${TAG}" -m "AuraRAG ${TAG}" --sign 2>/dev/null \
  || git tag -a "${TAG}" -m "AuraRAG ${TAG}"

echo "▶ Pushing commits + tag to origin..."
git push origin HEAD
git push origin "${TAG}"

echo ""
echo "✅  Release ${TAG} pushed."
echo ""
echo "   Next: create the GitHub release at"
echo "   https://github.com/thed700/enterprise-rag-pipeline/releases/new?tag=${TAG}"
echo "   and paste the contents of RELEASE_NOTES.md."
