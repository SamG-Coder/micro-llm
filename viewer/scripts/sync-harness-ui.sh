#!/usr/bin/env bash
# Maintainer snapshot only. SamG does not run this. The exe hosts harness/ui/.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root/viewer"
npm ci
npm test
npm run build
mkdir -p "$root/harness/ui"
find "$root/harness/ui" -mindepth 1 ! -name README.md -exec rm -rf {} +
cp -a "$root/viewer/dist/." "$root/harness/ui/"
echo "wrote $root/harness/ui (committed static snapshot)"
