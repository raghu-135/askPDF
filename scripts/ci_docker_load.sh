#!/usr/bin/env bash
# Pull pre-built CI images from GHCR and retag to local compose names.
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"

manifest="${ASKPDF_CI_IMAGE_MANIFEST:-/tmp/askpdf-ci-images.txt}"
if [ ! -f "$manifest" ]; then
  echo "missing image manifest: $manifest" >&2
  exit 1
fi

while IFS=$'\t' read -r local_tag remote_tag; do
  [ -n "$local_tag" ] || continue
  echo "Pulling $remote_tag -> $local_tag"
  docker pull "$remote_tag"
  docker tag "$remote_tag" "$local_tag"
done <"$manifest"

for tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "image not loaded: $tag" >&2
    exit 1
  fi
done

echo "All CI images loaded"