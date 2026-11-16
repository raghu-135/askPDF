#!/usr/bin/env bash
# Tag and push CI images to GHCR; write manifest for downstream jobs.
set -euo pipefail

# shellcheck source=ci_common.sh
source "$(dirname "$0")/ci_common.sh"

if [ -z "${ASKPDF_CI_IMAGE_TAG:-}" ]; then
  echo "ASKPDF_CI_IMAGE_TAG is required" >&2
  exit 1
fi

registry="${ASKPDF_CI_REGISTRY:-ghcr.io}"
owner_lc="$(echo "${GITHUB_REPOSITORY_OWNER:-}" | tr '[:upper:]' '[:lower:]')"
repo_lc="$(echo "${GITHUB_REPOSITORY:+${GITHUB_REPOSITORY#*/}}" | tr '[:upper:]' '[:lower:]')"
namespace="${ASKPDF_CI_REGISTRY_NAMESPACE:-${owner_lc}/${repo_lc}/ci}"

manifest="${ASKPDF_CI_IMAGE_MANIFEST:-/tmp/askpdf-ci-images.txt}"
: >"$manifest"

for local_tag in "${ASKPDF_CI_IMAGE_TAGS[@]}"; do
  safe="${local_tag//:/__}"
  remote="${registry}/${namespace}/${safe}:${ASKPDF_CI_IMAGE_TAG}"
  docker tag "$local_tag" "$remote"
  docker push "$remote"
  printf '%s\t%s\n' "$local_tag" "$remote" >>"$manifest"
done

echo "Published ${#ASKPDF_CI_IMAGE_TAGS[@]} images (manifest: $manifest)"