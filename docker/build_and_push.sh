#!/usr/bin/env bash
set -euo pipefail

# Build and optionally push the nanugpt Docker image.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

IMAGE_NAME=${IMAGE_NAME:-nanugpt}
REPO=${REPO:-}
TAG=${TAG:-latest}
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-0}
CUDA_INDEX_URL=${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --repo <name>            Repository, e.g. 'username/nanugpt' or 'ghcr.io/org/nanugpt'.
  --tag <tag>              Image tag (default: ${TAG}).
  --platform <plat>        Target platform (default: ${PLATFORM}).
  --push                   Push the image after building.
  --cuda-index-url <url>   PyTorch CUDA wheel index for xformers (default: ${CUDA_INDEX_URL}).
  -h, --help               Show this help.

Environment overrides:
  IMAGE_NAME, REPO, TAG, PLATFORM, PUSH, PYTORCH_CUDA_INDEX_URL

Examples:
  # Build locally only
  $(basename "$0") --repo username/nanugpt --tag 0.1.0

  # Build and push
  PUSH=1 $(basename "$0") --repo username/nanugpt --tag 0.1.0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --push) PUSH=1; shift 1 ;;
    --cuda-index-url) CUDA_INDEX_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${REPO}" ]]; then
  echo "Error: --repo is required (e.g., 'username/nanugpt')." >&2
  usage
  exit 1
fi

FULL_IMAGE="${REPO}:${TAG}"

echo "Building ${FULL_IMAGE} for platform ${PLATFORM}..."
echo "Using PYTORCH_CUDA_INDEX_URL=${CUDA_INDEX_URL}"

cd "${REPO_ROOT}"

# Ensure buildx is available
if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker buildx not found. Install Docker Buildx to continue." >&2
  exit 1
fi

BUILD_ARGS=(
  --platform "${PLATFORM}"
  --build-arg "PYTORCH_CUDA_INDEX_URL=${CUDA_INDEX_URL}"
  -t "${FULL_IMAGE}"
  -f docker/Dockerfile
  .
)

if [[ "${PUSH}" == "1" ]]; then
  echo "Building and pushing ${FULL_IMAGE}..."
  docker buildx build "${BUILD_ARGS[@]}" --push
else
  echo "Building ${FULL_IMAGE}" 
  docker buildx build "${BUILD_ARGS[@]}" --load
fi

echo "Done: ${FULL_IMAGE}"

