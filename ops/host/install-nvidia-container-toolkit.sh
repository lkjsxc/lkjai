#!/bin/sh
set -eu

APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --dry-run) APPLY=0 ;;
    *)
      echo "usage: $0 [--dry-run|--apply]" >&2
      exit 2
      ;;
  esac
done

TOOLKIT_VERSION="${NVIDIA_CONTAINER_TOOLKIT_VERSION:-}"
KEYRING=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
APT_LIST=/etc/apt/sources.list.d/nvidia-container-toolkit.list
DAEMON_JSON=/etc/docker/daemon.json
CUDA_SAMPLE_IMAGE="${CUDA_SAMPLE_IMAGE:-nvidia/cuda:12.8.1-base-ubuntu24.04}"

echo "NVIDIA Container Toolkit host setup"
echo "mode: $([ "$APPLY" -eq 1 ] && echo apply || echo dry-run)"
echo "mutations:"
echo "- install apt prerequisites: ca-certificates curl gnupg2"
echo "- write NVIDIA libnvidia-container keyring: $KEYRING"
echo "- write NVIDIA stable apt list: $APT_LIST"
if [ -n "$TOOLKIT_VERSION" ]; then
  echo "- install toolkit package set at version: $TOOLKIT_VERSION"
else
  echo "- install latest toolkit package set from the stable repository"
fi
echo "- back up $DAEMON_JSON before nvidia-ctk if it exists"
echo "- run nvidia-ctk runtime configure --runtime=docker"
echo "- restart Docker with systemctl"
echo "- verify Docker runtimes and run $CUDA_SAMPLE_IMAGE nvidia-smi"

if [ "$APPLY" -ne 1 ]; then
  echo "dry-run only; rerun with --apply to mutate the host"
  exit 0
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

tmp_key="$(mktemp)"
trap 'rm -f "$tmp_key"' EXIT
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
  gpg --dearmor > "$tmp_key"
sudo install -m 0644 "$tmp_key" "$KEYRING"

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
  sed "s#deb https://#deb [signed-by=$KEYRING] https://#g" |
  sudo tee "$APT_LIST" >/dev/null

sudo apt-get update
if [ -n "$TOOLKIT_VERSION" ]; then
  sudo apt-get install -y \
    "nvidia-container-toolkit=$TOOLKIT_VERSION" \
    "nvidia-container-toolkit-base=$TOOLKIT_VERSION" \
    "libnvidia-container-tools=$TOOLKIT_VERSION" \
    "libnvidia-container1=$TOOLKIT_VERSION"
else
  sudo apt-get install -y \
    nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1
fi

if [ -f "$DAEMON_JSON" ]; then
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  sudo cp "$DAEMON_JSON" "$DAEMON_JSON.bak.$stamp"
  echo "backed up $DAEMON_JSON to $DAEMON_JSON.bak.$stamp"
fi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker info --format '{{json .Runtimes}}'
docker run --rm --gpus all "$CUDA_SAMPLE_IMAGE" nvidia-smi
