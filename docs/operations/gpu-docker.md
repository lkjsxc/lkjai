# GPU Docker Setup

Owner: `docs/operations/gpu-docker.md`.
State: host setup runbook for WSL2 NVIDIA Docker access.

## Scope

This host setup is outside the product images. It installs NVIDIA Container
Toolkit into the WSL Ubuntu host so Docker Compose GPU reservations can reach
the Windows-provided NVIDIA driver.

The official WSL constraint remains important: install the NVIDIA Windows
driver on the Windows side and do not install a Linux display driver inside
WSL. The WSL CUDA guide notes that WSL receives the CUDA driver through the
Windows host. The Container Toolkit guide is the source for the apt repository,
`nvidia-ctk runtime configure --runtime=docker`, and Docker restart flow.

## Helper

Inspect the planned host mutations first:

```bash
ops/host/install-nvidia-container-toolkit.sh --dry-run
```

Apply them:

```bash
ops/host/install-nvidia-container-toolkit.sh --apply
```

The helper:

- installs `ca-certificates`, `curl`, and `gnupg2`;
- writes the NVIDIA `libnvidia-container` keyring and stable apt list;
- installs `nvidia-container-toolkit`, `nvidia-container-toolkit-base`,
  `libnvidia-container-tools`, and `libnvidia-container1`;
- backs up `/etc/docker/daemon.json` before `nvidia-ctk` changes it;
- runs `sudo nvidia-ctk runtime configure --runtime=docker`;
- restarts Docker with `sudo systemctl restart docker`;
- verifies Docker runtimes and runs a CUDA `nvidia-smi` container.

Set `NVIDIA_CONTAINER_TOOLKIT_VERSION=1.19.0-1` to match the currently
documented NVIDIA example exactly. Leaving it unset installs the latest stable
package set from NVIDIA's configured apt repository.

## Manual Checks

```bash
/usr/lib/wsl/lib/nvidia-smi
docker info --format '{{json .Runtimes}}'
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
docker compose --progress quiet --profile verify run --build --rm verify
docker compose --profile train run --rm train --smoke --steps 2
```

If `systemctl restart docker` fails, fix Docker/systemd first. The repository
Compose profiles assume Docker Engine is already healthy.
