#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run this script as root" >&2
  exit 2
fi

source /etc/os-release
if [[ ${ID} != ubuntu ]]; then
  echo "unsupported operating system: ${ID}" >&2
  exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  gnupg \
  jq \
  pciutils \
  python3 \
  rsync

if ! lspci -nn | grep -qi '10de:'; then
  echo "no NVIDIA PCI device is attached" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  install -d -m 0755 /opt/google/cuda-installer
  curl -fSsL \
    https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
    -o /opt/google/cuda-installer/cuda_installer.pyz
  chmod 0755 /opt/google/cuda-installer/cuda_installer.pyz
  sha256sum /opt/google/cuda-installer/cuda_installer.pyz
  python3 /opt/google/cuda-installer/cuda_installer.pyz list_driver_versions
  python3 /opt/google/cuda-installer/cuda_installer.pyz install_driver
fi

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat >/etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

toolkit_version=${NVIDIA_CONTAINER_TOOLKIT_VERSION:-1.19.1-1}
apt-get install -y \
  "nvidia-container-toolkit=${toolkit_version}" \
  "nvidia-container-toolkit-base=${toolkit_version}" \
  "libnvidia-container-tools=${toolkit_version}" \
  "libnvidia-container1=${toolkit_version}"

nvidia-ctk runtime configure --runtime=docker
systemctl enable --now docker
systemctl restart docker
usermod -aG docker ubuntu

docker version
docker compose version
nvidia-ctk --version
if nvidia-smi; then
  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
else
  echo "NVIDIA packages installed; reboot is required before the GPU can be validated" >&2
fi
