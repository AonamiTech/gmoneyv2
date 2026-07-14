#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run this script with sudo" >&2
  exit 2
fi

device=${1:?"usage: prepare_demo_data_disk.sh /dev/disk/azure/scsi1/lunN [mountpoint]"}
mountpoint=${2:-/mnt/gmoney-data}
device=$(readlink -f "${device}")

if [[ ! -b ${device} ]]; then
  echo "not a block device: ${device}" >&2
  exit 2
fi
if [[ $(lsblk -dnro TYPE "${device}") != disk ]]; then
  echo "refusing a non-disk device: ${device}" >&2
  exit 2
fi
size=$(lsblk -bdnro SIZE "${device}")
if (( size < 900000000000 )); then
  echo "refusing a disk smaller than 900 GB: ${device} (${size} bytes)" >&2
  exit 2
fi
if [[ -n $(lsblk -dnro FSTYPE "${device}") ]]; then
  echo "refusing a disk that already has a filesystem: ${device}" >&2
  exit 2
fi
if [[ $(lsblk -nrpo NAME "${device}" | wc -l) -ne 1 ]]; then
  echo "refusing a disk with existing partitions: ${device}" >&2
  exit 2
fi
if findmnt --source "${device}" >/dev/null 2>&1; then
  echo "refusing an already mounted disk: ${device}" >&2
  exit 2
fi

mkfs.ext4 -m 0 -L gmoney-demo-data "${device}"
uuid=$(blkid -s UUID -o value "${device}")
mkdir -p "${mountpoint}"
if ! grep -q "^UUID=${uuid}[[:space:]]" /etc/fstab; then
  printf 'UUID=%s %s ext4 defaults,nofail 0 2\n' "${uuid}" "${mountpoint}" >>/etc/fstab
fi
mount "${mountpoint}"

owner_uid=${SUDO_UID:-1000}
owner_gid=${SUDO_GID:-1000}
install -d -o "${owner_uid}" -g "${owner_gid}" -m 0755 \
  "${mountpoint}/gmoneyv2" \
  "${mountpoint}/gmoneyv2/archives" \
  "${mountpoint}/gmoneyv2/results" \
  "${mountpoint}/gmoneyv2/sources"
install -d -o 10001 -g 10001 -m 0755 "${mountpoint}/gmoneyv2/runtime"
install -d -o 10001 -g 10001 -m 0700 "${mountpoint}/gmoneyv2/runtime/jobs"

df -h "${mountpoint}"
echo "GMONEY_DATA_ROOT=${mountpoint}/gmoneyv2/runtime"
