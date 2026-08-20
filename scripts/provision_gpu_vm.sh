#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" != "0" ]]; then
  echo "Run as root: sudo bash scripts/provision_gpu_vm.sh PUBLIC_KEY_FILE" >&2
  exit 2
fi

readonly public_key_file="${1:?usage: provision_gpu_vm.sh PUBLIC_KEY_FILE}"
readonly benchmark_user="benchmark"
readonly tensorrt_package_version="10.4.0.26-1+cuda11.8"
readonly nvidia_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
readonly -a tensorrt_runtime_packages=(
  libnvinfer10
  libnvinfer-bin
  libnvinfer-dispatch10
  libnvinfer-lean10
  libnvinfer-plugin10
  libnvinfer-vc-plugin10
  libnvonnxparsers10
)
if [[ ! -f "${public_key_file}" || -L "${public_key_file}" ]]; then
  echo "PUBLIC_KEY_FILE must be a regular file" >&2
  exit 2
fi
public_key="$(<"${public_key_file}")"
if [[ ! "${public_key}" =~ ^ssh-ed25519\ [A-Za-z0-9+/]+={0,3}(\ .*)?$ ]]; then
  echo "PUBLIC_KEY_FILE must contain one SSH Ed25519 public key" >&2
  exit 2
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl python3-numpy

if ! apt-cache madison libnvinfer-bin | awk '{print $3}' | \
  grep -Fxq "${tensorrt_package_version}"; then
  keyring_deb="$(mktemp --suffix=.deb)"
  trap 'rm -f -- "${keyring_deb:-}"' EXIT
  curl --fail --location --silent --show-error \
    --output "${keyring_deb}" \
    "${nvidia_repo_url}/cuda-keyring_1.1-1_all.deb"
  dpkg -i "${keyring_deb}"
  rm -f -- "${keyring_deb}"
  trap - EXIT
  apt-get update
fi

if ! apt-cache madison libnvinfer-bin | awk '{print $3}' | \
  grep -Fxq "${tensorrt_package_version}"; then
  echo "TensorRT runtime ${tensorrt_package_version} is unavailable" >&2
  exit 1
fi

tensorrt_install_args=()
for package in "${tensorrt_runtime_packages[@]}"; do
  tensorrt_install_args+=("${package}=${tensorrt_package_version}")
done
DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-downgrades \
  "${tensorrt_install_args[@]}"

trtexec_source="$(
  dpkg-query -L libnvinfer-bin | awk '/\/trtexec$/ {print; exit}'
)"
if [[ -z "${trtexec_source}" || ! -x "${trtexec_source}" ]]; then
  echo "libnvinfer-bin did not provide an executable trtexec" >&2
  exit 1
fi
ln -sfn -- "${trtexec_source}" /usr/local/bin/trtexec

if ! id "${benchmark_user}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${benchmark_user}"
fi
install -d -m 700 -o "${benchmark_user}" -g "${benchmark_user}" \
  "/home/${benchmark_user}/.ssh"
printf '%s\n' "${public_key}" >"/home/${benchmark_user}/.ssh/authorized_keys"
chown "${benchmark_user}:${benchmark_user}" \
  "/home/${benchmark_user}/.ssh/authorized_keys"
chmod 600 "/home/${benchmark_user}/.ssh/authorized_keys"

printf '%s\n' \
  "${benchmark_user} ALL=(root) NOPASSWD: /usr/sbin/shutdown" \
  >"/etc/sudoers.d/${benchmark_user}-shutdown"
chmod 440 "/etc/sudoers.d/${benchmark_user}-shutdown"
visudo -cf "/etc/sudoers.d/${benchmark_user}-shutdown"

install -m 644 /dev/null /etc/systemd/system/benchmark-auto-shutdown.service
printf '%s\n' \
  '[Unit]' \
  'Description=Stop benchmark VM after three hours' \
  'After=multi-user.target' \
  '' \
  '[Service]' \
  'Type=oneshot' \
  'ExecStart=/usr/sbin/shutdown -h +180' \
  'RemainAfterExit=yes' \
  '' \
  '[Install]' \
  'WantedBy=multi-user.target' \
  >/etc/systemd/system/benchmark-auto-shutdown.service
systemctl daemon-reload
systemctl enable --now benchmark-auto-shutdown.service

nvidia-smi
trtexec --help >/dev/null
python3 -c 'import numpy; print("NumPy", numpy.__version__)'
systemctl is-enabled benchmark-auto-shutdown.service
echo "GPU benchmark VM provisioning completed"
