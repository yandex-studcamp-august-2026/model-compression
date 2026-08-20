#!/usr/bin/env bash
# The generated remote path is intentionally expanded by the local client.
# shellcheck disable=SC2029,SC2329
set -euo pipefail

: "${YC_GPU_INSTANCE_ID:?YC_GPU_INSTANCE_ID is required}"
: "${GPU_SSH_USER:?GPU_SSH_USER is required}"
: "${GPU_SSH_KEY_FILE:?GPU_SSH_KEY_FILE is required}"
: "${GPU_SSH_HOST_KEY:?GPU_SSH_HOST_KEY is required}"
: "${EXPECTED_GPU_MODEL:?EXPECTED_GPU_MODEL is required}"
: "${EXPECTED_TENSORRT_VERSION:?EXPECTED_TENSORRT_VERSION is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"

if [[ ! "${EXPECTED_TENSORRT_VERSION}" =~ ^[0-9]+(\.[0-9]+){2,3}$ ]]; then
  echo "EXPECTED_TENSORRT_VERSION must be an exact numeric version" >&2
  exit 2
fi
if [[ ! "${EXPECTED_GPU_MODEL}" =~ ^[A-Za-z0-9._-]{1,64}$ ]]; then
  echo "EXPECTED_GPU_MODEL has an invalid format" >&2
  exit 2
fi
if [[ ! "${YC_GPU_INSTANCE_ID}" =~ ^[a-z0-9]{10,64}$ ]]; then
  echo "YC_GPU_INSTANCE_ID has an invalid format" >&2
  exit 2
fi
if [[ ! "${GPU_SSH_USER}" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
  echo "GPU_SSH_USER has an invalid format" >&2
  exit 2
fi
if [[ ! "${GPU_SSH_HOST_KEY}" =~ ^ssh-ed25519\ [A-Za-z0-9+/]+={0,3}$ ]]; then
  echo "GPU_SSH_HOST_KEY must contain an SSH Ed25519 public key without a comment" >&2
  exit 2
fi
if [[ ! -f "${GPU_SSH_KEY_FILE}" || -L "${GPU_SSH_KEY_FILE}" ]]; then
  echo "GPU_SSH_KEY_FILE must be a regular file" >&2
  exit 2
fi

readonly bundles_dir="${1:?usage: run_remote_gpu.sh BUNDLES_DIR RESULTS_DIR RUNTIME_DIR}"
readonly results_dir="${2:?usage: run_remote_gpu.sh BUNDLES_DIR RESULTS_DIR RUNTIME_DIR}"
readonly runtime_dir="${3:?usage: run_remote_gpu.sh BUNDLES_DIR RESULTS_DIR RUNTIME_DIR}"
readonly known_hosts="${RUNNER_TEMP}/known_hosts"
readonly remote_dir="/tmp/model-bench-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"

bundle_count=0
while IFS= read -r -d '' bundle; do
  if [[ ! -f "${bundle}/bundle.json" || -L "${bundle}/bundle.json" ]]; then
    echo "Invalid GPU bundle directory: ${bundle}" >&2
    exit 2
  fi
  ((bundle_count += 1))
done < <(find "${bundles_dir}" -mindepth 1 -maxdepth 1 -type d -print0)
if [[ "${bundle_count}" == "0" ]]; then
  echo "No GPU bundle directories found under ${bundles_dir}" >&2
  exit 2
fi

owns_instance=0

instance_json() {
  yc compute instance get "${YC_GPU_INSTANCE_ID}" --format json
}

instance_status() {
  instance_json | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])'
}

stop_instance() {
  if [[ "${owns_instance}" == "1" ]]; then
    stop_requested=0
    for attempt in 1 2 3; do
      status="$(instance_status 2>/dev/null || true)"
      if [[ "${status}" == "STOPPED" ]]; then
        owns_instance=0
        return 0
      fi
      if yc compute instance stop "${YC_GPU_INSTANCE_ID}" --async >/dev/null; then
        stop_requested=1
        break
      fi
      echo "VM stop request ${attempt} failed" >&2
      sleep 5
    done
    if [[ "${stop_requested}" == "1" ]]; then
      for _ in {1..60}; do
        status="$(instance_status 2>/dev/null || true)"
        if [[ "${status}" == "STOPPED" ]]; then
          owns_instance=0
          return 0
        fi
        sleep 5
      done
    fi
    echo "GPU VM may still be running; manual intervention is required" >&2
    return 1
  fi
}

terminate() {
  stop_instance
  exit 143
}

trap stop_instance EXIT
trap terminate INT TERM

initial_status="$(instance_status)"
if [[ "${initial_status}" == "STOPPING" ]]; then
  for _ in {1..60}; do
    [[ "$(instance_status)" == "STOPPED" ]] && break
    sleep 10
  done
fi
if [[ "$(instance_status)" != "STOPPED" ]]; then
  echo "GPU VM must be STOPPED before the job starts" >&2
  exit 1
fi

yc compute instance start "${YC_GPU_INSTANCE_ID}" --async >/dev/null
owns_instance=1
for _ in {1..90}; do
  [[ "$(instance_status)" == "RUNNING" ]] && break
  sleep 10
done
if [[ "$(instance_status)" != "RUNNING" ]]; then
  echo "GPU VM did not reach RUNNING state" >&2
  exit 1
fi

instance_ip="$({ instance_json; } | python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data["network_interfaces"][0]["primary_v4_address"]["one_to_one_nat"]["address"])
')"
readonly instance_ip
readonly ssh_target="${GPU_SSH_USER}@${instance_ip}"
printf '%s %s\n' "${instance_ip}" "${GPU_SSH_HOST_KEY}" >"${known_hosts}"
ssh_options=(
  -i "${GPU_SSH_KEY_FILE}"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
  -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=${known_hosts}"
)

for _ in {1..60}; do
  if ssh "${ssh_options[@]}" "${ssh_target}" true 2>/dev/null; then
    break
  fi
  sleep 5
done
if ! ssh "${ssh_options[@]}" "${ssh_target}" true; then
  echo "SSH did not become available" >&2
  exit 1
fi

ssh "${ssh_options[@]}" "${ssh_target}" \
  "systemctl is-enabled --quiet benchmark-auto-shutdown.service; \
sudo shutdown -h +180 >/dev/null; \
find /tmp -mindepth 1 -maxdepth 1 -type d -name 'model-bench-*' -exec rm -rf -- {} +; \
mkdir -p '${remote_dir}/bundles' '${remote_dir}/results'"
tar -C "${runtime_dir}" -cf - src | \
  ssh "${ssh_options[@]}" "${ssh_target}" "tar -C '${remote_dir}' -xf -"
tar -C "${bundles_dir}" -cf - . | \
  ssh "${ssh_options[@]}" "${ssh_target}" "tar -C '${remote_dir}/bundles' -xf -"

remote_status=0
ssh "${ssh_options[@]}" "${ssh_target}" \
  "REMOTE_DIR='${remote_dir}' EXPECTED_GPU_MODEL='${EXPECTED_GPU_MODEL}' EXPECTED_TENSORRT_VERSION='${EXPECTED_TENSORRT_VERSION}' bash -s" <<'REMOTE' || remote_status=$?
set -euo pipefail
nvidia-smi
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#gpu_names[@]}" != "1" || "${gpu_names[0]}" != *"${EXPECTED_GPU_MODEL}"* ]]; then
  echo "Expected exactly one GPU matching '${EXPECTED_GPU_MODEL}', got: ${gpu_names[*]:-none}" >&2
  exit 1
fi
trt_version="$(trtexec --help 2>&1)"
printf '%s\n' "${trt_version}"
export TRT_VERSION_OUTPUT="${trt_version}"
trt_numeric="$(PYTHONPATH="${REMOTE_DIR}/src" python3 -c '
import os
from model_bench.tensorrt import parse_trtexec_version
print(parse_trtexec_version(os.environ["TRT_VERSION_OUTPUT"]))
')"
unset TRT_VERSION_OUTPUT
if [[ "${trt_numeric}" != "${EXPECTED_TENSORRT_VERSION}" ]]; then
  echo "Expected TensorRT ${EXPECTED_TENSORRT_VERSION}, got ${trt_numeric}" >&2
  exit 1
fi
python3 --version
python3 -c 'import numpy; print("NumPy", numpy.__version__)'
status=0
while IFS= read -r -d '' bundle; do
  if ! PYTHONPATH="${REMOTE_DIR}/src" python3 -m model_bench benchmark-gpu \
    --bundle "${bundle}" \
    --results "${REMOTE_DIR}/results" \
    --warmup-ms 5000 \
    --iterations 1000 \
    --throughput-streams 8; then
    status=1
  fi
done < <(find "${REMOTE_DIR}/bundles" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
tar -C "${REMOTE_DIR}/results" -cf "${REMOTE_DIR}/results.tar" .
exit "${status}"
REMOTE

mkdir -p "${results_dir}"
transfer_status=0
ssh "${ssh_options[@]}" "${ssh_target}" \
  "cat '${remote_dir}/results.tar'" | tar -C "${results_dir}" -xf - || transfer_status=$?
ssh "${ssh_options[@]}" "${ssh_target}" "rm -rf '${remote_dir}'" || true
if [[ "${remote_status}" != "0" ]]; then
  exit "${remote_status}"
fi
exit "${transfer_status}"
