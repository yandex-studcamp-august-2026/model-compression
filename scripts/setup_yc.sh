#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${GITHUB_PATH:?GITHUB_PATH is required}"
: "${YC_SA_JSON_CREDENTIALS:?YC_SA_JSON_CREDENTIALS is required}"

readonly yc_version="1.18.0"
readonly yc_sha256="6cc5dce883476fd1c6cf53d130231cac9d7021b22075969e981fb996e1d40bf5"
readonly install_dir="${RUNNER_TEMP}/yandex-cloud"
readonly yc_binary="${RUNNER_TEMP}/yc-${yc_version}"
readonly key_file="${RUNNER_TEMP}/yc-key.json"

curl --fail --silent --show-error --location --retry 3 \
  --output "${yc_binary}" \
  "https://storage.yandexcloud.net/yandexcloud-yc/release/${yc_version}/linux/amd64/yc"
actual_sha256="$(sha256sum "${yc_binary}" | cut -d ' ' -f 1)"
if [[ "${actual_sha256}" != "${yc_sha256}" ]]; then
  echo "Yandex Cloud CLI checksum mismatch" >&2
  exit 1
fi
mkdir -p "${install_dir}/bin"
install -m 755 "${yc_binary}" "${install_dir}/bin/yc"

install -m 600 /dev/null "${key_file}"
printf '%s' "${YC_SA_JSON_CREDENTIALS}" >"${key_file}"
"${install_dir}/bin/yc" config profile create benchmark-ci
"${install_dir}/bin/yc" config set service-account-key "${key_file}"
"${install_dir}/bin/yc" version
printf '%s\n' "${install_dir}/bin" >>"${GITHUB_PATH}"
