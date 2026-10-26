#!/usr/bin/env bash
# Download the exact public LCLM checkpoint without relying on Hub metadata APIs.
#
# This is useful on clusters where Hub API requests are rate-limited but the
# immutable ``/resolve/<revision>/<path>`` endpoint remains available.  Files
# are downloaded as Transformers BF16 Safetensors, never GGUF.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

readonly output_dir=$1
readonly repository="latent-context/0.6b-4b-LCLM-8x"
readonly revision="7a124e8bd607195b58e15c18231b99a2e546f69a"
# Default to the domestic Hub mirror while still allowing an explicitly supplied
# endpoint for sites that provide their own mirror.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
readonly base_url="${HF_ENDPOINT%/}/${repository}/resolve/${revision}"
readonly complete_file="${output_dir}/.downloaded_paths"

files=(
  .gitattributes README.md model_config.json processor_config.json
  adapter/adapter.safetensors
  encoder/added_tokens.json encoder/chat_template.jinja encoder/config.json
  encoder/model.safetensors encoder/special_tokens_map.json encoder/tokenizer.json
  encoder/tokenizer_config.json encoder/vocab.json encoder/merges.txt
  decoder/added_tokens.json decoder/chat_template.jinja decoder/config.json
  decoder/generation_config.json decoder/model-00001-of-00004.safetensors
  decoder/model-00002-of-00004.safetensors decoder/model-00003-of-00004.safetensors
  decoder/model-00004-of-00004.safetensors decoder/model.safetensors.index.json
  decoder/special_tokens_map.json decoder/tokenizer.json decoder/tokenizer_config.json
  decoder/vocab.json decoder/merges.txt
)

mkdir -p "${output_dir}"
touch "${complete_file}"

download_one() {
  local relative_path=$1
  local target="${output_dir}/${relative_path}"
  if grep -Fqx "${relative_path}" "${complete_file}"; then
    return
  fi
  mkdir -p "$(dirname "${target}")"
  echo "downloading ${relative_path}" >&2
  # -C - preserves partial large Safetensors across transient failures.  curl
  # returns 416 when the local object is already complete but the completion
  # marker was not written (for example after a shell interruption); that is
  # safe to mark complete because all paths are immutable at this revision.
  local http_code
  set +e
  http_code=$(curl --http1.1 --fail --location --continue-at - --retry 20 --retry-all-errors \
    --retry-delay 10 --connect-timeout 30 --write-out '%{http_code}' \
    "${base_url}/${relative_path}" --output "${target}")
  local status=$?
  set -e
  if [[ ${status} -ne 0 && !( ${status} -eq 22 && ${http_code} == 416 ) ]]; then
    return "${status}"
  fi
  printf '%s\n' "${relative_path}" >> "${complete_file}"
}

for file in "${files[@]}"; do
  download_one "${file}"
done
touch "${output_dir}/.download_complete"
echo "checkpoint ready: ${output_dir}" >&2
