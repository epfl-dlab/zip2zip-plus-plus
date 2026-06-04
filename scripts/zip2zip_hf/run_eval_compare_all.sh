#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU launcher for eval_compare.py.
# Example:
#   GPUS="0 1 2 3" QUESTION_FILE=/home/xinxian/Semester_Project/question.jsonl \
#     bash scripts/zip2zip_hf/run_eval_compare_all.sh
#
# Useful env vars:
#   RUN_DIR=outputs/eval_compare
#   BASE_MODEL=meta-llama/Llama-3.2-1B   # optional override
#   REVISION=hf
#   MAX_NEW_TOKENS=512
#   BATCH_SIZE=1
#   LIMIT=20
#   START=0
#   DO_SAMPLE=1
#   OVERWRITE=0
#   SKIP_GPT2=0
#   GPT2_DEVICE=cpu
#   EXTRA_ARGS="--torch-dtype bfloat16"
#   EXPORT_DEMO=1
#   DEMO_OUT_DIR=demo-data

RUN_DIR="${RUN_DIR:-outputs/eval_compare}"
BASE_MODEL="${BASE_MODEL:-}"
REVISION="${REVISION:-hf}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START="${START:-0}"
SEED="${SEED:-42}"
DO_SAMPLE="${DO_SAMPLE:-1}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_GPT2="${SKIP_GPT2:-0}"
GPT2_DEVICE="${GPT2_DEVICE:-cpu}"
GPUS="${GPUS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXPORT_DEMO="${EXPORT_DEMO:-1}"
DEMO_OUT_DIR="${DEMO_OUT_DIR:-demo-data}"

REPOS=(
  # "epfl-dlab/candidate-Llaza-MS2-FT-1BT-v1"
  # "epfl-dlab/candidate-Llaza-MS3-FT-1BT-v1"
  # "epfl-dlab/candidate-Llaza-MS4-FT-1BT-v1"
  "epfl-dlab/candidate-Llaza-MS2-FT-1BT-base-v1"
  "epfl-dlab/candidate-Llaza-MS3-FT-1BT-base-v1"
  "epfl-dlab/candidate-Llaza-MS4-FT-1BT-base-v1"
  "epfl-dlab/candidate-Llaza-MS2-flat-20BT-v1"
  "epfl-dlab/candidate-Llaza-MS3-flat-20BT-v1"
  "epfl-dlab/candidate-Llaza-MS4-flat-20BT-v1"
)

read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS is empty; set GPUS='0' or GPUS='0 1 2 3'." >&2
  exit 1
fi

COMMON_ARGS=(
  --revision "${REVISION}"
  --out-dir "${RUN_DIR}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --batch-size "${BATCH_SIZE}"
  --start "${START}"
  --seed "${SEED}"
  --gpt2-device "${GPT2_DEVICE}"
)

if [[ -z "${QUESTION_FILE:-}" ]]; then
  echo "Set QUESTION_FILE=/path/to/question.jsonl" >&2
  exit 1
fi
if [[ -n "${BASE_MODEL}" ]]; then
  COMMON_ARGS+=(--base-model "${BASE_MODEL}")
fi
COMMON_ARGS+=(--question-file "${QUESTION_FILE}")
if [[ -n "${LIMIT:-}" ]]; then
  COMMON_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${DO_SAMPLE}" == "1" ]]; then
  COMMON_ARGS+=(--do-sample)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi
if [[ "${SKIP_GPT2}" == "1" ]]; then
  COMMON_ARGS+=(--skip-gpt2)
fi

repo_index=0
while [[ "${repo_index}" -lt "${#REPOS[@]}" ]]; do
  pids=()
  for gpu in "${GPU_LIST[@]}"; do
    if [[ "${repo_index}" -ge "${#REPOS[@]}" ]]; then
      break
    fi
    repo="${REPOS[${repo_index}]}"
    echo "[launch] gpu=${gpu} repo=${repo}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" python scripts/zip2zip_hf/eval_compare.py generate \
      --repo "${repo}" \
      "${COMMON_ARGS[@]}" \
      ${EXTRA_ARGS} &
    pids+=("$!")
    repo_index=$((repo_index + 1))
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

python scripts/zip2zip_hf/eval_compare.py aggregate --run-dir "${RUN_DIR}"

if [[ "${EXPORT_DEMO}" == "1" ]]; then
  python scripts/zip2zip_hf/eval_compare.py export-demo --run-dir "${RUN_DIR}" --out-dir "${DEMO_OUT_DIR}"
fi

echo "[done] run_dir=${RUN_DIR}"
if [[ "${EXPORT_DEMO}" == "1" ]]; then
  echo "[done] demo_data=${RUN_DIR}/${DEMO_OUT_DIR}"
fi
