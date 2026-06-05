#!/usr/bin/env bash
set -euo pipefail

# Parallel 6-model demo runner for eval_compare.py + export-demo.
# Example:
#   cd /home/xinxian/Semester_Project/zip2zip-core
#   QUESTION_FILE=/home/xinxian/Semester_Project/question.jsonl \
#   GPUS="0 1 2" \
#   bash scripts/zip2zip_hf/run_demo_all_5_multistop.sh
#
# Useful env vars:
#   RUN_DIR=outputs/demo_all_5_multistop
#   DEMO_OUT_DIR=demo-data
#   REPOS="epfl-dlab/model-a epfl-dlab/model-b"  (space-separated; falls back to built-in list)
#   LIMIT=5
#   MAX_NEW_TOKENS=256
#   BATCH_SIZE=1
#   DO_SAMPLE=1
#   OVERWRITE=1
#   SKIP_GPT2=0
#   GPT2_DEVICE=cpu
#   GPUS="0 1 2"
#   EXTRA_ARGS="--device cuda --torch-dtype bfloat16"

RUN_DIR="${RUN_DIR:-outputs/demo_all_5_multistop}"
DEMO_OUT_DIR="${DEMO_OUT_DIR:-demo-data}"
REVISION="${REVISION:-hf}"
LIMIT="${LIMIT:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START="${START:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DO_SAMPLE="${DO_SAMPLE:-1}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_GPT2="${SKIP_GPT2:-0}"
GPT2_DEVICE="${GPT2_DEVICE:-cpu}"
GPUS="${GPUS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -n "${REPOS:-}" ]]; then
  read -r -a REPOS <<< "${REPOS}"
else
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
fi

if [[ -z "${QUESTION_FILE:-}" ]]; then
  echo "Set QUESTION_FILE=/path/to/question.jsonl" >&2
  exit 1
fi

read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS is empty; set GPUS='0' or GPUS='0 1 2'." >&2
  exit 1
fi

COMMON_ARGS=(
  --revision "${REVISION}"
  --question-file "${QUESTION_FILE}"
  --out-dir "${RUN_DIR}"
  --limit "${LIMIT}"
  --batch-size "${BATCH_SIZE}"
  --start "${START}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --gpt2-device "${GPT2_DEVICE}"
)

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

python scripts/zip2zip_hf/eval_compare.py export-demo --run-dir "${RUN_DIR}" --out-dir "${DEMO_OUT_DIR}"

echo "[done] run_dir=${RUN_DIR}"
echo "[done] demo_data=${RUN_DIR}/${DEMO_OUT_DIR}"
