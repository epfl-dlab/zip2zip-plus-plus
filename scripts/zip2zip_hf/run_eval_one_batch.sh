#!/usr/bin/env bash
set -euo pipefail

# Generic one-model batched eval launcher.
# Example:
#   REPO=epfl-dlab/candidate-Llaza-MS2-flat-20BT \
#   QUESTION_FILE=/home/xinxian/Semester_Project/question.jsonl \
#   GPU=0 BATCH_SIZE=4 LIMIT=20 \
#   bash scripts/zip2zip_hf/run_eval_one_batch.sh

if [[ -z "${QUESTION_FILE:-}" ]]; then
  echo "Set QUESTION_FILE=/path/to/question.jsonl" >&2
  exit 1
fi

REPO="${REPO:-epfl-dlab/candidate-Llaza-MS2-flat-20BT}"
RUN_DIR="${RUN_DIR:-outputs/eval_compare_one_batch}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT="${LIMIT:-20}"
START="${START:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BASE_MODEL="${BASE_MODEL:-}"
REVISION="${REVISION:-hf}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_GPT2="${SKIP_GPT2:-0}"
DO_SAMPLE="${DO_SAMPLE:-1}"
INSTRUCT="${INSTRUCT:-0}"
GPT2_DEVICE="${GPT2_DEVICE:-cpu}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  generate
  --repo "${REPO}"
  --revision "${REVISION}"
  --question-file "${QUESTION_FILE}"
  --out-dir "${RUN_DIR}"
  --batch-size "${BATCH_SIZE}"
  --limit "${LIMIT}"
  --start "${START}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --gpt2-device "${GPT2_DEVICE}"
)

if [[ -n "${BASE_MODEL}" ]]; then
  ARGS+=(--base-model "${BASE_MODEL}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SKIP_GPT2}" == "1" ]]; then
  ARGS+=(--skip-gpt2)
fi
if [[ "${DO_SAMPLE}" == "1" ]]; then
  ARGS+=(--do-sample)
fi
if [[ "${INSTRUCT}" == "1" ]]; then
  ARGS+=(--instruct)
fi

echo "[run] repo=${REPO} gpu=${GPU} batch_size=${BATCH_SIZE} limit=${LIMIT} question_file=${QUESTION_FILE}"
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/zip2zip_hf/eval_compare.py "${ARGS[@]}" ${EXTRA_ARGS}

python scripts/zip2zip_hf/eval_compare.py aggregate --run-dir "${RUN_DIR}"
