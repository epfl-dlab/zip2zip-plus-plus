#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# Build one frontend demo-data bundle directly from one zip2zip-core checkpoint.
# No HF model export, aggregate step, or plotting step is involved.
#
# Required:
#   CKPT_DIR=/path/to/checkpoint/step_N
#
# Optional:
#   QUESTION_FILE    default: /dlabscratch1/xinma/demo_question.jsonl
#   OUT_DIR          default: outputs/<checkpoint-name>_<step>_demo_data
#   GPU              default: 0
#   DEVICE           default: auto
#   SEED             default: 42
#   MAX_NEW_TOKENS   default: 512
#   TEMPERATURE      default: 1.0 (0 = greedy)
#   INSTRUCT         default: 1
#   LIMIT            smoke-test limit; unset = all questions
#   OVERWRITE        default: 0; otherwise resume an interrupted run
#   TOKENIZER        optional strict tokenizer override
#   EXTRA_ARGS       additional build_demo.py arguments

if [[ -z "${CKPT_DIR:-}" ]]; then
  echo "Set CKPT_DIR=/path/to/checkpoint/step_N" >&2
  exit 1
fi

QUESTION_FILE="${QUESTION_FILE:-/dlabscratch1/xinma/demo_question.jsonl}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
INSTRUCT="${INSTRUCT:-1}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  --ckpt-dir "${CKPT_DIR}"
  --question-file "${QUESTION_FILE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
)

if [[ -n "${OUT_DIR:-}" ]]; then
  ARGS+=(--out-dir "${OUT_DIR}")
fi
if [[ -n "${TOKENIZER:-}" ]]; then
  ARGS+=(--tokenizer "${TOKENIZER}")
fi
if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi
if [[ "${INSTRUCT}" == "0" ]]; then
  ARGS+=(--no-instruct)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi

# The core online-codebook runtime currently generates one independent prompt at
# a time.  Loading the model once and resuming from flushed traces provides the
# safe speedup without sharing mutable codebook/cache state across prompts.
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/build_demo.py "${ARGS[@]}" ${EXTRA_ARGS}
