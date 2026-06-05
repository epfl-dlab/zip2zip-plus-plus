#!/usr/bin/env bash
set -euo pipefail

# Unified eval launcher: generate → [aggregate] → [export-demo]
#
# Single-model:
#   REPO=epfl-dlab/my-model \
#   QUESTION_FILE=/path/to/question.jsonl \
#   bash scripts/zip2zip_hf/run_eval.sh
#
# Multi-model, multi-GPU:
#   REPOS="epfl-dlab/model-a epfl-dlab/model-b" \
#   QUESTION_FILE=/path/to/question.jsonl \
#   GPUS="0 1" \
#   bash scripts/zip2zip_hf/run_eval.sh
#
# Env vars:
#   REPO / REPOS        single repo or space-separated list (REPOS takes precedence)
#   QUESTION_FILE       required
#   RUN_DIR             output dir (default: outputs/eval_compare)
#   GPU / GPUS          GPU index or space-separated list (default: 0)
#   BATCH_SIZE          (default: 1)
#   LIMIT               max questions; unset = all
#   START               (default: 0)
#   SEED                (default: 42)
#   MAX_NEW_TOKENS      (default: 512)
#   REVISION            (default: hf)
#   BASE_MODEL          optional base model override
#   DO_SAMPLE           0/1 (default: 1)
#   OVERWRITE           0/1 (default: 0)
#   SKIP_GPT2           0/1 (default: 0)
#   INSTRUCT            0/1 (default: 0) — apply chat template for instruct models
#   GPT2_DEVICE         (default: cpu)
#   AGGREGATE           0/1 (default: 1) — run aggregate after generation
#   EXPORT_DEMO         0/1 (default: 0) — run export-demo after generation
#   DEMO_OUT_DIR        demo output dir (default: demo-data, relative to RUN_DIR)
#   EXTRA_ARGS          extra args passed verbatim to eval_compare.py generate

# ── resolve REPOS ──────────────────────────────────────────────────────────────
if [[ -n "${REPOS:-}" ]]; then
  read -r -a REPO_LIST <<< "${REPOS}"
elif [[ -n "${REPO:-}" ]]; then
  REPO_LIST=("${REPO}")
else
  echo "Set REPO=<repo> or REPOS='<repo1> <repo2> ...'" >&2
  exit 1
fi

# ── resolve GPU list ───────────────────────────────────────────────────────────
if [[ -n "${GPUS:-}" ]]; then
  read -r -a GPU_LIST <<< "${GPUS}"
elif [[ -n "${GPU:-}" ]]; then
  GPU_LIST=("${GPU}")
else
  GPU_LIST=("0")
fi

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty; set GPU=0 or GPUS='0 1 2'." >&2
  exit 1
fi

# ── required ───────────────────────────────────────────────────────────────────
if [[ -z "${QUESTION_FILE:-}" ]]; then
  echo "Set QUESTION_FILE=/path/to/question.jsonl" >&2
  exit 1
fi

# ── optional settings ──────────────────────────────────────────────────────────
RUN_DIR="${RUN_DIR:-outputs/eval_compare}"
BATCH_SIZE="${BATCH_SIZE:-1}"
START="${START:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
REVISION="${REVISION:-hf}"
BASE_MODEL="${BASE_MODEL:-}"
DO_SAMPLE="${DO_SAMPLE:-1}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_GPT2="${SKIP_GPT2:-0}"
INSTRUCT="${INSTRUCT:-0}"
GPT2_DEVICE="${GPT2_DEVICE:-cpu}"
AGGREGATE="${AGGREGATE:-1}"
EXPORT_DEMO="${EXPORT_DEMO:-0}"
DEMO_OUT_DIR="${DEMO_OUT_DIR:-demo-data}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ── common args ────────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --revision "${REVISION}"
  --question-file "${QUESTION_FILE}"
  --out-dir "${RUN_DIR}"
  --batch-size "${BATCH_SIZE}"
  --start "${START}"
  --seed "${SEED}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --gpt2-device "${GPT2_DEVICE}"
)

if [[ -n "${BASE_MODEL}" ]]; then
  COMMON_ARGS+=(--base-model "${BASE_MODEL}")
fi
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
if [[ "${INSTRUCT}" == "1" ]]; then
  COMMON_ARGS+=(--instruct)
fi

# ── parallel generate ──────────────────────────────────────────────────────────
repo_index=0
while [[ "${repo_index}" -lt "${#REPO_LIST[@]}" ]]; do
  pids=()
  for gpu in "${GPU_LIST[@]}"; do
    if [[ "${repo_index}" -ge "${#REPO_LIST[@]}" ]]; then
      break
    fi
    repo="${REPO_LIST[${repo_index}]}"
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

# ── post-processing ────────────────────────────────────────────────────────────
if [[ "${AGGREGATE}" == "1" ]]; then
  python scripts/zip2zip_hf/eval_compare.py aggregate --run-dir "${RUN_DIR}"
fi

if [[ "${EXPORT_DEMO}" == "1" ]]; then
  python scripts/zip2zip_hf/eval_compare.py export-demo \
    --run-dir "${RUN_DIR}" \
    --out-dir "${DEMO_OUT_DIR}"
fi

echo "[done] run_dir=${RUN_DIR}"
if [[ "${EXPORT_DEMO}" == "1" ]]; then
  echo "[done] demo_data=${RUN_DIR}/${DEMO_OUT_DIR}"
fi
