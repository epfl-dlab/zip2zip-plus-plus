"""Static and shell-syntax checks for the v0.7.1 RCP operator path."""

import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
FINETUNE = REPO / "scripts" / "finetune_phi35_rcp.sh"
PIPELINE = REPO / "scripts" / "pipeline_ft_eval_rcp.sh"
DOCS = REPO / "docs" / "finetuning.md"


def test_rcp_launchers_are_valid_bash():
    for launcher in (FINETUNE, PIPELINE):
        subprocess.run(["bash", "-n", str(launcher)], check=True)


def test_finetune_forwards_complete_v071_contract():
    source = FINETUNE.read_text()
    for fragment in (
        "GATED_COMPRESSED_ROPE=${GATED_COMPRESSED_ROPE:-}",
        "GATED_ROPE_START_LAYER=${GATED_ROPE_START_LAYER:-16}",
        "BASE_VIEW_REPLAY_PROB=${BASE_VIEW_REPLAY_PROB:-0}",
        'GATEDROPE_FLAG="--gated_compressed_rope"',
        '--gated_rope_start_layer "$GATED_ROPE_START_LAYER"',
        '--base_view_replay_prob "$BASE_VIEW_REPLAY_PROB"',
        'SEED=${SEED:-42}',
        '--seed "$SEED"',
    ):
        assert fragment in source


def test_pipeline_audits_every_gated_eval_and_docs_both_commands():
    source = PIPELINE.read_text()
    assert source.count('audit_eval_mode "$SMOKE_JSON"') == 1
    assert source.count('audit_eval_mode "$MC_JSON"') == 1
    assert source.count('audit_eval_mode "$PPL_JSON"') == 1
    assert 'args.get("gated_compressed_rope")' in source
    assert 'args.get("gated_rope_start_layer")' in source

    docs = DOCS.read_text()
    assert "smoke-z2z-v071" in docs
    assert "ft-z2z-v071" in docs
    assert docs.count("--environment RECIPE=v0.7.1") >= 2
    assert docs.count("--environment WANDB=0 --environment NUM_WORKERS=0") >= 2


def test_pipeline_gated_audit_accepts_match_and_rejects_mismatch(tmp_path):
    source = PIPELINE.read_text()
    audit_function = source.split("audit_eval_mode() {", 1)[1]
    audit_program = audit_function.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    result = tmp_path / "result.json"
    result.write_text(json.dumps({
        "args": {
            "gated_compressed_rope": True,
            "gated_rope_start_layer": 16,
        }
    }))

    common = [
        sys.executable,
        "-",
        str(result),
        "",  # eval mode
        "",  # digit protection
        "",  # online mask
        "",  # legacy two-axis
        "1",  # gated mode
    ]
    subprocess.run([*common, "16"], input=audit_program, text=True, check=True)
    bad = subprocess.run(
        [*common, "17"],
        input=audit_program,
        text=True,
        capture_output=True,
    )
    assert bad.returncode != 0
    assert "gated_rope_start_layer=16" in bad.stderr


def test_pipeline_offline_preflight_needs_no_api_key(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard_000.npy").touch()
    (data / "mask_000.npy").touch()

    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    inner_bash = stub_bin / "bash"
    inner_bash.write_text("#!/bin/sh\necho TRAIN_STUB_REACHED\nexit 42\n")
    inner_bash.chmod(0o755)

    base_env = {
        **os.environ,
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "PROJECT_DIR": str(REPO),
        "Z2Z_SCRATCH": str(tmp_path / "scratch"),
        "OUTPUT_BASE": str(tmp_path / "outputs"),
        "DATA_DIR": str(data),
        "RECIPE": "",
        "WANDB": "0",
        "RUN_NAME": "offline-preflight",
    }
    base_env.pop("WANDB_API_KEY", None)
    offline = subprocess.run(
        ["/bin/bash", str(PIPELINE)],
        env=base_env,
        text=True,
        capture_output=True,
    )
    assert offline.returncode == 42
    assert "TRAIN_STUB_REACHED" in offline.stdout
    assert "WANDB_API_KEY" not in offline.stdout

    online_env = {
        **base_env,
        "WANDB": "1",
        "RUN_NAME": "online-preflight",
    }
    online = subprocess.run(
        ["/bin/bash", str(PIPELINE)],
        env=online_env,
        text=True,
        capture_output=True,
    )
    assert online.returncode == 1
    assert "WANDB=1 requires WANDB_API_KEY" in online.stdout
    assert "TRAIN_STUB_REACHED" not in online.stdout


def test_pipeline_offline_routes_all_eval_and_upload_paths():
    source = PIPELINE.read_text()
    assert "EVAL_WANDB_ARGS=(--no_wandb --resume_wandb_id none)" in source
    assert '"${EVAL_WANDB_ARGS[@]}"' in source
    assert source.count('if [ "$WANDB" != "0" ]; then') >= 5
    assert "pipeline complete offline" in source


def test_wandb_entity_is_consistent_across_pipeline_phases():
    source = PIPELINE.read_text()
    assert "export WANDB_ENTITY=${WANDB_ENTITY:-epfl-dlab}" in source
    assert source.count('--entity "$WANDB_ENTITY"') == 4
    assert "--environment WANDB_ENTITY=$WANDB_ENTITY" in source

    eval_source = (REPO / "scripts" / "eval_ckpt_rcp.sh").read_text()
    assert "export WANDB_ENTITY=${WANDB_ENTITY:-epfl-dlab}" in eval_source
    assert "export WANDB_PROJECT=${WANDB_PROJECT:-llaza}" in eval_source
    assert '--entity "$WANDB_ENTITY"' in eval_source
    assert '--project "$WANDB_PROJECT"' in eval_source
    assert '"zip2zip-compression>=0.3.3"' in eval_source
    assert '[ "$WANDB" != "0" ]' in eval_source
    assert '${RUNAI_JOB_NAME:-local}-${HOSTNAME:-host}-$$' in eval_source
