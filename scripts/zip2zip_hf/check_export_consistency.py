"""Sanity-check exported HF config against training metadata.

Examples:
  python scripts/zip2zip_hf/check_export_consistency.py --repo epfl-dlab/candidate-Llaza-MS4-flat-20BT
  python scripts/zip2zip_hf/check_export_consistency.py --export_dir /path/to/export
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def _load_from_repo(repo: str, revision: str) -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id=repo, filename="zip2zip_config.json", revision=revision)
    meta_path = hf_hub_download(repo_id=repo, filename="meta.pt", revision=revision)

    with open(cfg_path) as f:
        cfg = json.load(f)
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    train_args = meta.get("args", {})
    return cfg, train_args


def _load_from_dir(export_dir: str) -> tuple[dict, dict]:
    cfg_path = os.path.join(export_dir, "zip2zip_config.json")
    meta_path = os.path.join(export_dir, "meta.pt")

    with open(cfg_path) as f:
        cfg = json.load(f)

    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        train_args = meta.get("args", {})
    else:
        train_args = {}
    return cfg, train_args


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=str, default=None, help="HF repo id")
    p.add_argument("--revision", type=str, default="hf", help="HF revision (default: hf)")
    p.add_argument("--export_dir", type=str, default=None, help="Local export directory")
    args = p.parse_args()

    if bool(args.repo) == bool(args.export_dir):
        raise ValueError("Pass exactly one of --repo or --export_dir")

    if args.repo:
        cfg, train_args = _load_from_repo(args.repo, args.revision)
        source = f"{args.repo}@{args.revision}"
    else:
        cfg, train_args = _load_from_dir(args.export_dir)
        source = args.export_dir

    mismatches: list[str] = []

    enc_cfg = cfg.get("encoder", {})
    got_causal = bool(enc_cfg.get("causal", False))
    got_residual = bool(enc_cfg.get("residual", True))

    expected_causal = bool(train_args.get("encoder_causal", False))
    expected_residual = not bool(train_args.get("no_encoder_residual", False))

    if got_causal != expected_causal:
        mismatches.append(
            f"encoder.causal mismatch: exported={got_causal} expected_from_train_args={expected_causal}"
        )
    if got_residual != expected_residual:
        mismatches.append(
            f"encoder.residual mismatch: exported={got_residual} expected_from_train_args={expected_residual}"
        )

    if "max_subtokens" in train_args:
        got_max_subtokens = cfg.get("compression", {}).get("max_subtokens")
        if got_max_subtokens != train_args["max_subtokens"]:
            mismatches.append(
                "compression.max_subtokens mismatch: "
                f"exported={got_max_subtokens} expected_from_train_args={train_args['max_subtokens']}"
            )

    if "max_codebook_size" in train_args:
        got_max_cb = cfg.get("compression", {}).get("max_codebook_size")
        if got_max_cb != train_args["max_codebook_size"]:
            mismatches.append(
                "compression.max_codebook_size mismatch: "
                f"exported={got_max_cb} expected_from_train_args={train_args['max_codebook_size']}"
            )

    print(f"[check_export_consistency] source={source}")
    print(f"  encoder.causal   : exported={got_causal} expected={expected_causal}")
    print(f"  encoder.residual : exported={got_residual} expected={expected_residual}")

    if mismatches:
        print("\n❌ Found mismatches:")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print("\n✅ Export config is consistent with training metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
