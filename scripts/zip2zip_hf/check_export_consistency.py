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
    from huggingface_hub.utils import EntryNotFoundError

    cfg_path = hf_hub_download(repo_id=repo, filename="zip2zip_config.json", revision=revision)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # export.py / export_phi.py now copy meta.pt alongside the exported
    # weights, so newer "hf"-branch pushes have it directly at `revision`.
    # Older exports (pushed before that) only ever had it on "main" (the
    # training-checkpoint branch) -- fall back there so both still work.
    meta = None
    for meta_revision in dict.fromkeys([revision, "main"]):
        try:
            meta_path = hf_hub_download(repo_id=repo, filename="meta.pt", revision=meta_revision)
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            break
        except EntryNotFoundError:
            continue
    if meta is None:
        print(f"  (no meta.pt on '{revision}' or 'main' -- skipping train-arg checks)")
        train_args = {}
    else:
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
        tokenizer_source, tokenizer_kwargs = args.repo, {"revision": args.revision}
    else:
        cfg, train_args = _load_from_dir(args.export_dir)
        source = args.export_dir
        tokenizer_source, tokenizer_kwargs = args.export_dir, {}

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

    # hidden_size // 64 is only a valid guess when head_dim == 64; catch the case
    # where export resolved a different head count than training actually used
    # (e.g. Phi-3.5-mini: encoder_n_heads=32, head_dim=96, //64 would silently give 48).
    expected_n_heads = train_args.get("encoder_n_heads")
    if isinstance(expected_n_heads, int) and expected_n_heads > 0:
        got_n_heads = enc_cfg.get("num_heads")
        if got_n_heads != expected_n_heads:
            mismatches.append(
                f"encoder.num_heads mismatch: exported={got_n_heads} expected_from_train_args={expected_n_heads}"
            )

    # untied checkpoints must export BOTH input_encoder.* and output_encoder.*
    # (tie_encoders=False) -- loading an untied checkpoint as tied silently
    # drops the output_encoder.* weights.
    if "untied_hyper_encoder" in train_args:
        expected_untied = bool(train_args["untied_hyper_encoder"])
        got_untied = not bool(enc_cfg.get("tie_encoders", True))
        if got_untied != expected_untied:
            mismatches.append(
                f"encoder.tie_encoders mismatch: exported tie_encoders={enc_cfg.get('tie_encoders')} "
                f"(untied={got_untied}) expected_from_train_args untied={expected_untied}"
            )

    # disable_digit_ids must carry over into compression.disabled_ids or the
    # export silently lets digits merge into hyper-tokens the model never saw
    # in training -- same mismatch class as the chat-token bug, for digits.
    digit_check_note = None
    if "disable_digit_ids" in train_args:
        expected_digit_protected = bool(train_args["disable_digit_ids"])
        try:
            from transformers import AutoTokenizer
            from zip2zip_core.disabled_ids import digit_ids

            tok = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
            vocab_size = cfg.get("compression", {}).get("initial_vocab_size") or len(tok)
            expected_ids = digit_ids(tok, vocab_size)
            exported_ids = set(cfg.get("compression", {}).get("disabled_ids", []))
            got_digit_protected = bool(expected_ids) and expected_ids.issubset(exported_ids)
            if got_digit_protected != expected_digit_protected:
                mismatches.append(
                    "compression.disabled_ids digit-protection mismatch: exported digit ids "
                    f"{'present' if got_digit_protected else 'MISSING'} "
                    f"expected_from_train_args disable_digit_ids={expected_digit_protected}"
                )
            digit_check_note = f"digit ids {'present' if got_digit_protected else 'absent'} in disabled_ids"
        except Exception as e:
            digit_check_note = f"could not verify ({e})"

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
    print(f"  encoder.causal       : exported={got_causal} expected={expected_causal}")
    print(f"  encoder.residual     : exported={got_residual} expected={expected_residual}")
    if isinstance(expected_n_heads, int) and expected_n_heads > 0:
        print(f"  encoder.num_heads    : exported={enc_cfg.get('num_heads')} expected={expected_n_heads}")
    if "untied_hyper_encoder" in train_args:
        print(f"  encoder.tie_encoders : exported={enc_cfg.get('tie_encoders')} "
              f"expected_untied={bool(train_args['untied_hyper_encoder'])}")
    if digit_check_note is not None:
        print(f"  digit protection     : {digit_check_note} "
              f"(train disable_digit_ids={bool(train_args.get('disable_digit_ids'))})")

    if mismatches:
        print("\n❌ Found mismatches:")
        for m in mismatches:
            print(f"  - {m}")
        return 1

    print("\n✅ Export config is consistent with training metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
