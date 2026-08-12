"""Quick test: load a zip2zip checkpoint and run a forward pass.

Supports local checkpoints and HuggingFace Hub:
    python scripts/test_checkpoint.py                                    # uses default local path
    python scripts/test_checkpoint.py --ckpt_dir /path/to/step_6000
    python scripts/test_checkpoint.py --hf_repo Saibo-creator/zip2zip_1b_finemath_10bt_ms4 --hf_revision step_6000
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import dataclasses

import torch

from zip2zip_core.checkpoint import (
    prepare_inference_state_dict,
    resolve_gated_rope_start_pair,
)
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import Zip2ZipLlama3Model, restore_encoder_residual

# ── parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default=None, help="Local checkpoint directory")
parser.add_argument("--hf_repo", type=str, default=None, help="HuggingFace repo ID")
parser.add_argument("--hf_revision", type=str, default=None, help="HF revision (e.g. step_6000)")
parser.add_argument("--model_config", type=str, default="1B", choices=list(zip2zip_llama_configs.keys()))
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── resolve checkpoint directory ─────────────────────────────────────────────
if args.hf_repo:
    from huggingface_hub import hf_hub_download

    print(f"Downloading from {args.hf_repo} (revision: {args.hf_revision or 'main'})...")
    ckpt_dir = None
    for filename in ["model.pt", "optimizer.pt", "meta.pt"]:
        try:
            path = hf_hub_download(
                repo_id=args.hf_repo,
                filename=filename,
                revision=args.hf_revision,
            )
            if ckpt_dir is None:
                ckpt_dir = os.path.dirname(path)
        except Exception:
            if filename == "model.pt":
                raise
    print(f"  Downloaded to {ckpt_dir}")
elif args.ckpt_dir:
    ckpt_dir = args.ckpt_dir
else:
    ckpt_dir = "/mnt/scratch/checkpoints/zip2zip_1b_finemath_10bt_ms3/step_1000"

# ── read training args saved alongside the checkpoint ─────────────────────────
meta_path = os.path.join(ckpt_dir, "meta.pt")
if not os.path.exists(meta_path):
    raise FileNotFoundError(
        f"required checkpoint metadata is missing: {meta_path}"
    )
meta = torch.load(meta_path, map_location="cpu", weights_only=False)
if not isinstance(meta, dict) or not isinstance(meta.get("args"), dict):
    raise ValueError(f"invalid checkpoint metadata: {meta_path}")
train_args = meta["args"]
print(f"Checkpoint step: {meta.get('step', '<unknown>')}")
print(f"  max_subtokens  : {train_args.get('max_subtokens')}")
print(f"  max_codebook_size: {train_args.get('max_codebook_size')}")

# ── build model (override fields that differ from the default config) ──────────
MODEL_KEY = train_args.get("model_config") or args.model_config
print(f"\nBuilding {MODEL_KEY} model on {DEVICE}...")
cfg = zip2zip_llama_configs[MODEL_KEY]
model_path = os.path.join(ckpt_dir, "model.pt")
print(f"\nLoading checkpoint from {model_path} ...")
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
state_dict = prepare_inference_state_dict(state_dict, train_args)
overrides = {
    key: train_args[key]
    for key in (
        "max_subtokens",
        "max_codebook_size",
        "hyper_encoder_type",
        "encoder_dim",
        "encoder_n_layers",
        "encoder_n_heads",
        "encoder_intermediate_size",
        "token_type_loss_weight",
        "gated_rope_start_layer",
        "gated_rope_start_pair",
    )
    if train_args.get(key) is not None
}
overrides.update(
    tie_hyper_encoder=not bool(train_args.get("untied_hyper_encoder")),
    share_hyper_encoder_weights=bool(
        train_args.get("share_hyper_encoder_weights")
    ),
    base_token_positions=bool(train_args.get("base_token_positions")),
    two_axis_rope=bool(train_args.get("two_axis_rope")),
    gated_compressed_rope=bool(train_args.get("gated_compressed_rope")),
)
if train_args.get("gated_compressed_rope"):
    head_dim = (
        getattr(cfg.layer.attention, "head_dim", None)
        or cfg.dim // cfg.layer.attention.n_heads
    )
    overrides["gated_rope_start_pair"] = resolve_gated_rope_start_pair(
        train_args,
        state_dict,
        n_complex_pairs=head_dim // 2,
        default=cfg.gated_rope_start_pair,
    )
if overrides:
    cfg = dataclasses.replace(cfg, **overrides)
model = Zip2ZipLlama3Model(cfg)
restore_encoder_residual(model, train_args)
model = model.to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params/1e6:.1f}M  (max_subtokens={cfg.max_subtokens})")

# ── load checkpoint ───────────────────────────────────────────────────────────
model.load_state_dict(state_dict, strict=True)
print("  All keys matched perfectly (strict load).")

# ── forward pass ──────────────────────────────────────────────────────────────
print("\nRunning forward pass (no codebook)...")
model.eval()
B, T = 2, 64
tokens = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE)
with torch.no_grad():
    out = model(tokens)

# out may be (logits,) or logits directly
logits = out[0] if isinstance(out, tuple) else out
print(f"  logits shape: {logits.shape}")   # expected (B, T, vocab_size)
print(f"  logits range: [{logits.min():.3f}, {logits.max():.3f}]")

if cfg.gated_compressed_rope:
    print("\nRunning compressed forward through gated RoPE...")
    compressed_tokens = tokens[:1].clone()
    compressed_tokens[0, T // 2] = cfg.vocab_size
    codebook = torch.full(
        (1, cfg.max_codebook_size, cfg.max_subtokens),
        cfg.pad_token_id,
        device=DEVICE,
        dtype=torch.long,
    )
    codebook[0, 0, :2] = torch.tensor([1, 2], device=DEVICE)
    before = model.gated_rope_forwards
    with torch.no_grad():
        compressed_out = model(
            compressed_tokens,
            codebook=codebook,
            hyper_causal_mask=bool(
                train_args.get("hyper_causal_mask", True)
            ),
        )
    compressed_logits = (
        compressed_out[0]
        if isinstance(compressed_out, tuple)
        else compressed_out
    )
    if model.gated_rope_forwards != before + 1:
        raise RuntimeError("gated compressed forward did not activate RoPE gates")
    if not torch.isfinite(compressed_logits).all():
        raise RuntimeError("gated compressed forward produced non-finite logits")
    print(
        "  gated path active: "
        f"shape={tuple(model.compressed_rope_gate.shape)} "
        f"forwards={model.gated_rope_forwards}"
    )
print("\nDone — checkpoint loads and runs correctly.")
