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

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import Zip2ZipLlama3Model

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
if os.path.exists(meta_path):
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    train_args = meta.get("args", {})
    print(f"Checkpoint step: {meta['step']}")
    print(f"  max_subtokens  : {train_args.get('max_subtokens')}")
    print(f"  max_codebook_size: {train_args.get('max_codebook_size')}")
else:
    train_args = {}
    print("No meta.pt found — using default config")

# ── build model (override fields that differ from the default config) ──────────
MODEL_KEY = args.model_config
print(f"\nBuilding {MODEL_KEY} model on {DEVICE}...")
cfg = zip2zip_llama_configs[MODEL_KEY]
overrides = {k: train_args[k] for k in ("max_subtokens", "max_codebook_size") if k in train_args}
if overrides:
    cfg = dataclasses.replace(cfg, **overrides)
model = Zip2ZipLlama3Model(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params/1e6:.1f}M  (max_subtokens={cfg.max_subtokens})")

# ── load checkpoint ───────────────────────────────────────────────────────────
model_path = os.path.join(ckpt_dir, "model.pt")
print(f"\nLoading checkpoint from {model_path} ...")
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"  WARNING missing keys ({len(missing)}): {missing[:5]}")
if unexpected:
    print(f"  WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}")
if not missing and not unexpected:
    print("  All keys matched perfectly.")

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
print("\nDone — checkpoint loads and runs correctly.")
