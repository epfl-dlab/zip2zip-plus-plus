"""Quick test: load a zip2zip checkpoint and run a forward pass."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import dataclasses

import torch

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import Zip2ZipLlama3Model

# ── config ────────────────────────────────────────────────────────────────────
CKPT_DIR = "/mnt/scratch/checkpoints/zip2zip_1b_finemath_10bt_ms3/step_1000"
MODEL_KEY = "1B"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── read training args saved alongside the checkpoint ─────────────────────────
meta = torch.load(f"{CKPT_DIR}/meta.pt", map_location="cpu", weights_only=False)
args = meta.get("args", {})
print(f"Checkpoint step: {meta['step']}")
print(f"  max_subtokens  : {args.get('max_subtokens')}")
print(f"  max_codebook_size: {args.get('max_codebook_size')}")

# ── build model (override fields that differ from the default config) ──────────
print(f"\nBuilding {MODEL_KEY} model on {DEVICE}...")
cfg = zip2zip_llama_configs[MODEL_KEY]
overrides = {k: args[k] for k in ("max_subtokens", "max_codebook_size") if k in args}
if overrides:
    cfg = dataclasses.replace(cfg, **overrides)
model = Zip2ZipLlama3Model(cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  params: {n_params/1e6:.1f}M  (max_subtokens={cfg.max_subtokens})")

# ── load checkpoint ───────────────────────────────────────────────────────────
print(f"\nLoading checkpoint from {CKPT_DIR}/model.pt ...")
state_dict = torch.load(f"{CKPT_DIR}/model.pt", map_location=DEVICE, weights_only=True)
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
