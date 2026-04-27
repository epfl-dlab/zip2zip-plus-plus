"""Print exact parameter counts for each encoder ablation configuration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import Zip2ZipLlama3Model

BASE_CONFIG = "1B"
MS = 2  # must match ablation script

ablations = [
    # (label, encoder_dim, encoder_n_heads, encoder_intermediate_size, encoder_n_layers)
    ("dim=128  (heads=2)",  128,  2,  512, 2),
    ("dim=256  (heads=4)",  256,  4, 1024, 2),
    ("dim=512  (default)", 512,  8, 2048, 2),
    ("dim=1024 (heads=16)", 1024, 16, 4096, 2),
    ("dim=2048 (heads=32)", 2048, 32, 8192, 2),
    ("layers=1",            512,  8, 2048, 1),
    ("layers=2 (default)",  512,  8, 2048, 2),
]

import dataclasses

base_cfg = zip2zip_llama_configs[BASE_CONFIG]

# Total base model params (with default encoder, ms=MS)
base_cfg_ms = dataclasses.replace(base_cfg, max_subtokens=MS)
base_model = Zip2ZipLlama3Model(base_cfg_ms)
total_base = sum(p.numel() for p in base_model.parameters())

print(f"Base model ({BASE_CONFIG}, ms={MS}) total params: {total_base:,}  ({total_base/1e6:.1f}M)\n")
print(f"{'Label':<28} {'Enc params':>12} {'Enc (M)':>9} {'% of base':>10}")
print("-" * 65)

for label, enc_dim, enc_heads, enc_inter, enc_layers in ablations:
    cfg = dataclasses.replace(
        base_cfg,
        max_subtokens=MS,
        encoder_dim=enc_dim,
        encoder_n_heads=enc_heads,
        encoder_intermediate_size=enc_inter,
        encoder_n_layers=enc_layers,
    )
    model = Zip2ZipLlama3Model(cfg)
    enc_params = sum(p.numel() for p in model.hyper_encoder.parameters())
    pct = enc_params / total_base * 100
    print(f"{label:<28} {enc_params:>12,} {enc_params/1e6:>8.1f}M {pct:>9.1f}%")
