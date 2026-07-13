"""Verify a zip2zip-core finetune checkpoint's weights against ground truth.

Context: the step_8000 Phi-3.5 repro ckpt scores 5.19 nats/base-token on its own
training data through the exact training forward (W&B logged 1.63), and 6.9
nats/base-token in base mode — the decoder as loaded is broken while the code
path is provably unchanged since training. So the weights on disk are suspect.

Ground truths this script exploits:
  1. FROZEN params (decoder base_layer weights, tok_embeddings, norms, lm head)
     never trained -> they must be BIT-IDENTICAL to a fresh HF->torchtitan
     conversion of the base model (same code path as train.py's
     --init_from_hf: _split_phi3_to_llama + Llama3StateDictAdapter.from_hf).
  2. FROZEN params must also be identical ACROSS all step_* checkpoints of a run.

Interpretation:
  frozen == reference           -> save path fine for frozen params; damage is in
                                   the trained tensors (LoRA/hyper) or elsewhere.
  frozen != ref, cosine ~ 1     -> decoder actually TRAINED (freeze failed) or
                                   numeric corruption.
  frozen != ref, cosine ~ 0     -> checkpoint is NOT from an --init_from_hf run
                                   (e.g. artifact of a from-scratch/misconfigured
                                   attempt).

If a frozen mismatch is found and --write_repaired is given, writes a repaired
checkpoint: reference frozen weights + the ckpt's trained tensors (LoRA A/B +
hyper_encoder). Re-run scripts/diagnose_ckpt_replay.py on it to test recovery.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob as _glob
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

import torch

from zip2zip_core.configs import zip2zip_llama_configs


def log(msg: str):
    print(f"[weights] {msg}", flush=True)


def load_ckpt_sd(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "model.pt")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        out[k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")] = v
    return out


def build_cfg(ckpt_dir: str):
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    train_args = {}
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        train_args = meta.get("args", {}) or {}
    cfg = zip2zip_llama_configs[train_args.get("model_config", "1B")]
    overrides = {}
    for key in ("max_subtokens", "max_codebook_size", "hyper_encoder_type",
                "encoder_dim", "encoder_n_layers", "encoder_n_heads",
                "encoder_intermediate_size"):
        v = train_args.get(key)
        if v is not None:
            overrides[key] = v
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg, train_args


def build_reference_sd(hf_model: str, cfg):
    """Exact same conversion train.py --init_from_hf performs (minus dist)."""
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from zip2zip_core.train import _split_phi3_to_llama
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    cache_dir = snapshot_download(hf_model, allow_patterns=["*.safetensors", "*.json"])
    hf_sd = {}
    for f in sorted(_glob.glob(os.path.join(cache_dir, "*.safetensors"))):
        hf_sd.update(load_file(f, device="cpu"))
    if any(k.endswith("self_attn.qkv_proj.weight") for k in hf_sd):
        log("splitting fused Phi-3 qkv_proj/gate_up_proj to Llama layout")
        hf_sd = _split_phi3_to_llama(hf_sd, cfg)
    return Llama3StateDictAdapter(cfg, None).from_hf(hf_sd)


def frozen_key_pairs(ckpt_sd: dict, ref_sd: dict):
    """(ckpt_key, ref_key) pairs for every frozen parameter present in both."""
    pairs = []
    for ck in ckpt_sd:
        if ck.startswith("hyper_encoder.") or ".lora_A." in ck or ".lora_B." in ck:
            continue
        rk = ck.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        if rk in ref_sd:
            pairs.append((ck, rk))
    return pairs


def compare(ckpt_sd, ref_sd, pairs):
    n_exact = 0
    worst = []
    for ck, rk in pairs:
        a = ckpt_sd[ck].float()
        b = ref_sd[rk].float()
        if a.shape != b.shape:
            worst.append((ck, float("inf"), 0.0, f"SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"))
            continue
        diff = (a - b).abs().max().item()
        if diff == 0.0:
            n_exact += 1
            continue
        rel = ((a - b).norm() / (b.norm() + 1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
        worst.append((ck, diff, rel, f"cos={cos:.4f}"))
    worst.sort(key=lambda t: -t[1] if t[1] != float("inf") else float("inf"))
    return n_exact, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--hf_model", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--compare_steps", default=None,
                   help="Run dir containing step_* checkpoints: cross-check that "
                        "frozen tensors are identical across steps.")
    p.add_argument("--write_repaired", default=None,
                   help="If a frozen mismatch is found, write a repaired ckpt "
                        "(reference frozen + ckpt LoRA/hyper) to this dir.")
    args = p.parse_args()

    cfg, train_args = build_cfg(args.ckpt_dir)
    log(f"config: {train_args.get('model_config')} vocab={cfg.vocab_size}")
    log(f"meta step={torch.load(os.path.join(args.ckpt_dir, 'meta.pt'), map_location='cpu', weights_only=False).get('step')}")
    log(f"init_from_hf (train arg): {train_args.get('init_from_hf')}")
    for k in ("lora_rank", "lora_alpha", "freeze_decoder", "lr", "seq_len", "max_steps", "steps"):
        if k in train_args:
            log(f"  train_args[{k}] = {train_args[k]}")

    log("loading checkpoint state dict (mmap)...")
    ckpt_sd = load_ckpt_sd(args.ckpt_dir)
    log(f"building reference conversion from {args.hf_model} ...")
    ref_sd = build_reference_sd(args.hf_model, cfg)

    pairs = frozen_key_pairs(ckpt_sd, ref_sd)
    log(f"comparing {len(pairs)} frozen tensors (decoder bases, embeddings, norms, head)...")
    n_exact, worst = compare(ckpt_sd, ref_sd, pairs)
    log(f"RESULT: {n_exact}/{len(pairs)} frozen tensors exactly match the reference")
    if worst:
        log(f"MISMATCHED: {len(worst)} tensors. Worst 12:")
        for ck, diff, rel, extra in worst[:12]:
            log(f"  {ck}: max|diff|={diff:.6g} rel_frob={rel:.6g} {extra}")
        mean_rel = sum(w[2] for w in worst if w[1] != float("inf")) / max(1, len(worst))
        log(f"  mean rel_frob over mismatched: {mean_rel:.6g}")
        log("  cos~1 & small rel -> decoder trained/corrupted numerically; "
            "cos~0 -> not an --init_from_hf artifact at all.")
    else:
        log("All frozen weights are exactly as converted from HF — save path is "
            "clean for frozen params; damage must be in trained tensors or elsewhere.")

    if args.compare_steps:
        probe_keys = None
        steps = sorted(
            _glob.glob(os.path.join(args.compare_steps, "step_*")),
            key=lambda d: int(os.path.basename(d).split("_")[1]),
        )
        log(f"cross-step frozen check over {len(steps)} checkpoints in {args.compare_steps}")
        base_vals = {}
        for sdir in steps:
            if not os.path.exists(os.path.join(sdir, "model.pt")):
                continue
            sd = load_ckpt_sd(sdir)
            if probe_keys is None:
                probe_keys = [k for k in (
                    "tok_embeddings.weight",
                    "layers.0.attention.wq.base_layer.weight",
                    "layers.31.feed_forward.w2.base_layer.weight",
                    "output.weight",
                ) if k in sd]
            row = []
            for k in probe_keys:
                v = sd[k].float()
                if k not in base_vals:
                    base_vals[k] = v
                    row.append("ref")
                else:
                    row.append(f"{(v - base_vals[k]).abs().max().item():.3g}")
            log(f"  {os.path.basename(sdir)}: max|diff vs first ckpt| = "
                f"{dict(zip(probe_keys, row))}")
            del sd

    if worst and args.write_repaired:
        out_dir = args.write_repaired
        log(f"writing repaired checkpoint to {out_dir} ...")
        os.makedirs(out_dir, exist_ok=True)
        repaired = {}
        for k, v in ckpt_sd.items():
            if k.startswith("hyper_encoder.") or ".lora_A." in k or ".lora_B." in k:
                repaired[k] = v.clone()  # trained tensors: keep from ckpt
        for ck, rk in pairs:
            repaired[ck] = ref_sd[rk].float().clone()  # frozen: take reference
        # carry over anything not covered (e.g. buffers saved by mistake)
        for k, v in ckpt_sd.items():
            if k not in repaired:
                repaired[k] = v.clone()
        torch.save(repaired, os.path.join(out_dir, "model.pt"))
        shutil.copy(os.path.join(args.ckpt_dir, "meta.pt"), os.path.join(out_dir, "meta.pt"))
        log(f"repaired ckpt written. Now run scripts/diagnose_ckpt_replay.py with "
            f"--ckpt_dir {out_dir} — if train-style replay drops to ~1.6-2.0, the "
            f"trained LoRA/hyper tensors are intact and the run is fully recovered.")
    elif not worst:
        log("no repair needed/possible on the frozen side.")


if __name__ == "__main__":
    main()
