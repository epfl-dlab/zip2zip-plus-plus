"""Test the stale-save hypothesis: do the TRAINED tensors evolve across step_* ckpts?

Background: the step_8000 repro ckpt replays at 5.19 nats/base-token on training
data — matching the training log around step ~500 (5.33), not step 8000 (1.63).
Frozen params are bit-exact vs the HF conversion (so saves LOOK healthy), but
frozen params can't distinguish a fresh gather from a stale one. The trained
tensors can:

  - If lora_A/lora_B/hyper_encoder are IDENTICAL across step_500...step_8000,
    every save after the first wrote stale values -> the run's true final state
    was never persisted (fix save_checkpoint, retrain ~2.3h).
  - If they evolve, replaying step_500/step_1000 tells us whether each ckpt
    matches its logged training loss (5.33 / 3.15) — localizing when/where the
    saved state diverges from the live one.

Also checks shard-structure (world_size=4 row quarters, zero rows) to detect
partial gathers, and optimizer.pt probe norms across steps.

Usage:
    python scripts/diagnose_ckpt_evolution.py \
        --run_dir '/path/to/run' --data_dir /path/to/shards \
        --tokenizer microsoft/Phi-3.5-mini-instruct
"""

from __future__ import annotations

import argparse
import glob as _glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

PROBE_KEYS = [
    "layers.0.attention.wq.lora_A.weight",
    "layers.0.attention.wq.lora_B.weight",
    "layers.15.feed_forward.w1.lora_B.weight",
    "layers.31.feed_forward.w2.lora_B.weight",
    "hyper_encoder.pos_embed.weight",
]


def log(msg: str):
    print(f"[evolution] {msg}", flush=True)


def load_sd(step_dir: str):
    path = os.path.join(step_dir, "model.pt")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", ""): v
            for k, v in sd.items()}


def step_dirs(run_dir: str):
    dirs = [d for d in _glob.glob(os.path.join(run_dir, "step_*"))
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "model.pt"))
            and not d.endswith("_repaired")]
    return sorted(dirs, key=lambda d: int(os.path.basename(d).split("_")[1]))


def quarter_norms(t: torch.Tensor, parts: int = 4):
    rows = t.shape[0]
    q = rows // parts
    return [round(t[i * q:(i + 1) * q].float().norm().item(), 4) for i in range(parts)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--replay_steps", default="500,1000,8000",
                   help="Comma-separated step numbers to replay on GPU.")
    p.add_argument("--n_chunks", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dirs = step_dirs(args.run_dir)
    log(f"found {len(dirs)} checkpoints in {args.run_dir}")

    # ── Part 1: trained-tensor evolution across steps ──────────────────────
    log("=== part 1: trained-tensor evolution (bit-identical => stale saves) ===")
    first_vals = {}
    per_step_norms = {}
    for d in dirs:
        name = os.path.basename(d)
        sd = load_sd(d)
        norms, diffs = {}, {}
        for k in PROBE_KEYS:
            if k not in sd:
                continue
            v = sd[k].float()
            norms[k] = round(v.norm().item(), 6)
            if k not in first_vals:
                first_vals[k] = v.clone()
                diffs[k] = "ref"
            else:
                diffs[k] = f"{(v - first_vals[k]).abs().max().item():.6g}"
        per_step_norms[name] = norms
        log(f"  {name}: norms={norms}")
        log(f"  {name}: max|diff vs step_{os.path.basename(dirs[0]).split('_')[1]}|={diffs}")
        # optimizer probe (same staleness question on the optimizer path)
        opt_path = os.path.join(d, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                osd = torch.load(opt_path, map_location="cpu", weights_only=True, mmap=True)
                state = osd.get("state", {})
                probe = None
                for pk, pv in state.items():
                    if "lora_B" in str(pk) and isinstance(pv, dict) and "exp_avg" in pv:
                        probe = (str(pk), pv["exp_avg"].float().norm().item(),
                                 pv.get("step"))
                        break
                if probe:
                    log(f"  {name}: optimizer probe {probe[0]}: ||exp_avg||={probe[1]:.6g} step={probe[2]}")
            except Exception as e:
                log(f"  {name}: optimizer probe failed ({type(e).__name__})")
        del sd

    # ── Part 2: shard-structure on the last ckpt ───────────────────────────
    log("=== part 2: shard structure of trained tensors (last ckpt) ===")
    sd = load_sd(dirs[-1])
    for k in PROBE_KEYS:
        if k not in sd:
            continue
        t = sd[k].float()
        zero_rows = int((t.abs().sum(dim=tuple(range(1, t.dim()))) == 0).sum())
        log(f"  {k}: shape={tuple(t.shape)} zero_rows={zero_rows}/{t.shape[0]} "
            f"quarter_norms={quarter_norms(t)}")
    del sd

    # ── Part 3: GPU replay per step ─────────────────────────────────────────
    steps_to_replay = [int(s) for s in args.replay_steps.split(",") if s.strip()]
    log(f"=== part 3: train-style replay on steps {steps_to_replay} "
        f"(training log: 5.33@500, 3.15@1000, 1.63@8000) ===")
    from transformers import AutoTokenizer
    from diagnose_ckpt_replay import load_chunks, train_style_replay
    from zip2zip_core.lm_eval_adapter import Zip2ZipLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special = set(tokenizer.all_special_ids or [])
    added = set(tokenizer.get_added_vocab().values())

    chunks = load_chunks(args.data_dir, args.n_chunks, 4096, 0.5)
    for step in steps_to_replay:
        d = os.path.join(args.run_dir, f"step_{step}")
        if not os.path.isdir(d):
            log(f"  step_{step}: missing, skipped")
            continue
        lm = Zip2ZipLM(pretrained=d, tokenizer=args.tokenizer,
                       max_length=1024, device=args.device, dtype="bfloat16")
        train_disabled = sorted(i for i in (special | added) if 0 <= i < lm.cfg.vocab_size)
        ce = base = valid = 0
        accs = []
        for chunk, mask in chunks:
            m = train_style_replay(lm, chunk, mask, 2048, train_disabled, tokenizer)
            if m is None:
                continue
            ce += m["ce_sum"]; base += m["n_base"]; valid += m["n_valid"]
            accs.append(m["acc"])
        log(f"  step_{step}: nats/base-token={ce / base:.4f} "
            f"nats/valid_comp={ce / valid:.4f} acc={sum(accs) / len(accs):.3f}")
        del lm
        torch.cuda.empty_cache()

    log("=== interpretation ===")
    log("identical probes across steps + flat replay  -> stale saves: only the first")
    log("  gather was real; fix save_checkpoint, retrain (~2.3h on 4 GPUs).")
    log("evolving probes + replay tracking the log    -> saves honest; problem is")
    log("  elsewhere (report back for next steps).")


if __name__ == "__main__":
    main()
