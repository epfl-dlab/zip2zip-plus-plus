"""Diagnostic: does the tied hyper-encoder miscalibrate hypertoken logits?

zip2zip-core scores hypertokens against the SAME vectors that embed them on the
input side (tied), while base tokens are scored against Phi's separate lm_head.
If those two vector families have different scales, hypertoken logits are
systematically inflated/deflated inside the shared softmax. This script
measures that directly on a trained checkpoint with real training-distribution
data — no training needed.

  A. Norm statistics: hyper logit vectors f(c_k) (input-table-derived) vs
     lm_head rows vs tok_embeddings rows.
  B. Teacher-forced calibration on real compressed chunks:
       - probability mass assigned to the hyper region vs the empirical
         fraction of hypertoken targets
       - CE split: base-token targets vs hypertoken targets
       - P(re-emit the just-consumed hypertoken) vs its empirical frequency
       - argmax share: how often the greedy token is a hypertoken

Reads shard_00000.npy of DATA_DIR and mirrors the training chunking (4096-token
chunks, LZW, keep first seq_len+1). Usage:

  python scripts/diagnose_tied_hyper.py \
      --ckpt_dir /path/to/step_8000 \
      --data_dir /path/to/phi-1B-sft-8shards-eosfix \
      [--n_chunks 64] [--out_json /path/to/out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

import numpy as np
import torch
import torch.nn.functional as F

from zip2zip_core.lm_eval_adapter import Zip2ZipLM  # noqa: E402
from zip2zip_compression import LZWCompressor  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--n_chunks", type=int, default=64)
    p.add_argument("--chunk_len", type=int, default=4096)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    lm = Zip2ZipLM(
        pretrained=args.ckpt_dir,
        tokenizer=args.tokenizer,
        max_length=4096,
        device="cuda",
        dtype="bfloat16",
        eval_mode="compressed",
        batch_size=1,
        hyper_causal_mask=True,
    )
    model, cfg, dev = lm.model, lm.cfg, lm._device
    V = cfg.vocab_size

    # ---------- A: static norm statistics ----------
    with torch.no_grad():
        lm_head_norms = model.output.weight.detach().float().norm(dim=1).cpu()
        tok_emb_norms = model.tok_embeddings.weight.detach().float().norm(dim=1).cpu()

    shard = np.load(os.path.join(args.data_dir, "shard_00000.npy"), mmap_mode="r")
    rng = random.Random(args.seed)
    offsets = [rng.randrange(0, len(shard) - args.chunk_len) for _ in range(args.n_chunks)]

    hyper_norms = []
    ce_base_sum = ce_hyper_sum = 0.0
    n_base_tgt = n_hyper_tgt = 0
    hyper_mass_sum = 0.0
    n_positions = 0
    argmax_hyper = 0
    p_self_sum = 0.0
    n_self_pos = 0
    n_self_repeat_empirical = 0
    n_chunks_used = 0
    n_chunks_dropped = 0

    for off in offsets:
        ids = [int(t) for t in shard[off : off + args.chunk_len]]
        compressor = LZWCompressor(**lm._compressor_kwargs)
        compressed, _, codebook = compressor.encode(
            ids, padding="do_not_pad", truncation=False
        )
        if len(compressed) < args.seq_len + 1:
            n_chunks_dropped += 1  # mirrors the training filter (data.py:272)
            continue
        compressed = compressed[: args.seq_len + 1]
        n_chunks_used += 1

        x = torch.tensor(compressed[:-1], dtype=torch.long, device=dev).unsqueeze(0)
        cb = lm._codebook_to_tensor(codebook).to(dev).unsqueeze(0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hyper_embeds = model._encode_codebook_with_weights(
                cb, model.tok_embeddings.weight
            )
            out = model(x, codebook=cb, hyper_causal_mask=True)
        logits = (out[0] if isinstance(out, tuple) else out)[0].float()  # (T, V+K)
        logp = F.log_softmax(logits, dim=-1)

        n_entries = len(codebook.to_dict())
        if n_entries:
            hyper_norms.append(
                hyper_embeds[0, :n_entries].detach().float().norm(dim=1).cpu()
            )

        T = x.shape[1]
        targets = torch.tensor(compressed[1:], dtype=torch.long, device=dev)
        is_hyper_tgt = targets >= V
        tgt_logp = logp[torch.arange(T, device=dev), targets]

        ce_base_sum += float(-tgt_logp[~is_hyper_tgt].sum())
        ce_hyper_sum += float(-tgt_logp[is_hyper_tgt].sum())
        n_base_tgt += int((~is_hyper_tgt).sum())
        n_hyper_tgt += int(is_hyper_tgt.sum())

        hyper_mass_sum += float(logp[:, V:].exp().sum(dim=-1).mean()) * T
        n_positions += T
        argmax_hyper += int((logp.argmax(dim=-1) >= V).sum())

        # self-repetition: at positions whose INPUT token is a hypertoken h,
        # model P(next == h) vs whether the data actually repeats h.
        in_tokens = x[0]
        self_pos = (in_tokens >= V).nonzero(as_tuple=True)[0]
        if len(self_pos):
            p_self_sum += float(logp[self_pos, in_tokens[self_pos]].exp().sum())
            n_self_pos += int(len(self_pos))
            n_self_repeat_empirical += int((targets[self_pos] == in_tokens[self_pos]).sum())

    hyper_norms = torch.cat(hyper_norms) if hyper_norms else torch.zeros(0)

    result = {
        "ckpt_dir": args.ckpt_dir,
        "chunks_used": n_chunks_used,
        "chunks_dropped_ge2x": n_chunks_dropped,
        "norms": {
            "lm_head_mean": float(lm_head_norms.mean()),
            "lm_head_median": float(lm_head_norms.median()),
            "tok_emb_mean": float(tok_emb_norms.mean()),
            "hyper_vec_mean": float(hyper_norms.mean()) if len(hyper_norms) else None,
            "hyper_vec_median": float(hyper_norms.median()) if len(hyper_norms) else None,
            "hyper_over_lm_head_mean_ratio": (
                float(hyper_norms.mean() / lm_head_norms.mean()) if len(hyper_norms) else None
            ),
        },
        "calibration": {
            "empirical_hyper_target_fraction": n_hyper_tgt / max(n_base_tgt + n_hyper_tgt, 1),
            "mean_hyper_prob_mass": hyper_mass_sum / max(n_positions, 1),
            "argmax_hyper_fraction": argmax_hyper / max(n_positions, 1),
            "ce_per_base_target": ce_base_sum / max(n_base_tgt, 1),
            "ce_per_hyper_target": ce_hyper_sum / max(n_hyper_tgt, 1),
        },
        "self_repetition": {
            "positions_after_hypertoken": n_self_pos,
            "mean_model_p_reemit": p_self_sum / max(n_self_pos, 1),
            "empirical_reemit_rate": n_self_repeat_empirical / max(n_self_pos, 1),
        },
    }

    print("\n" + "=" * 70)
    print("TIED HYPER-ENCODER DIAGNOSTIC")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    print("=" * 70)
    print("Reading guide:")
    print("  norms.hyper_over_lm_head_mean_ratio ~1.0  -> scales match, tying benign")
    print("  mean_hyper_prob_mass >> empirical_hyper_target_fraction -> inflated hyper logits")
    print("  mean_model_p_reemit >> empirical_reemit_rate -> tied self-repetition pathology")
    print("  ce_per_hyper_target vs ce_per_base_target -> where the model struggles")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
