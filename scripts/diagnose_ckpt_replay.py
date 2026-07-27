"""Diagnose a zip2zip-core checkpoint by replaying TRAINING conditions through
the EVAL loading + forward path, on actual training shards.

Motivation: the step_8000 Phi-3.5 repro ckpt logged W&B loss=1.63 nats/base-token
during training, yet the lm-eval adapter measures ~1.24 nats/byte (~5 nats/base-token)
on raw text — a 3-4x contradiction. This script scores training-formatted data
(4096-base-token chunks -> 2049 compressed tokens, exact data.py recipe) with the
checkpoint as loaded by the eval adapter (incl. LoRA fold), reproducing train.py's
loss definition exactly. It also scores the same data through the adapter's real
rolling-window path and in base (uncompressed) mode.

Verdict guide printed at the end:
  train-style replay ~= W&B loss (~1.63)  -> ckpt + eval forward healthy; the ppl gap
                                             lives in the eval windowing/methodology.
  train-style replay >> 1.63              -> the loaded weights are not the trained
                                             state (save/load defect) or the forward
                                             path is broken; finetuning itself is fine.
  base-mode much worse than ~2-2.5        -> folded decoder itself damaged.

Usage (see scripts/diagnose_ckpt_rcp.sh for the Run:AI launcher):
    python scripts/diagnose_ckpt_replay.py \
        --ckpt_dir /path/to/step_8000 \
        --data_dir /path/to/phi-1B-sft-8shards \
        --tokenizer microsoft/Phi-3.5-mini-instruct
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from zip2zip_compression import LZWCompressor

from zip2zip_core.data import online_codebook_counts, online_unavailable_targets
from zip2zip_core.lm_eval_adapter import Zip2ZipLM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


def log(msg: str):
    print(f"[diagnose] {msg}", flush=True)


def inspect_lora_weights(ckpt_dir: str):
    """Cheap mmap pass over model.pt: prove the LoRA adapters actually trained."""
    path = os.path.join(ckpt_dir, "model.pt")
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    lora_keys = [k for k in sd if k.endswith(".lora_B.weight")]
    log(f"model.pt keys={len(sd)}, lora_B tensors={len(lora_keys)}")
    for k in lora_keys[:3] + lora_keys[-2:]:
        base = k[: -len(".lora_B.weight")]
        a = sd[f"{base}.lora_A.weight"].float()
        b = sd[k].float()
        w = sd[f"{base}.base_layer.weight"].float()
        delta = b @ a
        log(f"  {base}: ||A||={a.norm():.4f} ||B||={b.norm():.4f} "
            f"||B@A||/||W||={delta.norm() / w.norm():.5f}")
    del sd


def load_chunks(data_dir: str, n_chunks: int, chunk_len: int, start_frac: float):
    """Mirror Zip2ZipDataset shard listing/mask pairing; return list of
    (base_token_chunk, mask_chunk_or_None)."""
    shard_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npy") and not f.startswith("mask_")
    )
    if not shard_files:
        raise SystemExit(f"No .npy shards in {data_dir}")
    shard_path = shard_files[0]
    basename = os.path.basename(shard_path)
    mask_name = basename.replace("shard_", "mask_", 1) if basename.startswith("shard_") \
        else f"mask_{basename}"
    mask_path = os.path.join(data_dir, mask_name)

    try:
        shard = np.load(shard_path, mmap_mode="r")
    except ValueError:
        size = os.path.getsize(shard_path)
        shard = np.memmap(shard_path, dtype=np.uint32, mode="r", shape=(size // 4,))
    mask = np.load(mask_path, mmap_mode="r") if os.path.exists(mask_path) else None
    log(f"shard: {basename} tokens={len(shard)} mask_present={mask is not None}")

    start = int(len(shard) * start_frac)
    start -= start % chunk_len
    chunks = []
    for i in range(n_chunks):
        lo = start + i * chunk_len
        if lo + chunk_len > len(shard):
            break
        m = np.asarray(mask[lo:lo + chunk_len]) if mask is not None else None
        chunks.append((shard[lo:lo + chunk_len].astype(np.int64).tolist(), m))
    log(f"loaded {len(chunks)} chunks of {chunk_len} base tokens from offset {start} "
        f"({start_frac:.0%} into shard 0)")
    return chunks


def compressed_loss_mask(compressed, base_mask, cb_dict, vocab):
    """data.py._compressed_loss_mask: a compressed token is a valid label only if
    ALL its base tokens have mask=1."""
    offset, out = 0, []
    for tok in compressed:
        span = len(cb_dict[tok]) if tok >= vocab else 1
        out.append(bool(np.all(base_mask[offset:offset + span])))
        offset += span
    return torch.BoolTensor(out)


@torch.no_grad()
def train_style_replay(lm, chunk, mask, seq_len, disabled_ids, tokenizer):
    """Exact data.py -> train.py recipe: fresh LZW over the 4096-token chunk,
    truncate to seq_len+1, forward with codebook + hyper_causal_mask, CE like train.py.
    Returns metrics dict or None if the chunk compresses too well (train would skip it)."""
    cfg = lm.cfg
    compressor = LZWCompressor(
        initial_vocab_size=cfg.vocab_size,
        max_codebook_size=cfg.max_codebook_size,
        max_subtokens=cfg.max_subtokens,
        pad_token_id=cfg.pad_token_id,
        disabled_ids=list(disabled_ids),
    )
    compressed, _, codebook = compressor.encode(chunk, padding="do_not_pad", truncation=False)
    if len(compressed) < seq_len + 1:
        return None
    compressed = compressed[: seq_len + 1]
    cb_dict = codebook.to_dict()
    n_base = sum(len(cb_dict[t]) if t >= cfg.vocab_size else 1 for t in compressed)

    x = torch.LongTensor(compressed[:-1]).unsqueeze(0).to(lm.device)
    y = torch.LongTensor(compressed[1:]).unsqueeze(0).to(lm.device)
    valid_frac = 1.0
    if mask is not None:
        lm_mask = compressed_loss_mask(compressed, mask, cb_dict, cfg.vocab_size)[1:]
        valid_frac = lm_mask.float().mean().item()
        if not lm_mask.any():
            return None
        y[0][~lm_mask.to(lm.device)] = -100
    recorded_active_k = lm.train_args.get("max_active_codebook_size")
    active_k = (
        cfg.max_codebook_size
        if recorded_active_k is None
        else int(recorded_active_k)
    )
    cb = (
        lm._codebook_to_tensor(codebook)[:active_k]
        .unsqueeze(0)
        .to(lm.device)
    )
    out_of_range_input = x >= cfg.vocab_size + active_k
    if out_of_range_input.any():
        bad_id = int(x[out_of_range_input][0].item())
        raise RuntimeError(
            f"train-style replay input hyper-token {bad_id} exceeds "
            f"max_active_codebook_size={active_k}"
        )
    codebook_counts = None
    online_skipped = 0
    if lm.online_codebook_mask:
        compressor_args = {
            "initial_vocab_size": cfg.vocab_size,
            "max_codebook_size": cfg.max_codebook_size,
            "max_subtokens": cfg.max_subtokens,
            "pad_token_id": cfg.pad_token_id,
            "disabled_ids": list(disabled_ids),
        }
        counts_cpu = online_codebook_counts(
            compressed[:-1], codebook, compressor_args
        )
        unavailable = online_unavailable_targets(
            y[0].cpu(), counts_cpu, cfg.vocab_size, cb.shape[1]
        )
        online_skipped = int(unavailable.sum().item())
        y[0][unavailable.to(lm.device)] = -100
        if not (y != -100).any():
            return None
        codebook_counts = counts_cpu.unsqueeze(0).to(lm.device)

    with torch.autocast(device_type=lm.device.type, dtype=lm._dtype):
        out = lm.model(
            x,
            codebook=cb,
            hyper_causal_mask=True,
            codebook_counts=codebook_counts,
        )
    logits = (out[0] if isinstance(out, tuple) else out).flatten(0, 1).float()
    labels = y.flatten(0, 1)
    per_tok = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
    valid = labels != -100
    ce_sum = per_tok.sum().item()
    n_target_base = sum(
        len(cb_dict[int(tok)]) if int(tok) >= cfg.vocab_size else 1
        for tok in labels[valid].tolist()
    )

    n_bytes = len(tokenizer.decode(chunk[:n_base], skip_special_tokens=False).encode("utf-8"))
    return dict(
        ce_sum=ce_sum,
        n_base=n_base,
        n_target_base=n_target_base,
        n_valid=int(valid.sum()),
        n_comp=len(compressed),
        n_bytes=n_bytes,
        valid_frac=valid_frac,
        online_skipped=online_skipped,
        acc=(logits[valid].argmax(-1) == labels[valid]).float().mean().item(),
    )


@torch.no_grad()
def eval_style_rolling(lm, ids, max_len):
    """The adapter's actual loglikelihood_rolling windowing over `ids`,
    scored via the real lm._score_compressed. Returns (nll_sum, n_scored_base)."""
    total, scored = 0.0, 0
    for prefix, pred in map(
        make_disjoint_window,
        get_rolling_token_windows(
            list(ids), prefix_token=lm.eot_token_id, max_seq_len=max_len, context_len=1
        ),
    ):
        full = list(prefix) + list(pred)
        if len(full) < 2:
            continue
        lp, _, _, n_base = lm._score_compressed(full, cont_start_base=len(prefix))
        total += -lp
        scored += n_base
    return total, scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--tokenizer", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--seq_len", type=int, default=2048,
                   help="Training seq_len in COMPRESSED tokens (chunk = 2*seq_len base).")
    p.add_argument("--n_chunks", type=int, default=8)
    p.add_argument("--start_frac", type=float, default=0.5,
                   help="Where in shard 0 to sample chunks (0.5 = middle; the 262M-token "
                        "run consumed only part of each shard, so this is near-held-out).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--output_json",
        default=None,
        help="Optional dual-denominator aggregate artifact path.",
    )
    p.add_argument(
        "--replay_only",
        action="store_true",
        help="Stop after train-style replay (useful for a quick denominator map).",
    )
    args = p.parse_args()

    log("=== step 0: LoRA adapter weight inspection (mmap, no GPU) ===")
    inspect_lora_weights(args.ckpt_dir)

    log("=== step 1: load checkpoint through the eval adapter (fold path) ===")
    lm = Zip2ZipLM(
        pretrained=args.ckpt_dir,
        tokenizer=args.tokenizer,
        max_length=1024,
        device=args.device,
        dtype=args.dtype,
        eval_mode="compressed",
    )
    log(f"cfg: vocab={lm.cfg.vocab_size} max_codebook={lm.cfg.max_codebook_size} "
        f"max_subtokens={lm.cfg.max_subtokens} pad={lm.cfg.pad_token_id}")
    replay_arg_keys = (
        "model_config", "seq_len", "max_subtokens", "max_codebook_size",
        "max_active_codebook_size", "lora_rank", "lora_alpha",
        "hyper_causal_mask", "base_token_positions", "two_axis_rope",
        "gated_compressed_rope", "gated_rope_start_layer",
        "gated_rope_start_pair",
        "base_view_replay_prob",
        "online_codebook_mask", "no_remap_codebook", "data_dir", "tokenizer",
    )
    replay_args = {key: lm.train_args.get(key) for key in replay_arg_keys}
    log(f"train_args from meta.pt: {replay_args}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    special = set(tokenizer.all_special_ids or [])
    added = set(tokenizer.get_added_vocab().values())
    train_disabled = {
        i for i in (special | added) if 0 <= i < lm.cfg.vocab_size
    }
    if lm.train_args.get("disable_digit_ids"):
        digit_pieces = {str(d) for d in range(10)} | {
            f"▁{d}" for d in range(10)
        }
        train_disabled.update(
            i for piece, i in tokenizer.get_vocab().items()
            if piece in digit_pieces and 0 <= i < lm.cfg.vocab_size
        )
    train_disabled = sorted(train_disabled)
    log(f"training disabled_ids ({len(train_disabled)}): {train_disabled}")
    log(f"adapter disabled_ids  ({len(lm._disabled_ids)}): {lm._disabled_ids} "
        f"(match={list(lm._disabled_ids) == train_disabled})")

    chunks = load_chunks(args.data_dir, args.n_chunks, args.seq_len * 2, args.start_frac)

    log("=== step 2: train-style replay (exact data.py/train.py recipe) ===")
    agg = dict(
        ce=0.0,
        base=0,
        target_base=0,
        valid=0,
        comp=0,
        bytes=0,
        online_skipped=0,
    )
    accs = []
    for i, (chunk, mask) in enumerate(chunks):
        m = train_style_replay(lm, chunk, mask, args.seq_len, train_disabled, tokenizer)
        if m is None:
            log(f"  chunk {i}: skipped (compressed < seq_len+1 or fully masked — train.py would skip too)")
            continue
        log(f"  chunk {i}: nats/legacy-base={m['ce_sum'] / m['n_base']:.4f} "
            f"nats/target-base={m['ce_sum'] / m['n_target_base']:.4f} "
            f"nats/valid_comp={m['ce_sum'] / max(m['n_valid'], 1):.4f} "
            f"nats/byte={m['ce_sum'] / m['n_bytes']:.4f} "
            f"acc={m['acc']:.3f} "
            f"compression_legacy={m['n_base'] / m['n_valid']:.3f} "
            f"compression_target={m['n_target_base'] / m['n_valid']:.3f} "
            f"assistant_valid_frac_before_online={m['valid_frac']:.3f} "
            f"online_skipped={m['online_skipped']}")
        agg["ce"] += m["ce_sum"]; agg["base"] += m["n_base"]
        agg["target_base"] += m["n_target_base"]; agg["valid"] += m["n_valid"]
        agg["comp"] += m["n_comp"]; agg["bytes"] += m["n_bytes"]
        agg["online_skipped"] += m["online_skipped"]; accs.append(m["acc"])

    if agg["base"] == 0:
        raise SystemExit("No scorable chunks — check DATA_DIR / start_frac.")

    train_nats_base = agg["ce"] / agg["base"]
    train_nats_target_base = agg["ce"] / agg["target_base"]
    train_nats_byte = agg["ce"] / agg["bytes"]
    log(f"TRAIN-STYLE REPLAY AGGREGATE: "
        f"nats/legacy-base-token={train_nats_base:.4f} "
        f"| nats/target-base-token={train_nats_target_base:.4f} "
        f"(W&B step-8000 'loss' was ~1.63) | nats/valid-comp-token={agg['ce'] / agg['valid']:.4f} "
        f"| bits/byte={train_nats_byte / math.log(2):.4f} | byte_ppl={math.exp(train_nats_byte):.4f} "
        f"| acc={sum(accs) / len(accs):.3f} (W&B acc ~0.5x) "
        f"| compression_legacy={agg['base'] / agg['valid']:.3f} "
        f"| compression_target={agg['target_base'] / agg['valid']:.3f} "
        f"| online_skipped={agg['online_skipped']}")
    dual_denominator = {
        "checkpoint": os.path.abspath(args.ckpt_dir),
        "data_dir": os.path.abspath(args.data_dir),
        "start_frac": args.start_frac,
        "requested_chunks": args.n_chunks,
        "scored_chunks": len(accs),
        "ce_sum": agg["ce"],
        "valid_compressed_targets": agg["valid"],
        "legacy_base_tokens": agg["base"],
        "target_base_tokens": agg["target_base"],
        "legacy_nats_per_base_token": train_nats_base,
        "target_nats_per_base_token": train_nats_target_base,
        "legacy_compression": agg["base"] / agg["valid"],
        "target_compression": agg["target_base"] / agg["valid"],
    }
    if args.output_json:
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(dual_denominator, f, indent=2, sort_keys=True)
            f.write("\n")
        log(f"wrote dual-denominator artifact: {args.output_json}")
    if args.replay_only:
        log("replay-only requested; skipping rolling-window and base-mode passes")
        return

    log("=== step 3: eval-style rolling windows on the SAME data (real adapter scorer) ===")
    for max_len in (1024, 2048):
        nll, scored = 0.0, 0
        n_bytes = 0
        for chunk, _ in chunks:
            t, s = eval_style_rolling(lm, chunk, max_len)
            nll += t; scored += s
            n_bytes += len(tokenizer.decode(chunk, skip_special_tokens=False).encode("utf-8"))
        log(f"  window={max_len}: nats/base-token={nll / scored:.4f} "
            f"bits/byte={nll / n_bytes / math.log(2):.4f} byte_ppl={math.exp(nll / n_bytes):.4f} "
            f"(eval log: wikitext byte_ppl 3.45, pile 3.83 @ window 1024)")

    log("=== step 4: base (uncompressed) mode on the same data — folded decoder only ===")
    nll, n_tok, n_bytes = 0.0, 0, 0
    for chunk, _ in chunks:
        ids = chunk[:4096]
        lp, _, n, _ = lm._score_base(ids, cont_start_base=1)
        nll += -lp; n_tok += n
        n_bytes += len(tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8"))
    log(f"  base-mode: nats/base-token={nll / n_tok:.4f} bits/byte={nll / n_bytes / math.log(2):.4f} "
        f"byte_ppl={math.exp(nll / n_bytes):.4f} (healthy Phi-3.5 ballpark: ~2.0-2.5 nats/tok)")

    log("=== verdict guide ===")
    log(f"train-style replay nats/base = {train_nats_base:.3f} vs W&B 1.63:")
    log("  ~1.6-1.9  -> ckpt healthy + eval forward healthy: the byte-ppl gap is windowing/"
        "methodology; compare step-3 numbers across window sizes.")
    log("  >>1.63 (e.g. 4+) -> loaded weights != trained state or forward-path defect; "
        "check step-0 LoRA norms and step-4 base-mode to localize decoder vs hyper path.")


if __name__ == "__main__":
    main()
