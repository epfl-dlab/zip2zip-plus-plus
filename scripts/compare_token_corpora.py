"""Compare two pretokenized corpora by measuring the bytes actually on disk.

Every number this prints is measured from the shard files, not inferred from a
directory name or a recipe. Both corpora are put through the identical LZW
configuration, so any difference in compression ratio is a property of the data.

Usage:
    python scripts/compare_token_corpora.py \
        --dirs /path/to/corpus_a /path/to/corpus_b \
        --tokenizer microsoft/Phi-3.5-mini-instruct \
        --disable_digit_ids
"""

import argparse
import json
import os
import re

import numpy as np

CODE_MARKERS = ("def ", "function ", "#include", "import ", "};", "</", "printf",
                "public class", "return ", "$ ", "()", "=>")
MATH_MARKERS = ("\\frac", "\\begin{", "$$", "\\sum", "\\int", "\\sqrt", "\\cdot")


def classify(text):
    """Crude content bucket for one document. A heuristic, not a ground truth."""
    if sum(1 for ch in text if ord(ch) > 0x24F) > 0.10 * max(len(text), 1):
        return "non_latin"
    if any(m in text for m in MATH_MARKERS):
        return "math_markup"
    hits = sum(1 for m in CODE_MARKERS if m in text)
    punct = sum(1 for ch in text if ch in "{};<>") / max(len(text), 1)
    if hits >= 2 or punct > 0.02:
        return "code_like"
    return "prose"


def load_manifest(d):
    p = os.path.join(d, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def segment_lengths(ids, eos_id):
    """Token counts of the complete eos-terminated segments inside one window."""
    ends = np.flatnonzero(ids == eos_id)
    if ends.size < 2:
        return np.array([], dtype=np.int64)
    return np.diff(ends)


def analyse(d, tok, args, compressor, initial_vocab):
    out = {"dir": d, "name": os.path.basename(d)}

    shards = sorted(f for f in os.listdir(d)
                    if f.startswith("shard_") and f.endswith(".npy"))
    masks = sorted(f for f in os.listdir(d)
                   if f.startswith("mask_") and f.endswith(".npy"))
    out["num_shards"] = len(shards)
    out["num_masks"] = len(masks)
    out["manifest"] = load_manifest(d)

    total = 0
    for f in shards:
        total += int(np.load(os.path.join(d, f), mmap_mode="r").shape[0])
    out["total_tokens"] = total
    out["tokens_per_shard"] = int(
        np.load(os.path.join(d, shards[0]), mmap_mode="r").shape[0]
    )

    # Sample a few shards spread across the corpus rather than only the first.
    picks = sorted(set(
        int(round(i * (len(shards) - 1) / max(args.sample_shards - 1, 1)))
        for i in range(args.sample_shards)
    ))
    out["sampled_shards"] = [shards[i] for i in picks]

    seg_lens, vocab_counts = [], np.zeros(initial_vocab, dtype=np.int64)
    windows, buckets, excerpts = [], {}, []
    mask_in_loss = []
    chars, chars_tokens = 0, 0
    base_len = args.seq_len * 2

    for si in picks:
        arr = np.load(os.path.join(d, shards[si]), mmap_mode="r")
        block = np.asarray(arr[: args.scan_tokens])
        seg_lens.append(segment_lengths(block, tok.eos_token_id))
        vocab_counts += np.bincount(block, minlength=initial_vocab)[:initial_vocab]

        for k in range(args.windows // len(picks)):
            start = k * base_len
            if start + base_len > block.size:
                break
            windows.append(np.asarray(block[start:start + base_len]).tolist())

        # Documents for decoding: slice between consecutive eos markers.
        ends = np.flatnonzero(block == tok.eos_token_id)
        for j in range(min(args.docs_per_shard, max(ends.size - 1, 0))):
            seg = block[ends[j] + 1: ends[j + 1] + 1]
            text = tok.decode(seg.tolist(), skip_special_tokens=True)
            if text.strip():
                buckets[classify(text)] = buckets.get(classify(text), 0) + 1
                if len(excerpts) < args.excerpts:
                    excerpts.append((shards[si], len(seg), text[:400]))

        # chars/token on a contiguous slice, a proxy for tokenizer efficiency.
        probe = block[: args.probe_tokens].tolist()
        chars += len(tok.decode(probe, skip_special_tokens=True))
        chars_tokens += len(probe)

        if masks:
            m = np.load(os.path.join(d, "mask_" + shards[si].split("_", 1)[1]),
                        mmap_mode="r")
            mask_in_loss.append(float(np.asarray(m[: args.scan_tokens]).mean()))

    lens = np.concatenate([s for s in seg_lens if s.size]) if any(
        s.size for s in seg_lens) else np.array([0])
    out["doc_len"] = {
        "mean": float(lens.mean()),
        "p10": float(np.percentile(lens, 10)),
        "median": float(np.median(lens)),
        "p90": float(np.percentile(lens, 90)),
        "max": int(lens.max()),
    }
    out["distinct_ids_used"] = int(np.count_nonzero(vocab_counts))
    top = np.argsort(vocab_counts)[::-1][:12]
    out["top_tokens"] = [
        (int(i), tok.decode([int(i)]), int(vocab_counts[i])) for i in top
    ]
    out["chars_per_token"] = chars / max(chars_tokens, 1)
    out["content_mix"] = buckets
    out["excerpts"] = excerpts
    out["mask_fraction_in_loss"] = (
        float(np.mean(mask_in_loss)) if mask_in_loss else None
    )

    # Identical LZW settings on both corpora: the ratio difference is the data.
    ratios = []
    step = 32
    for i in range(0, len(windows), step):
        batch = windows[i:i + step]
        try:
            comp, _, _ = compressor.batch_encode(
                batch, padding="do_not_pad", truncation=False, max_length=None
            )
        except TypeError:
            comp, _, _ = compressor.batch_encode(batch)
        for src, dst in zip(batch, comp):
            dst = [t for t in list(dst) if t != tok.pad_token_id]
            if dst:
                ratios.append(len(src) / len(dst))
    r = np.array(ratios) if ratios else np.array([0.0])
    out["lzw"] = {
        "windows": len(ratios),
        "mean_ratio": float(r.mean()),
        "p10": float(np.percentile(r, 10)),
        "median": float(np.median(r)),
        "p90": float(np.percentile(r, 90)),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs=2, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--max_codebook_size", type=int, default=4096)
    ap.add_argument("--max_subtokens", type=int, default=4)
    ap.add_argument("--disable_digit_ids", action="store_true")
    ap.add_argument("--sample_shards", type=int, default=3)
    ap.add_argument("--scan_tokens", type=int, default=20_000_000)
    ap.add_argument("--probe_tokens", type=int, default=200_000)
    ap.add_argument("--windows", type=int, default=192)
    ap.add_argument("--docs_per_shard", type=int, default=200)
    ap.add_argument("--excerpts", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from zip2zip_compression import LZWCompressor
    from zip2zip_core.disabled_ids import compute_disabled_ids

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    initial_vocab = len(tok)
    disabled = compute_disabled_ids(tok, initial_vocab,
                                    disable_digit_ids=args.disable_digit_ids)
    compressor = LZWCompressor(
        initial_vocab_size=initial_vocab,
        max_codebook_size=args.max_codebook_size,
        max_subtokens=args.max_subtokens,
        pad_token_id=tok.pad_token_id,
        disabled_ids=disabled,
    )
    print(f"tokenizer={args.tokenizer} vocab={initial_vocab} "
          f"bos={tok.bos_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id}")
    print(f"LZW: max_codebook_size={args.max_codebook_size} "
          f"max_subtokens={args.max_subtokens} disabled_ids={len(disabled)} "
          f"(digits={'on' if args.disable_digit_ids else 'off'})")
    print(f"window = seq_len*2 = {args.seq_len * 2} base tokens\n")

    reports = [analyse(d, tok, args, compressor, initial_vocab) for d in args.dirs]

    for r in reports:
        print(f"\n{'=' * 70}\n{r['name']}\n{'=' * 70}")
        print(f"shards:               {r['num_shards']} "
              f"({r['tokens_per_shard']} tokens each)")
        print(f"mask files:           {r['num_masks']}")
        print(f"total tokens:         {r['total_tokens']} "
              f"({r['total_tokens']/1e9:.3f}B)")
        if r["manifest"]:
            print(f"manifest:             {json.dumps(r['manifest'])}")
        print(f"sampled shards:       {r['sampled_shards']}")
        d = r["doc_len"]
        print(f"document length:      mean {d['mean']:.0f}  p10 {d['p10']:.0f}  "
              f"median {d['median']:.0f}  p90 {d['p90']:.0f}  max {d['max']}")
        print(f"distinct ids used:    {r['distinct_ids_used']} of {initial_vocab}")
        print(f"chars per token:      {r['chars_per_token']:.3f}")
        if r["mask_fraction_in_loss"] is not None:
            print(f"tokens in loss:       {r['mask_fraction_in_loss']*100:.1f}%")
        print(f"content mix (heuristic, {sum(r['content_mix'].values())} docs): "
              f"{r['content_mix']}")
        print(f"top tokens:           "
              f"{[(t, c) for _, t, c in r['top_tokens'][:8]]}")
        l = r["lzw"]
        print(f"LZW ratio:            mean {l['mean_ratio']:.4f}  p10 {l['p10']:.4f}"
              f"  median {l['median']:.4f}  p90 {l['p90']:.4f}  "
              f"({l['windows']} windows)")
        for shard, n, text in r["excerpts"]:
            print(f"\n  --- {shard}, {n} tokens ---")
            print("  " + re.sub(r"\s+", " ", text)[:300])

    a, b = reports
    print(f"\n{'=' * 70}\nSIDE BY SIDE\n{'=' * 70}")
    print(f"{'metric':<26}{a['name'][:20]:>21}{b['name'][:20]:>21}")
    rows = [
        ("total tokens (B)", f"{a['total_tokens']/1e9:.3f}", f"{b['total_tokens']/1e9:.3f}"),
        ("shards", a["num_shards"], b["num_shards"]),
        ("has loss masks", bool(a["num_masks"]), bool(b["num_masks"])),
        ("mean doc length", f"{a['doc_len']['mean']:.0f}", f"{b['doc_len']['mean']:.0f}"),
        ("median doc length", f"{a['doc_len']['median']:.0f}", f"{b['doc_len']['median']:.0f}"),
        ("distinct ids used", a["distinct_ids_used"], b["distinct_ids_used"]),
        ("chars per token", f"{a['chars_per_token']:.3f}", f"{b['chars_per_token']:.3f}"),
        ("LZW mean ratio", f"{a['lzw']['mean_ratio']:.4f}", f"{b['lzw']['mean_ratio']:.4f}"),
    ]
    for name, x, y in rows:
        print(f"{name:<26}{str(x):>21}{str(y):>21}")


if __name__ == "__main__":
    main()
