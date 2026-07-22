"""Shared mechanics for the hyper-encoder invariant tests.

Each test file keeps its own small_cfg() — the config under test IS the
experiment — and imports the model-call contract, codebook fixture layout,
and check harness from here so they exist exactly once.

Importing this module also puts src/ and ext/torchtitan on sys.path.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg, seed=0):
    torch.manual_seed(seed)
    model = cfg.build()
    with torch.no_grad():
        model.init_weights()
    model = model.to(device=DEV, dtype=torch.float32).eval()
    # force the CPU/GPU-safe padded attention path (flash-attn varlen is CUDA-only)
    model.hyper_encoder.disable_varlen = True
    if getattr(model, "hyper_output", None) is not None:
        model.hyper_output.disable_varlen = True
    return model


def make_input(cfg, n_hyper=3, T=16):
    V = cfg.vocab_size
    torch.manual_seed(42)
    # tokens: mostly base, a few hypertokens referencing entries 0..n_hyper-1
    toks = torch.randint(0, V, (1, T), device=DEV)
    for i in range(n_hyper):
        toks[0, 2 + i] = V + i
    # codebook: n_hyper real entries (2 base subtokens + pad), rest all-pad
    pad = cfg.pad_token_id
    K = cfg.max_codebook_size
    cb = torch.full((1, K, cfg.max_subtokens), pad, device=DEV)
    for i in range(n_hyper):
        cb[0, i, 0] = 10 + i
        cb[0, i, 1] = 20 + i
    return toks, cb


def fwd(model, toks, cb):
    with torch.no_grad():
        out = model(toks, codebook=cb, hyper_causal_mask=True)
    return (out[0] if isinstance(out, tuple) else out).float()


def make_harness():
    """Return (results, check): a fresh result dict and its recorder."""
    results = {}

    def check(name, cond, detail=""):
        results[name] = bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    return results, check


def summarize(results):
    """Print the summary block; return the list of failed invariant names."""
    print("\n==== SUMMARY ====")
    n_pass = sum(results.values())
    print(f"{n_pass}/{len(results)} passed")
    failed = [k for k, v in results.items() if not v]
    for k in failed:
        print(f"  FAILED: {k}")
    return failed
