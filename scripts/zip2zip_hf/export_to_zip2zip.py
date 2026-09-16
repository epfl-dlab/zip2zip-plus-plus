"""Export a zip2zip-core checkpoint to ext/zip2zip format (CLI wrapper).

Usage:
    python scripts/zip2zip_hf/export_to_zip2zip.py \\
        --ckpt_dir /path/to/step_6000 \\
        --output_dir /path/to/export \\
        --base_model meta-llama/Llama-3.1-8B \\
        --model_config 1B
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

from zip2zip_core.export import export


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="zip2zip-core checkpoint directory (contains model.pt)")
    p.add_argument("--output_dir", required=True, help="Destination directory for ext/zip2zip format")
    p.add_argument("--base_model", default=None,
                   help="HuggingFace base model name, e.g. meta-llama/Llama-3.1-8B")
    p.add_argument("--model_config", default=None,
                   help="zip2zip-core model config key (e.g. '1B'). "
                        "If omitted, n_heads/n_kv_heads are auto-detected from weight shapes.")
    p.add_argument("--encoder_n_heads", type=int, default=None,
                   help="Number of encoder attention heads. Default: resolved from "
                        "meta.pt's train args, falling back to hidden_size // 64 "
                        "only if meta.pt is missing or lacks encoder_n_heads.")
    p.add_argument("--max_codebook_size", type=int, default=4096)
    p.add_argument("--disabled_ids", type=str, default=None,
                   help="Comma-separated list of token IDs to disable for LZW compression. "
                        "If omitted, derived from the base model tokenizer (honoring "
                        "--disable_digit_ids / meta.pt's auto-detected value).")
    p.add_argument("--disable_digit_ids", action="store_true", default=None,
                   help="Digits were protected from LZW merges during training. "
                        "Default: auto-detected from meta.pt's --disable_digit_ids.")
    p.add_argument("--no_disable_digit_ids", dest="disable_digit_ids", action="store_false",
                   help="Force digit protection off even if meta.pt recorded it on.")
    p.add_argument("--residual", action="store_true", default=None,
                   help="Force encoder residual on. Default: restore from meta.pt.")
    p.add_argument("--no_residual", dest="residual", action="store_false")
    p.add_argument("--causal", action="store_true", default=False)
    p.add_argument("--max_shard_size", default="5GB")
    args = p.parse_args()

    export(
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        model_config=args.model_config,
        encoder_n_heads=args.encoder_n_heads,
        max_codebook_size=args.max_codebook_size,
        disabled_ids=[int(x) for x in args.disabled_ids.split(",")] if args.disabled_ids else None,
        disable_digit_ids=args.disable_digit_ids,
        residual=args.residual,
        causal=args.causal,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
