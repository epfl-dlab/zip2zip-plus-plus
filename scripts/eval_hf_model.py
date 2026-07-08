"""Evaluate a released zip2zip HF model (adapter format) via the zip2zip package.

This uses Zip2ZipModel.from_pretrained() which handles the PEFT adapter +
zip2zip encoder loading from HuggingFace repos like
epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1.

Usage:
    python scripts/eval_hf_model.py \
        --model epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1 \
        --tasks arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,gsm8k \
        --num_fewshot 2 --limit 20
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip.tools.harness import Zip2ZipForLMEval
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager


from load_preset import apply_preset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="default")
    p.add_argument("--preset_file", default=None)
    p.add_argument("--model", default="epfl-dlab/zip2zip-Phi-3.5-mini-instruct-v0.1")
    p.add_argument("--tasks",
                   default="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,winogrande,gsm8k")
    p.add_argument("--num_fewshot", type=int, default=2)
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_path", default=None)
    p.add_argument("--include_path", default=None,
                   help="Directory of custom task YAMLs. "
                        "Default: scripts/lm_eval_tasks/ next to this script.")
    p.add_argument("--seed", type=int, default=1234)

    preset_info = apply_preset(p, default="default")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    include_path = args.include_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lm_eval_tasks"
    )
    if not os.path.isdir(include_path):
        include_path = None

    if preset_info:
        print(f"[eval_hf] preset:      {preset_info[0]} — {preset_info[1]}")
    print(f"[eval_hf] model:       {args.model}")
    print(f"[eval_hf] tasks:       {tasks}")
    print(f"[eval_hf] num_fewshot: {args.num_fewshot}")
    print(f"[eval_hf] limit:       {args.limit}")
    print(f"[eval_hf] max_length:  {args.max_length}")

    print(f"[eval_hf] Loading model...")
    model = Zip2ZipModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    tokenizer = Zip2ZipTokenizer.from_pretrained(args.model)

    lm = Zip2ZipForLMEval(
        model, tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    task_manager = TaskManager(include_path=include_path) if include_path else TaskManager()

    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        task_manager=task_manager,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        apply_chat_template=getattr(args, 'apply_chat_template', True),
        fewshot_as_multiturn=getattr(args, 'fewshot_as_multiturn', True),
    )

    print("\n" + "=" * 72)
    print("Results:")
    print(json.dumps(results.get("results", results), indent=2, default=str))
    print("=" * 72)

    compression = lm.compression_summary()
    print("Compression (base tokens per compressed token, >1 = more compression):")
    print(json.dumps(compression, indent=2))
    print("=" * 72)

    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(
                {
                    "results": results.get("results"),
                    "configs": results.get("configs"),
                    "compression": compression,
                    "model": args.model,
                    "tasks": tasks,
                    "args": vars(args),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[eval_hf] Saved to {args.output_path}")


if __name__ == "__main__":
    main()
