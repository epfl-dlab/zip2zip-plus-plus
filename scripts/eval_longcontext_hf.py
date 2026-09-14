"""Evaluate the official HF baseline on RULER with its native LongRoPE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from eval_harness import _use_ruler_hotpot_mirror
from load_preset import apply_preset


def _parse_lengths(value: str | None) -> list[int]:
    return [int(item.strip()) for item in (value or "").split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True)
    parser.add_argument("--preset_file")
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ruler_lengths", default=None)
    parser.add_argument("--force_greedy", action="store_true")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_log_samples", action="store_true")
    preset_info = apply_preset(parser)
    args = parser.parse_args()

    if args.preset != "longcontext_ruler":
        raise ValueError("this entry point only accepts longcontext_ruler")
    if not args.tasks:
        raise ValueError(f"preset {args.preset!r} has no tasks")
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    tokenizer = args.tokenizer or args.model
    metadata = None
    ruler_lengths = _parse_lengths(args.ruler_lengths)
    if ruler_lengths:
        _use_ruler_hotpot_mirror()
        metadata = {"max_seq_lengths": ruler_lengths, "tokenizer": tokenizer}

    print(f"[eval_longcontext_hf] preset: {preset_info}")
    print(f"[eval_longcontext_hf] model: {args.model}")
    print(f"[eval_longcontext_hf] tasks: {tasks}")
    print(f"[eval_longcontext_hf] RULER lengths: {ruler_lengths or '<none>'}")
    print("[eval_longcontext_hf] RoPE: official model config (LongRoPE)")

    lm = HFLM(
        pretrained=args.model,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    rope_scaling = getattr(lm.model.config, "rope_scaling", None)
    rope_type = (
        rope_scaling.get("rope_type", rope_scaling.get("type"))
        if isinstance(rope_scaling, dict)
        else None
    )
    if rope_type != "longrope":
        raise ValueError(
            f"baseline must use official LongRoPE, got rope_scaling={rope_scaling!r}"
        )
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        task_manager=TaskManager(metadata=metadata),
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        log_samples=not args.no_log_samples,
        apply_chat_template=getattr(args, "apply_chat_template", False),
        fewshot_as_multiturn=getattr(args, "fewshot_as_multiturn", False),
        gen_kwargs=(
            {"do_sample": False, "temperature": 0.0}
            if args.force_greedy
            else None
        ),
    )

    payload = {
        "results": results.get("results"),
        "configs": results.get("configs"),
        "versions": results.get("versions"),
        "config": results.get("config"),
        "model": args.model,
        "tasks": tasks,
        "args": vars(args),
        "rope_mode": "official-phi3-longrope",
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(results.get("results", results), indent=2, default=str))
    print(f"[eval_longcontext_hf] saved: {output}")


if __name__ == "__main__":
    main()
