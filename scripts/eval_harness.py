"""Run lm-eval-harness over a Zip2Zip checkpoint.

Default tasks cover the standard pretraining suite:
    arc_challenge, arc_easy, hellaswag, openbookqa, piqa, winogrande,
    commonsense_qa, medqa_4options, wikitext  (built-in, loglikelihood)
    zip2zip_pile, zip2zip_mc4, zip2zip_dc4    (custom YAMLs, loglikelihood)

Generation tasks are also supported (gsm8k, humaneval, mbpp, ifeval) but are
opt-in via --tasks because they require generate_until and, for code tasks,
HF_ALLOW_CODE_EVAL=1 in env.

Usage:
    python scripts/eval_harness.py --ckpt_dir /path/to/step_6000
    python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 \\
        --tasks gsm8k,humaneval,mbpp,ifeval
    python scripts/eval_harness.py --hf_repo epfl-dlab/Llaza-3.2-1B-v0.1 --hf_revision step_6000

The checkpoint's `meta.pt` is read to recover the model config, max_subtokens,
max_codebook_size, and other architectural fields, so passing the right model
flags is not required.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

# Make src/ and torchtitan importable when run as a standalone script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))

from zip2zip_core.lm_eval_adapter import Zip2ZipLM  # noqa: E402  (registers "zip2zip")
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.tasks import TaskManager  # noqa: E402

DEFAULT_MC_TASKS = [
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "commonsense_qa",
    "medqa_4options",
]
DEFAULT_PPL_TASKS_BUILTIN = ["wikitext"]
DEFAULT_PPL_TASKS_CUSTOM = ["zip2zip_pile", "zip2zip_mc4", "zip2zip_dc4"]
DEFAULT_TASKS = DEFAULT_MC_TASKS #+ DEFAULT_PPL_TASKS_BUILTIN + DEFAULT_PPL_TASKS_CUSTOM


def _resolve_ckpt_dir(args: argparse.Namespace) -> str:
    if args.hf_repo:
        from huggingface_hub import hf_hub_download

        local_dir = None
        for fn in ("model.pt", "meta.pt"):
            try:
                p = hf_hub_download(args.hf_repo, fn, revision=args.hf_revision)
                if local_dir is None:
                    local_dir = os.path.dirname(p)
            except Exception:
                if fn == "model.pt":
                    raise
        return local_dir

    ckpt = args.ckpt_dir
    if ckpt is None:
        raise SystemExit("Must pass --ckpt_dir or --hf_repo.")
    if os.path.basename(os.path.normpath(ckpt)) == "latest":
        parent = os.path.dirname(os.path.normpath(ckpt))
        steps = glob.glob(os.path.join(parent, "step_*"))
        if not steps:
            raise SystemExit(f"No step_* directories under {parent}")
        steps.sort(key=lambda d: int(os.path.basename(d).split("_")[1]))
        ckpt = steps[-1]
        print(f"[eval_harness] Resolved 'latest' → {ckpt}")
    return ckpt


from load_preset import apply_preset as _apply_preset
from sample_logging import print_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", default=None,
                   help="Local checkpoint directory (or .../latest).")
    p.add_argument("--hf_repo", default=None, help="HuggingFace repo ID, e.g. user/repo.")
    p.add_argument("--hf_revision", default=None, help="HF revision/branch (e.g. step_6000).")
    p.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--tasks", default=None,
                   help=f"Comma-separated. Default: {','.join(DEFAULT_TASKS)}")
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--limit", type=float, default=None,
                   help="Per-task sample limit for quick smoke runs.")
    p.add_argument("--max_length", "--max_seq_len", dest="max_length", type=int, default=4096,
                   help="Max base-token context length per scoring call. Default 4096 "
                        "(matches the 4k training window for these checkpoints). "
                        "--max_seq_len is a deprecated alias.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--eval_mode", default="compressed", choices=["compressed", "base"],
                   help="'compressed': LZW-compress text, score compressed positions "
                        "(matches training distribution). 'base': vanilla LM scoring.")
    p.add_argument("--no_hyper_causal_mask", action="store_true",
                   help="Disable hyper_causal_mask in the model forward.")
    p.add_argument("--disable_digit_ids", action="store_true",
                   help="Diagnostic: add digit tokens to disabled_ids so numbers "
                        "are never LZW-merged into hypertokens (compressed mode).")
    p.add_argument("--disable_mathsym_ids", action="store_true",
                   help="Diagnostic: also keep math operators/symbols (=+-*/%%$^<>) "
                        "out of LZW merges — triage for extending the protected set.")
    p.add_argument("--include_path", default=None,
                   help="Directory of custom task YAMLs. "
                        "Defaults to scripts/lm_eval_tasks/ next to this script.")
    p.add_argument("--output_path", default=None,
                   help="If set, write results JSON here.")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project name. Defaults to WANDB_PROJECT from project.py.")
    p.add_argument("--wandb_name", default=None, help="W&B run name.")
    p.add_argument("--resume_wandb_id", type=str, required=True,
                   help="W&B run ID to log eval results into (e.g. '8d11iyds'). "
                        "Pass 'none' to create a new run instead.")
    p.add_argument("--no_log_samples", action="store_true",
                   help="Disable logging per-sample results to W&B.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--preset", default=None,
                   help="Named evaluation preset from the presets YAML file.")
    p.add_argument("--preset_file", default=None,
                   help="Path to presets YAML file. "
                        "Default: scripts/eval_presets.yaml next to this script.")

    preset_info = _apply_preset(p)
    args = p.parse_args()

    ckpt_dir = _resolve_ckpt_dir(args)
    tasks = (
        DEFAULT_TASKS
        if args.tasks is None
        else [t.strip() for t in args.tasks.split(",") if t.strip()]
    )
    include_path = args.include_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lm_eval_tasks"
    )
    if not os.path.isdir(include_path):
        include_path = None

    lm = Zip2ZipLM(
        pretrained=ckpt_dir,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
        eval_mode=args.eval_mode,
        batch_size=args.batch_size,
        hyper_causal_mask=not args.no_hyper_causal_mask,
        disable_digit_ids=args.disable_digit_ids,
        disable_mathsym_ids=args.disable_mathsym_ids,
    )
    # The adapter auto-enables digit protection for checkpoints trained with it
    # and auto-switches control checkpoints (max_codebook_size=0) to base mode —
    # reflect the effective settings so the results JSON records what actually ran.
    args.disable_digit_ids = lm.disable_digit_ids
    args.disable_mathsym_ids = lm.disable_mathsym_ids
    args.eval_mode = lm.eval_mode
    args.online_codebook_mask = lm.online_codebook_mask
    args.online_codebook_mask_active = lm.online_codebook_mask_active

    if preset_info:
        print(f"[eval_harness] preset: {preset_info[0]} — {preset_info[1]}")
    print(f"[eval_harness] checkpoint:   {ckpt_dir}")
    print(f"[eval_harness] tasks:        {tasks}")
    print(f"[eval_harness] eval_mode:    {args.eval_mode}")
    print(
        f"[eval_harness] online mask:  checkpoint={args.online_codebook_mask} "
        f"active={args.online_codebook_mask_active}"
    )
    print(f"[eval_harness] include_path: {include_path}")

    task_manager = TaskManager(include_path=include_path) if include_path else TaskManager()

    eval_start = time.perf_counter()
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
        log_samples=not args.no_log_samples,
        apply_chat_template=getattr(args, 'apply_chat_template', False),
        fewshot_as_multiturn=getattr(args, 'fewshot_as_multiturn', False),
    )
    eval_wall_seconds = time.perf_counter() - eval_start

    print("\n" + "=" * 72)
    print("Results:")
    print(json.dumps(results.get("results", results), indent=2, default=str))
    print("=" * 72)

    compression = lm.compression_summary()
    print("Compression (base tokens per compressed token, >1 = more compression):")
    print(json.dumps(compression, indent=2))
    print(f"Evaluation wall time: {eval_wall_seconds:.1f}s")
    print("=" * 72)

    if not args.no_log_samples:
        print_samples(results)

    if not args.no_wandb:
        from zip2zip_core.project import WANDB_ENTITY, WANDB_PROJECT
        from lm_eval.loggers import WandbLogger
        import wandb

        if args.wandb_name:
            wandb_name = f"eval-{args.wandb_name}"
        elif args.hf_repo:
            repo_short = args.hf_repo.split("/")[-1]
            rev = args.hf_revision or "main"
            wandb_name = f"eval-{repo_short}-{rev}"
        else:
            ckpt_name = os.path.basename(os.path.normpath(ckpt_dir))
            wandb_name = f"eval-{ckpt_name}"

        resume_id = args.resume_wandb_id
        if resume_id and resume_id.lower() != "none":
            wandb.init(
                entity=WANDB_ENTITY,
                project=args.wandb_project or WANDB_PROJECT,
                id=resume_id,
                resume="must",
                tags=["eval"],
                # Nested under one key: when resuming a TRAINING run (the
                # finetune->eval pipeline), flat eval args could collide with
                # same-named training config keys (seed, tokenizer, ...).
                config={"eval_args": vars(args)},
            )
        else:
            wandb.init(
                entity=WANDB_ENTITY,
                project=args.wandb_project or WANDB_PROJECT,
                name=wandb_name,
                job_type="eval",
                tags=["eval"],
                config=vars(args),
            )
        wandb_logger = WandbLogger()
        if "versions" in results:
            results["versions"] = {k: str(v) for k, v in results["versions"].items()}
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        eval_stats = {
            f"eval/{k}": v
            for k, v in compression.items()
            if (
                k.endswith("_ratio")
                or k == "online_skipped_targets"
                or k.startswith("online_replay_")
            )
        }
        eval_stats["eval/wall_seconds"] = eval_wall_seconds
        wandb.log(eval_stats)
        if not args.no_log_samples and "samples" in results:
            wandb_logger.log_eval_samples(results["samples"])
        print(f"[eval_harness] Results logged to W&B: {wandb_logger.run.url}")

    if args.output_path:
        out_dir = os.path.dirname(os.path.abspath(args.output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(
                {
                    "results": results.get("results"),
                    "configs": results.get("configs"),
                    "compression": compression,
                    "eval_wall_seconds": eval_wall_seconds,
                    "ckpt_dir": ckpt_dir,
                    "tasks": tasks,
                    "args": vars(args),
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Saved results JSON to {args.output_path}")


if __name__ == "__main__":
    main()
