"""Run lm-eval-harness over a Zip2Zip checkpoint.

Default tasks cover the standard pretraining suite:
    arc_challenge, arc_easy, hellaswag, openbookqa, piqa, winogrande,
    commonsense_qa, medqa_4options, wikitext  (built-in, loglikelihood)
    zip2zip_pile, zip2zip_mc4, zip2zip_dc4    (custom YAMLs, loglikelihood)

Generation tasks are also supported (gsm8k, triviaqa, humaneval, mbpp, ifeval)
but are opt-in via --tasks/--preset because they require generate_until. Code
tasks execute model-written code: they need HF_ALLOW_CODE_EVAL=1 in the
environment (eval_ckpt_rcp.sh exports it for the `postsft` preset), and
lm-eval's confirm_run_unsafe_code is passed from that same variable.

Usage:
    python scripts/eval_harness.py --ckpt_dir /path/to/step_6000
    python scripts/eval_harness.py --ckpt_dir /path/to/step_6000 \\
        --tasks gsm8k,humaneval,mbpp,ifeval
    python scripts/eval_harness.py --hf_repo epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct --hf_revision main

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
from zip2zip_core.multi_view import derive_multi_view_metrics  # noqa: E402
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

_RULER_HOTPOT_MIRROR = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
    "7e54db4656209750ff487f6fdf8e39a66dba136b/"
    "hotpot_dev_distractor_v1.json"
)


def _use_ruler_hotpot_mirror() -> None:
    """Backport the pinned lm-eval 0.4.13 mirror to the 0.4.9 task."""
    from functools import cache
    from pathlib import Path

    from lm_eval import utils as lm_eval_utils
    from lm_eval.tasks.ruler import qa_utils

    def pin_callback(callback):
        namespace = callback.__globals__
        current = namespace["read_hotpotqa"]
        if getattr(current, "_z2z_hf_mirror", False):
            return
        original = getattr(current, "__wrapped__", current)

        @cache
        def read_hotpotqa():
            return original(_RULER_HOTPOT_MIRROR)

        setattr(read_hotpotqa, "_z2z_hf_mirror", _RULER_HOTPOT_MIRROR)
        namespace["read_hotpotqa"] = read_hotpotqa

    # Keep direct package imports correct as well.
    pin_callback(qa_utils.get_hotpotqa)

    # lm-eval 0.4.9 does not reuse that imported module when resolving a
    # YAML !function. It executes qa_utils.py into a fresh module namespace,
    # so patch each freshly loaded Hotpot callback at the loader boundary.
    current_import = lm_eval_utils.import_function
    if not getattr(current_import, "_z2z_ruler_hotpot_hook", False):
        original_import = current_import

        def import_function(loader, node, yaml_path):
            callback = original_import(loader, node, yaml_path)
            if (
                Path(yaml_path).name == "qa_hotpot.yaml"
                and callback.__name__ == "get_hotpotqa"
            ):
                pin_callback(callback)
            return callback

        setattr(import_function, "_z2z_ruler_hotpot_hook", True)
        lm_eval_utils.import_function = import_function
    print(f"[eval_harness] RULER HotpotQA source: {_RULER_HOTPOT_MIRROR}")


def _parse_ruler_lengths(value: str | None) -> list[int]:
    if not value:
        return []
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"invalid RULER lengths: {value!r}")
    if lengths != sorted(set(lengths)):
        raise ValueError(
            "RULER lengths must be unique and strictly increasing, got "
            f"{lengths}"
        )
    return lengths


def _resolve_ckpt_dir(args: argparse.Namespace) -> str:
    if args.hf_repo:
        from huggingface_hub import hf_hub_download

        local_dir = None
        for fn in ("model.pt", "meta.pt"):
            p = hf_hub_download(args.hf_repo, fn, revision=args.hf_revision)
            if local_dir is None:
                local_dir = os.path.dirname(p)
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
    p.add_argument("--num_fewshot", type=int, default=None,
                   help="Global few-shot count forced on EVERY task (the MC and "
                        "perplexity presets set it explicitly). Default None "
                        "keeps each task's own protocol, e.g. math500's four "
                        "fixed Minerva exemplars next to 0-shot ifeval.")
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
    p.add_argument("--rope_mode", default="checkpoint",
                   choices=["checkpoint", "phi3-longrope"],
                   help="Inference-only RoPE adaptation. phi3-longrope loads the "
                        "official factors from the checkpoint meta.pt init_from_hf model.")
    p.add_argument("--rope_source", default=None,
                   help="Optional HF config used only as the source of LongRoPE "
                        "factors. Model weights and tokenizer still come from the "
                        "checkpoint. Intended for explicit inference-only "
                        "extrapolation, e.g. Phi-3-medium-128k factors on a "
                        "Phi-3-medium-4k checkpoint.")
    p.add_argument("--fail_on_truncation", action="store_true",
                   help="Abort instead of silently left-truncating an overlength request.")
    p.add_argument("--no_kv_cache", action="store_true",
                   help="Generate by re-running the full prefix for every sampled "
                        "token instead of caching keys and values. The pre-2026-09 "
                        "behaviour; keep it only for A/B checks, it costs one "
                        "prefill per generated token.")
    p.add_argument("--ruler_lengths", default=None,
                   help="Comma-separated RULER lengths, e.g. "
                        "4096,8192,16384,32768,65536,131072.")
    p.add_argument("--force_greedy", action="store_true",
                   help="Override task sampling settings with deterministic greedy decoding.")
    p.add_argument("--eval_max_subtokens", type=int, default=None,
                   help="Eval-only override for LZW max_subtokens. Unset follows "
                        "the checkpoint meta.pt; changing it is supported for "
                        "hierarchical hyper-encoders only.")
    p.add_argument("--no_hyper_causal_mask", action="store_true",
                   help="Disable hyper_causal_mask in the model forward.")
    p.add_argument("--no_online_codebook_mask", action="store_true",
                   help="For a v0.6.5 checkpoint, score with the legacy k<=t mask "
                        "instead of the exact decoder-time mask it was trained with. "
                        "Use this to compare against v0.1-v0.6.4 numbers, which were "
                        "all produced with the legacy mask. No effect on older "
                        "checkpoints or in base mode.")
    p.add_argument("--no_multi_view", action="store_true",
                   help="Disable the multi-view perplexity columns. On by "
                        "default: rolling-perplexity tasks additionally report "
                        "multi_view_* metrics using the exact complete-"
                        "segmentation marginal. Strict metrics are never "
                        "affected either way.")
    p.add_argument("--legacy_untrimmed_stops", action="store_true",
                   help="Return generated text without cutting it at the first "
                        "stop-string occurrence, as all evals did before 2026-08. "
                        "Only for bit-exact reproduction of those results — wrong "
                        "for text-scored tasks like triviaqa.")
    p.add_argument("--legacy_stripped_generation", action="store_true",
                   help="Decode generations standalone, as all evals did before "
                        "2026-09: SentencePiece tokenizers (Phi) then lose one "
                        "leading space, which breaks HumanEval indentation. Only "
                        "for bit-exact reproduction of old generation samples.")
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
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity. Defaults to WANDB_ENTITY from the environment, "
                        "then the shared project constant.")
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
        # None = follow the checkpoint; False = force the legacy mask for a
        # like-for-like comparison with pre-v0.6.5 numbers.
        online_codebook_mask=False if args.no_online_codebook_mask else None,
        disable_digit_ids=args.disable_digit_ids,
        disable_mathsym_ids=args.disable_mathsym_ids,
        eval_max_subtokens=args.eval_max_subtokens,
        trim_stop_strings=not args.legacy_untrimmed_stops,
        multi_view=not args.no_multi_view,
        preserve_leading_space=not args.legacy_stripped_generation,
        rope_mode=args.rope_mode,
        rope_source=args.rope_source,
        fail_on_truncation=args.fail_on_truncation,
        use_kv_cache=not args.no_kv_cache,
    )
    # The adapter auto-enables digit protection for checkpoints trained with it
    # and auto-switches control checkpoints (max_codebook_size=0) to base mode —
    # reflect the effective settings so the results JSON records what actually ran.
    args.disable_digit_ids = lm.disable_digit_ids
    args.disable_mathsym_ids = lm.disable_mathsym_ids
    args.eval_mode = lm.eval_mode
    args.checkpoint_max_subtokens = lm.checkpoint_max_subtokens
    args.eval_max_subtokens = lm.eval_max_subtokens
    args.base_token_positions = lm.cfg.base_token_positions
    args.two_axis_rope = lm.cfg.two_axis_rope
    args.gated_compressed_rope = bool(
        getattr(lm.cfg, "gated_compressed_rope", False)
    )
    args.gated_rope_start_layer = int(
        getattr(lm.cfg, "gated_rope_start_layer", 0)
    )
    args.gated_rope_start_pair = int(
        getattr(lm.cfg, "gated_rope_start_pair", 0)
    )
    args.online_codebook_mask = lm.online_codebook_mask
    args.online_codebook_mask_active = lm.online_codebook_mask_active
    args.trim_stop_strings = lm.trim_stop_strings
    args.preserve_leading_space = lm.preserve_leading_space
    args.multi_view = lm.multi_view
    args.rope_mode = lm.rope_mode
    args.rope_source = lm.rope_source
    args.fail_on_truncation = lm.fail_on_truncation
    args.use_kv_cache = lm.use_kv_cache
    # Distinguishes "inactive because base mode" from "inactive because this eval
    # deliberately asked for the legacy mask" — otherwise a results JSON cannot
    # be audited for which regime produced its numbers.
    args.online_codebook_mask_requested = lm.online_codebook_mask_requested

    if preset_info:
        print(f"[eval_harness] preset: {preset_info[0]} — {preset_info[1]}")
    print(f"[eval_harness] checkpoint:   {ckpt_dir}")
    print(f"[eval_harness] tasks:        {tasks}")
    print(f"[eval_harness] eval_mode:    {args.eval_mode}")
    print(f"[eval_harness] rope_mode:    {args.rope_mode}")
    print(f"[eval_harness] rope_source:  {args.rope_source or '<checkpoint base>'}")
    print(f"[eval_harness] fail truncation: {args.fail_on_truncation}")
    print(f"[eval_harness] kv cache:     {args.use_kv_cache}")
    print(
        f"[eval_harness] max_subtokens: checkpoint="
        f"{args.checkpoint_max_subtokens} eval={args.eval_max_subtokens}"
    )
    print(
        f"[eval_harness] decoder RoPE: base_positions="
        f"{args.base_token_positions} two_axis={args.two_axis_rope} "
        f"gated_compressed={args.gated_compressed_rope} "
        f"gated_start_layer={args.gated_rope_start_layer} "
        f"gated_start_pair={args.gated_rope_start_pair}"
    )
    print(
        f"[eval_harness] online mask:  checkpoint={args.online_codebook_mask} "
        f"active={args.online_codebook_mask_active}"
    )
    print(f"[eval_harness] include_path: {include_path}")

    ruler_lengths = _parse_ruler_lengths(args.ruler_lengths)
    task_metadata = None
    if ruler_lengths:
        _use_ruler_hotpot_mirror()
        task_metadata = {
            "max_seq_lengths": ruler_lengths,
            "tokenizer": lm.tokenizer.name_or_path,
        }
        print(f"[eval_harness] RULER lengths: {ruler_lengths}")
    if task_metadata is not None:
        task_manager = TaskManager(
            include_path=include_path, metadata=task_metadata
        )
    else:
        task_manager = (
            TaskManager(include_path=include_path)
            if include_path
            else TaskManager()
        )

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
        **(
            {"gen_kwargs": {"do_sample": False, "temperature": 0.0}}
            if args.force_greedy
            else {}
        ),
        # Tasks flagged unsafe_code (humaneval*) execute generated code and are
        # refused unless this is True; HF evaluate's code_eval independently
        # refuses unless HF_ALLOW_CODE_EVAL=1. One env var drives both.
        confirm_run_unsafe_code=os.environ.get("HF_ALLOW_CODE_EVAL") == "1",
    )
    eval_wall_seconds = time.perf_counter() - eval_start

    # lm-eval lets a built-in task of the same name shadow an include_path
    # YAML without warning; the paper's math500 protocol lives in the local
    # YAML, so refuse a result that came from anything else.
    math500_cfg = (results.get("configs") or {}).get("math500")
    if math500_cfg is not None:
        revision = str((math500_cfg.get("dataset_kwargs") or {}).get("revision", ""))
        if not revision.startswith("6e4ed1a2"):
            raise RuntimeError(
                "math500 resolved to an unexpected task definition "
                f"(dataset revision {revision!r}); scripts/lm_eval_tasks/math500.yaml was not used"
            )

    # Multi-view perplexity: turn the adapter's per-task loglikelihood sums
    # into metrics by rescaling the harness-reported strict values (the
    # denominator cancels; see zip2zip_core.multi_view), then attach the
    # columns next to each task's strict metrics so every downstream consumer
    # — the stdout dump below, the results JSON, the W&B logger, and
    # log_results_to_wandb.py's backfill — inherits them with no extra
    # plumbing.
    multi_view = {}
    for mv_task, mv_sums in lm.multi_view_summary().items():
        row = (results.get("results") or {}).get(mv_task)
        reported = {}
        for k, v in (row or {}).items():
            name = k.split(",")[0]
            if name in ("word_perplexity", "byte_perplexity", "bits_per_byte"):
                reported.setdefault(name, v)
        derived = derive_multi_view_metrics(mv_sums, reported)
        derived.update(
            derive_multi_view_metrics(
                mv_sums,
                reported,
                loglik_key="first_token_multi_view_loglik_sum",
                metric_prefix="first_token_multi_view",
                gap_prefix="first_token_segmentation_gap",
            )
        )
        multi_view[mv_task] = {**mv_sums, **derived}
        if row is not None:
            for k, v in derived.items():
                if k.startswith(
                    (
                        "multi_view_",
                        "segmentation_gap_",
                        "first_token_multi_view_",
                        "first_token_segmentation_gap_",
                    )
                ) and isinstance(
                    v, (int, float)
                ):
                    row[f"{k},none"] = v

    print("\n" + "=" * 72)
    print("Results:")
    print(json.dumps(results.get("results", results), indent=2, default=str))
    print("=" * 72)

    compression = lm.compression_summary()
    print("Compression (base tokens per compressed token, >1 = more compression):")
    print(json.dumps(compression, indent=2))
    if multi_view:
        print("Multi-view perplexity (exact complete-segmentation marginal "
              "plus first-token upper bound):")
        print(json.dumps(multi_view, indent=2))
        for mv_task, mv_metrics in multi_view.items():
            # Back-solving the byte denominator from our strict sum and the
            # reported byte perplexity must land on a (near-)integer byte
            # count; anything else means our sums are not the ones behind the
            # reported metric and the multi-view columns cannot be trusted.
            implied = mv_metrics.get("implied_byte_denominator")
            if (
                isinstance(implied, (int, float))
                and abs(implied - round(implied)) > 1e-6 * max(1.0, abs(implied))
            ):
                print(f"[eval_harness] WARNING: multi-view alignment "
                      f"self-check FAILED for {mv_task}: implied byte "
                      f"denominator {implied!r} is not an integer — the "
                      f"accumulated sums do not match the reported "
                      f"byte_perplexity, multi_view_* values for this task "
                      f"are suspect.")
    print(f"Evaluation wall time: {eval_wall_seconds:.1f}s")
    print("=" * 72)

    if not args.no_log_samples:
        print_samples(results)

    if not args.no_wandb:
        from zip2zip_core.project import WANDB_ENTITY, WANDB_PROJECT
        from lm_eval.loggers import WandbLogger
        import wandb

        wandb_entity = (
            args.wandb_entity
            or os.environ.get("WANDB_ENTITY")
            or WANDB_ENTITY
        )
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
                entity=wandb_entity,
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
                entity=wandb_entity,
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
        if resume_id and resume_id.lower() != "none":
            # Pipeline path: log_results_to_wandb.py logs these same numbers as
            # final/<task>/<metric> on their own checkpoint-step x-axis. lm-eval
            # then writes an unprefixed <task>/<metric> copy on the training
            # run's step axis, i.e. a duplicate panel per metric. Hide those
            # auto-panels; the values stay in the run. A standalone eval run
            # (resume_id none) keeps them visible, because there they are the
            # only copy of the result.
            for task_name in (results.get("results") or {}):
                wandb.define_metric(f"{task_name}/*", hidden=True)
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
                    # Task versions and the harness's own config record (model
                    # args, harness/transformers versions): the protocol a paper
                    # number was produced under must be readable from the JSON.
                    "versions": results.get("versions"),
                    "config": results.get("config"),
                    "compression": compression,
                    "multi_view": multi_view,
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
        # Per-sample generations next to the results. With RESUME_WANDB_ID the
        # launcher runs the harness with --no_wandb, so the W&B sample tables
        # that gsm8k's paired (McNemar) analyses relied on are never written;
        # without this file a follow-up eval's generations are gone for good.
        # Generation tasks only: MC and perplexity samples have no downstream
        # consumer and would add ~1e2 MB per default eval to the PVC.
        configs = results.get("configs") or {}
        gen_samples = {
            task: rows
            for task, rows in (results.get("samples") or {}).items()
            if (configs.get(task) or {}).get("output_type") == "generate_until"
        }
        if not args.no_log_samples and gen_samples:
            stem, _ = os.path.splitext(args.output_path)
            samples_path = f"{stem}_samples.json"
            with open(samples_path, "w") as f:
                json.dump(gen_samples, f, default=str)
            print(f"Saved per-sample generations to {samples_path}")


if __name__ == "__main__":
    main()
