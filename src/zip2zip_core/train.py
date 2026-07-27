"""Standalone zip2zip pretraining script using FSDP2 + torch.compile.

Usage:
    torchrun --nnodes=N --nproc_per_node=4 -m zip2zip_core.train \
        --data_dir /path/to/tokens \
        --max_subtokens 2 \
        --steps 19000

Token-budget training:
    torchrun --nnodes=N --nproc_per_node=4 -m zip2zip_core.train \
        --data_dir /path/to/tokens \
        --max_subtokens 4 \
        --max_tokens 2000000000

Curriculum training:
    Phase 1: --max_subtokens 2 --steps 6000
    Phase 2: --max_subtokens 3 --steps 12000 --resume_from checkpoint_dir
    Phase 3: --max_subtokens 4 --steps 19000 --resume_from checkpoint_dir

Resume from latest checkpoint:
    --resume_from latest  (auto-finds latest step_* in output_dir)

Resume from HuggingFace:
    --resume_from_hf user/repo  (downloads checkpoint from HF Hub)
    --resume_from_hf user/repo --resume_hf_revision step_5000
"""

import argparse
import contextlib
import dataclasses
import glob
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.checkpoint import strip_wrapper_prefixes
from zip2zip_core.data import build_dataloader
from zip2zip_core.model import rope_cache_len

torch.set_float32_matmul_precision('high') 


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def base_view_replay_microsteps(
    step: int,
    gradient_accumulation_steps: int,
    probability: float,
) -> frozenset[int]:
    """Select whole base-view microbatches deterministically for one step.

    The cumulative-floor schedule is stateless: every rank and a resumed run
    derive the same answer from the optimizer step alone. The final microstep
    is deliberately kept compressed so every trainable compressed-only module
    participates in FSDP's gradient-sync forward/backward.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1, got "
            f"{gradient_accumulation_steps}"
        )
    if not 0.0 <= probability < 1.0:
        raise ValueError(
            f"base-view replay probability must be in [0, 1), got {probability}"
        )
    expected_per_step = gradient_accumulation_steps * probability
    if math.ceil(expected_per_step - 1e-12) > gradient_accumulation_steps - 1:
        raise ValueError(
            "base-view replay leaves no guaranteed compressed final microstep: "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"probability={probability}"
        )
    before = math.floor((step - 1) * expected_per_step + 1e-12)
    through = math.floor(step * expected_per_step + 1e-12)
    count = through - before
    return frozenset(range(count))


_VIEW_METRIC_FIELDS = (
    "nll_sum",
    "valid_count",
    "legacy_base_count",
    "target_base_count",
    "correct",
    "base_correct",
    "base_count",
    "hyper_correct",
    "hyper_count",
    "relaxed_hyper_correct",
    "type_nll_sum",
    "type_count",
    "type_correct",
    "base_type_correct",
    "base_type_count",
    "hyper_type_correct",
    "hyper_type_count",
    "micro_count",
    # Historical per-rank-microbatch ratios. These preserve the established
    # top-level W&B weighting formula when replay is disabled (minor reduction
    # rounding can differ), and become compressed-only averages when replay is
    # enabled. For class-specific compatibility metrics, a rank-microbatch with
    # no such class contributes zero, matching the old accumulator behavior.
    # The canonical raw-count metrics above instead exclude absent classes from
    # their class-specific denominators.
    "compat_backward_loss_sum",
    "compat_loss_legacy_sum",
    "compat_loss_target_sum",
    "compat_compression_legacy_sum",
    "compat_compression_target_sum",
    "compat_acc_sum",
    "compat_base_acc_sum",
    "compat_hyper_acc_sum",
    "compat_relaxed_hyper_acc_sum",
    "compat_type_loss_sum",
    "compat_type_acc_sum",
    "compat_base_type_acc_sum",
    "compat_hyper_type_acc_sum",
    "compat_hyper_ratio_sum",
)


def new_view_metric_sums() -> dict[str, float]:
    """Empty raw-count bucket for one input view."""
    return {field: 0.0 for field in _VIEW_METRIC_FIELDS}


def finalize_view_metric_sums(sums: dict[str, float]) -> dict[str, float]:
    """Turn one all-reduced raw-count bucket into safe view-specific metrics."""

    def ratio(num: str, den: str) -> float:
        denominator = sums[den]
        return sums[num] / denominator if denominator else 0.0

    micro_count = sums["micro_count"]

    def micro(field: str) -> float:
        return sums[field] / micro_count if micro_count else 0.0

    return {
        "loss_legacy_base_token": ratio("nll_sum", "legacy_base_count"),
        "loss_target_base_token": ratio("nll_sum", "target_base_count"),
        "loss_per_valid_token": ratio("nll_sum", "valid_count"),
        "compression_legacy": ratio("legacy_base_count", "valid_count"),
        "compression_target": ratio("target_base_count", "valid_count"),
        "acc": ratio("correct", "valid_count"),
        "base_token_acc": ratio("base_correct", "base_count"),
        "hyper_token_acc": ratio("hyper_correct", "hyper_count"),
        "relaxed_hyper_acc": ratio(
            "relaxed_hyper_correct", "hyper_count"
        ),
        "relaxed_acc": (
            (sums["base_correct"] + sums["relaxed_hyper_correct"])
            / sums["valid_count"]
            if sums["valid_count"]
            else 0.0
        ),
        "type_loss": ratio("type_nll_sum", "type_count"),
        "type_acc": ratio("type_correct", "type_count"),
        "base_type_acc": ratio("base_type_correct", "base_type_count"),
        "hyper_type_acc": ratio(
            "hyper_type_correct", "hyper_type_count"
        ),
        "hyper_ratio": ratio("hyper_type_count", "type_count"),
        "micro_count": micro_count,
        "compat_backward_loss": micro("compat_backward_loss_sum"),
        "compat_loss_legacy": micro("compat_loss_legacy_sum"),
        "compat_loss_target": micro("compat_loss_target_sum"),
        "compat_compression_legacy": micro(
            "compat_compression_legacy_sum"
        ),
        "compat_compression_target": micro(
            "compat_compression_target_sum"
        ),
        "compat_acc": micro("compat_acc_sum"),
        "compat_base_acc": micro("compat_base_acc_sum"),
        "compat_hyper_acc": micro("compat_hyper_acc_sum"),
        "compat_relaxed_hyper_acc": micro(
            "compat_relaxed_hyper_acc_sum"
        ),
        "compat_type_loss": micro("compat_type_loss_sum"),
        "compat_type_acc": micro("compat_type_acc_sum"),
        "compat_base_type_acc": micro("compat_base_type_acc_sum"),
        "compat_hyper_type_acc": micro("compat_hyper_type_acc_sum"),
        "compat_hyper_ratio": micro("compat_hyper_ratio_sum"),
    }


def finalize_mixed_objective(
    *view_sums: dict[str, float],
) -> float:
    """Mean backward objective over all all-reduced view rank-microbatches."""
    objective_sum = sum(
        sums["compat_backward_loss_sum"] for sums in view_sums
    )
    rank_microbatches = sum(sums["micro_count"] for sums in view_sums)
    return (
        objective_sum / rank_microbatches
        if rank_microbatches
        else 0.0
    )


def _all_reduce_view_metric_sums(
    sums: dict[str, float], device: torch.device
) -> dict[str, float]:
    values = torch.tensor(
        [sums[field] for field in _VIEW_METRIC_FIELDS],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return dict(zip(_VIEW_METRIC_FIELDS, values.tolist()))


def _unwrap_model(model):
    """Unwrap DDP and torch.compile wrappers to get the raw model.

    FSDP2 (fully_shard) modifies modules in place and adds no wrapper, so for the
    FSDP path this returns the model directly (per-block torch.compile only wraps
    the inner blocks, not the root)."""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


def apply_fsdp(model, world_size):
    """Shard the model with FSDP2 (fully_shard) across all DP ranks.

    Params/grads/optimizer states are sharded, so a 3.8B model needs only
    ~footprint/world_size per GPU (vs DDP which replicates everything).

    Params stay in fp32 (no MixedPrecisionPolicy) and bf16 compute is provided by
    the ``torch.autocast`` context in the training loop — same numerics as the old
    DDP path. We deliberately avoid param_dtype=bf16 here because the zip2zip
    hyper-encoder reads ``tok_embeddings.weight`` directly (see
    Zip2ZipLlama3Model._encode_codebook_with_weights), and mixing an autocast/bf16
    activation path with an mp-casted weight is brittle; keeping params fp32 +
    autocast is the robust combination.

    NOTE: per-block torch.compile is applied *after* checkpoint load (see
    compile_transformer_blocks), not here — compiling first renames params to
    ``layers.N._orig_mod.*`` which breaks DCP set_model_state_dict on resume.
    """
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, mesh=mesh)
    fully_shard(model.hyper_encoder, mesh=mesh)
    if getattr(model, "hyper_output", None) is not None:
        fully_shard(model.hyper_output, mesh=mesh)
    for block in model.layers.values():
        fully_shard(block, mesh=mesh)
    if model.norm is not None and model.output is not None:
        fully_shard([model.norm, model.output], mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh)
    return model


def compile_transformer_blocks(model):
    """torch.compile each transformer block in place. Call AFTER checkpoint load so
    the on-disk (canonical) param names match the model at load time."""
    for layer_id, block in list(model.layers.items()):
        model.layers.register_module(layer_id, torch.compile(block))


def _relaxed_prefix_correct(hyper_preds, hyper_labels, batch_idx, cb, vocab_size, pad_token_id):
    """Count predictions whose base tokens are a prefix of the target's base tokens.

    E.g. target=[127,78,278] → predicting [127,78] or just 127 counts as correct.
    """
    if len(hyper_preds) == 0:
        return 0
    max_ms = cb.shape[-1]
    device = cb.device

    target_cb_idx = (hyper_labels - vocab_size).clamp(min=0, max=cb.shape[1] - 1)
    target_entries = cb[batch_idx, target_cb_idx]

    pred_is_base = hyper_preds < vocab_size
    pred_entries = torch.full((len(hyper_preds), max_ms), pad_token_id, device=device, dtype=cb.dtype)
    if pred_is_base.any():
        pred_entries[pred_is_base, 0] = hyper_preds[pred_is_base]
    pred_is_hyper = ~pred_is_base
    if pred_is_hyper.any():
        pred_cb_idx = (hyper_preds[pred_is_hyper] - vocab_size).clamp(min=0, max=cb.shape[1] - 1)
        pred_entries[pred_is_hyper] = cb[batch_idx[pred_is_hyper], pred_cb_idx]

    pred_has_content = pred_entries != pad_token_id
    matches = (pred_entries == target_entries) | ~pred_has_content
    is_prefix = matches.all(dim=-1) & pred_has_content.any(dim=-1)
    return is_prefix.sum().item()


def _relaxed_target_nll(valid_logits, valid_labels, batch_idx, cb, vocab_size, pad_token_id):
    """Per-token relaxed NLL.

    For base-token labels, this is standard NLL.
    For hyper-token labels, sum probability mass over all tokens whose expanded
    base-token sequence is a prefix of the target sequence.
    """
    if len(valid_labels) == 0:
        return torch.empty(0, device=valid_logits.device, dtype=valid_logits.dtype)

    log_probs = F.log_softmax(valid_logits, dim=-1)
    nll = -log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

    hyper_mask = valid_labels >= vocab_size
    if not hyper_mask.any():
        return nll

    device = valid_logits.device
    max_ms = cb.shape[-1]
    hyper_labels = valid_labels[hyper_mask]
    hyper_batch_idx = batch_idx[hyper_mask]
    hyper_log_probs = log_probs[hyper_mask]

    target_cb_idx = (hyper_labels - vocab_size).clamp(min=0, max=cb.shape[1] - 1)
    target_entries = cb[hyper_batch_idx, target_cb_idx]  # (H, max_ms)
    target_lengths = (target_entries != pad_token_id).sum(dim=-1)

    candidate_ids = torch.arange(valid_logits.shape[-1], device=device)
    cand_is_base = candidate_ids < vocab_size
    cand_entries = torch.full(
        (candidate_ids.numel(), max_ms), pad_token_id, device=device, dtype=cb.dtype
    )
    if cand_is_base.any():
        cand_entries[cand_is_base, 0] = candidate_ids[cand_is_base].to(cb.dtype)

    cand_is_hyper = ~cand_is_base
    if cand_is_hyper.any():
        cand_cb_idx = (candidate_ids[cand_is_hyper] - vocab_size).clamp(
            min=0, max=cb.shape[1] - 1
        )
        cand_entries[cand_is_hyper] = cb[hyper_batch_idx[0], cand_cb_idx]

    cand_lengths = (cand_entries != pad_token_id).sum(dim=-1)

    relaxed_nll = []
    for i in range(hyper_labels.shape[0]):
        tgt = target_entries[i]
        tgt_len = target_lengths[i]
        if tgt_len == 0:
            relaxed_nll.append(-hyper_log_probs[i, hyper_labels[i]])
            continue

        batch_cand_entries = cand_entries
        if cand_is_hyper.any():
            batch_cand_entries = cand_entries.clone()
            batch_cand_entries[cand_is_hyper] = cb[hyper_batch_idx[i], cand_cb_idx]

        prefix_ok = cand_lengths <= tgt_len
        prefix_ok &= cand_lengths > 0

        matches = batch_cand_entries == tgt.unsqueeze(0)
        valid_prefix = (
            torch.arange(max_ms, device=device).unsqueeze(0) < cand_lengths.unsqueeze(1)
        )
        prefix_ok &= (matches | ~valid_prefix).all(dim=-1)

        relaxed_logprob = torch.logsumexp(hyper_log_probs[i, prefix_ok], dim=0)
        relaxed_nll.append(-relaxed_logprob)

    nll[hyper_mask] = torch.stack(relaxed_nll)
    return nll


def _count_target_bytes(labels, cb, vocab_size, pad_token_id, tokenizer):
    """Count UTF-8 bytes represented by valid target labels."""
    total_bytes = 0
    B, T = labels.shape
    for b in range(B):
        valid_labels = labels[b][labels[b] != -100]
        if valid_labels.numel() == 0:
            continue

        base_ids = []
        for tok in valid_labels.tolist():
            if tok < vocab_size:
                base_ids.append(tok)
            else:
                cb_idx = min(max(tok - vocab_size, 0), cb.shape[1] - 1)
                entry = cb[b, cb_idx]
                base_ids.extend(entry[entry != pad_token_id].tolist())

        text = tokenizer.decode(
            base_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        total_bytes += len(text.encode("utf-8"))
    return total_bytes


_hf_upload_thread = None


def _debug_rank_log(enabled: bool, rank: int, step: int, micro_step: int, message: str):
    """Emit a flushed per-rank debug log line for early-step hang diagnosis."""
    if enabled:
        now = time.strftime("%H:%M:%S")
        print(f"[debug {now}] [rank{rank}] [step {step}] [micro {micro_step}] {message}", flush=True)

def _gather_loader_state(dataloader, args):
    """Every rank's position in its own data stream, gathered onto all ranks.

    Returns None when the position cannot be trusted, so a later resume can say
    so loudly instead of seeking to a wrong spot. COLLECTIVE: every rank must
    reach the all_gather below, so call this outside any rank-0-only block.
    """
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "state_dict"):
        return None
    if args.num_workers > 0:
        # __iter__ runs inside worker processes; the position it advances there
        # never reaches this object, so state_dict() here is a stale zero. Every
        # launcher in this repo defaults NUM_WORKERS to 1 or 4, so this is the
        # COMMON case, not a corner: such a run is simply not resumable, and the
        # startup banner says so before the GPUs are spent.
        return None
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dataset.state_dict())
    return {
        "world_size": dist.get_world_size(),
        "num_workers": args.num_workers,
        "ranks": gathered,
    }


def _should_restore_data_position(resume_dir, args):
    """Whether this resume must seek the data stream back to the checkpoint.

    Not every --resume_from continues a training stream:
      --reset_step  a fresh LR schedule is a NEW run that happens to warm-start
                    its weights, so it reads the data from the beginning.
      --eval        scores a checkpoint and writes nothing. Every checkpoint of a
                    run must be scored on the SAME data (from shard 0) or per-step
                    curves compare different samples — and scripts/eval_lm.sh has
                    a fixed argument list, so it could not opt out of a restore.
    """
    return resume_dir is not None and not args.reset_step and not args.eval


def _restore_loader_state(resume_dir, dataloader, args):
    """Seek this rank's data stream back to where the checkpoint left it.

    Without this a resumed run continues the LR schedule and optimizer from step
    N but re-reads the shards from the beginning: it then trains twice on the
    first steps' data and never sees the tail. That is silent — no exception, no
    log line — and it destroys comparability with a clean baseline, so a
    checkpoint that cannot be positioned is a hard error unless the operator
    explicitly accepts the replay.
    """
    rank = dist.get_rank()
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "load_state_dict"):
        return
    meta_path = os.path.join(resume_dir, "meta.pt")
    saved = None
    if os.path.exists(meta_path):
        saved = torch.load(meta_path, map_location="cpu", weights_only=False).get(
            "loader_state"
        )

    problem = None
    if saved is None:
        problem = (
            f"{resume_dir} records no data-stream position (checkpoint predates "
            "this feature, or was written with --num_workers > 0)"
        )
    elif saved.get("world_size") != dist.get_world_size():
        problem = (
            f"position was saved for world_size={saved.get('world_size')} but this "
            f"run has {dist.get_world_size()}; shards are split rank::world_size, "
            "so each rank would resume inside a different file"
        )
    elif args.num_workers > 0:
        problem = (
            f"--num_workers={args.num_workers} keeps the data position inside "
            "worker processes, where it cannot be restored; set NUM_WORKERS=0 "
            "(launcher) or --num_workers 0 for a resumable run"
        )
    else:
        # Dry-run the stream-identity checks (mode, shard files, index range) so
        # their failures go through the same decision below instead of escaping
        # as a bare ValueError that --allow_data_replay cannot reach.
        try:
            dataset.check_state_dict(saved["ranks"][rank])
        except ValueError as exc:
            problem = str(exc)

    # The decision must be RANK-UNIFORM. check_state_dict compares this rank's
    # own shard list, so ranks can disagree; letting some raise while the rest
    # restore and march into the next collective would hang the job instead of
    # failing it.
    verdicts = [None] * dist.get_world_size()
    dist.all_gather_object(verdicts, problem)
    failed = [(r, p) for r, p in enumerate(verdicts) if p is not None]

    if failed:
        detail = "; ".join(f"rank {r}: {p}" for r, p in failed[:4])
        message = (
            f"cannot restore the data-stream position: {detail}. Resuming anyway "
            "would REPLAY the shards from the beginning while the step counter "
            "continues, so this run would train on repeated data and never see "
            "the tail — its numbers would not be comparable to a clean baseline. "
            "Start from step 0, or pass --allow_data_replay (ALLOW_DATA_REPLAY=1 "
            "in the launchers) to accept it."
        )
        if not args.allow_data_replay:
            raise ValueError(message)
        if rank == 0:
            print(f"[resume] WARNING: {message}")
        return

    dataset.load_state_dict(saved["ranks"][rank])
    positions = [None] * dist.get_world_size()
    dist.all_gather_object(positions, dataset.state_dict())
    if rank == 0:
        summary = ", ".join(
            f"rank{r}=(shard {p['shard_idx']}, offset {p['offset']})"
            for r, p in enumerate(positions)
        )
        print(f"[resume] data stream restored: {summary}")


def save_checkpoint(model, optimizer, step, args, output_dir, hf_repo=None,
                    dataloader=None):
    """Save a consolidated (full, unsharded) checkpoint.

    The model is FSDP-sharded, so we gather the full state via DCP's
    get_*_state_dict(full_state_dict=True). The resulting model.pt / optimizer.pt
    are plain state dicts with canonical keys (no FSDP/compile prefixes), so they
    stay compatible with eval/export/resume.
    """
    global _hf_upload_thread
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict, get_optimizer_state_dict, StateDictOptions,
    )
    rank = dist.get_rank()
    ckpt_dir = os.path.join(output_dir, f"step_{step}")

    # Collective, like the state gather below: keep it out of the rank-0 block.
    loader_state = _gather_loader_state(dataloader, args) if dataloader is not None else None

    # Collective: every rank must participate in gathering the full state.
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=opts)
    optim_sd = (
        get_optimizer_state_dict(model, optimizer, options=opts)
        if optimizer is not None else None
    )

    if rank == 0:
        if _hf_upload_thread is not None and _hf_upload_thread.is_alive():
            print(f"[Rank 0] Waiting for previous HF upload to finish ...")
            _hf_upload_thread.join()

        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model_sd, os.path.join(ckpt_dir, "model.pt"))
        if optim_sd is not None:
            torch.save(optim_sd, os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(
            {"step": step, "args": vars(args), "loader_state": loader_state},
            os.path.join(ckpt_dir, "meta.pt"),
        )
        if loader_state is None:
            print(f"[Rank 0] Saved checkpoint at step {step} "
                  f"(WITHOUT a data position — a resume from it will replay data)")
        else:
            print(f"[Rank 0] Saved checkpoint at step {step} "
                  f"(data position: {loader_state['ranks'][0]} on rank 0)")

        if hf_repo:
            import threading
            _hf_upload_thread = threading.Thread(
                target=_push_checkpoint_to_hub,
                args=(ckpt_dir, hf_repo, step),
                daemon=True,
            )
            _hf_upload_thread.start()
            print(f"[Rank 0] HF upload started in background for step {step}")
    dist.barrier()


def _resolve_latest_checkpoint(output_dir):
    """Find the latest step_* checkpoint directory in output_dir."""
    pattern = os.path.join(output_dir, "step_*")
    ckpt_dirs = glob.glob(pattern)
    if not ckpt_dirs:
        raise FileNotFoundError(f"No step_* checkpoints found in {output_dir}")
    # Sort by step number
    def _step_num(d):
        try:
            return int(os.path.basename(d).split("_")[1])
        except (IndexError, ValueError):
            return -1
    latest = max(ckpt_dirs, key=_step_num)
    return latest


def _download_checkpoint_from_hf(repo_id, revision=None):
    """Download checkpoint files from a HuggingFace Hub repo.

    Returns the local directory containing model.pt, optimizer.pt (if available), and meta.pt (if available).
    """
    from huggingface_hub import hf_hub_download

    local_dir = None
    for filename in ["model.pt", "optimizer.pt", "meta.pt"]:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
            if local_dir is None:
                local_dir = os.path.dirname(path)
        except Exception:
            if filename == "model.pt":
                raise
            # optimizer.pt and meta.pt are optional for HF checkpoints
            pass

    return local_dir




def _push_checkpoint_to_hub(ckpt_dir, repo_id, step):
    """Push a checkpoint directory to HuggingFace Hub."""
    from zip2zip_core.hub import push_checkpoint
    push_checkpoint(ckpt_dir, repo_id, step)


def _clean_state_dict(model_state):
    """Strip FSDP wrapper prefixes for compatibility."""
    return strip_wrapper_prefixes(model_state)


def validate_resume_args(resume_dir, args):
    """Fail fast when a resume would silently change the training recipe.

    Compression semantics must never change inside one checkpoint lineage:
    a resume that drops --disable_digit_ids (or changes the codebook size or
    tokenizer) would continue the run on a different data distribution with no
    error anywhere. Architecture, objective, and initialization lineage must
    also remain explicit. Curriculum phases legitimately change max_subtokens /
    seq_len / data_dir, so those only print a note. Deliberate changes to the
    hard set require --allow_resume_mismatch.
    """
    meta_path = os.path.join(resume_dir, "meta.pt")
    if not os.path.exists(meta_path):
        return
    prev = torch.load(meta_path, map_location="cpu", weights_only=False).get("args", {}) or {}
    hard = ("disable_digit_ids", "max_codebook_size",
            "max_active_codebook_size", "tokenizer",
            "untied_hyper_encoder", "share_hyper_encoder_weights",
            "base_token_positions", "two_axis_rope",
            "gated_compressed_rope", "gated_rope_start_layer",
            "gated_rope_start_pair",
            "base_view_replay_prob",
            "zero_init_encoder_output", "no_encoder_residual",
            "token_type_loss_weight", "online_codebook_mask",
            # lora_alpha sets scaling = alpha/rank as a RUNTIME attribute
            # (lora.py), not a state-dict entry: changing it on resume loads the
            # weights happily and silently changes the model's function. The
            # other two change what the objective is and what the codebook means.
            "lora_alpha", "hyper_causal_mask", "no_remap_codebook",
            "hyper_encoder_type", "encoder_dim", "encoder_n_layers",
            "encoder_n_heads", "encoder_intermediate_size")
    if (
        float(prev.get("base_view_replay_prob", 0.0) or 0.0) > 0.0
        or float(getattr(args, "base_view_replay_prob", 0.0) or 0.0) > 0.0
    ):
        # Replay selection is defined over microsteps, so changing accumulation
        # changes which examples receive the base view even at the same step.
        hard += ("gradient_accumulation_steps",)
    soft = ("max_subtokens", "seq_len", "data_dir", "warmstart_steps")
    # Hard keys whose ABSENCE from an old meta.pt has an unambiguous meaning:
    # the flag/objective did not exist, i.e. it was OFF. Treat absence as that
    # default so e.g. resuming a v0.5 lineage with --base_token_positions or a
    # fresh --token_type_loss_weight is caught instead of silently changing
    # semantics mid-lineage.
    absent_defaults = {
        "disable_digit_ids": False,
        "untied_hyper_encoder": False,
        "share_hyper_encoder_weights": False,
        "base_token_positions": False,
        "two_axis_rope": False,
        "gated_compressed_rope": False,
        "gated_rope_start_layer": 0,
        # The first gated-RoPE implementation covered every complex pair and
        # predates this metadata field. Absence therefore means pair 0.
        "gated_rope_start_pair": 0,
        "base_view_replay_prob": 0.0,
        "zero_init_encoder_output": False,
        "no_encoder_residual": False,
        "token_type_loss_weight": 0.0,
        "online_codebook_mask": False,
    }
    # Legacy metas record None for the encoder_* args (they predate the
    # effective-value resolution at startup): unknown is not a conflict there,
    # and the strict checkpoint load backstops real architecture mismatches.
    # For every other hard key a recorded None IS a difference worth stopping on.
    mismatches = [
        (k, prev.get(k, absent_defaults.get(k)),
         getattr(args, k, absent_defaults.get(k)))
        for k in hard
        if (k in prev or k in absent_defaults)
        and prev.get(k, absent_defaults.get(k)) != getattr(
            args, k, absent_defaults.get(k)
        )
        and not (k.startswith("encoder_") and prev.get(k) is None)
    ]
    if mismatches and not args.allow_resume_mismatch:
        raise ValueError(
            f"resume args differ from {meta_path} on recipe-critical settings "
            f"{mismatches} (recorded, requested) — resuming would silently change the "
            "training recipe or mislabel its initialization lineage. Pass the "
            "recorded values, or "
            "--allow_resume_mismatch for a deliberate change."
        )
    if dist.get_rank() == 0:
        for k, old, new in mismatches:
            print(f"[resume] OVERRIDE: {k} changes from {old!r} to {new!r} (--allow_resume_mismatch)")
        for k in soft:
            if k in prev and prev[k] != getattr(args, k):
                print(f"[resume] note: {k} changes from {prev[k]!r} to {getattr(args, k)!r} "
                      f"(legitimate for curriculum phases; verify it is intended)")


def load_checkpoint(model, optimizer, resume_dir, device, random_weights=False):
    """Load model and optimizer state from a local directory.

    Handles curriculum transitions where max_subtokens changes between phases
    by zero-padding the hyper_encoder position embedding.

    If random_weights=True, skip loading model weights (control experiment:
    same step/LR position, same optimizer state, but randomly initialized model).
    """
    meta_path = os.path.join(resume_dir, "meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu")
        step = meta["step"]
    else:
        step = 0
        if dist.get_rank() == 0:
            print("No meta.pt found — starting from step 0")

    from torch.distributed.checkpoint.state_dict import (
        set_model_state_dict, set_optimizer_state_dict, StateDictOptions,
    )
    set_opts = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)

    if random_weights:
        if dist.get_rank() == 0:
            print(f"[random_weights] Skipping model weight load — keeping random init (step={step})")
        curriculum_transition = False
    else:
        model_state = torch.load(os.path.join(resume_dir, "model.pt"), map_location="cpu")
        model_state = _clean_state_dict(model_state)

        # Handle pos_embed size mismatch from curriculum phase transitions. Read the
        # current (global) shape from the sharded param without gathering weights.
        # Both encoders carry a pos_embed when untied.
        named_params = dict(_unwrap_model(model).named_parameters())
        curriculum_transition = False
        for pos_key in ("hyper_encoder.pos_embed.weight", "hyper_output.pos_embed.weight"):
            if pos_key in model_state and pos_key in named_params:
                current_pos = named_params[pos_key]
                saved_pos = model_state[pos_key]
                if saved_pos.shape[0] < current_pos.shape[0]:
                    padded = torch.zeros((current_pos.shape[0], saved_pos.shape[1]), dtype=saved_pos.dtype)
                    padded[:saved_pos.shape[0]] = saved_pos
                    model_state[pos_key] = padded
                    curriculum_transition = True
                    if dist.get_rank() == 0:
                        print(f"Padded {pos_key} from {saved_pos.shape[0]} to {current_pos.shape[0]} positions")

        # Scatter the full state dict onto the FSDP-sharded model.
        set_model_state_dict(model, model_state, options=set_opts)

    opt_path = os.path.join(resume_dir, "optimizer.pt")
    if optimizer is None:
        pass  # eval mode — skip optimizer load
    elif random_weights:
        if dist.get_rank() == 0:
            print("[random_weights] Skipping optimizer state load — fresh optimizer for random init")
    elif curriculum_transition:
        if dist.get_rank() == 0:
            print("Curriculum transition detected — skipping optimizer state load (will re-init)")
    elif os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location="cpu")
        set_optimizer_state_dict(model, optimizer, optim_state_dict=opt_state, options=set_opts)
    else:
        if dist.get_rank() == 0:
            print("No optimizer.pt found — optimizer will start fresh")

    if dist.get_rank() == 0:
        print(f"Resumed from step {step}")
    return step


def _split_phi3_to_llama(hf_sd, config):
    """Split Phi3 fused weights into HF-Llama layout so Llama3StateDictAdapter works.

    Phi-3 stores fused ``self_attn.qkv_proj`` and ``mlp.gate_up_proj``; the
    torchtitan decoder (and Llama3StateDictAdapter) expects separate q/k/v and
    gate/up. This is the exact inverse of ``_fuse_llama_to_phi3`` in
    ``scripts/zip2zip_hf/export_phi.py`` — split here, then ``from_hf`` applies
    the RoPE reverse-permute, mirroring how export fuses *after* ``to_hf``.
    """
    n_layers = config.n_layers
    n_heads = config.layer.attention.n_heads
    n_kv_heads = config.layer.attention.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = config.dim // n_heads
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    out = {}
    # pass-through tensors (same names in HF-Llama)
    for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if k in hf_sd:
            out[k] = hf_sd[k]
    for i in range(n_layers):
        p = f"model.layers.{i}."
        # Phi3 concatenates qkv_proj as [q | k | v] along the output dim.
        qkv = hf_sd[f"{p}self_attn.qkv_proj.weight"]
        q, k, v = torch.split(qkv, [q_dim, kv_dim, kv_dim], dim=0)
        out[f"{p}self_attn.q_proj.weight"] = q
        out[f"{p}self_attn.k_proj.weight"] = k
        out[f"{p}self_attn.v_proj.weight"] = v
        out[f"{p}self_attn.o_proj.weight"] = hf_sd[f"{p}self_attn.o_proj.weight"]
        # Phi3 chunks gate_up_proj as [gate | up].
        gate_up = hf_sd[f"{p}mlp.gate_up_proj.weight"]
        g, u = torch.chunk(gate_up, 2, dim=0)
        out[f"{p}mlp.gate_proj.weight"] = g
        out[f"{p}mlp.up_proj.weight"] = u
        out[f"{p}mlp.down_proj.weight"] = hf_sd[f"{p}mlp.down_proj.weight"]
        out[f"{p}input_layernorm.weight"] = hf_sd[f"{p}input_layernorm.weight"]
        out[f"{p}post_attention_layernorm.weight"] = hf_sd[f"{p}post_attention_layernorm.weight"]
    return out


def load_hf_pretrained(model, hf_model_name, device):
    """Load decoder weights from a HuggingFace pretrained Llama or Phi-3 model.

    Hyper-encoder and other zip2zip-specific parameters are left at their
    current (randomly initialized) values. Phi-3 checkpoints (fused qkv_proj /
    gate_up_proj) are split to Llama layout first via ``_split_phi3_to_llama``.
    """
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    raw_model = _unwrap_model(model)
    config = raw_model.zip2zip_config

    # Download model files
    if dist.get_rank() == 0:
        cache_dir = snapshot_download(hf_model_name, allow_patterns=["*.safetensors", "*.json"])
    else:
        cache_dir = None
    cache_dir_list = [cache_dir]
    dist.broadcast_object_list(cache_dir_list, src=0)
    cache_dir = cache_dir_list[0]

    # Load all safetensor shards
    import glob as _glob
    shard_files = sorted(_glob.glob(os.path.join(cache_dir, "*.safetensors")))
    hf_sd = {}
    for f in shard_files:
        hf_sd.update(load_file(f, device="cpu"))

    # Phi-3 stores fused qkv_proj / gate_up_proj — split to Llama layout first
    # so Llama3StateDictAdapter (Llama-only) can convert + RoPE-permute them.
    if any(k.endswith("self_attn.qkv_proj.weight") for k in hf_sd):
        if dist.get_rank() == 0:
            print("[init_from_hf] Detected fused Phi-3 qkv_proj/gate_up_proj — splitting to Llama layout")
        hf_sd = _split_phi3_to_llama(hf_sd, config)

    # Convert HF keys to torchtitan keys using Llama3StateDictAdapter
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
    adapter = Llama3StateDictAdapter(config, None)
    tt_sd = adapter.from_hf(hf_sd)

    # Load with strict=False: hyper_encoder keys won't be in the checkpoint
    missing, unexpected = raw_model.load_state_dict(tt_sd, strict=False)

    if dist.get_rank() == 0:
        hyper_missing = [
            k
            for k in missing
            if k.startswith("hyper_encoder")
            or k.startswith("token_type_head")
            or k.startswith("hyper_output")
            or k == "compressed_rope_gate"
        ]
        other_missing = [k for k in missing if k not in hyper_missing]
        print(f"[init_from_hf] Loaded decoder weights from {hf_model_name}")
        print(
            "  Zip2Zip-specific params (randomly initialized): "
            f"{len(hyper_missing)}"
        )
        if other_missing:
            print(f"  ⚠️ Other missing keys: {other_missing}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {unexpected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_config", type=str, default="1B", choices=list(zip2zip_llama_configs.keys()))
    parser.add_argument("--max_subtokens", type=int, default=2)
    parser.add_argument("--max_codebook_size", type=int, default=4096)
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--encoder_intermediate_size", type=int, default=None)
    parser.add_argument("--encoder_n_heads", type=int, default=None)
    parser.add_argument("--hyper_encoder_type", type=str, default="flat", choices=["flat", "hierarchical", "fast_hierarchical"])
    parser.add_argument("--encoder_n_layers", type=int, default=None)
    parser.add_argument("--max_active_codebook_size", type=int, default=4096)
    parser.add_argument("--disable_digit_ids", action="store_true",
                        help="Add digit tokens to the LZW disabled_ids so numbers are "
                             "never merged into hypertokens during training.")
    parser.add_argument("--untied_hyper_encoder", action="store_true",
                        help="Use a separate output-role hyper-encoder (reading lm_head "
                             "rows) instead of reusing the input hyper-encoder for logits. "
                             "Matches the released model. Default off = tied (legacy).")
    parser.add_argument("--share_hyper_encoder_weights", action="store_true",
                        help="With --untied_hyper_encoder, re-encode the output role from "
                             "lm_head rows but reuse the input hyper-encoder's weights. "
                             "Default off = train a separate output encoder.")
    encoder_residual_group = parser.add_mutually_exclusive_group()
    encoder_residual_group.add_argument(
        "--zero_init_encoder_output", action="store_true",
        help="Make the hyper-encoder emit exactly zero at init, so the "
             "first hypertoken embedding equals its first base token's "
             "embedding (the encoder_residual design intent). Fixes a "
             "silent no-op: the existing zero-init only touches "
             "proj_out, which does not exist when encoder_dim == dim "
             "(every Phi run), leaving the encoder ~54x too large at "
             "step 0. Default off = v0.1-v0.6.3 behavior.")
    parser.add_argument("--base_token_positions", action="store_true",
                        help="RoPE positions follow the uncompressed stream (each token "
                             "sits at the base-space index of its last constituent) "
                             "instead of one position per compressed token, so relative "
                             "distances keep their pretrained meaning. Default off = "
                             "compressed-index positions (v0.5 and released behavior).")
    parser.add_argument(
        "--two_axis_rope",
        action="store_true",
        help="Split complex RoPE pairs 50/50 between base-stream positions "
             "(even pairs) and compressed-token positions (odd pairs). Requires "
             "--base_token_positions. Default off preserves single-axis RoPE.",
    )
    parser.add_argument(
        "--gated_compressed_rope",
        action="store_true",
        help="Learn a per-layer interpolation from base-stream RoPE (exact at "
             "zero initialization) toward compressed-index RoPE for the "
             "configured low-frequency pair suffix. Requires "
             "--base_token_positions and is incompatible with --two_axis_rope.",
    )
    parser.add_argument(
        "--gated_rope_start_layer",
        type=int,
        default=0,
        help="Zero-based first decoder layer using gated compressed RoPE.",
    )
    parser.add_argument(
        "--gated_rope_start_pair",
        type=int,
        default=0,
        help="Zero-based first complex RoPE pair with a learned gate. The "
             "generic default gates every pair; v0.7.1's named launcher "
             "profile sets 32 for Phi-3.5's lowest-frequency third.",
    )
    parser.add_argument(
        "--base_view_replay_prob",
        type=float,
        default=0.0,
        help="Deterministic fraction of whole training microbatches replayed as "
             "ordinary uncompressed base-token views from the same source chunk. "
             "The final accumulation microstep always remains compressed.",
    )
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--local_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=19000,
                        help="Total steps for LR schedule computation")
    parser.add_argument("--stop_at", type=int, default=None,
                        help="Stop training at this step (for curriculum phases). Defaults to --steps.")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Stop after processing this many global training tokens. Overrides --steps as stop criterion.")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--warmstart_steps", type=int, default=0,
                        help="Phased warm-start: for the first N optimizer steps, freeze "
                             "the decoder-LoRA (null its grads) so only the hyper-encoder(s) "
                             "train and stabilize before the LoRA adapts to them. 0 = off.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--hyper_lr", type=float, default=None,
                        help="LR for hyper_encoder (random init). Defaults to --lr if not set.")
    parser.add_argument("--min_hyper_lr", type=float, default=None,
                        help="Min LR for hyper_encoder cosine schedule. Defaults to --min_lr if not set.")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta2", type=float, default=0.95,
                        help="AdamW beta2. The released ozz-main finetunes used the torch "
                             "default 0.999; zip2zip-core pretraining runs used 0.95.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--allow_resume_mismatch", action="store_true",
                        help="Permit resuming with recipe-critical args (data/compression "
                             "semantics, architecture, objective, or initialization lineage) "
                             "that differ from the checkpoint's meta.pt. Off = hard error.")
    parser.add_argument("--allow_data_replay", action="store_true",
                        default=os.environ.get("ALLOW_DATA_REPLAY", "") not in ("", "0"),
                        help="Permit resuming from a checkpoint whose data-stream position "
                             "cannot be restored. The run then re-reads the shards from the "
                             "start while the step counter continues, so it trains on repeated "
                             "data and never sees the tail — its numbers are NOT comparable to "
                             "a clean baseline. Off = hard error. Also settable as "
                             "ALLOW_DATA_REPLAY=1 in the environment, so the launchers that do "
                             "not forward extra arguments can still opt out.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Local checkpoint dir, or 'latest' to auto-find latest step_* in output_dir")
    parser.add_argument("--resume_from_hf", type=str, default=None,
                        help="HuggingFace repo ID to download checkpoint from (e.g. user/model-name)")
    parser.add_argument("--resume_hf_revision", type=str, default=None,
                        help="HF revision/branch to download from (e.g. step0). Defaults to main.")
    parser.add_argument("--random_weights", action="store_true",
                        help="Control experiment: resume step/optimizer from checkpoint but reinit model weights randomly")
    parser.add_argument("--reset_step", action="store_true",
                        help="After loading checkpoint weights, reset step to 0 (fresh LR schedule). "
                             "Useful for finetuning: warm-start weights but new cosine schedule.")
    parser.add_argument("--init_from_hf", type=str, default=None,
                        help="HuggingFace model name to initialize decoder weights from (e.g. meta-llama/Llama-3.2-1B). "
                             "Hyper-encoder stays randomly initialized.")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HF tokenizer used for byte-PPL/BPB eval decoding stats. "
                             "Must match the tokenizer the data was pretokenized with "
                             "(e.g. microsoft/Phi-3.5-mini-instruct for the Phi3.5-mini config).")
    parser.add_argument("--activation_checkpoint", action="store_true",
                        help="Recompute transformer-layer activations during backward "
                             "(gradient checkpointing) to cut activation memory. Optional with "
                             "FSDP (which already shards params/grads/optimizer); useful for extra "
                             "headroom or very long sequences. Default off.")
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze all decoder parameters, only train hyper-encoder.")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="Apply LoRA to decoder attention layers with this rank. Implies frozen base weights.")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA alpha scaling factor.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for torch, cuda, and numpy")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="HuggingFace repo ID to push checkpoints to. "
                             "Defaults to {HF_ORG}/{run_name} when --wandb is enabled.")
    parser.add_argument("--no_hf_repo", action="store_true",
                        help="Disable automatic push to HuggingFace Hub")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None, help="wandb run group")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="wandb tags")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="wandb run ID (for resuming into the same run across curriculum phases)")
    parser.add_argument("--wandb_resume", type=str, default=None, choices=["allow", "must", "never"],
                        help="wandb resume mode. Use 'must' with --wandb_id to continue a previous run.")
    parser.add_argument("--mode", type=str, default="lm", choices=["lm", "compress"],
                        help="Training mode: 'lm' for language modeling, 'compress' for compression/decompression task")
    parser.add_argument("--no_remap_codebook", action="store_true",
                        help="Disable compact codebook remapping (ablation: use full codebook with original hyper IDs)")
    parser.add_argument("--hyper_causal_mask", action="store_true",
                        help="Enable hyper causal mask: at position t, only codebook entries k <= t are available")
    parser.add_argument(
        "--online_codebook_mask",
        action="store_true",
        help="Replace the position-based hyper causal approximation with exact "
             "decoder-time row availability replayed by the dataloader. Requires "
             "--no_remap_codebook and --mode lm. Default off preserves historical "
             "training behavior.",
    )
    parser.add_argument("--token_type_loss_weight", type=float, default=0.0,
                        help="Weight for token type (base vs hyper) prediction head. 0 = disabled.")
    parser.add_argument("--eval", action="store_true",
                        help="Eval-only mode: load checkpoint, run --steps batches with no gradient, print stats.")
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler for the first N training steps and save a Chrome trace.")
    parser.add_argument("--profile_steps", type=int, default=5,
                        help="Number of active profiling steps (after warmup)")
    parser.add_argument("--profile_warmup", type=int, default=3,
                        help="Number of profiler warmup steps before active profiling")
    parser.add_argument("--disable_varlen", action="store_true",
                        help="Disable varlen attention in hyper-encoder (use padded path instead)")
    encoder_residual_group.add_argument(
        "--no_encoder_residual", action="store_true",
        help="Disable residual connection in hyper-encoder (ablation experiment). "
             "Incompatible with --zero_init_encoder_output because a zero encoder "
             "output would make the complete hypertoken embedding zero.")
    parser.add_argument("--debug_first_steps", type=int, default=0,
                        help="Print per-rank progress markers for the first N training steps. "
                             "Useful for diagnosing hangs before step 1 logging.")
    args = parser.parse_args()

    if args.no_compile:
        args.compile = False
    if args.two_axis_rope and args.gated_compressed_rope:
        parser.error(
            "--two_axis_rope and --gated_compressed_rope are mutually exclusive"
        )
    if args.two_axis_rope and not args.base_token_positions:
        parser.error("--two_axis_rope requires --base_token_positions")
    if args.gated_compressed_rope and not args.base_token_positions:
        parser.error("--gated_compressed_rope requires --base_token_positions")
    if not 0.0 <= args.base_view_replay_prob < 1.0:
        parser.error("--base_view_replay_prob must be in [0, 1)")
    if args.base_view_replay_prob and args.mode != "lm":
        parser.error("--base_view_replay_prob requires --mode lm")
    try:
        base_view_replay_microsteps(
            1,
            args.gradient_accumulation_steps,
            args.base_view_replay_prob,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.online_codebook_mask and not args.no_remap_codebook:
        parser.error("--online_codebook_mask requires --no_remap_codebook")
    if args.online_codebook_mask and args.mode != "lm":
        parser.error("--online_codebook_mask requires --mode lm")
    if args.online_codebook_mask and args.max_codebook_size == 0:
        parser.error("--online_codebook_mask requires compression (max_codebook_size > 0)")
    if args.online_codebook_mask and args.max_active_codebook_size <= 0:
        parser.error(
            "--online_codebook_mask requires max_active_codebook_size > 0"
        )
    if args.share_hyper_encoder_weights and not args.untied_hyper_encoder:
        parser.error("--share_hyper_encoder_weights requires --untied_hyper_encoder")
    if args.stop_at is None:
        args.stop_at = args.steps
    if args.hyper_lr is None:
        args.hyper_lr = args.lr
    if args.min_hyper_lr is None:
        args.min_hyper_lr = args.min_lr

    # Resolve the encoder architecture args to their EFFECTIVE values (None ->
    # the model config's default) before anything records vars(args): the stdout
    # config dump, wandb.init, and meta.pt then all agree, the eval adapter
    # rebuilds from concrete values, and validate_resume_args compares them.
    _cfg_defaults = zip2zip_llama_configs[args.model_config]
    if args.gated_compressed_rope and not (
        0 <= args.gated_rope_start_layer < _cfg_defaults.n_layers
    ):
        parser.error(
            "--gated_rope_start_layer must be in "
            f"[0, {_cfg_defaults.n_layers}) for {args.model_config}"
        )
    _head_dim = (
        getattr(_cfg_defaults.layer.attention, "head_dim", None)
        or _cfg_defaults.dim // _cfg_defaults.layer.attention.n_heads
    )
    _n_rope_pairs = _head_dim // 2
    if args.gated_compressed_rope and not (
        0 <= args.gated_rope_start_pair < _n_rope_pairs
    ):
        parser.error(
            "--gated_rope_start_pair must be in "
            f"[0, {_n_rope_pairs}) for {args.model_config}"
        )
    for _f in ("encoder_dim", "encoder_n_layers", "encoder_n_heads",
               "encoder_intermediate_size"):
        if getattr(args, _f) is None:
            setattr(args, _f, getattr(_cfg_defaults, _f))

    # Initialize distributed. Generous timeout so the first step's torch.compile
    # (which can take minutes and skews across ranks at large scale) doesn't trip
    # the NCCL watchdog before the first collective completes.
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    wall_clock_start = time.time()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # When launched via srun with CUDA_VISIBLE_DEVICES, each process sees only 1 GPU
    if torch.cuda.device_count() == 1:
        local_rank = 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Set global seed (rank-offset so each process gets different data order)
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np; np.random.seed(seed)
    import random as _random; _random.seed(seed)

    # Deduplicate output_dir: if it already exists, append (1), (2), ... on rank 0,
    # then broadcast the resolved path to all ranks.
    # Skip deduplication when resuming — we want to write back to the same directory.
    if rank == 0:
        if not args.resume_from and not args.resume_from_hf:
            base_dir = args.output_dir
            if os.path.exists(base_dir):
                n = 1
                while os.path.exists(f"{base_dir}({n})"):
                    n += 1
                args.output_dir = f"{base_dir}({n})"
                print(f"Output dir {base_dir} already exists, using {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
    output_dir_list = [args.output_dir if rank == 0 else None]
    dist.broadcast_object_list(output_dir_list, src=0)
    args.output_dir = output_dir_list[0]

    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Config: {json.dumps(vars(args), indent=2)}")

        if args.wandb:
            import wandb
            from zip2zip_core.project import WANDB_ENTITY, WANDB_PROJECT, HF_ORG

            if args.wandb_name and not args.wandb_id:
                import uuid
                uid = uuid.uuid4().hex[:4]
                args.wandb_name = f"{args.wandb_name}_{uid}"

            wandb.init(
                entity=args.wandb_entity or WANDB_ENTITY,
                project=args.wandb_project or WANDB_PROJECT,
                name=args.wandb_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                id=args.wandb_id,
                resume=args.wandb_resume,
                config=vars(args),
            )

            run_name = wandb.run.name
            if not args.no_hf_repo and args.hf_repo is None:
                args.hf_repo = f"{HF_ORG}/candidate-{run_name}"
                print(f"[hf_repo] Auto-set to {args.hf_repo}")

    # Build model
    config = zip2zip_llama_configs[args.model_config]
    replace_kwargs = dict(
        max_codebook_size=args.max_codebook_size,
        max_subtokens=args.max_subtokens,
        hyper_encoder_type=args.hyper_encoder_type,
        token_type_loss_weight=args.token_type_loss_weight,
        # Inverted at the arg boundary: --untied_hyper_encoder (default off) maps
        # to tie_hyper_encoder=False. The adapter applies the same inversion.
        tie_hyper_encoder=not args.untied_hyper_encoder,
        share_hyper_encoder_weights=args.share_hyper_encoder_weights,
        base_token_positions=args.base_token_positions,
        two_axis_rope=args.two_axis_rope,
        gated_compressed_rope=args.gated_compressed_rope,
        gated_rope_start_layer=args.gated_rope_start_layer,
        gated_rope_start_pair=args.gated_rope_start_pair,
        zero_init_encoder_output=args.zero_init_encoder_output,
        rope=dataclasses.replace(
            config.rope,
            max_seq_len=rope_cache_len(
                args.seq_len, args.max_subtokens, args.base_token_positions
            ),
        ),
    )
    if args.encoder_dim is not None:
        replace_kwargs["encoder_dim"] = args.encoder_dim
    if args.encoder_intermediate_size is not None:
        replace_kwargs["encoder_intermediate_size"] = args.encoder_intermediate_size
    if args.encoder_n_heads is not None:
        replace_kwargs["encoder_n_heads"] = args.encoder_n_heads
    if args.encoder_n_layers is not None:
        replace_kwargs["encoder_n_layers"] = args.encoder_n_layers
    config = dataclasses.replace(config, **replace_kwargs)

    model = config.build()
    model.gradient_checkpointing = args.activation_checkpoint
    if args.activation_checkpoint and rank == 0:
        print("[activation_checkpoint] gradient checkpointing enabled on transformer layers")
    if args.disable_varlen:
        model.hyper_encoder.disable_varlen = True
        if getattr(model, "hyper_output", None) is not None:
            model.hyper_output.disable_varlen = True
    if args.no_encoder_residual:
        model.encoder_residual = False
    with torch.no_grad():
        model.init_weights()
    if rank == 0:
        # Print what the encoder zero-init actually matched. With the flag on,
        # residual mode cannot silently report an empty list: it either names
        # the output gate or fails during initialization.
        print(f"[encoder_zero_init] zeroed={model.zero_init_report} "
              f"(flag={'on' if args.zero_init_encoder_output else 'off'})")
    # Keep params in fp32 on device; FSDP's MixedPrecisionPolicy casts to bf16 for
    # compute and the optimizer runs on the (sharded) fp32 params.
    model = model.to(device=device)

    # Load pretrained HF decoder weights if specified
    if args.init_from_hf:
        load_hf_pretrained(model, args.init_from_hf, device)

    # Freeze decoder / apply LoRA
    if args.freeze_decoder or args.lora_rank:
        for name, param in model.named_parameters():
            if (
                not name.startswith("hyper_encoder")
                and not name.startswith("hyper_output")
                and not name.startswith("token_type_head")
                and name != "compressed_rope_gate"
            ):
                param.requires_grad_(False)
        if (
            args.gated_compressed_rope
            and not model.compressed_rope_gate.requires_grad
        ):
            raise RuntimeError("compressed_rope_gate was unexpectedly frozen")
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[freeze_decoder] Decoder frozen. Trainable params: {trainable:,}")

    if args.lora_rank:
        from zip2zip_core.lora import apply_lora
        lora_params = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        model = model.to(device=device)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[lora] Applied LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}), added {lora_params:,} params. Trainable: {trainable:,}")

    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {param_count:,} ({param_count/1e9:.2f}B)")

    # Shard with FSDP2 (params/grads/optimizer sharded across all DP ranks),
    # with optional per-block torch.compile. Replaces DDP, which replicated the
    # full model + fp32 optimizer on every GPU and cannot fit a 3.8B model.
    if rank == 0:
        print(f"Applying FSDP2 (fully_shard) across {world_size} ranks...")
    model = apply_fsdp(model, world_size)

    if not args.eval:
        # AdamW runs directly on the FSDP-sharded fp32 params (no fp32 master copy
        # needed — the params already are fp32 and sharded, so optimizer states are
        # fp32 and sharded too). Split into decoder vs hyper_encoder param groups
        # for differential LR (the hyper_encoder is randomly init'd and benefits
        # from a higher LR).
        hyper_trainable = []
        decoder_trainable = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "hyper_encoder" in n or "hyper_output" in n or "token_type_head" in n:
                hyper_trainable.append(p)
            else:
                decoder_trainable.append(p)

        if rank == 0:
            print(f"Optimizer param groups: decoder={len(decoder_trainable)} params "
                  f"(lr={args.lr:.2e}), hyper={len(hyper_trainable)} params "
                  f"(lr={args.hyper_lr:.2e}) | FSDP-sharded fp32 AdamW")

        optimizer = torch.optim.AdamW(
            [
                {"params": decoder_trainable, "lr": args.lr},
                {"params": hyper_trainable,   "lr": args.hyper_lr},
            ],
            betas=(0.9, args.adam_beta2),
            weight_decay=args.weight_decay,
        )

    # Resume if specified. resume_dir stays None when starting fresh; the data
    # stream is repositioned after the dataloader exists (see below).
    start_step = 0
    resume_dir = None
    if args.resume_from_hf:
        if rank == 0:
            print(f"Downloading checkpoint from HuggingFace: {args.resume_from_hf}")
            resume_dir = _download_checkpoint_from_hf(args.resume_from_hf, args.resume_hf_revision)
        else:
            resume_dir = None
        # Broadcast the path from rank 0 to all ranks
        resume_dir_list = [resume_dir]
        dist.broadcast_object_list(resume_dir_list, src=0)
        resume_dir = resume_dir_list[0]
        validate_resume_args(resume_dir, args)
        start_step = load_checkpoint(model, optimizer if not args.eval else None, resume_dir, device)
    elif args.resume_from:
        resume_dir = args.resume_from
        if resume_dir == "latest":
            resume_dir = _resolve_latest_checkpoint(args.output_dir)
            if rank == 0:
                print(f"Auto-resolved latest checkpoint: {resume_dir}")
        validate_resume_args(resume_dir, args)
        start_step = load_checkpoint(model, optimizer if not args.eval else None, resume_dir, device, random_weights=args.random_weights)

    if args.reset_step and start_step != 0:
        if rank == 0:
            print(f"[reset_step] Resetting start_step from {start_step} to 0 (fresh LR schedule)")
        start_step = 0

    # Compile transformer blocks AFTER loading the checkpoint (compiling first
    # renames params to layers.N._orig_mod.* and breaks DCP resume). Param identity
    # is unchanged by compile, so the already-built optimizer stays valid.
    if args.compile:
        if rank == 0:
            print("Compiling transformer blocks with torch.compile...")
        compile_transformer_blocks(model)

    # Special-token ids the LZW compressor must never merge into the codebook,
    # derived from the active tokenizer so this is correct for Llama, Phi, etc.
    # (For Llama-3.1 this reproduces the reserved 128000-128255 range.)
    # Single source of truth in zip2zip_core.disabled_ids -- eval (lm_eval_adapter)
    # and HF export (export_phi.py / export.py) must derive the exact same set.
    from transformers import AutoTokenizer
    from zip2zip_core.disabled_ids import compute_disabled_ids, digit_ids as _digit_ids_fn

    _dl_tok = AutoTokenizer.from_pretrained(args.tokenizer)
    disabled_ids = compute_disabled_ids(
        _dl_tok, config.vocab_size, disable_digit_ids=args.disable_digit_ids
    )
    if args.disable_digit_ids and rank == 0:
        _digit_ids = sorted(_digit_ids_fn(_dl_tok, config.vocab_size))
        print(f"[data] digit ids disabled for LZW ({len(_digit_ids)}): {_digit_ids}")
    if rank == 0:
        print(f"[data] initial_vocab_size={config.vocab_size} "
              f"pad_token_id={config.pad_token_id} "
              f"disabled_ids={disabled_ids[:6]}{'...' if len(disabled_ids) > 6 else ''} "
              f"({len(disabled_ids)} ids)")

    # Dataset and dataloader
    dataloader = build_dataloader(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        max_subtokens=args.max_subtokens,
        max_codebook_size=args.max_codebook_size,
        max_active_codebook_size=args.max_active_codebook_size,
        local_batch_size=args.local_batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        pad_token_id=config.pad_token_id,
        initial_vocab_size=config.vocab_size,
        disabled_ids=disabled_ids,
        mode=args.mode,
        remap_codebook=not args.no_remap_codebook,
        online_codebook_mask=args.online_codebook_mask,
        include_base_view=(
            args.base_view_replay_prob > 0.0 and not args.eval
        ),
        debug_samples=max(0, args.debug_first_steps * max(1, args.gradient_accumulation_steps)),
    )

    # Reposition the data stream to match the checkpoint. Must happen here: the
    # dataloader does not exist yet when load_checkpoint() runs above.
    #   --reset_step: a fresh LR schedule is a new run that happens to warm-start
    #     its weights, so it starts the data from the beginning by design.
    #   --eval: scores a checkpoint and writes nothing. It must read the SAME data
    #     for every checkpoint (from shard 0), or per-step curves compare
    #     different samples; and scripts/eval_lm.sh cannot pass an override.
    if _should_restore_data_position(resume_dir, args):
        _restore_loader_state(resume_dir, dataloader, args)
    elif rank == 0 and resume_dir is not None:
        reason = "--eval" if args.eval else "--reset_step"
        print(f"[resume] data stream starts at shard 0 ({reason})")

    if rank == 0 and not args.eval:
        if args.num_workers > 0:
            print(
                f"[checkpoint] NOT RESUMABLE: --num_workers={args.num_workers} keeps "
                "the data position in worker processes. If this run is preempted, "
                "resuming it would replay data. Use --num_workers 0 "
                "(NUM_WORKERS=0) for a resumable run."
            )
        else:
            print("[checkpoint] resumable: checkpoints will record the data position")

    use_token_type_head = args.token_type_loss_weight > 0

    # ---------- Eval-only mode ----------
    if args.eval:
        model.eval()
        data_iter = iter(dataloader)
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Byte-PPL/BPB evaluation requires `transformers`. "
                "Install it in the current environment, e.g. `uv pip install transformers`."
            ) from e
        byte_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        total_loss_sum = 0.0
        total_valid_tokens = 0
        total_base_tokens = 0
        total_target_base_tokens = 0
        total_target_bytes = 0
        total_correct = 0
        total_base_correct = 0
        total_base_count = 0
        total_hyper_correct = 0
        total_hyper_count = 0
        total_relaxed_hyper_correct = 0
        total_relaxed_loss_sum = 0.0
        total_online_skipped = 0
        # Per-merge-size accuracy: index 0 unused, index k = merge size k
        max_ms = config.max_subtokens
        merge_correct = [0] * (max_ms + 1)
        merge_count = [0] * (max_ms + 1)

        if rank == 0:
            print(f"Running eval for {args.steps} steps...")

        with torch.no_grad():
            for eval_step in range(1, args.steps + 1):
                input_dict, labels = next(data_iter)
                x = input_dict["input"].to(device)
                cb = input_dict["codebook"].to(device)
                codebook_counts = input_dict.get("codebook_counts")
                if codebook_counts is not None:
                    codebook_counts = codebook_counts.to(device)
                n_base_tokens = input_dict["n_base_tokens"].to(device)
                target_n_base_tokens = input_dict["target_n_base_tokens"].to(
                    device
                )
                labels = labels.to(device)
                if args.online_codebook_mask:
                    total_online_skipped += int(
                        input_dict["online_skipped_targets"].sum().item()
                    )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        x,
                        codebook=cb,
                        hyper_causal_mask=args.hyper_causal_mask,
                        codebook_counts=codebook_counts,
                    )
                    if use_token_type_head:
                        logits, _ = output
                    else:
                        logits = output

                flat_logits = logits.flatten(0, 1).float()
                flat_labels = labels.flatten(0, 1)
                per_token_loss = F.cross_entropy(
                    flat_logits, flat_labels, reduction="none", ignore_index=-100,
                )
                valid_mask = flat_labels != -100
                total_loss_sum += per_token_loss.sum().item()
                total_valid_tokens += valid_mask.sum().item()
                total_base_tokens += n_base_tokens.sum().item()
                total_target_base_tokens += target_n_base_tokens.sum().item()
                total_target_bytes += _count_target_bytes(
                    labels, cb, config.vocab_size, config.pad_token_id, byte_tokenizer
                )

                B_eval, T_eval = labels.shape
                flat_valid_idx = torch.arange(B_eval * T_eval, device=device)[valid_mask]
                valid_batch_idx = flat_valid_idx // T_eval
                relaxed_nll = _relaxed_target_nll(
                    flat_logits[valid_mask], flat_labels[valid_mask], valid_batch_idx,
                    cb, config.vocab_size, config.pad_token_id,
                )
                total_relaxed_loss_sum += relaxed_nll.sum().item()

                valid_preds = flat_logits[valid_mask].argmax(-1)
                valid_labels = flat_labels[valid_mask]
                total_correct += (valid_preds == valid_labels).sum().item()

                base_tok_mask = valid_labels < config.vocab_size
                hyper_tok_mask = valid_labels >= config.vocab_size
                total_base_correct += (valid_preds[base_tok_mask] == valid_labels[base_tok_mask]).sum().item()
                total_base_count += base_tok_mask.sum().item()
                total_hyper_correct += (valid_preds[hyper_tok_mask] == valid_labels[hyper_tok_mask]).sum().item()
                total_hyper_count += hyper_tok_mask.sum().item()

                # Relaxed prefix accuracy for hyper tokens
                if hyper_tok_mask.any():
                    hyper_batch_idx = flat_valid_idx[hyper_tok_mask] // T_eval
                    total_relaxed_hyper_correct += _relaxed_prefix_correct(
                        valid_preds[hyper_tok_mask], valid_labels[hyper_tok_mask],
                        hyper_batch_idx, cb, config.vocab_size, config.pad_token_id,
                    )

                # Per-merge-size accuracy for hyper tokens
                if hyper_tok_mask.any():
                    # Compute merge size for each label before flattening
                    cb_indices = (labels - config.vocab_size).clamp(min=0)  # (B, T)
                    cb_idx_exp = cb_indices.unsqueeze(-1).expand(-1, -1, cb.shape[-1])  # (B, T, ms)
                    entries = cb.gather(1, cb_idx_exp)  # (B, T, ms)
                    ms_per_token = (entries != config.pad_token_id).sum(dim=-1).flatten()  # (B*T,)
                    valid_ms = ms_per_token[valid_mask]  # only valid positions
                    hyper_ms = valid_ms[hyper_tok_mask]
                    hyper_preds_correct = (valid_preds[hyper_tok_mask] == valid_labels[hyper_tok_mask])
                    for k in range(2, max_ms + 1):
                        k_mask = hyper_ms == k
                        if k_mask.any():
                            merge_correct[k] += hyper_preds_correct[k_mask].sum().item()
                            merge_count[k] += k_mask.sum().item()

                if rank == 0 and eval_step % args.log_freq == 0:
                    print(f"  eval step {eval_step}/{args.steps}")

        # All-reduce across ranks
        stats = torch.tensor([
            total_loss_sum, total_valid_tokens, total_base_tokens,
            total_target_base_tokens, total_target_bytes,
            total_correct, total_base_correct, total_base_count,
            total_hyper_correct, total_hyper_count, total_relaxed_hyper_correct,
            total_relaxed_loss_sum, total_online_skipped,
        ], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        merge_stats = torch.tensor(
            merge_correct + merge_count, device=device, dtype=torch.float64
        )
        dist.all_reduce(merge_stats, op=dist.ReduceOp.SUM)
        mc = merge_stats[: max_ms + 1].tolist()
        mn = merge_stats[max_ms + 1 :].tolist()

        (
            loss_sum, valid_tok, base_tok, target_base_tok, target_bytes,
            correct, base_correct, base_count, hyper_correct, hyper_count,
            relaxed_hyper_correct, relaxed_loss_sum, online_skipped,
        ) = stats.tolist()

        if rank == 0:
            avg_loss = loss_sum / valid_tok
            base_loss = loss_sum / base_tok
            ppl = math.exp(base_loss)
            target_base_loss = loss_sum / target_base_tok
            target_base_ppl = math.exp(target_base_loss)
            byte_ppl = math.exp(loss_sum / target_bytes)
            bpb = loss_sum / (target_bytes * math.log(2))
            relaxed_loss_valid = relaxed_loss_sum / valid_tok
            relaxed_loss = relaxed_loss_sum / base_tok
            relaxed_ppl = math.exp(relaxed_loss)
            target_relaxed_loss = relaxed_loss_sum / target_base_tok
            target_relaxed_ppl = math.exp(target_relaxed_loss)
            relaxed_byte_ppl = math.exp(relaxed_loss_sum / target_bytes)
            relaxed_bpb = relaxed_loss_sum / (target_bytes * math.log(2))
            acc = correct / valid_tok
            base_token_acc = base_correct / base_count if base_count > 0 else 0.0
            hyper_token_acc = hyper_correct / hyper_count if hyper_count > 0 else 0.0
            relaxed_hyper_acc = relaxed_hyper_correct / hyper_count if hyper_count > 0 else 0.0
            relaxed_acc = (base_correct + relaxed_hyper_correct) / valid_tok
            compression = base_tok / valid_tok
            target_compression = target_base_tok / valid_tok

            print("=" * 60)
            print("Eval Results:")
            print(f"  loss (legacy base-token denominator) = {base_loss:.4f}")
            print(f"  ppl (legacy denominator)             = {ppl:.2f}")
            print(f"  loss (target base token)              = {target_base_loss:.4f}")
            print(f"  ppl (target base token)               = {target_base_ppl:.2f}")
            print(f"  byte_ppl              = {byte_ppl:.4f}")
            print(f"  bpb                   = {bpb:.4f}")
            print(f"  loss (per valid token) = {avg_loss:.4f}")
            print(f"  relaxed_loss (legacy denominator) = {relaxed_loss:.4f}")
            print(f"  relaxed_ppl (legacy denominator)  = {relaxed_ppl:.2f}")
            print(f"  relaxed_loss (target base token)  = {target_relaxed_loss:.4f}")
            print(f"  relaxed_ppl (target base token)   = {target_relaxed_ppl:.2f}")
            print(f"  relaxed_byte_ppl      = {relaxed_byte_ppl:.4f}")
            print(f"  relaxed_bpb           = {relaxed_bpb:.4f}")
            print(f"  relaxed_loss (per valid token) = {relaxed_loss_valid:.4f}")
            print(f"  acc                   = {acc:.4f}")
            print(f"  base_token_acc        = {base_token_acc:.4f}")
            print(f"  hyper_token_acc       = {hyper_token_acc:.4f}")
            print(f"  relaxed_hyper_acc     = {relaxed_hyper_acc:.4f}")
            print(f"  relaxed_acc           = {relaxed_acc:.4f}")
            print(f"  compression (legacy)  = {compression:.2f}")
            print(f"  compression (target)  = {target_compression:.2f}")
            print(f"  total tokens evaluated = {int(valid_tok):,}")
            if args.online_codebook_mask:
                online_skip_rate = online_skipped / (valid_tok + online_skipped)
                print(
                    f"  online targets skipped = {int(online_skipped):,} "
                    f"({online_skip_rate:.6f})"
                )
            for k in range(2, max_ms + 1):
                if mn[k] > 0:
                    print(f"  hyper_acc_ms{k}         = {mc[k] / mn[k]:.4f}  (n={int(mn[k]):,})")
            print("=" * 60)

            if args.wandb:
                import wandb

                eval_dict = {
                    "eval/loss": base_loss,
                    "eval/ppl": ppl,
                    "eval/loss_target_base_token": target_base_loss,
                    "eval/ppl_target_base_token": target_base_ppl,
                    "eval/byte_ppl": byte_ppl,
                    "eval/bpb": bpb,
                    "eval/loss_per_valid_token": avg_loss,
                    "eval/relaxed_loss": relaxed_loss,
                    "eval/relaxed_ppl": relaxed_ppl,
                    "eval/relaxed_loss_target_base_token": target_relaxed_loss,
                    "eval/relaxed_ppl_target_base_token": target_relaxed_ppl,
                    "eval/relaxed_byte_ppl": relaxed_byte_ppl,
                    "eval/relaxed_bpb": relaxed_bpb,
                    "eval/relaxed_loss_per_valid_token": relaxed_loss_valid,
                    "eval/acc": acc,
                    "eval/base_token_acc": base_token_acc,
                    "eval/hyper_token_acc": hyper_token_acc,
                    "eval/relaxed_hyper_acc": relaxed_hyper_acc,
                    "eval/relaxed_acc": relaxed_acc,
                    "eval/compression": compression,
                    "eval/compression_target": target_compression,
                }
                if args.online_codebook_mask:
                    eval_dict["eval/online_skipped_targets"] = online_skipped
                    eval_dict["eval/online_skipped_target_rate"] = (
                        online_skipped / (valid_tok + online_skipped)
                    )
                for k in range(2, max_ms + 1):
                    if mn[k] > 0:
                        eval_dict[f"eval/hyper_acc_ms{k}"] = mc[k] / mn[k]
                wandb.log(eval_dict)
                wandb.finish()

        dist.destroy_process_group()
        return

    # Training loop
    global_batch_tokens = args.local_batch_size * args.seq_len * world_size * args.gradient_accumulation_steps
    schedule_total_steps = args.steps
    if args.max_tokens is not None:
        schedule_total_steps = max(1, math.ceil(args.max_tokens / global_batch_tokens))
    if rank == 0:
        print(f"Global batch: {global_batch_tokens:,} tokens/step")
        if args.max_tokens is not None:
            print(
                "Stopping by token budget: "
                f"max_tokens={args.max_tokens:,} "
                f"(estimated {schedule_total_steps:,} steps at current global batch)"
            )
        else:
            print(f"Stopping by step budget: steps={args.steps:,}")
        print(f"Starting training from step {start_step + 1}...")
        if args.base_token_positions:
            print(f"[base_token_positions] enabled: RoPE positions follow the "
                  f"uncompressed stream (rope cache = seq_len*max_subtokens = "
                  f"{args.seq_len * args.max_subtokens} positions)")
        if args.two_axis_rope:
            print(
                "[two_axis_rope] enabled: even complex pairs use base-stream "
                "positions; odd pairs use compressed-token positions"
            )
        if args.gated_compressed_rope:
            print(
                "[gated_compressed_rope] enabled: zero-initialized learned "
                f"compressed-position delta in layers "
                f"{args.gated_rope_start_layer}-{config.n_layers - 1}, "
                f"complex pairs {args.gated_rope_start_pair}-"
                f"{config.rope.dim // 2 - 1}"
            )
        if args.base_view_replay_prob:
            print(
                "[base_view_replay] enabled: "
                f"target={args.base_view_replay_prob:.2%}, "
                f"gradient_accumulation_steps={args.gradient_accumulation_steps}; "
                "rank-identical stateless schedule, final microstep compressed"
            )
        if args.warmstart_steps:
            print(f"[warmstart] enabled: decoder-LoRA frozen for first {args.warmstart_steps} steps")
            if args.warmstart_steps >= args.warmup_steps:
                print(f"[warmstart] WARNING: warmstart_steps ({args.warmstart_steps}) >= "
                      f"warmup_steps ({args.warmup_steps}) — LoRA will unfreeze onto an "
                      f"already-decaying LR and skip its warmup. Prefer warmstart_steps < warmup_steps.")

    model.train()
    data_iter = iter(dataloader)
    step = start_step
    log_view_sums = {
        "compressed": new_view_metric_sums(),
        "base_view": new_view_metric_sums(),
    }
    log_online_skipped = 0
    log_online_targets = 0
    log_tokens = 0
    start_time = time.time()

    # Profiler setup
    if args.profile:
        profile_trace_dir = os.path.join(args.output_dir, "profiler_traces")
        if rank == 0:
            os.makedirs(profile_trace_dir, exist_ok=True)
            print(f"Profiling enabled: 3 warmup + {args.profile_steps} active steps → {profile_trace_dir}")
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=args.profile_warmup, active=args.profile_steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_trace_dir),
            record_shapes=True,
        )
        profiler.start()
    else:
        profiler = None

    while True:
        if args.max_tokens is not None:
            tokens_seen_before_step = step * global_batch_tokens
            if tokens_seen_before_step >= args.max_tokens:
                break
        elif step >= args.stop_at:
            break

        step += 1

        # Set learning rate (differential: decoder vs hyper_encoder)
        lr       = get_lr(step, args.warmup_steps, schedule_total_steps, args.lr,       args.min_lr)
        hyper_lr = get_lr(step, args.warmup_steps, schedule_total_steps, args.hyper_lr, args.min_hyper_lr)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = hyper_lr

        optimizer.zero_grad()
        model.zero_grad()
        replay_microsteps = base_view_replay_microsteps(
            step,
            args.gradient_accumulation_steps,
            args.base_view_replay_prob,
        )

        for micro_step in range(args.gradient_accumulation_steps):
          debug_enabled = step <= args.debug_first_steps
          with torch.profiler.record_function("fwd+bwd"):
            _debug_rank_log(debug_enabled, rank, step, micro_step, "before next(data_iter)")
            input_dict, labels = next(data_iter)
            _debug_rank_log(
                debug_enabled,
                rank,
                step,
                micro_step,
                f"after next(data_iter) input={tuple(input_dict['input'].shape)} "
                f"codebook={tuple(input_dict['codebook'].shape)} labels={tuple(labels.shape)}",
            )

            use_base_view = micro_step in replay_microsteps
            x = input_dict[
                "base_input" if use_base_view else "input"
            ].to(device)
            cb = input_dict["codebook"].to(device)
            codebook_counts = input_dict.get("codebook_counts")
            if codebook_counts is not None and not use_base_view:
                codebook_counts = codebook_counts.to(device)
            else:
                codebook_counts = None
            n_base_tokens = input_dict[
                "base_n_base_tokens" if use_base_view else "n_base_tokens"
            ].to(device)
            target_n_base_tokens = (
                input_dict["base_n_base_tokens"]
                if use_base_view
                else input_dict["target_n_base_tokens"]
            ).to(device)
            labels = (
                input_dict["base_labels"] if use_base_view else labels
            ).to(device)
            _debug_rank_log(
                debug_enabled,
                rank,
                step,
                micro_step,
                f"after to(device) n_base_tokens_sum={int(n_base_tokens.sum().item())}",
            )

            # FSDP2 grad accumulation: only reduce-scatter grads on the last micro-step
            is_last = micro_step == args.gradient_accumulation_steps - 1
            model.set_requires_gradient_sync(is_last)
            ctx = contextlib.nullcontext()

            with ctx:
              with torch.profiler.record_function("fwd"):
                _debug_rank_log(debug_enabled, rank, step, micro_step, "before model forward")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(
                        x,
                        codebook=None if use_base_view else cb,
                        hyper_causal_mask=(
                            args.hyper_causal_mask and not use_base_view
                        ),
                        codebook_counts=codebook_counts,
                    )
                    if use_token_type_head:
                        logits, token_type_logits = output
                    else:
                        logits = output
                _debug_rank_log(debug_enabled, rank, step, micro_step, f"after model forward logits={tuple(logits.shape)}")
                flat_logits = logits.flatten(0, 1).float()
                flat_labels = labels.flatten(0, 1)
                with torch.profiler.record_function("cross_entropy"):
                    _debug_rank_log(debug_enabled, rank, step, micro_step, "before cross_entropy")
                    per_token_loss = F.cross_entropy(
                        flat_logits,
                        flat_labels,
                        reduction="none",
                        ignore_index=-100,
                    )
                _debug_rank_log(debug_enabled, rank, step, micro_step, "after cross_entropy")
                valid_mask = flat_labels != -100
                if args.online_codebook_mask and not use_base_view:
                    skipped = int(
                        input_dict["online_skipped_targets"].sum().item()
                    )
                    log_online_skipped += skipped
                    log_online_targets += int(valid_mask.sum().item()) + skipped
                loss_sum = per_token_loss.sum()
                valid_count = valid_mask.sum()
                backward_loss = loss_sum / valid_count
                legacy_base_token_loss = loss_sum / n_base_tokens.sum()
                target_base_token_loss = (
                    loss_sum / target_n_base_tokens.sum()
                )
                # Token type loss: predict if next token is base (0) or hyper (1)
                if use_token_type_head:
                    type_targets = (labels >= config.vocab_size).float().flatten(0, 1)
                    valid_targets = type_targets[valid_mask]
                    valid_type_logits = token_type_logits.flatten(0, 1).float()[valid_mask]
                    type_loss = F.binary_cross_entropy_with_logits(
                        valid_type_logits,
                        valid_targets,
                        reduction="mean",
                    )
                    backward_loss = backward_loss + args.token_type_loss_weight * type_loss
                loss = backward_loss / args.gradient_accumulation_steps
              with torch.profiler.record_function("bwd"):
                _debug_rank_log(debug_enabled, rank, step, micro_step, "before backward")
                loss.backward()
                _debug_rank_log(debug_enabled, rank, step, micro_step, "after backward")

            # Keep compressed and replay metrics in disjoint raw-count buckets.
            # The optimizer still sees the mean over every accumulation
            # microstep; only reporting is stratified.
            view_sums = log_view_sums[
                "base_view" if use_base_view else "compressed"
            ]
            backward_value = backward_loss.item()
            n_valid = int(valid_count.item())
            legacy_base_n = int(n_base_tokens.sum().item())
            target_base_n = int(target_n_base_tokens.sum().item())
            view_sums["nll_sum"] += loss_sum.item()
            view_sums["valid_count"] += n_valid
            view_sums["legacy_base_count"] += legacy_base_n
            view_sums["target_base_count"] += target_base_n
            view_sums["micro_count"] += 1
            view_sums["compat_backward_loss_sum"] += backward_value
            view_sums["compat_loss_legacy_sum"] += (
                legacy_base_token_loss.item()
            )
            view_sums["compat_loss_target_sum"] += (
                target_base_token_loss.item()
            )
            view_sums["compat_compression_legacy_sum"] += (
                legacy_base_n / n_valid
            )
            view_sums["compat_compression_target_sum"] += (
                target_base_n / n_valid
            )
            with torch.no_grad():
                valid_preds = flat_logits[valid_mask].argmax(-1)
                valid_labels = flat_labels[valid_mask]
                correct_n = int((valid_preds == valid_labels).sum().item())
                view_sums["correct"] += correct_n
                view_sums["compat_acc_sum"] += correct_n / n_valid
                base_tok_mask = valid_labels < config.vocab_size
                hyper_tok_mask = valid_labels >= config.vocab_size
                base_n = int(base_tok_mask.sum().item())
                hyper_n = int(hyper_tok_mask.sum().item())
                if base_tok_mask.any():
                    base_correct_n = int(
                        (
                            valid_preds[base_tok_mask]
                            == valid_labels[base_tok_mask]
                        ).sum().item()
                    )
                    view_sums["base_correct"] += base_correct_n
                    view_sums["base_count"] += base_n
                    view_sums["compat_base_acc_sum"] += (
                        base_correct_n / base_n
                    )
                if hyper_tok_mask.any():
                    hyper_correct_n = int(
                        (
                            valid_preds[hyper_tok_mask]
                            == valid_labels[hyper_tok_mask]
                        ).sum().item()
                    )
                    view_sums["hyper_correct"] += hyper_correct_n
                    view_sums["hyper_count"] += hyper_n
                    view_sums["compat_hyper_acc_sum"] += (
                        hyper_correct_n / hyper_n
                    )
                    # Relaxed prefix accuracy
                    B_t, T_t = labels.shape
                    flat_valid_idx = torch.arange(B_t * T_t, device=device)[valid_mask]
                    hyper_batch_idx = flat_valid_idx[hyper_tok_mask] // T_t
                    relaxed_n = _relaxed_prefix_correct(
                        valid_preds[hyper_tok_mask], valid_labels[hyper_tok_mask],
                        hyper_batch_idx, cb, config.vocab_size, config.pad_token_id,
                    )
                    view_sums["relaxed_hyper_correct"] += relaxed_n
                    view_sums["compat_relaxed_hyper_acc_sum"] += (
                        relaxed_n / hyper_n
                    )
            if use_token_type_head:
                with torch.no_grad():
                    type_preds = (valid_type_logits > 0).float()
                    type_correct_n = int(
                        (type_preds == valid_targets).sum().item()
                    )
                    view_sums["type_nll_sum"] += (
                        type_loss.item() * n_valid
                    )
                    view_sums["type_count"] += n_valid
                    view_sums["type_correct"] += type_correct_n
                    view_sums["compat_type_loss_sum"] += type_loss.item()
                    view_sums["compat_type_acc_sum"] += (
                        type_correct_n / n_valid
                    )
                    # Per-class accuracy
                    base_mask_cls = valid_targets == 0
                    hyper_mask_cls = valid_targets == 1
                    base_type_n = int(base_mask_cls.sum().item())
                    hyper_type_n = int(hyper_mask_cls.sum().item())
                    if base_mask_cls.any():
                        base_type_correct_n = int(
                            (type_preds[base_mask_cls] == 0).sum().item()
                        )
                        view_sums["base_type_correct"] += base_type_correct_n
                        view_sums["base_type_count"] += base_type_n
                        view_sums["compat_base_type_acc_sum"] += (
                            base_type_correct_n / base_type_n
                        )
                    if hyper_mask_cls.any():
                        hyper_type_correct_n = int(
                            (type_preds[hyper_mask_cls] == 1).sum().item()
                        )
                        view_sums["hyper_type_correct"] += (
                            hyper_type_correct_n
                        )
                        view_sums["hyper_type_count"] += hyper_type_n
                        view_sums["compat_hyper_type_acc_sum"] += (
                            hyper_type_correct_n / hyper_type_n
                        )
                    view_sums["compat_hyper_ratio_sum"] += (
                        hyper_type_n / n_valid
                    )

        # Phased warm-start: for the first --warmstart_steps optimizer steps, null
        # the decoder-LoRA grads AFTER backward+grad-sync but BEFORE clip/step, so
        # AdamW skips them (no update, no momentum accumulation) while the hyper-
        # encoder(s) train alone and stabilize. requires_grad stays True on every
        # param, so FSDP's reduce-scatter fires uniformly on all ranks every step
        # (no rank-divergence). Nulling before clip means the hyper grads are
        # clipped on their own norm, exactly as if the LoRA weren't there.
        if args.warmstart_steps and step <= args.warmstart_steps:
            for p in optimizer.param_groups[0]["params"]:
                p.grad = None
            if rank == 0 and step == 1:
                print(f"[warmstart] decoder-LoRA frozen for first {args.warmstart_steps} "
                      f"steps; hyper-encoder(s) training alone")
        elif args.warmstart_steps and rank == 0 and step == args.warmstart_steps + 1:
            print(f"[warmstart] step {step}: unfreezing decoder-LoRA (full training)")

        # Clip + step directly on the FSDP-sharded fp32 params.
        debug_enabled = step <= args.debug_first_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # clip_grad_norm_ may return a DTensor under FSDP; reduce to a python float.
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        _debug_rank_log(debug_enabled, rank, step, -1, f"after grad clip grad_norm={float(grad_norm):.4f}")

        _debug_rank_log(debug_enabled, rank, step, -1, "before optimizer.step")
        optimizer.step()
        _debug_rank_log(debug_enabled, rank, step, -1, "after optimizer.step")

        log_tokens += args.local_batch_size * args.seq_len * args.gradient_accumulation_steps

        # Logging
        if step % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_step_time = elapsed / args.log_freq
            _debug_rank_log(debug_enabled, rank, step, -1, "before metric all_reduce")
            compressed_sums = _all_reduce_view_metric_sums(
                log_view_sums["compressed"], device
            )
            base_view_sums = _all_reduce_view_metric_sums(
                log_view_sums["base_view"], device
            )
            compressed_metrics = finalize_view_metric_sums(compressed_sums)
            base_view_metrics = finalize_view_metric_sums(base_view_sums)
            mixed_backward_loss = finalize_mixed_objective(
                compressed_sums, base_view_sums
            )
            replay_count = base_view_sums["micro_count"]
            total_micro_count = (
                compressed_sums["micro_count"] + replay_count
            )
            replay_rate = (
                replay_count / total_micro_count
                if total_micro_count
                else 0.0
            )
            _debug_rank_log(debug_enabled, rank, step, -1, "after metric all_reduce")

            gate_stats = None
            if args.gated_compressed_rope:
                gate_values = _unwrap_model(model).compressed_rope_gate.detach()
                if hasattr(gate_values, "full_tensor"):
                    gate_values = gate_values.full_tensor()
                gate_values = gate_values.float()
                gate_stats = {
                    "min": gate_values.min().item(),
                    "max": gate_values.max().item(),
                    "mean": gate_values.mean().item(),
                    "rms": gate_values.square().mean().sqrt().item(),
                    "layer_mean": gate_values.mean(dim=1).tolist(),
                    "layer_rms": gate_values.square().mean(dim=1).sqrt().tolist(),
                    "pair_mean": gate_values.mean(dim=0).tolist(),
                    "pair_rms": gate_values.square().mean(dim=0).sqrt().tolist(),
                }

            if args.online_codebook_mask:
                online_counts_tensor = torch.tensor(
                    [log_online_skipped, log_online_targets],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(online_counts_tensor, op=dist.ReduceOp.SUM)
                online_skipped_value, online_targets_value = (
                    online_counts_tensor.tolist()
                )
                online_skip_rate = (
                    online_skipped_value / online_targets_value
                    if online_targets_value
                    else 0.0
                )

            tokens_per_sec = log_tokens * world_size / elapsed

            if rank == 0:
                total_tokens_seen = step * global_batch_tokens
                # Historical top-level keys are now explicitly compressed-only.
                # They retain the old per-micro averaging formula, while the
                # slash-prefixed view metrics below use exact raw counts.
                _base_loss_val = compressed_metrics["compat_loss_legacy"]
                _target_loss_val = compressed_metrics["compat_loss_target"]

                def _safe_ppl(value):
                    if not math.isfinite(value) or value > 60:
                        return float("inf")
                    return math.exp(value)

                # Guard ppl: a diverged/inf loss must not crash the logging (and
                # the whole job). Print inf instead and flag it.
                _ppl_val = _safe_ppl(_base_loss_val)
                _target_ppl_val = _safe_ppl(_target_loss_val)
                if not math.isfinite(_base_loss_val):
                    print(f"[WARN] step {step}: non-finite base_loss={_base_loss_val} "
                          f"(training diverged or numerical issue)")
                log_msg = (
                    f"step={step:6d} | loss={_base_loss_val:.4f} | "
                    f"ppl={_ppl_val:.2f} | "
                    f"target_loss={_target_loss_val:.4f} | "
                    f"acc={compressed_metrics['compat_acc']:.4f} | "
                    f"base_token_acc={compressed_metrics['compat_base_acc']:.4f} | "
                    f"hyper_token_acc={compressed_metrics['compat_hyper_acc']:.4f} | "
                    f"relaxed_acc={compressed_metrics['compat_relaxed_hyper_acc']:.4f} | "
                    f"backward_loss={compressed_metrics['compat_backward_loss']:.4f} | "
                    f"compression={compressed_metrics['compat_compression_legacy']:.2f} | "
                    f"target_compression={compressed_metrics['compat_compression_target']:.2f} | "
                )
                if use_token_type_head:
                    log_msg += (
                        f"type_loss={compressed_metrics['compat_type_loss']:.4f} | "
                        f"type_acc={compressed_metrics['compat_type_acc']:.4f} | "
                        f"base_acc={compressed_metrics['compat_base_type_acc']:.4f} | "
                        f"hyper_acc={compressed_metrics['compat_hyper_type_acc']:.4f} | "
                        f"hyper_ratio={compressed_metrics['compat_hyper_ratio']:.4f} | "
                    )
                if args.online_codebook_mask:
                    log_msg += (
                        f"online_skip={online_skip_rate:.6f} "
                        f"(n={int(online_skipped_value)}) | "
                    )
                if args.base_view_replay_prob:
                    log_msg += (
                        f"mixed_backward_loss={mixed_backward_loss:.4f} | "
                        f"base_view_loss="
                        f"{base_view_metrics['loss_target_base_token']:.4f} | "
                        f"base_replay={replay_rate:.4f} | "
                    )
                if gate_stats is not None:
                    log_msg += (
                        f"rope_gate_mean={gate_stats['mean']:.3e} | "
                        f"rope_gate_rms={gate_stats['rms']:.3e} | "
                    )
                hyper_lr_suffix = f" | hyper_lr={hyper_lr:.2e}" if hyper_lr != lr else ""
                log_msg += (
                    f"lr={lr:.2e}{hyper_lr_suffix} | grad_norm={grad_norm:.4f} | "
                    f"avg_step_time={avg_step_time:.2f}s | "
                    f"tok/s={tokens_per_sec:.0f} | "
                    f"tokens={total_tokens_seen/1e9:.2f}B"
                )
                print(log_msg)

                if args.wandb:
                    log_dict = {
                        "loss": _base_loss_val,
                        "ppl": _ppl_val,
                        "loss_target_base_token": _target_loss_val,
                        "ppl_target_base_token": _target_ppl_val,
                        "acc": compressed_metrics["compat_acc"],
                        "base_token_acc": compressed_metrics["compat_base_acc"],
                        "hyper_token_acc": compressed_metrics["compat_hyper_acc"],
                        "relaxed_acc": compressed_metrics[
                            "compat_relaxed_hyper_acc"
                        ],
                        "backward_loss": compressed_metrics[
                            "compat_backward_loss"
                        ],
                        "compression": compressed_metrics[
                            "compat_compression_legacy"
                        ],
                        "compression_target": compressed_metrics[
                            "compat_compression_target"
                        ],
                        "objective/backward_loss": mixed_backward_loss,
                        "compressed/loss_legacy_base_token": compressed_metrics[
                            "loss_legacy_base_token"
                        ],
                        "compressed/ppl_legacy_base_token": _safe_ppl(
                            compressed_metrics["loss_legacy_base_token"]
                        ),
                        "compressed/loss_target_base_token": compressed_metrics[
                            "loss_target_base_token"
                        ],
                        "compressed/ppl_target_base_token": _safe_ppl(
                            compressed_metrics["loss_target_base_token"]
                        ),
                        "compressed/loss_per_valid_token": compressed_metrics[
                            "loss_per_valid_token"
                        ],
                        "compressed/compression_legacy": compressed_metrics[
                            "compression_legacy"
                        ],
                        "compressed/compression_target": compressed_metrics[
                            "compression_target"
                        ],
                        "compressed/acc": compressed_metrics["acc"],
                        "compressed/base_token_acc": compressed_metrics[
                            "base_token_acc"
                        ],
                        "compressed/hyper_token_acc": compressed_metrics[
                            "hyper_token_acc"
                        ],
                        "compressed/relaxed_hyper_acc": compressed_metrics[
                            "relaxed_hyper_acc"
                        ],
                        "compressed/relaxed_acc": compressed_metrics[
                            "relaxed_acc"
                        ],
                        "compressed/backward_loss": compressed_metrics[
                            "compat_backward_loss"
                        ],
                        # Counts are summed over distributed ranks, so name
                        # their unit explicitly rather than implying logical
                        # accumulation microbatches.
                        "compressed/rank_microbatches": compressed_metrics[
                            "micro_count"
                        ],
                        "lr": lr,
                        "hyper_lr": hyper_lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "avg_step_time": avg_step_time,
                        "tokens_per_sec": tokens_per_sec,
                        "total_tokens": total_tokens_seen,
                    }
                    if use_token_type_head:
                        log_dict["type_loss"] = compressed_metrics[
                            "compat_type_loss"
                        ]
                        log_dict["type_acc"] = compressed_metrics[
                            "compat_type_acc"
                        ]
                        log_dict["base_type_acc"] = compressed_metrics[
                            "compat_base_type_acc"
                        ]
                        log_dict["hyper_type_acc"] = compressed_metrics[
                            "compat_hyper_type_acc"
                        ]
                        log_dict["hyper_ratio"] = compressed_metrics[
                            "compat_hyper_ratio"
                        ]
                        for key in (
                            "type_loss",
                            "type_acc",
                            "base_type_acc",
                            "hyper_type_acc",
                            "hyper_ratio",
                        ):
                            log_dict[f"compressed/{key}"] = compressed_metrics[
                                key
                            ]
                    if args.online_codebook_mask:
                        log_dict["online_skipped_targets"] = online_skipped_value
                        log_dict["online_skipped_target_rate"] = online_skip_rate
                    if args.base_view_replay_prob:
                        log_dict["base_view_replay_rate"] = replay_rate
                        log_dict.update(
                            {
                                "base_view/loss_target_base_token":
                                    base_view_metrics[
                                        "loss_target_base_token"
                                    ],
                                "base_view/ppl_target_base_token": _safe_ppl(
                                    base_view_metrics[
                                        "loss_target_base_token"
                                    ]
                                ),
                                "base_view/loss_per_valid_token":
                                    base_view_metrics[
                                        "loss_per_valid_token"
                                    ],
                                "base_view/acc": base_view_metrics["acc"],
                                "base_view/base_token_acc":
                                    base_view_metrics["base_token_acc"],
                                "base_view/rank_microbatches":
                                    base_view_metrics["micro_count"],
                            }
                        )
                        if use_token_type_head:
                            for key in (
                                "type_loss",
                                "type_acc",
                                "base_type_acc",
                            ):
                                log_dict[f"base_view/{key}"] = (
                                    base_view_metrics[key]
                                )
                    if gate_stats is not None:
                        log_dict.update(
                            {
                                "rope_gate_min": gate_stats["min"],
                                "rope_gate_max": gate_stats["max"],
                                "rope_gate_mean": gate_stats["mean"],
                                "rope_gate_rms": gate_stats["rms"],
                            }
                        )
                        for layer_offset, (mean, rms) in enumerate(
                            zip(
                                gate_stats["layer_mean"],
                                gate_stats["layer_rms"],
                            )
                        ):
                            layer_id = (
                                args.gated_rope_start_layer + layer_offset
                            )
                            log_dict[
                                f"rope_gate/layer_{layer_id:02d}_mean"
                            ] = mean
                            log_dict[
                                f"rope_gate/layer_{layer_id:02d}_rms"
                            ] = rms
                        for pair_offset, (mean, rms) in enumerate(
                            zip(
                                gate_stats["pair_mean"],
                                gate_stats["pair_rms"],
                            )
                        ):
                            pair_id = (
                                args.gated_rope_start_pair + pair_offset
                            )
                            log_dict[
                                f"rope_gate/pair_{pair_id:02d}_mean"
                            ] = mean
                            log_dict[
                                f"rope_gate/pair_{pair_id:02d}_rms"
                            ] = rms
                    wandb.log(log_dict, step=step)

            log_view_sums = {
                "compressed": new_view_metric_sums(),
                "base_view": new_view_metric_sums(),
            }
            log_online_skipped = 0
            log_online_targets = 0
            log_tokens = 0
            start_time = time.time()

        # Save checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args, args.output_dir,
                            dataloader=dataloader)

        # Profiler step
        if profiler is not None:
            profiler.step()
            if step - start_step >= args.profile_warmup + args.profile_steps:
                if rank == 0:
                    print("Profiling complete — stopping early.")
                break

    # Final save
    if profiler is not None:
        profiler.stop()
        if rank == 0:
            labels = {"fwd+bwd", "fwd", "bwd", "hyper_encoder", "Main LM", "hyper_lm_head", "lm_head", "cross_entropy"}
            sub_prefixes = ("layer_", "hyper_encoder.", "he.", "logit_cat")

            # Merge duplicate events (CPU vs CUDA rows share the same key)
            merged = {}
            for e in profiler.key_averages():
                if e.key in labels or e.key.startswith(sub_prefixes):
                    if e.key not in merged:
                        merged[e.key] = 0
                    merged[e.key] = max(merged[e.key], e.device_time_total)

            # Divide by active steps to get per-step time
            n_steps = args.profile_steps
            for k in merged:
                merged[k] /= n_steps
            fwd_bwd = merged.get("fwd+bwd", 0)
            fwd = merged.get("fwd", 0)
            bwd_estimated = fwd_bwd - fwd
            merged["bwd"] = bwd_estimated

            top_level = {k: v for k, v in merged.items() if k in labels}
            children = {k: v for k, v in merged.items() if k not in labels}

            fwd_time = top_level.get("fwd", 1)

            # Parent → child mapping
            fwd_children = ["hyper_encoder", "Main LM", "lm_head", "hyper_lm_head", "cross_entropy"]
            parent_children = {
                "Main LM": "layer_",
                "hyper_encoder": ("hyper_encoder.", "he."),
                "hyper_lm_head": "logit_cat",
            }

            def print_entry(name, value, indent=0, ref=None):
                ref = ref or fwd_time
                pct = value / ref * 100 if ref > 0 else 0
                prefix = "  " * indent
                print(f"{prefix}{name:<{40 - indent * 2}} {value / 1e3:>10.1f}ms {pct:>9.1f}%")

            # Table 1: Overview
            print(f"\n{'Component':<40} {'CUDA total':>12} {'% of fwd':>10}")
            print("-" * 64)
            for name in ["fwd+bwd", "fwd", "bwd (est.)"]:
                lookup = "bwd" if name == "bwd (est.)" else name
                if lookup not in top_level:
                    continue
                print_entry(name, top_level[lookup])
                if lookup == "fwd":
                    for child_name in fwd_children:
                        if child_name not in top_level:
                            continue
                        print_entry(child_name, top_level[child_name], indent=1)

            # Table 2: Detailed breakdown
            all_items = {**top_level, **children}
            print(f"\n{'Detailed breakdown':<40} {'CUDA total':>12} {'% of parent':>10}")
            print("-" * 64)
            for child_name in fwd_children:
                if child_name not in top_level:
                    continue
                parent_val = top_level[child_name]
                print_entry(child_name, parent_val, ref=fwd_time)
                gc_prefixes = parent_children.get(child_name, None)
                if gc_prefixes:
                    if isinstance(gc_prefixes, str):
                        gc_prefixes = (gc_prefixes,)
                    kids = sorted(
                        [(k, v) for k, v in all_items.items()
                         if any(k.startswith(p) or k == p for p in gc_prefixes) and k != child_name],
                        key=lambda kv: kv[1], reverse=True,
                    )
                    for ck, cv in kids:
                        print_entry(ck, cv, indent=1, ref=parent_val)
    else:
        save_checkpoint(model, optimizer, step, args, args.output_dir, hf_repo=args.hf_repo,
                        dataloader=dataloader)

    if rank == 0:
        torch.cuda.synchronize()
        wall_clock_elapsed = time.time() - wall_clock_start
        hours, rem = divmod(wall_clock_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Training complete! Wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
        if args.wandb:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
