"""Standalone zip2zip pretraining script using DDP + torch.compile.

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
from torch.nn.parallel import DistributedDataParallel as DDP

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.data import build_dataloader

torch.set_float32_matmul_precision('high') 


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _unwrap_model(model):
    """Unwrap DDP and torch.compile wrappers to get the raw model."""
    raw = model.module if hasattr(model, "module") else model
    if hasattr(raw, "_orig_mod"):
        raw = raw._orig_mod
    return raw


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


def save_checkpoint(model, optimizer, step, args, output_dir, push_to_hub=None):
    """Save model and optimizer state. Only rank 0 saves."""
    rank = dist.get_rank()
    ckpt_dir = os.path.join(output_dir, f"step_{step}")

    dist.barrier()
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        raw_model = _unwrap_model(model)
        torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save({"step": step, "args": vars(args)}, os.path.join(ckpt_dir, "meta.pt"))
        print(f"[Rank 0] Saved checkpoint at step {step}")

        if push_to_hub:
            _push_checkpoint_to_hub(ckpt_dir, push_to_hub, step)
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
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)

    revision = f"step_{step}"
    try:
        api.create_branch(repo_id, branch=revision)
    except Exception:
        pass  # branch already exists

    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=repo_id,
        path_in_repo=".",
        revision=revision,
        commit_message=f"Checkpoint at step {step}",
    )
    # Also update main branch with latest checkpoint
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=repo_id,
        path_in_repo=".",
        commit_message=f"Checkpoint at step {step}",
    )
    print(f"[Rank 0] Pushed checkpoint to {repo_id} (revision: {revision})")


def _clean_state_dict(model_state):
    """Strip FSDP wrapper prefixes for compatibility."""
    cleaned = {}
    for k, v in model_state.items():
        cleaned[k.replace("_fsdp_wrapped_module.", "")] = v
    return cleaned


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

    raw_model = _unwrap_model(model)

    if random_weights:
        if dist.get_rank() == 0:
            print(f"[random_weights] Skipping model weight load — keeping random init (step={step})")
        curriculum_transition = False
    else:
        model_state = torch.load(os.path.join(resume_dir, "model.pt"), map_location="cpu")
        model_state = _clean_state_dict(model_state)

        # Handle pos_embed size mismatch from curriculum phase transitions
        pos_key = "hyper_encoder.pos_embed.weight"
        curriculum_transition = False
        if pos_key in model_state:
            current_pos = raw_model.state_dict()[pos_key]
            saved_pos = model_state[pos_key]
            if saved_pos.shape[0] < current_pos.shape[0]:
                padded = torch.zeros_like(current_pos)
                padded[:saved_pos.shape[0]] = saved_pos
                model_state[pos_key] = padded
                curriculum_transition = True
                if dist.get_rank() == 0:
                    print(f"Padded pos_embed from {saved_pos.shape[0]} to {current_pos.shape[0]} positions")

        raw_model.load_state_dict(model_state)

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
        optimizer.load_state_dict(opt_state)
    else:
        if dist.get_rank() == 0:
            print("No optimizer.pt found — optimizer will start fresh")

    if dist.get_rank() == 0:
        print(f"Resumed from step {step}")
    return step


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
    parser.add_argument("--hyper_encoder_type", type=str, default="flat", choices=["flat", "hierarchical"])
    parser.add_argument("--encoder_n_layers", type=int, default=None)
    parser.add_argument("--max_active_codebook_size", type=int, default=4096)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Local checkpoint dir, or 'latest' to auto-find latest step_* in output_dir")
    parser.add_argument("--resume_from_hf", type=str, default=None,
                        help="HuggingFace repo ID to download checkpoint from (e.g. user/model-name)")
    parser.add_argument("--resume_hf_revision", type=str, default=None,
                        help="HF revision/branch to download from (e.g. step_5000). Defaults to main.")
    parser.add_argument("--random_weights", action="store_true",
                        help="Control experiment: resume step/optimizer from checkpoint but reinit model weights randomly")
    parser.add_argument("--reset_step", action="store_true",
                        help="After loading checkpoint weights, reset step to 0 (fresh LR schedule). "
                             "Useful for finetuning: warm-start weights but new cosine schedule.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for torch, cuda, and numpy")
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="HuggingFace repo ID to push checkpoints to (e.g. user/model-name)")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="zip2zip-core")
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
    parser.add_argument("--token_type_loss_weight", type=float, default=0.0,
                        help="Weight for token type (base vs hyper) prediction head. 0 = disabled.")
    parser.add_argument("--eval", action="store_true",
                        help="Eval-only mode: load checkpoint, run --steps batches with no gradient, print stats.")
    args = parser.parse_args()

    if args.no_compile:
        args.compile = False
    if args.stop_at is None:
        args.stop_at = args.steps

    # Initialize distributed
    dist.init_process_group("nccl")
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

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                group=args.wandb_group,
                tags=args.wandb_tags,
                id=args.wandb_id,
                resume=args.wandb_resume,
                config=vars(args),
            )

    # Build model
    config = zip2zip_llama_configs[args.model_config]
    replace_kwargs = dict(
        max_codebook_size=args.max_codebook_size,
        max_subtokens=args.max_subtokens,
        hyper_encoder_type=args.hyper_encoder_type,
        token_type_loss_weight=args.token_type_loss_weight,
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
    with torch.no_grad():
        model.init_weights()
    model = model.to(device=device, dtype=torch.bfloat16)

    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {param_count:,} ({param_count/1e9:.2f}B)")

    # torch.compile the model before DDP wrapping
    if args.compile:
        if rank == 0:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    if not args.eval:
        # fp32 master weights: optimizer states stay in fp32, model stays in bf16
        master_params = [p.detach().float().requires_grad_(True) for p in model.parameters()]

        # Optimizer
        optimizer = torch.optim.AdamW(
            master_params,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    # Resume if specified
    start_step = 0
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
        start_step = load_checkpoint(model, optimizer if not args.eval else None, resume_dir, device)
    elif args.resume_from:
        resume_dir = args.resume_from
        if resume_dir == "latest":
            resume_dir = _resolve_latest_checkpoint(args.output_dir)
            if rank == 0:
                print(f"Auto-resolved latest checkpoint: {resume_dir}")
        start_step = load_checkpoint(model, optimizer if not args.eval else None, resume_dir, device, random_weights=args.random_weights)

    if args.reset_step and start_step != 0:
        if rank == 0:
            print(f"[reset_step] Resetting start_step from {start_step} to 0 (fresh LR schedule)")
        start_step = 0

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
        mode=args.mode,
        remap_codebook=not args.no_remap_codebook,
    )

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
        byte_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        total_loss_sum = 0.0
        total_valid_tokens = 0
        total_base_tokens = 0
        total_target_bytes = 0
        total_correct = 0
        total_base_correct = 0
        total_base_count = 0
        total_hyper_correct = 0
        total_hyper_count = 0
        total_relaxed_hyper_correct = 0
        total_relaxed_loss_sum = 0.0
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
                n_base_tokens = input_dict["n_base_tokens"].to(device)
                labels = labels.to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(x, codebook=cb, hyper_causal_mask=args.hyper_causal_mask)
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
            total_loss_sum, total_valid_tokens, total_base_tokens, total_target_bytes,
            total_correct, total_base_correct, total_base_count,
            total_hyper_correct, total_hyper_count, total_relaxed_hyper_correct,
            total_relaxed_loss_sum,
        ], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        merge_stats = torch.tensor(
            merge_correct + merge_count, device=device, dtype=torch.float64
        )
        dist.all_reduce(merge_stats, op=dist.ReduceOp.SUM)
        mc = merge_stats[: max_ms + 1].tolist()
        mn = merge_stats[max_ms + 1 :].tolist()

        loss_sum, valid_tok, base_tok, target_bytes, correct, base_correct, base_count, hyper_correct, hyper_count, relaxed_hyper_correct, relaxed_loss_sum = stats.tolist()

        if rank == 0:
            avg_loss = loss_sum / valid_tok
            base_loss = loss_sum / base_tok
            ppl = math.exp(base_loss)
            byte_ppl = math.exp(loss_sum / target_bytes)
            bpb = loss_sum / (target_bytes * math.log(2))
            relaxed_loss_valid = relaxed_loss_sum / valid_tok
            relaxed_loss = relaxed_loss_sum / base_tok
            relaxed_ppl = math.exp(relaxed_loss)
            relaxed_byte_ppl = math.exp(relaxed_loss_sum / target_bytes)
            relaxed_bpb = relaxed_loss_sum / (target_bytes * math.log(2))
            acc = correct / valid_tok
            base_token_acc = base_correct / base_count if base_count > 0 else 0.0
            hyper_token_acc = hyper_correct / hyper_count if hyper_count > 0 else 0.0
            relaxed_hyper_acc = relaxed_hyper_correct / hyper_count if hyper_count > 0 else 0.0
            relaxed_acc = (base_correct + relaxed_hyper_correct) / valid_tok
            compression = base_tok / valid_tok

            print("=" * 60)
            print("Eval Results:")
            print(f"  loss (per base token) = {base_loss:.4f}")
            print(f"  ppl                   = {ppl:.2f}")
            print(f"  byte_ppl              = {byte_ppl:.4f}")
            print(f"  bpb                   = {bpb:.4f}")
            print(f"  loss (per valid token) = {avg_loss:.4f}")
            print(f"  relaxed_loss (per base token) = {relaxed_loss:.4f}")
            print(f"  relaxed_ppl           = {relaxed_ppl:.2f}")
            print(f"  relaxed_byte_ppl      = {relaxed_byte_ppl:.4f}")
            print(f"  relaxed_bpb           = {relaxed_bpb:.4f}")
            print(f"  relaxed_loss (per valid token) = {relaxed_loss_valid:.4f}")
            print(f"  acc                   = {acc:.4f}")
            print(f"  base_token_acc        = {base_token_acc:.4f}")
            print(f"  hyper_token_acc       = {hyper_token_acc:.4f}")
            print(f"  relaxed_hyper_acc     = {relaxed_hyper_acc:.4f}")
            print(f"  relaxed_acc           = {relaxed_acc:.4f}")
            print(f"  compression           = {compression:.2f}")
            print(f"  total tokens evaluated = {int(valid_tok):,}")
            for k in range(2, max_ms + 1):
                if mn[k] > 0:
                    print(f"  hyper_acc_ms{k}         = {mc[k] / mn[k]:.4f}  (n={int(mn[k]):,})")
            print("=" * 60)

            if args.wandb:
                import wandb

                eval_dict = {
                    "eval/loss": base_loss,
                    "eval/ppl": ppl,
                    "eval/byte_ppl": byte_ppl,
                    "eval/bpb": bpb,
                    "eval/loss_per_valid_token": avg_loss,
                    "eval/relaxed_loss": relaxed_loss,
                    "eval/relaxed_ppl": relaxed_ppl,
                    "eval/relaxed_byte_ppl": relaxed_byte_ppl,
                    "eval/relaxed_bpb": relaxed_bpb,
                    "eval/relaxed_loss_per_valid_token": relaxed_loss_valid,
                    "eval/acc": acc,
                    "eval/base_token_acc": base_token_acc,
                    "eval/hyper_token_acc": hyper_token_acc,
                    "eval/relaxed_hyper_acc": relaxed_hyper_acc,
                    "eval/relaxed_acc": relaxed_acc,
                    "eval/compression": compression,
                }
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

    model.train()
    data_iter = iter(dataloader)
    step = start_step
    log_loss = 0.0
    log_base_loss = 0.0
    log_compression = 0.0
    log_acc = 0.0
    log_base_token_acc = 0.0
    log_hyper_token_acc = 0.0
    log_relaxed_acc = 0.0
    log_type_loss = 0.0
    log_type_acc = 0.0
    log_base_type_acc = 0.0
    log_hyper_type_acc = 0.0
    log_hyper_ratio = 0.0
    log_tokens = 0
    start_time = time.time()

    while True:
        if args.max_tokens is not None:
            tokens_seen_before_step = step * global_batch_tokens
            if tokens_seen_before_step >= args.max_tokens:
                break
        elif step >= args.stop_at:
            break

        step += 1

        # Set learning rate
        lr = get_lr(step, args.warmup_steps, schedule_total_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        model.zero_grad()
        accum_loss = 0.0
        accum_base_loss = 0.0
        accum_compression = 0.0
        accum_acc = 0.0
        accum_base_token_acc = 0.0
        accum_hyper_token_acc = 0.0
        accum_relaxed_acc = 0.0
        accum_type_loss = 0.0
        accum_type_acc = 0.0
        accum_base_type_acc = 0.0
        accum_hyper_type_acc = 0.0
        accum_hyper_ratio = 0.0

        for micro_step in range(args.gradient_accumulation_steps):
            input_dict, labels = next(data_iter)

            x = input_dict["input"].to(device)
            cb = input_dict["codebook"].to(device)
            n_base_tokens = input_dict["n_base_tokens"].to(device)
            labels = labels.to(device)

            # Only sync gradients on last micro-step
            is_last = micro_step == args.gradient_accumulation_steps - 1
            ctx = contextlib.nullcontext() if is_last else model.no_sync()

            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(x, codebook=cb, hyper_causal_mask=args.hyper_causal_mask)
                    if use_token_type_head:
                        logits, token_type_logits = output
                    else:
                        logits = output
                flat_logits = logits.flatten(0, 1).float()
                flat_labels = labels.flatten(0, 1)
                per_token_loss = F.cross_entropy(
                    flat_logits,
                    flat_labels,
                    reduction="none",
                    ignore_index=-100,
                )
                valid_mask = flat_labels != -100
                loss_sum = per_token_loss.sum()
                backward_loss = loss_sum / valid_mask.sum()
                base_token_loss = loss_sum / n_base_tokens.sum()
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
                loss.backward()
            accum_loss += backward_loss.item() / args.gradient_accumulation_steps
            accum_base_loss += base_token_loss.item() / args.gradient_accumulation_steps
            accum_compression += (n_base_tokens.float().sum() / (valid_mask.sum())).item() / args.gradient_accumulation_steps
            with torch.no_grad():
                valid_preds = flat_logits[valid_mask].argmax(-1)
                valid_labels = flat_labels[valid_mask]
                accum_acc += (valid_preds == valid_labels).float().mean().item() / args.gradient_accumulation_steps
                base_tok_mask = valid_labels < config.vocab_size
                hyper_tok_mask = valid_labels >= config.vocab_size
                if base_tok_mask.any():
                    accum_base_token_acc += (valid_preds[base_tok_mask] == valid_labels[base_tok_mask]).float().mean().item() / args.gradient_accumulation_steps
                if hyper_tok_mask.any():
                    accum_hyper_token_acc += (valid_preds[hyper_tok_mask] == valid_labels[hyper_tok_mask]).float().mean().item() / args.gradient_accumulation_steps
                    # Relaxed prefix accuracy
                    B_t, T_t = labels.shape
                    flat_valid_idx = torch.arange(B_t * T_t, device=device)[valid_mask]
                    hyper_batch_idx = flat_valid_idx[hyper_tok_mask] // T_t
                    relaxed_n = _relaxed_prefix_correct(
                        valid_preds[hyper_tok_mask], valid_labels[hyper_tok_mask],
                        hyper_batch_idx, cb, config.vocab_size, config.pad_token_id,
                    )
                    accum_relaxed_acc += (relaxed_n / hyper_tok_mask.sum().item()) / args.gradient_accumulation_steps
            if use_token_type_head:
                accum_type_loss += type_loss.item() / args.gradient_accumulation_steps
                with torch.no_grad():
                    type_preds = (valid_type_logits > 0).float()
                    accum_type_acc += (type_preds == valid_targets).float().mean().item() / args.gradient_accumulation_steps
                    # Per-class accuracy
                    base_mask_cls = valid_targets == 0
                    hyper_mask_cls = valid_targets == 1
                    if base_mask_cls.any():
                        accum_base_type_acc += (type_preds[base_mask_cls] == 0).float().mean().item() / args.gradient_accumulation_steps
                    if hyper_mask_cls.any():
                        accum_hyper_type_acc += (type_preds[hyper_mask_cls] == 1).float().mean().item() / args.gradient_accumulation_steps
                    accum_hyper_ratio += hyper_mask_cls.float().mean().item() / args.gradient_accumulation_steps

        # Copy bf16 grads → fp32 master params, clip, step, copy back
        for mp, p in zip(master_params, model.parameters()):
            mp.grad = p.grad.float() if p.grad is not None else None
        grad_norm = torch.nn.utils.clip_grad_norm_(master_params, args.max_grad_norm)

        optimizer.step()

        with torch.no_grad():
            for mp, p in zip(master_params, model.parameters()):
                p.copy_(mp.to(p.dtype))

        log_loss += accum_loss
        log_base_loss += accum_base_loss
        log_compression += accum_compression
        log_acc += accum_acc
        log_base_token_acc += accum_base_token_acc
        log_hyper_token_acc += accum_hyper_token_acc
        log_relaxed_acc += accum_relaxed_acc
        if use_token_type_head:
            log_type_loss += accum_type_loss
            log_type_acc += accum_type_acc
            log_base_type_acc += accum_base_type_acc
            log_hyper_type_acc += accum_hyper_type_acc
            log_hyper_ratio += accum_hyper_ratio
        log_tokens += args.local_batch_size * args.seq_len * args.gradient_accumulation_steps

        # Logging
        if step % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_step_time = elapsed / args.log_freq
            avg_loss = log_loss / args.log_freq
            avg_base_loss = log_base_loss / args.log_freq
            avg_compression = log_compression / args.log_freq
            avg_acc = log_acc / args.log_freq
            avg_base_token_acc = log_base_token_acc / args.log_freq
            avg_hyper_token_acc = log_hyper_token_acc / args.log_freq
            avg_relaxed_acc = log_relaxed_acc / args.log_freq

            # All-reduce loss for global average
            loss_tensor = torch.tensor(avg_loss, device=device)
            base_loss_tensor = torch.tensor(avg_base_loss, device=device)
            compression_tensor = torch.tensor(avg_compression, device=device)
            acc_tensor = torch.tensor(avg_acc, device=device)
            base_token_acc_tensor = torch.tensor(avg_base_token_acc, device=device)
            hyper_token_acc_tensor = torch.tensor(avg_hyper_token_acc, device=device)
            relaxed_acc_tensor = torch.tensor(avg_relaxed_acc, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(base_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(compression_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(base_token_acc_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(hyper_token_acc_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(relaxed_acc_tensor, op=dist.ReduceOp.AVG)

            if use_token_type_head:
                avg_type_loss = log_type_loss / args.log_freq
                avg_type_acc = log_type_acc / args.log_freq
                avg_base_type_acc = log_base_type_acc / args.log_freq
                avg_hyper_type_acc = log_hyper_type_acc / args.log_freq
                avg_hyper_ratio = log_hyper_ratio / args.log_freq
                type_loss_tensor = torch.tensor(avg_type_loss, device=device)
                type_acc_tensor = torch.tensor(avg_type_acc, device=device)
                base_type_acc_tensor = torch.tensor(avg_base_type_acc, device=device)
                hyper_type_acc_tensor = torch.tensor(avg_hyper_type_acc, device=device)
                hyper_ratio_tensor = torch.tensor(avg_hyper_ratio, device=device)
                dist.all_reduce(type_loss_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(type_acc_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(base_type_acc_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(hyper_type_acc_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(hyper_ratio_tensor, op=dist.ReduceOp.AVG)

            tokens_per_sec = log_tokens * world_size / elapsed

            if rank == 0:
                total_tokens_seen = step * global_batch_tokens
                log_msg = (
                    f"step={step:6d} | loss={base_loss_tensor.item():.4f} | "
                    f"ppl={math.exp(base_loss_tensor.item()):.2f} | "
                    f"acc={acc_tensor.item():.4f} | "
                    f"base_token_acc={base_token_acc_tensor.item():.4f} | "
                    f"hyper_token_acc={hyper_token_acc_tensor.item():.4f} | "
                    f"relaxed_acc={relaxed_acc_tensor.item():.4f} | "
                    f"backward_loss={loss_tensor.item():.4f} | "
                    f"compression={compression_tensor.item():.2f} | "
                )
                if use_token_type_head:
                    log_msg += (
                        f"type_loss={type_loss_tensor.item():.4f} | type_acc={type_acc_tensor.item():.4f} | "
                        f"base_acc={base_type_acc_tensor.item():.4f} | hyper_acc={hyper_type_acc_tensor.item():.4f} | "
                        f"hyper_ratio={hyper_ratio_tensor.item():.4f} | "
                    )
                log_msg += (
                    f"lr={lr:.2e} | grad_norm={grad_norm:.4f} | "
                    f"avg_step_time={avg_step_time:.2f}s | "
                    f"tok/s={tokens_per_sec:.0f} | "
                    f"tokens={total_tokens_seen/1e9:.2f}B"
                )
                print(log_msg)

                if args.wandb:
                    log_dict = {
                        "loss": base_loss_tensor.item(),
                        "ppl": math.exp(base_loss_tensor.item()),
                        "acc": acc_tensor.item(),
                        "base_token_acc": base_token_acc_tensor.item(),
                        "hyper_token_acc": hyper_token_acc_tensor.item(),
                        "relaxed_acc": relaxed_acc_tensor.item(),
                        "backward_loss": loss_tensor.item(),
                        "compression": compression_tensor.item(),
                        "lr": lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "avg_step_time": avg_step_time,
                        "tokens_per_sec": tokens_per_sec,
                        "total_tokens": total_tokens_seen,
                    }
                    if use_token_type_head:
                        log_dict["type_loss"] = type_loss_tensor.item()
                        log_dict["type_acc"] = type_acc_tensor.item()
                        log_dict["base_type_acc"] = base_type_acc_tensor.item()
                        log_dict["hyper_type_acc"] = hyper_type_acc_tensor.item()
                        log_dict["hyper_ratio"] = hyper_ratio_tensor.item()
                    wandb.log(log_dict, step=step)

            log_loss = 0.0
            log_base_loss = 0.0
            log_compression = 0.0
            log_acc = 0.0
            log_base_token_acc = 0.0
            log_hyper_token_acc = 0.0
            log_relaxed_acc = 0.0
            log_type_loss = 0.0
            log_type_acc = 0.0
            log_base_type_acc = 0.0
            log_hyper_type_acc = 0.0
            log_hyper_ratio = 0.0
            log_tokens = 0
            start_time = time.time()

        # Save checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args, args.output_dir, push_to_hub=args.push_to_hub)

    # Final save
    save_checkpoint(model, optimizer, step, args, args.output_dir, push_to_hub=args.push_to_hub)

    if rank == 0:
        print("Training complete!")
        if args.wandb:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
