"""Standalone zip2zip pretraining script using DDP + torch.compile.

Usage:
    torchrun --nnodes=N --nproc_per_node=4 -m zip2zip_core.train \
        --data_dir /path/to/tokens \
        --max_subtokens 2 \
        --steps 19000

Curriculum training:
    Phase 1: --max_subtokens 2 --steps 6000
    Phase 2: --max_subtokens 3 --steps 12000 --resume_from checkpoint_dir
    Phase 3: --max_subtokens 4 --steps 19000 --resume_from checkpoint_dir
"""

import argparse
import contextlib
import dataclasses
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


def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(model, optimizer, step, args, output_dir):
    """Save model and optimizer state. Only rank 0 saves."""
    rank = dist.get_rank()
    ckpt_dir = os.path.join(output_dir, f"step_{step}")

    dist.barrier()
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save the unwrapped model (without DDP wrapper)
        raw_model = model.module if hasattr(model, "module") else model
        # For compiled models, unwrap further
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod
        torch.save(raw_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save({"step": step, "args": vars(args)}, os.path.join(ckpt_dir, "meta.pt"))
        print(f"[Rank 0] Saved checkpoint at step {step}")
    dist.barrier()


def load_checkpoint(model, optimizer, resume_dir, device):
    """Load model and optimizer state.

    Handles curriculum transitions where max_subtokens changes between phases
    by zero-padding the hyper_encoder position embedding.
    """
    meta = torch.load(os.path.join(resume_dir, "meta.pt"), map_location="cpu")
    step = meta["step"]

    model_state = torch.load(os.path.join(resume_dir, "model.pt"), map_location="cpu")

    # Get the raw model (unwrap DDP and compile)
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # Strip FSDP wrapper prefixes if present (for loading FSDP checkpoints into DDP)
    cleaned_state = {}
    for k, v in model_state.items():
        new_key = k.replace("_fsdp_wrapped_module.", "")
        cleaned_state[new_key] = v
    model_state = cleaned_state

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

    if curriculum_transition:
        if dist.get_rank() == 0:
            print("Curriculum transition detected — skipping optimizer state load (will re-init)")
    else:
        opt_state = torch.load(os.path.join(resume_dir, "optimizer.pt"), map_location="cpu")
        optimizer.load_state_dict(opt_state)

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
    parser.add_argument("--max_active_codebook_size", type=int, default=2048)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--local_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=19000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="zip2zip-core")
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    if args.no_compile:
        args.compile = False

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

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"World size: {world_size}")
        print(f"Config: {json.dumps(vars(args), indent=2)}")

        if args.wandb:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )

    # Build model
    config = zip2zip_llama_configs[args.model_config]
    replace_kwargs = dict(max_codebook_size=args.max_codebook_size, max_subtokens=args.max_subtokens)
    if args.encoder_dim is not None:
        replace_kwargs["encoder_dim"] = args.encoder_dim
    if args.encoder_intermediate_size is not None:
        replace_kwargs["encoder_intermediate_size"] = args.encoder_intermediate_size
    if args.encoder_n_heads is not None:
        replace_kwargs["encoder_n_heads"] = args.encoder_n_heads
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

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Resume if specified
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from, device)

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
    )

    # Training loop
    global_batch_tokens = args.local_batch_size * args.seq_len * world_size * args.gradient_accumulation_steps
    if rank == 0:
        print(f"Global batch: {global_batch_tokens:,} tokens/step")
        print(f"Starting training from step {start_step + 1}...")

    model.train()
    data_iter = iter(dataloader)
    step = start_step
    log_loss = 0.0
    log_tokens = 0
    start_time = time.time()

    while step < args.steps:
        step += 1

        # Set learning rate
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(args.gradient_accumulation_steps):
            input_dict, labels = next(data_iter)

            x = input_dict["input"].to(device)
            cb = input_dict["codebook"].to(device)
            labels = labels.to(device)

            # Only sync gradients on last micro-step
            is_last = micro_step == args.gradient_accumulation_steps - 1
            ctx = contextlib.nullcontext() if is_last else model.no_sync()

            with ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x, codebook=cb)
                loss = F.cross_entropy(
                    logits.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    reduction="mean",
                )
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        log_loss += accum_loss
        log_tokens += args.local_batch_size * args.seq_len * args.gradient_accumulation_steps

        # Logging
        if step % args.log_freq == 0:
            elapsed = time.time() - start_time
            avg_loss = log_loss / args.log_freq

            # All-reduce loss for global average
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

            tokens_per_sec = log_tokens * world_size / elapsed

            if rank == 0:
                total_tokens_seen = step * global_batch_tokens
                print(
                    f"step={step:6d} | loss={loss_tensor.item():.4f} | "
                    f"lr={lr:.2e} | grad_norm={grad_norm:.4f} | "
                    f"tok/s={tokens_per_sec:.0f} | "
                    f"tokens={total_tokens_seen/1e9:.2f}B"
                )

                if args.wandb:
                    wandb.log(
                        {
                            "loss": loss_tensor.item(),
                            "lr": lr,
                            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "tokens_per_sec": tokens_per_sec,
                            "total_tokens": total_tokens_seen,
                        },
                        step=step,
                    )

            log_loss = 0.0
            log_tokens = 0
            start_time = time.time()

        # Save checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(model, optimizer, step, args, args.output_dir)

    # Final save
    save_checkpoint(model, optimizer, step, args, args.output_dir)

    if rank == 0:
        print("Training complete!")
        if args.wandb:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
