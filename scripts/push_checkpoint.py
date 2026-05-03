"""Push a local checkpoint directory to HuggingFace Hub.

Automatically detects checkpoint format and pushes to the appropriate branch:
  - Training (model.pt)    → main branch (core format, for resume)
  - Exported (safetensors) → hf branch (for inference)

For training checkpoints, also auto-exports to HF format and pushes to the hf branch.

Usage:
    python scripts/push_checkpoint.py \
        --ckpt_dir /mnt/scratch/checkpoints/ft/step_2000 \
        --repo_id epfl-dlab/Llaza-3.2-1B-v0.1

    # Skip auto-export:
    python scripts/push_checkpoint.py \
        --ckpt_dir /mnt/scratch/checkpoints/ft/step_2000 \
        --repo_id epfl-dlab/Llaza-3.2-1B-v0.1 \
        --no_export
"""

import argparse
import os
import re
import sys
import tempfile


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "ext", "torchtitan"))


def main():
    parser = argparse.ArgumentParser(description="Push a local checkpoint to HuggingFace Hub")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. epfl-dlab/Llaza-3.2-1B-v0.1)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step number. Default: extracted from ckpt_dir name (e.g. step_6000 -> 6000)")
    parser.add_argument("--branch", type=str, default=None,
                        help="Target branch: 'main' for training checkpoints, 'hf' for exported safetensors. "
                             "Default: auto-detected from directory contents.")
    parser.add_argument("--no_export", action="store_true",
                        help="Skip auto-export to HF format for training checkpoints.")
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    step = args.step
    if step is None:
        match = re.search(r"step_(\d+)", os.path.basename(ckpt_dir))
        if match:
            step = int(match.group(1))
        else:
            raise ValueError(
                f"Could not extract step from '{os.path.basename(ckpt_dir)}'. "
                "Pass --step explicitly."
            )

    files = set(os.listdir(ckpt_dir))
    is_exported = "zip2zip_config.json" in files and "model.safetensors" in files
    is_training = "model.pt" in files and "meta.pt" in files

    if args.branch is None:
        if is_training and not is_exported:
            args.branch = "main"
        elif is_exported and not is_training:
            args.branch = "hf"
        else:
            raise ValueError(
                f"Cannot auto-detect branch from {ckpt_dir}. "
                "Pass --branch main (training) or --branch hf (exported)."
            )
        print(f"Auto-detected branch: {args.branch}")

    from zip2zip_core.hub import upload_folder
    from zip2zip_core.model_card import write_model_card
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True)

    # Generate model card for training checkpoints
    train_args = {}
    if is_training:
        import torch
        meta = torch.load(os.path.join(ckpt_dir, "meta.pt"), map_location="cpu", weights_only=False)
        train_args = meta.get("args", {})
        write_model_card(os.path.join(ckpt_dir, "README.md"), args.repo_id, step, train_args)

    # Upload the checkpoint
    upload_folder(args.repo_id, ckpt_dir, branch=args.branch, step=step,
                  label="exported" if args.branch == "hf" else "training checkpoint", api=api)

    # Auto-export training checkpoints to hf branch
    if is_training and not args.no_export and args.branch != "hf":
        from zip2zip_core.export import export

        base_model = train_args.get("init_from_hf")
        if not base_model:
            print("WARNING: meta.pt missing 'init_from_hf' — skipping auto-export.")
            return

        with tempfile.TemporaryDirectory(prefix="zip2zip_export_") as export_dir:
            print(f"\n{'='*60}")
            print(f"Auto-exporting to HF format...")
            print(f"{'='*60}")
            export(
                ckpt_dir=ckpt_dir,
                output_dir=export_dir,
                base_model=base_model,
                model_config=train_args.get("model_config"),
                max_codebook_size=train_args.get("max_codebook_size", 4096),
                causal=train_args.get("hyper_causal_mask", False),
            )
            write_model_card(os.path.join(export_dir, "README.md"), args.repo_id, step, train_args)
            upload_folder(args.repo_id, export_dir, branch="hf", step=step,
                          label="exported HF format", api=api)


if __name__ == "__main__":
    main()
