"""Push a local checkpoint directory to HuggingFace Hub.

Automatically detects checkpoint format and pushes to the appropriate branch:
  - Training (model.pt)    → main branch (core format, for resume)
  - Exported (safetensors) → hf branch (for inference)

Usage:
    # Auto-detect format:
    python scripts/push_checkpoint.py \
        --ckpt_dir /mnt/scratch/checkpoints/ft/step_2000 \
        --repo_id epfl-dlab/zip2zip-Llama-3.2-1B-preview

    python scripts/push_checkpoint.py \
        --ckpt_dir /mnt/scratch/export/zip2zip_1b_step1908 \
        --repo_id epfl-dlab/zip2zip-Llama-3.2-1B-preview \
        --step 1908
"""

import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description="Push a local checkpoint to HuggingFace Hub")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. epfl-dlab/zip2zip-Llama-3.2-1B-preview)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step number. Default: extracted from ckpt_dir name (e.g. step_6000 -> 6000)")
    parser.add_argument("--branch", type=str, default=None,
                        help="Target branch: 'main' for training checkpoints, 'hf' for exported safetensors. "
                             "Default: auto-detected from directory contents.")
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

    if args.branch is None:
        files = set(os.listdir(ckpt_dir))
        is_exported = "zip2zip_config.json" in files and "model.safetensors" in files
        is_training = "model.pt" in files and "meta.pt" in files
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

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True)

    if args.branch != "main":
        try:
            api.create_branch(args.repo_id, branch=args.branch)
        except Exception:
            pass

    print(f"Pushing {ckpt_dir} to {args.repo_id} (branch: {args.branch}, step: {step})...")
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=args.repo_id,
        path_in_repo=".",
        revision=args.branch,
        commit_message=f"Step {step} ({'exported' if args.branch == 'hf' else 'training'} checkpoint)",
    )

    print(f"Done. Pushed to {args.repo_id} branch={args.branch}")


if __name__ == "__main__":
    main()
