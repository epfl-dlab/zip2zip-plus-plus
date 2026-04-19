"""Push a local checkpoint directory to HuggingFace Hub.

Usage:
    python scripts/push_checkpoint.py \
        --ckpt_dir /mnt/scratch/checkpoints/zip2zip_1b_finemath_10bt_ms4/step_6000 \
        --repo_id user/model-name

    # Custom step number (default: extracted from directory name):
    python scripts/push_checkpoint.py \
        --ckpt_dir /path/to/checkpoint \
        --repo_id user/model-name \
        --step 6000
"""

import argparse
import os
import re


def main():
    parser = argparse.ArgumentParser(description="Push a local checkpoint to HuggingFace Hub")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to checkpoint directory (containing model.pt, etc.)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. user/model-name)")
    parser.add_argument("--step", type=int, default=None,
                        help="Step number. Default: extracted from ckpt_dir name (e.g. step_6000 -> 6000)")
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Extract step from directory name if not provided
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

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True)

    revision = f"step_{step}"
    try:
        api.create_branch(args.repo_id, branch=revision)
    except Exception:
        pass  # branch already exists

    print(f"Pushing {ckpt_dir} to {args.repo_id} (revision: {revision})...")
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=args.repo_id,
        path_in_repo=".",
        revision=revision,
        commit_message=f"Checkpoint at step {step}",
    )

    # Also update main branch with latest checkpoint
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=args.repo_id,
        path_in_repo=".",
        commit_message=f"Checkpoint at step {step}",
    )

    print(f"Done. Pushed to {args.repo_id} (revision: step_{step} + main)")


if __name__ == "__main__":
    main()
