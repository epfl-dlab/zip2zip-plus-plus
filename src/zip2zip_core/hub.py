"""HuggingFace Hub upload utilities for zip2zip-core checkpoints."""

from __future__ import annotations

import os


def upload_folder(
    repo_id: str,
    folder_path: str,
    branch: str = "main",
    step: int | None = None,
    label: str = "checkpoint",
    api=None,
):
    """Upload a folder to a HuggingFace Hub repo branch.

    Creates the repo and branch if they don't exist.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()

    api.create_repo(repo_id, exist_ok=True)

    if branch != "main":
        try:
            api.create_branch(repo_id, branch=branch)
        except Exception:
            pass

    commit_msg = f"Step {step} ({label})" if step is not None else label
    print(f"Pushing {label} to {repo_id} (branch: {branch})...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        path_in_repo=".",
        revision=branch,
        commit_message=commit_msg,
    )
    print(f"Done: {repo_id} branch={branch}")


def push_checkpoint(ckpt_dir: str, repo_id: str, step: int):
    """Push a training checkpoint to HuggingFace Hub.

    Uploads to both a step-specific branch (step_{N}) and main branch,
    with an auto-generated model card. Optionally logs the HF URL to wandb.
    """
    import torch
    from zip2zip_core.model_card import write_model_card

    # Read training args for model card
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        train_args = meta.get("args", {})
    else:
        train_args = {}

    # Generate model card
    write_model_card(os.path.join(ckpt_dir, "README.md"), repo_id, step, train_args)

    from huggingface_hub import HfApi
    api = HfApi()

    # Upload to step branch
    revision = f"step_{step}"
    upload_folder(repo_id, ckpt_dir, branch=revision, step=step,
                  label="training checkpoint", api=api)

    # Also update main branch with latest checkpoint
    upload_folder(repo_id, ckpt_dir, branch="main", step=step,
                  label="training checkpoint (latest)", api=api)

    print(f"Pushed checkpoint to {repo_id} (step_{step} + main)")

    # Log to wandb if active
    try:
        import wandb
        if wandb.run is not None:
            hf_url = f"https://huggingface.co/{repo_id}/tree/{revision}"
            wandb.run.summary["hf_repo"] = repo_id
            wandb.run.summary["hf_url"] = hf_url
    except Exception:
        pass
