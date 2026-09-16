"""HuggingFace Hub upload utilities for zip2zip-core checkpoints."""

from __future__ import annotations

import os


def _ensure_branch(api, repo_id: str, branch: str, *, recreate: bool = False) -> None:
    """Create ``branch`` from the repository's root commit if it does not exist.

    ``create_branch`` defaults to the head of ``main``. For the release layout
    that is exactly wrong: ``hf`` would inherit ``model.pt``/``optimizer.pt``
    from the checkpoint upload that runs first. Starting from the root commit
    (the ``.gitattributes`` created with the repo) gives an empty branch.

    ``recreate`` deletes an existing branch first, so its history no longer
    carries files inherited before this logic existed. Deleting files with
    ``delete_patterns`` cleans only the branch tip.
    """
    refs = api.list_repo_refs(repo_id)
    exists = any(ref.name == branch for ref in refs.branches)
    if exists and not recreate:
        return
    if exists:
        print(f"Recreating branch {branch} of {repo_id} from the root commit...")
        api.delete_branch(repo_id, branch=branch)
    commits = api.list_repo_commits(repo_id, revision="main")
    if not commits:
        raise RuntimeError(f"{repo_id} has no commits on main; cannot root {branch}")
    # Newest first, so the last entry is the initial commit.
    root_commit = commits[-1].commit_id
    api.create_branch(repo_id, branch=branch, revision=root_commit, exist_ok=True)


def upload_folder(
    repo_id: str,
    folder_path: str,
    branch: str = "main",
    step: int | None = None,
    label: str = "checkpoint",
    api=None,
    *,
    delete_patterns: list[str] | None = None,
    private: bool | None = None,
    recreate_branch: bool = False,
):
    """Upload a folder to a HuggingFace Hub repo branch.

    Creates the repo and branch if they don't exist. ``delete_patterns`` removes
    matching remote files that the folder does not re-add, in the same commit,
    so a branch can be made to hold exactly the folder's contents. ``private``
    applies only when this call creates the repo. ``recreate_branch`` rebuilds a
    non-main branch from the root commit before uploading.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()

    create_kwargs = {} if private is None else {"private": private}
    api.create_repo(repo_id, exist_ok=True, **create_kwargs)

    if branch != "main":
        _ensure_branch(api, repo_id, branch, recreate=recreate_branch)

    commit_msg = f"Step {step} ({label})" if step is not None else label
    print(f"Pushing {label} to {repo_id} (branch: {branch})...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        path_in_repo=".",
        revision=branch,
        commit_message=commit_msg,
        delete_patterns=delete_patterns,
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
