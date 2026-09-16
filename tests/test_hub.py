"""Tests for the Hub branch and upload helpers behind the release flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from zip2zip_core.hub import _ensure_branch, upload_folder


class StubApi:
    def __init__(self, branches=("main",), commits=("newest", "middle", "root")):
        self.branches = list(branches)
        self.commits = [SimpleNamespace(commit_id=c) for c in commits]
        self.calls = []

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(
            branches=[SimpleNamespace(name=name) for name in self.branches]
        )

    def list_repo_commits(self, repo_id, revision="main"):
        return list(self.commits)

    def create_branch(self, repo_id, *, branch, revision, exist_ok=False):
        self.calls.append(("create_branch", branch, revision))
        self.branches.append(branch)

    def delete_branch(self, repo_id, *, branch):
        self.calls.append(("delete_branch", branch))
        self.branches.remove(branch)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


def test_missing_branch_is_created_from_the_root_commit():
    api = StubApi()
    _ensure_branch(api, "org/repo", "hf")
    assert api.calls == [("create_branch", "hf", "root")]


def test_existing_branch_is_left_alone_by_default():
    api = StubApi(branches=("main", "hf"))
    _ensure_branch(api, "org/repo", "hf")
    assert api.calls == []


def test_recreate_deletes_then_roots_the_branch():
    api = StubApi(branches=("main", "hf"))
    _ensure_branch(api, "org/repo", "hf", recreate=True)
    assert api.calls == [("delete_branch", "hf"), ("create_branch", "hf", "root")]


def test_repo_without_commits_is_an_error():
    api = StubApi(commits=())
    with pytest.raises(RuntimeError, match="no commits"):
        _ensure_branch(api, "org/repo", "hf")


def test_upload_folder_forwards_private_delete_patterns_and_branch_options(tmp_path):
    api = StubApi()
    upload_folder(
        "org/repo",
        str(tmp_path),
        branch="hf",
        step=8000,
        label="export",
        api=api,
        delete_patterns=["model.pt"],
        private=True,
    )
    assert api.calls[0] == ("create_repo", {"exist_ok": True, "private": True})
    assert api.calls[1] == ("create_branch", "hf", "root")
    kind, kwargs = api.calls[2]
    assert kind == "upload_folder"
    assert kwargs["revision"] == "hf"
    assert kwargs["delete_patterns"] == ["model.pt"]
    assert kwargs["commit_message"] == "Step 8000 (export)"


def test_upload_folder_to_main_never_touches_branches(tmp_path):
    api = StubApi()
    upload_folder("org/repo", str(tmp_path), api=api)
    assert api.calls[0] == ("create_repo", {"exist_ok": True})
    assert api.calls[1][0] == "upload_folder"
    assert api.calls[1][1]["delete_patterns"] is None
    assert len(api.calls) == 2
