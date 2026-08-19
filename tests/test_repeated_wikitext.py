"""Tests for the opt-in repeated-window WikiText corpus and task."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_repeated_wikitext as repeated  # noqa: E402


class CharacterTokenizer:
    """Minimal reversible tokenizer; one Unicode code point is one token."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in token_ids)


def load_task_utils():
    path = SCRIPTS / "lm_eval_tasks" / "utils.py"
    spec = importlib.util.spec_from_file_location("_repeated_task_utils", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detokenizer_matches_wikitext_v2_rules():
    page = (
        ' = A = \n A @-@ B , C . He said " spaced " and it \'s 5 ° C . '
        "The value is N ."
    )
    got = repeated.wikitext_detokenize(page)
    assert got == (
        ' = A =\nA-B, C. He said "spaced" and it\'s 5°C. The value is 1 .'
    )


def test_default_output_is_repo_relative_not_cwd_relative(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    assert repeated.default_output_path(8) == (
        repeated.REPO_ROOT.parent
        / "datasets"
        / "wikitext-repeat8-phi35"
        / "test.jsonl"
    )
    assert repeated.default_output_path(8).is_absolute()


def test_repeated_rows_are_token_bounded_and_cover_source_once():
    tokenizer = CharacterTokenizer()
    source = "abcdefghijklmnopqrstuvwxyz"
    rows = repeated.repeated_rows_for_text(
        source,
        tokenizer,
        source_doc_id=7,
        repeat_n=4,
        max_tokens=23,
        separator="|",
    )

    # A nominal five-character block would occupy 23 tokens (4*5 + 3), while
    # every row must stay within the explicit bound.
    assert len(rows) > 1
    assert sum(row["source_base_tokens"] for row in rows) == len(source)
    assert [row["source_block_id"] for row in rows] == list(range(len(rows)))
    assert all(row["source_doc_id"] == 7 for row in rows)

    recovered_source = []
    for row in rows:
        copies = row["text"].split("|")
        assert len(copies) == 4
        assert copies == [copies[0]] * 4
        recovered_source.append(copies[0])
        assert row["repeat_n"] == 4
        assert row["repeated_base_tokens"] == len(row["text"])
        assert row["repeated_base_tokens"] <= 23
        assert row["n_bytes"] == len(row["text"].encode("utf-8"))
        assert row["n_words"] == 1

    assert "".join(recovered_source) == source


def test_builder_writes_auditable_jsonl_and_refuses_implicit_overwrite(tmp_path):
    tokenizer = CharacterTokenizer()
    output = tmp_path / "repeat4" / "test.jsonl"
    documents = [{"page": "alpha @-@ beta"}, {"page": " \n "}]

    manifest = repeated.build_corpus(
        documents,
        tokenizer,
        output=output,
        tokenizer_name="character-test",
        repeat_n=4,
        max_tokens=40,
        separator="\n\n",
    )

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    stored_manifest = json.loads(
        (output.parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == manifest
    assert manifest["tokenizer"] == "character-test"
    assert manifest["repeat_n"] == 4
    assert manifest["num_rows"] == len(rows) > 0
    assert manifest["total_repeated_base_tokens"] == sum(
        row["repeated_base_tokens"] for row in rows
    )
    assert all(row["repeated_base_tokens"] <= 40 for row in rows)
    assert all(row["text"].count("alpha-beta") in (0, 4) for row in rows)

    with pytest.raises(FileExistsError, match="--overwrite"):
        repeated.build_corpus(
            documents,
            tokenizer,
            output=output,
            tokenizer_name="character-test",
            repeat_n=4,
            max_tokens=40,
            separator="\n\n",
        )


def test_main_prints_ready_hint(tmp_path, monkeypatch, capsys):
    output = tmp_path / "repeat8" / "test.jsonl"

    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: [
        {"page": "alpha @-@ beta"}
    ]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, use_fast):
            assert name == repeated.DEFAULT_TOKENIZER
            assert use_fast is True
            return CharacterTokenizer()

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    repeated.main(
        [
            "--repeat_n",
            "8",
            "--max_tokens",
            "160",
            "--output",
            str(output),
        ]
    )

    stdout = capsys.readouterr().out
    assert "READY: repeated-window corpus (repeat_n=8)" in stdout
    assert f"Dataset: {output.resolve()}" in stdout
    assert f"Manifest: {(output.parent / 'manifest.json').resolve()}" in stdout
    assert "--tasks zip2zip_wikitext_repeat8" in stdout


def test_non_phi_tokenizer_requires_explicit_output(capsys):
    with pytest.raises(SystemExit):
        repeated.parse_args(["--tokenizer", "example/other-tokenizer"])

    stderr = capsys.readouterr().err
    assert "--output is required with a non-default tokenizer" in stderr


def test_task_denominators_use_and_validate_exact_repeated_text():
    utils = load_task_utils()
    text = "café\n\ncafé"
    doc = {
        "text": text,
        "n_bytes": len(text.encode("utf-8")),
        "n_words": 2,
    }

    got = utils.repeated_wikitext_process_results(doc, (-12.5,))
    assert got == {
        "word_perplexity": (-12.5, 2),
        "byte_perplexity": (-12.5, len(text.encode("utf-8"))),
        "bits_per_byte": (-12.5, len(text.encode("utf-8"))),
    }

    with pytest.raises(ValueError, match="denominator mismatch"):
        utils.repeated_wikitext_process_results(
            {**doc, "n_bytes": doc["n_bytes"] + 1}, (-12.5,)
        )


@pytest.mark.parametrize("repeat_n", [4, 8])
def test_repeated_task_is_not_wired_into_defaults_or_pipeline(repeat_n):
    task_name = f"zip2zip_wikitext_repeat{repeat_n}"
    task_yaml = SCRIPTS / "lm_eval_tasks" / f"{task_name}.yaml"
    task_source = task_yaml.read_text(encoding="utf-8")

    assert f"task: {task_name}" in task_source
    assert (
        "dataset_path: epfl-dlab/zip2zip-wikitext-repeat-phi35"
        in task_source
    )
    assert f"dataset_name: repeat{repeat_n}" in task_source
    assert "revision: v1.0.0" in task_source
    assert "../datasets/" not in task_source
    assert "utils.repeated_wikitext_process_results" in task_source
    assert "expected_tokenizer: microsoft/Phi-3.5-mini-instruct" in task_source
    assert f"repeat_n: {repeat_n}" in task_source

    for path in (
        SCRIPTS / "eval_harness.py",
        SCRIPTS / "eval_presets.yaml",
        SCRIPTS / "pipeline_ft_eval_rcp.sh",
        SCRIPTS / "eval_ckpt_rcp.sh",
    ):
        assert task_name not in path.read_text(encoding="utf-8")
