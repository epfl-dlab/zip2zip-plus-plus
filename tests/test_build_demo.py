from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest


def load_build_demo_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "build_demo.py"
    )
    spec = importlib.util.spec_from_file_location("_build_demo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 3

    def decode(self, token_ids, **_kwargs):
        pieces = {1: "p", 4: "a", 5: "b"}
        return "".join(pieces.get(int(token_id), "") for token_id in token_ids)

    def get_added_vocab(self):
        return {}


def test_default_question_file_and_checkpoint_output_name():
    module = load_build_demo_module()
    assert str(module.DEFAULT_QUESTION_FILE) == "/dlabscratch1/xinma/demo_question.jsonl"
    checkpoint = module.Path("/tmp/run-name/step_8000")
    assert module.checkpoint_slug(checkpoint) == "run-name_step_8000"


def test_load_questions_preserves_id_category_and_first_turn(tmp_path):
    module = load_build_demo_module()
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"question_id": 7, "category": "math", "turns": ["1+1?"]}) + "\n",
        encoding="utf-8",
    )
    assert module.load_questions(path, None) == [
        {"question_id": 7, "category": "math", "prompt": "1+1?"}
    ]


def test_model_result_uses_recorded_compressed_ids_and_online_codebook():
    module = load_build_demo_module()
    runtime = types.SimpleNamespace(tokenizer=FakeTokenizer())
    metadata = {
        "zip_token_start": 8,
        "zip_vocab_size": 16,
        "pad_token_id": 0,
        "special_token_ids": [0, 1, 2, 3],
    }
    trace = {
        "raw_generated_ids": [8, 2],
        "generated_compressed_ids": [8],
        "generated_base_ids": [4, 5],
        "codebook": {"8": [4, 5]},
    }

    result = module.model_result_view(trace, runtime=runtime, metadata=metadata)

    assert result["raw_generated_token_ids"] == [8, 2]
    assert result["compressed_token_ids"] == [8]
    assert result["reconstructed_token_ids"] == [4, 5]
    assert result["reconstructed_text"] == "ab"
    assert result["tokens"][0]["kind"] == "zip"
    assert result["tokens"][0]["expands_to_token_ids"] == [4, 5]


def test_build_demo_end_to_end_without_hf_export(tmp_path, monkeypatch):
    module = load_build_demo_module()
    checkpoint = tmp_path / "trained-model" / "step_8"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").touch()
    (checkpoint / "meta.pt").touch()
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps({"question_id": 9, "category": "qa", "turns": ["prompt"]}) + "\n",
        encoding="utf-8",
    )
    config = types.SimpleNamespace(
        vocab_size=8,
        max_codebook_size=16,
        max_subtokens=4,
        pad_token_id=0,
        base_token_positions=True,
        two_axis_rope=False,
        gated_compressed_rope=False,
        tie_hyper_encoder=False,
    )
    runtime = module.CoreRuntime(
        model=types.SimpleNamespace(zip2zip_config=config),
        train_args={"untied_hyper_encoder": True},
        tokenizer=FakeTokenizer(),
        tokenizer_name="fake-tokenizer",
        codebook_manager=None,
        compressor=None,
        compression_kwargs={
            "initial_vocab_size": 8,
            "max_codebook_size": 16,
            "max_subtokens": 4,
            "pad_token_id": 0,
            "disabled_ids": [],
        },
        stop_token_ids={2},
        device="cpu",
    )
    trace = types.SimpleNamespace(
        text="ab",
        colored_text="ab",
        prompt_base_ids=[1],
        prompt_compressed_ids=[1],
        generated_base_ids=[4, 5],
        raw_generated_ids=[8, 2],
        generated_compressed_ids=[8],
        codebook={8: [4, 5]},
        stop_reason="eos",
    )
    monkeypatch.setattr(module, "load_core_runtime", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(module, "generate_trace", lambda *_args, **_kwargs: trace)
    out_dir = tmp_path / "output"
    args = types.SimpleNamespace(
        ckpt_dir=checkpoint,
        question_file=questions,
        out_dir=out_dir,
        tokenizer=None,
        device="cpu",
        seed=42,
        max_new_tokens=8,
        temperature=0.0,
        instruct=False,
        limit=None,
        overwrite=False,
    )

    assert module.build_demo(args) == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    result_path = out_dir / manifest["models"][0]["results_file"]
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_path.name == "trained-model_step_8_demo_data.json"
    assert payload["examples"][0]["model_result"]["compressed_token_ids"] == [8]
    assert payload["examples"][0]["model_result"]["reconstructed_text"] == "ab"
    assert json.loads((out_dir / "validation.json").read_text())["num_examples"] == 1

    monkeypatch.setattr(
        module,
        "generate_trace",
        lambda *_args, **_kwargs: pytest.fail("completed question should be resumed"),
    )
    assert module.build_demo(args) == 0

    args.temperature = 0.5
    with pytest.raises(ValueError, match="changed settings"):
        module.build_demo(args)
