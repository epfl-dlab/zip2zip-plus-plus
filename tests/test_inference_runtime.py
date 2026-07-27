"""Standalone compressed-generation regression tests."""

from __future__ import annotations

import importlib.util
import os
import types

import pytest
import torch


@pytest.fixture(scope="module")
def inference_module():
    path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "inference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_standalone_inference_runtime_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCodebook:
    def __init__(self, mapping):
        self.mapping = mapping

    def to_dict(self):
        return dict(self.mapping)


class _FakeManager:
    def __init__(self, vocab_size: int, entry: list[int], pad_id: int = 0):
        self.vocab_size = vocab_size
        self.entry = list(entry)
        self.pad_id = pad_id
        self.internal_codebook_manager = self
        self.reset()

    def reset(self):
        self.calls: list[list[int]] = []
        self._seeded = False
        self.mapping: dict[int, list[int]] = {}
        self.updates = None
        self.indices = None

    def update_codebooks(self, ids):
        self.calls.append([int(token) for token in ids[0].tolist()])
        if not self._seeded:
            self._seeded = True
            padded = self.entry + [self.pad_id] * (4 - len(self.entry))
            self.updates = torch.tensor([[padded]], dtype=torch.long)
            self.indices = [[0]]
            self.mapping[self.vocab_size] = list(self.entry)
        else:
            self.updates = torch.empty((1, 0, 4), dtype=torch.long)
            self.indices = [[]]

    def get_new_codes(self):
        return self.updates, self.indices

    def get_codebooks(self):
        return [_FakeCodebook(self.mapping)]


class _FakeCompressor:
    def __init__(self, compressed_prompt: list[int]):
        self.compressed_prompt = list(compressed_prompt)
        self.calls: list[list[int]] = []

    def encode(self, ids, **kwargs):
        self.calls.append(list(ids))
        return list(self.compressed_prompt), [1] * len(self.compressed_prompt), None


class _FakeTokenizer:
    eos_token_id = 2

    def __init__(self, prompt_ids: list[int]):
        self.prompt_ids = list(prompt_ids)
        self.decoded: list[list[int]] = []

    def encode(self, text, add_special_tokens=False):
        return list(self.prompt_ids)

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(token) for token in ids]
        self.decoded.append(ids)
        return " ".join(str(token) for token in ids if token != self.eos_token_id)

    def convert_ids_to_tokens(self, ids):
        return [f"t{int(token)}" for token in ids]

    def get_added_vocab(self):
        return {}


class _FakeModel:
    def __init__(self, choices: list[int], vocab_size: int = 8, max_cb: int = 4):
        self.choices = list(choices)
        self.calls: list[list[int]] = []
        self.zip2zip_config = types.SimpleNamespace(
            vocab_size=vocab_size,
            max_codebook_size=max_cb,
            pad_token_id=0,
        )

    def reset_inference_cache(self):
        self.calls.clear()

    def __call__(self, tokens, **kwargs):
        self.calls.append([int(token) for token in tokens[0].tolist()])
        choice = self.choices[len(self.calls) - 1]
        logits = torch.full(
            (
                1,
                tokens.shape[1],
                self.zip2zip_config.vocab_size
                + self.zip2zip_config.max_codebook_size,
            ),
            -100.0,
        )
        logits[0, -1, choice] = 100.0
        return logits

    forward = __call__


def _run(inference_module, *, entry, choices):
    vocab_size = 8
    prompt_ids = [1, 4, 5, 4, 5]
    manager = _FakeManager(vocab_size, entry)
    compressor = _FakeCompressor([1, vocab_size])
    tokenizer = _FakeTokenizer(prompt_ids)
    model = _FakeModel(choices, vocab_size=vocab_size)
    output, _ = inference_module.generate(
        "prompt",
        model,
        manager,
        compressor,
        tokenizer,
        {tokenizer.eos_token_id},
        max_new_tokens=4,
        temperature=0.0,
        device="cpu",
    )
    return output, model, manager, compressor, tokenizer


def test_generated_hypertoken_is_expanded_but_context_stays_compressed(
    inference_module,
):
    output, model, manager, compressor, tokenizer = _run(
        inference_module,
        entry=[4, 5],
        choices=[8, 2],
    )

    assert output == "4 5"
    assert compressor.calls == [[1, 4, 5, 4, 5]]
    assert manager.calls == [[1, 4, 5, 4, 5], [4, 5]]
    assert model.calls == [[1, 8], [1, 8, 8]]
    assert [4, 5] in tokenizer.decoded
    assert all(
        token < model.zip2zip_config.vocab_size
        for decoded in tokenizer.decoded
        for token in decoded
    )


def test_eos_inside_hypertoken_stops_before_eos(inference_module):
    output, model, manager, _, _ = _run(
        inference_module,
        entry=[4, 2, 5],
        choices=[8],
    )
    assert output == "4"
    assert manager.calls == [[1, 4, 5, 4, 5]]
    assert model.calls == [[1, 8]]


def test_unavailable_hypertoken_stops_without_decoding_it(inference_module):
    output, model, manager, _, tokenizer = _run(
        inference_module,
        entry=[4, 5],
        choices=[9],
    )
    assert output == ""
    assert manager.calls == [[1, 4, 5, 4, 5]]
    assert model.calls == [[1, 8]]
    assert not any(9 in decoded for decoded in tokenizer.decoded)


def test_temperature_zero_is_greedy(inference_module):
    logits = torch.tensor([-3.0, 1.0, 9.0, 2.0])
    assert inference_module._sample_next(logits, 0.0) == 2
