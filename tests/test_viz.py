import os
import pytest
from pathlib import Path
from transformers import AutoTokenizer

from zip2zip_core.viz import (
    BLUE, YELLOW, ORANGE, RED, DARK_RED, BROWN, BLACK, RESET,
    ColoredToken,
    colorize_by_ngram,
    colorize_by_ppl,
    colorize_random,
    render_colored_tokens,
    _decode_preserving_leading_space,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = str(REPO_ROOT / "assets" / "hf_tokenizer" / "Llama-3.1-8B")


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


@pytest.fixture
def codebook():
    """Codebook where 1000→[10,11] (2-gram), 1001→[20,21,22] (3-gram), 1002→[30,31,32,33] (4-gram)."""
    return {
        1000: [10, 11],
        1001: [20, 21, 22],
        1002: [30, 31, 32, 33],
    }


# ── colorize_by_ngram ────────────────────────────────────────────────────────

class TestColorizeByNgram:
    def test_base_token_is_blue(self, codebook):
        result = colorize_by_ngram([42], codebook)
        assert len(result) == 1
        assert result[0].token_ids == [42]
        assert result[0].color == BLUE

    def test_2gram_is_yellow(self, codebook):
        result = colorize_by_ngram([1000], codebook)
        assert result[0].token_ids == [10, 11]
        assert result[0].color == YELLOW

    def test_3gram_is_orange(self, codebook):
        result = colorize_by_ngram([1001], codebook)
        assert result[0].token_ids == [20, 21, 22]
        assert result[0].color == ORANGE

    def test_4gram_is_red(self, codebook):
        result = colorize_by_ngram([1002], codebook)
        assert result[0].token_ids == [30, 31, 32, 33]
        assert result[0].color == RED

    def test_5gram_is_dark_red(self):
        cb = {2000: [1, 2, 3, 4, 5]}
        result = colorize_by_ngram([2000], cb)
        assert result[0].color == DARK_RED

    def test_6gram_falls_back_to_brown(self):
        cb = {3000: [1, 2, 3, 4, 5, 6]}
        result = colorize_by_ngram([3000], cb)
        assert result[0].color == BROWN

    def test_special_token_is_black(self, codebook):
        result = colorize_by_ngram([42], codebook, special_token_ids={42})
        assert result[0].color == BLACK

    def test_special_token_overrides_ngram_color(self, codebook):
        result = colorize_by_ngram([1000], codebook, special_token_ids={1000})
        assert result[0].color == BLACK

    def test_mixed_sequence(self, codebook):
        result = colorize_by_ngram([42, 1000, 1001], codebook)
        assert len(result) == 3
        assert result[0].color == BLUE
        assert result[1].color == YELLOW
        assert result[2].color == ORANGE

    def test_empty_input(self, codebook):
        assert colorize_by_ngram([], codebook) == []

    def test_none_special_tokens_defaults_to_empty(self, codebook):
        result = colorize_by_ngram([42], codebook, special_token_ids=None)
        assert result[0].color == BLUE


# ── colorize_by_ppl ──────────────────────────────────────────────────────────

class TestColorizeByPpl:
    def test_lowest_ppl_is_brightest(self):
        result = colorize_by_ppl([1, 2], [0.0, 10.0], {})
        assert "255;255;255" in result[0].color
        assert "0;0;255" in result[1].color

    def test_expands_via_codebook(self):
        cb = {100: [1, 2]}
        result = colorize_by_ppl([100], [5.0], cb)
        assert result[0].token_ids == [1, 2]

    def test_length_matches_input(self):
        result = colorize_by_ppl([1, 2, 3], [1.0, 2.0, 3.0], {})
        assert len(result) == 3


# ── colorize_random ──────────────────────────────────────────────────────────

class TestColorizeRandom:
    def test_returns_one_per_token(self):
        result = colorize_random([1, 2, 3])
        assert len(result) == 3

    def test_each_has_ansi_color(self):
        result = colorize_random([1])
        assert result[0].color.startswith("\033[38;5;")

    def test_expands_via_codebook(self):
        result = colorize_random([100], codebook={100: [1, 2]})
        assert result[0].token_ids == [1, 2]

    def test_no_codebook_keeps_original(self):
        result = colorize_random([42])
        assert result[0].token_ids == [42]

    def test_empty_input(self):
        assert colorize_random([]) == []


# ── render_colored_tokens ────────────────────────────────────────────────────

class TestRenderColoredTokens:
    def test_single_token(self, tokenizer):
        # token 791 = "the"
        tokens = [ColoredToken(token_ids=[279], color=BLUE)]
        result = render_colored_tokens(tokens, tokenizer)
        assert BLUE in result
        assert RESET in result
        assert "the" in result

    def test_multiple_tokens(self, tokenizer):
        tokens = [
            ColoredToken(token_ids=[9906], color=BLUE),   # "hello"
            ColoredToken(token_ids=[1917], color=YELLOW),  # "world"
        ]
        result = render_colored_tokens(tokens, tokenizer)
        assert BLUE in result
        assert YELLOW in result

    def test_empty_list(self, tokenizer):
        assert render_colored_tokens([], tokenizer) == ""


# ── _decode_preserving_leading_space ─────────────────────────────────────────

class TestDecodePreservingLeadingSpace:
    def test_normal_token(self, tokenizer):
        result = _decode_preserving_leading_space(tokenizer, [15339])
        assert result == "hello"

    def test_leading_space_preserved(self, tokenizer):
        # token 279 = " the" in Llama-3 (has leading space marker)
        result = _decode_preserving_leading_space(tokenizer, [279])
        assert result.startswith(" ")

    def test_non_decodable_raises(self, tokenizer):
        with pytest.raises(ValueError, match="Non-decodable"):
            _decode_preserving_leading_space(tokenizer, [9999999])


# ── end-to-end: colorizer → renderer ────────────────────────────────────────

class TestEndToEnd:
    def test_ngram_colorize_then_render(self, tokenizer):
        codebook = {200000: [15339, 1917]}
        colored = colorize_by_ngram([200000], codebook)
        result = render_colored_tokens(colored, tokenizer)
        assert YELLOW in result  # 2-gram → yellow
        assert "hello" in result
        assert "world" in result

    def test_base_tokens_roundtrip(self, tokenizer):
        ids = tokenizer.encode("The cat sat", add_special_tokens=False)
        colored = colorize_by_ngram(ids, {})
        result = render_colored_tokens(colored, tokenizer)
        for color_code in [BLUE]:
            assert color_code in result
