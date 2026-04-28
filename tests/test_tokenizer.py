import pytest
from pathlib import Path
from transformers import AutoTokenizer

from zip2zip_core.tokenizer import Zip2ZipTokenizer, LZWVerificationResult

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = str(REPO_ROOT / "assets" / "hf_tokenizer" / "Llama-3.1-8B")

REPEATED_TEXT = "The quick brown fox jumps over the lazy dog. " * 20


@pytest.fixture(scope="session")
def hf_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


@pytest.fixture(scope="session")
def tokenizer(hf_tokenizer):
    return Zip2ZipTokenizer(
        hf_bpe_tokenizer=hf_tokenizer,
        max_codebook_size=4096,
        max_subtokens=4,
    )


class TestBannedMethods:
    def test_call_raises(self, tokenizer):
        with pytest.raises(NotImplementedError):
            tokenizer("hello")

    def test_encode_plus_raises(self, tokenizer):
        with pytest.raises(NotImplementedError):
            tokenizer._encode_plus("hello")


class TestBatchEncodePlus:
    def test_returns_input_ids_and_attention_mask(self, tokenizer):
        enc = tokenizer.batch_encode_plus(["hello world"])
        assert "input_ids" in enc
        assert "attention_mask" in enc

    def test_return_codebook(self, tokenizer):
        enc = tokenizer.batch_encode_plus(["hello world"], return_codebook=True)
        assert "codebooks" in enc
        cb = enc["codebooks"][0]
        for key, val in cb.to_dict().items():
            assert key >= tokenizer.initial_vocab_size
            assert all(v < tokenizer.initial_vocab_size for v in val)
            assert len(val) <= tokenizer.max_subtokens

    def test_compression(self, tokenizer):
        raw_hf = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        base_len = len(raw_hf.encode(REPEATED_TEXT, add_special_tokens=False))
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT])
        comp_len = len(enc["input_ids"][0])
        assert comp_len < base_len

    def test_batch(self, tokenizer):
        texts = ["hello", "world", "hello world"]
        enc = tokenizer.batch_encode_plus(texts)
        assert len(enc["input_ids"]) == 3

    def test_padding(self, tokenizer):
        texts = ["hi", "The quick brown fox jumps over the lazy dog"]
        enc = tokenizer.batch_encode_plus(texts, padding=True)
        assert len(enc["input_ids"][0]) == len(enc["input_ids"][1])
        assert 0 in enc["attention_mask"][0]


class TestRoundtrip:
    @pytest.mark.parametrize("text", [
        "Hello, world!",
        REPEATED_TEXT,
        "def foo():\n    return 42\n",
        "Unicode: cafe\u0301 \u2603 \u00e9\u00e8\u00ea",
    ])
    def test_encode_decode_matches(self, tokenizer, text):
        enc = tokenizer.batch_encode_plus([text])
        decoded = tokenizer.decode(
            enc["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        assert decoded == text

    def test_batch_roundtrip(self, tokenizer):
        texts = ["Hello!", "The quick brown fox.", "1 + 1 = 2"]
        enc = tokenizer.batch_encode_plus(texts)
        for ids, original in zip(enc["input_ids"], texts):
            decoded = tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            assert decoded == original

    def test_decode_return_codebook(self, tokenizer):
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT])
        result = tokenizer.decode(enc["input_ids"][0], return_codebook=True)
        assert isinstance(result, tuple)
        text, codebook = result
        assert isinstance(text, str)
        assert hasattr(codebook, "to_dict")


class TestPprint:
    def test_prints_output(self, tokenizer, capsys):
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT])
        tokenizer.pprint(enc["input_ids"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_returns_none(self, tokenizer):
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT])
        result = tokenizer.pprint(enc["input_ids"])
        assert result is None


class TestVerifyLzw:
    def test_perfect_sequence(self, tokenizer):
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT])
        result = tokenizer.verify_lzw(enc["input_ids"][0])
        assert isinstance(result, LZWVerificationResult)
        assert result.is_perfect is True
        assert result.mismatches == []
        assert result.num_tokens == result.num_canonical_tokens

    def test_faulty_sequence(self, tokenizer):
        enc = tokenizer.batch_encode_plus([REPEATED_TEXT], return_codebook=True)
        compressed_ids = list(enc["input_ids"][0])
        cb = enc["codebooks"][0].to_dict()
        # find a hypertoken and expand it back to base tokens (simulate a missed merge)
        for i, tid in enumerate(compressed_ids):
            if tid in cb:
                base_tokens = cb[tid]
                faulty_ids = compressed_ids[:i] + base_tokens + compressed_ids[i + 1:]
                break
        result = tokenizer.verify_lzw(faulty_ids)
        assert result.is_perfect is False
        assert len(result.mismatches) > 0
