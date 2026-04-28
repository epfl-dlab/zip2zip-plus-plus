from __future__ import annotations

import dataclasses
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from transformers.utils import PushToHubMixin
from transformers import PreTrainedTokenizerBase, AutoTokenizer, BatchEncoding
from zip2zip_compression import Codebook
from zip2zip_compression import LZWCompressor

from zip2zip_core.viz import colorize_by_ngram, render_colored_tokens
from zip2zip_core.utils import levenshtein_distance


@dataclasses.dataclass
class LZWMismatch:
    position: int
    actual: int
    expected: int


@dataclasses.dataclass
class LZWVerificationResult:
    is_perfect: bool
    num_base_tokens: int
    num_tokens: int
    num_canonical_tokens: int
    compression_ratio: float
    canonical_compression_ratio: float
    compression_efficiency: float
    edit_distance: int
    normalized_edit_distance: float
    mismatches: List[LZWMismatch]
    compressed_ids: List[int]
    canonical_ids: List[int]

    def __str__(self) -> str:
        lines = [
            f"is_perfect:        {self.is_perfect}",
            f"base tokens:       {self.num_base_tokens}",
            f"compressed tokens: {self.num_tokens} (canonical: {self.num_canonical_tokens})",
            f"compression ratio: {self.compression_ratio:.2f}x (canonical: {self.canonical_compression_ratio:.2f}x, efficiency: {self.compression_efficiency:.3f})",
            f"edit_distance:     {self.edit_distance} (normalized: {self.normalized_edit_distance:.3f})",
            f"mismatches:        {len(self.mismatches)}",
        ]
        return "\n".join(lines)

    def pprint(self, tokenizer) -> None:
        print(str(self))
        codebook_actual = tokenizer.compressor.decode(self.compressed_ids)[1].to_dict()
        codebook_canonical = tokenizer.compressor.decode(self.canonical_ids)[1].to_dict()
        special_ids = set(tokenizer.hf_bpe_tokenizer.get_added_vocab().values())
        actual_colored = colorize_by_ngram(self.compressed_ids, codebook_actual, special_ids)
        canonical_colored = colorize_by_ngram(self.canonical_ids, codebook_canonical, special_ids)
        print(f"\nactual:    {render_colored_tokens(actual_colored, tokenizer.hf_bpe_tokenizer)}")
        print(f"canonical: {render_colored_tokens(canonical_colored, tokenizer.hf_bpe_tokenizer)}")


def get_base_vocab_size(tokenizer) -> int:
    return len(tokenizer.vocab)



class Zip2ZipTokenizer(PushToHubMixin):
    """Wrapper that adds LZW compression layer to any tokenizer.

    Protocol stack:

        Text
         ↓
      [Base Tokenizer]  ← Converts text to base token IDs
         ↓
      [Zip2Zip Layer]   ← Compresses token IDs using LZW
         ↓
    Compressed Token IDs
    """
    def __init__(
        self,
        hf_bpe_tokenizer: PreTrainedTokenizerBase,
        max_codebook_size: int = 4096,
        max_subtokens: int = 4,
        disabled_ids: Optional[List[int]] = None,
    ) -> None:

        set_pad_token_if_none(hf_bpe_tokenizer)

        self.initial_vocab_size = get_base_vocab_size(hf_bpe_tokenizer)
        self.max_codebook_size = max_codebook_size
        self.max_subtokens = max_subtokens
        self.disabled_ids = disabled_ids

        self.old_batch_encode_plus = hf_bpe_tokenizer._batch_encode_plus
        hf_bpe_tokenizer._batch_encode_plus = self._batch_encode_plus

        self.old_decode = hf_bpe_tokenizer._decode
        hf_bpe_tokenizer._decode = self._decode

        self.hf_bpe_tokenizer = hf_bpe_tokenizer
        self.compressor = LZWCompressor(
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            pad_token_id=self.hf_bpe_tokenizer.pad_token_id,
            disabled_ids=self.disabled_ids,
        )

    def __getattr__(self, attr):
        return getattr(self.hf_bpe_tokenizer, attr)

    def __call__(self, *args, **kwargs) -> BatchEncoding:
        raise NotImplementedError(
            "__call__ is banned to avoid confusion between batch and single encoding. "
            "Use batch_encode_plus([text], ...) explicitly."
        )

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        raise NotImplementedError(
            "_encode_plus is banned to avoid inconsistency in batch encoding/decoding. "
            "Use batch_encode_plus([text], ...) instead."
        )

    def _lzw_encode(
        self, *args, **kwargs
    ) -> Tuple[List[List[int]], List[torch.Tensor], List[Codebook]]:
        return self.compressor.batch_encode(*args, **kwargs)

    def _lzw_decode(self, *args, **kwargs) -> List[Tuple[List[int], Codebook]]:
        return self.compressor.batch_decode(*args, **kwargs)

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        return_tensors = kwargs.pop("return_tensors", None)
        padding = kwargs.pop("padding_strategy").value
        truncation = kwargs.pop("truncation_strategy").value
        max_length = kwargs.pop("max_length", None)
        return_codebook = kwargs.pop("return_codebook", False)

        encoding = self.old_batch_encode_plus(*args, **kwargs)

        (
            encoding["input_ids"],
            encoding["attention_mask"],
            codebooks,
        ) = self._lzw_encode(
            encoding["input_ids"],
            padding=padding,
            truncation=truncation != "do_not_truncate",
            max_length=max_length,
        )

        if return_tensors:
            encoding = encoding.convert_to_tensors(return_tensors)

        if return_codebook:
            encoding["codebooks"] = codebooks

        return encoding

    def _decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> Union[str, Tuple[str, Codebook]]:
        # we add a dimension to the token_ids to make it a list of lists, which is required by _lzw_decode
        if isinstance(token_ids, int):
            token_ids = [[token_ids]]
        else:
            token_ids = [token_ids]

        return_codebook = kwargs.pop("return_codebook", False)

        base_token_ids, codebook = self._lzw_decode(token_ids)[0]

        text = self.old_decode(
            base_token_ids, skip_special_tokens, clean_up_tokenization_spaces, **kwargs
        )
        if return_codebook:
            return text, codebook
        else:
            return text



    def pprint(
        self,
        compressed_ids: Union[List[int], List[List[int]], np.ndarray, torch.Tensor],
    ) -> None:
        if isinstance(compressed_ids, torch.Tensor):
            compressed_ids = compressed_ids.tolist()
        elif isinstance(compressed_ids, np.ndarray):
            compressed_ids = compressed_ids.tolist()

        token_ids_codebook_pairs = self._lzw_decode(compressed_ids)
        special_token_ids = set(self.hf_bpe_tokenizer.get_added_vocab().values())

        for i, (_, codebook) in enumerate(token_ids_codebook_pairs):
            codebook_map = codebook.to_dict()
            colored_tokens = colorize_by_ngram(compressed_ids[i], codebook_map, special_token_ids)
            print(render_colored_tokens(colored_tokens, self.hf_bpe_tokenizer))

    def verify_lzw(self, compressed_ids: List[int]) -> LZWVerificationResult:
        base_ids, _ = self.compressor.decode(compressed_ids)
        canonical_ids, _, _ = self.compressor.encode(base_ids)
        ed = levenshtein_distance(compressed_ids, canonical_ids)
        max_len = max(len(compressed_ids), len(canonical_ids))
        mismatches = [
            LZWMismatch(position=i, actual=compressed_ids[i], expected=canonical_ids[i])
            for i in range(min(len(compressed_ids), len(canonical_ids)))
            if compressed_ids[i] != canonical_ids[i]
        ]
        return LZWVerificationResult(
            is_perfect=(compressed_ids == canonical_ids),
            num_base_tokens=len(base_ids),
            num_tokens=len(compressed_ids),
            num_canonical_tokens=len(canonical_ids),
            compression_ratio=len(base_ids) / len(compressed_ids),
            canonical_compression_ratio=len(base_ids) / len(canonical_ids),
            compression_efficiency=len(canonical_ids) / len(compressed_ids),
            edit_distance=ed,
            normalized_edit_distance=ed / max_len if max_len > 0 else 0.0,
            mismatches=mismatches,
            compressed_ids=compressed_ids,
            canonical_ids=canonical_ids,
        )


def set_pad_token_if_none(
    tokenizer: PreTrainedTokenizerBase, pad_token_id: Optional[int] = None
) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            pad_token_id if pad_token_id is not None else tokenizer.eos_token_id
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B")
    tokenizer = Zip2ZipTokenizer(
        max_codebook_size=4096,
        max_subtokens=4,
        hf_bpe_tokenizer=tokenizer
    )
    # Read this script's own source code
    with open(__file__, "r") as f:
        text = f.read()

    compressed_ids = tokenizer.batch_encode_plus([text])["input_ids"][0]
    

    # It is important to have clean_up_tokenization_spaces=False to ensure that the decoded text matches the original text, as the hf tokenizer has some messiness 
    # with spaces that can cause discrepancies between the original and decoded text. 
    re_decoded_text = tokenizer.decode(compressed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    assert text == re_decoded_text, "Decoded text does not match original"
