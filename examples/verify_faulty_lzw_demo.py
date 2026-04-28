"""Demonstrate verify_lzw: perfect vs faulty compressed sequences."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer
from zip2zip_core.tokenizer import Zip2ZipTokenizer

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "hf_tokenizer", "Llama-3.1-8B")


def main():
    hf_tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer = Zip2ZipTokenizer(
        hf_bpe_tokenizer=hf_tok,
        max_codebook_size=4096,
        max_subtokens=4,
    )

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
        "The quick brown fox jumps over the lazy dog. "
    )

    enc = tokenizer.batch_encode_plus([text], return_codebook=True)
    compressed_ids = enc["input_ids"][0]
    cb = enc["codebooks"][0].to_dict()

    # ── 1. Perfect sequence ──
    print("=== Perfect LZW sequence ===")
    result = tokenizer.verify_lzw(compressed_ids)
    result.pprint(tokenizer)

    # ── 2. Faulty sequence: expand one hypertoken back to base tokens ──
    print("\n=== Faulty sequence (missed merge) ===")
    faulty_ids = list(compressed_ids)
    for i, tid in enumerate(faulty_ids):
        if tid in cb:
            base_tokens = cb[tid]
            faulty_ids = faulty_ids[:i] + base_tokens + faulty_ids[i + 1:]
            print(f"  Expanded hypertoken {tid} at position {i} -> base tokens {base_tokens}")
            break

    result = tokenizer.verify_lzw(faulty_ids)
    result.pprint(tokenizer)


if __name__ == "__main__":
    main()
