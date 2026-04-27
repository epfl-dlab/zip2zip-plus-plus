"""Show colorized output of zip2zip compression using pprint."""

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
        "The quick brown fox jumps over the lazy dog again! "
        "The quick brown fox is tired of jumping. "
        "The quick brown fox jumps over the lazy dog once more. "
        "Finally, the quick brown fox decides to take a nap instead of jumping. "
        "This is the last time the quick brown fox jumps over the lazy dog."
    )

    print(f"Original text:\n  {text}\n")

    encoding = tokenizer(text, return_codebook=True)
    compressed_ids = encoding["input_ids"]

    base_len = len(hf_tok.encode(text, add_special_tokens=False))
    comp_len = len(compressed_ids)
    print(f"Base tokens: {base_len}  |  Compressed tokens: {comp_len}  |  Ratio: {base_len / comp_len:.2f}x\n")

    # pprint: blue=base, yellow=2-gram, orange=3-gram, red=4-gram
    print("Colored output (blue=base, yellow=2-gram, orange=3-gram, red=4-gram):")
    tokenizer.pprint([compressed_ids])


if __name__ == "__main__":
    main()
