"""Quick inference test for exported zip2zip models on HF Hub.

Usage:
    python scripts/zip2zip_hf/run.py --repo epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1
    python scripts/zip2zip_hf/run.py --repo epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1 --prompt "Hello world"
"""

import argparse

import torch
from transformers import AutoModelForCausalLM
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="epfl-dlab/candidate-Llaza-3.2-1B-MS3F-4K-FT-1BT-v0.1")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--prompt", type=str, default="Please write a MultiHeadAttention layer in PyTorch.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    # Load base model without revision (base_model repo doesn't have an "hf" branch)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    model = Zip2ZipModel.from_pretrained(
        args.repo,
        revision="hf",
        base_model=base_model,
        dtype=torch.bfloat16,
    ).to("cuda").eval()

    tokenizer = Zip2ZipTokenizer.from_pretrained(args.repo, revision="hf")

    inputs = tokenizer([args.prompt], return_tensors="pt", padding="longest").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            top_k=50,
        )

    for text in tokenizer.batch_decode(outputs, skip_special_tokens=True):
        print(text)
        print("=" * 40)

    print("\n--- Color decode ---\n")
    for text in tokenizer.color_decode(outputs, color_scheme="finegrained"):
        print(text)
        print("=" * 40)


if __name__ == "__main__":
    main()
