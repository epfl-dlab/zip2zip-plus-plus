"""Quick inference test for a published Zip2Zip++ model.

Example:
    python scripts/zip2zip_hf/run.py \
        --repo epfl-dlab/zip2zip-pp-Llama-3.2-1B-Instruct \
        --prompt "Hello world"
"""

from __future__ import annotations

import argparse

import torch
from zip2zip import Zip2ZipModel, Zip2ZipTokenizer


CHAT_TURN_END_TOKENS = ("<|end|>", "<|eot_id|>", "<|im_end|>")


def get_base_tokenizer(tokenizer):
    return getattr(tokenizer, "hf_tokenizer", getattr(tokenizer, "tokenizer", tokenizer))


def get_generation_stop_token_ids(
    tokenizer: Zip2ZipTokenizer, *, instruct: bool
) -> int | list[int]:
    base_tokenizer = get_base_tokenizer(tokenizer)
    stop_ids: list[int] = []
    candidates = [getattr(tokenizer, "eos_token_id", None)]
    if instruct:
        for token in CHAT_TURN_END_TOKENS:
            token_id = base_tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != base_tokenizer.unk_token_id:
                candidates.append(int(token_id))
    for token_id in candidates:
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id not in stop_ids:
            stop_ids.append(token_id)
    if not stop_ids:
        raise ValueError("No generation stop token ids found")
    return stop_ids[0] if len(stop_ids) == 1 else stop_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="hf")
    parser.add_argument(
        "--prompt", default="Please write a MultiHeadAttention layer in PyTorch."
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--instruct", action="store_true", help="Apply the base chat template."
    )
    args = parser.parse_args()

    tokenizer = Zip2ZipTokenizer.from_pretrained(
        args.repo, revision=args.revision
    )
    model = Zip2ZipModel.from_pretrained(
        args.repo,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    prompt = args.prompt
    if args.instruct:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding="longest",
        add_special_tokens=not args.instruct,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            top_k=50,
            eos_token_id=get_generation_stop_token_ids(
                tokenizer, instruct=args.instruct
            ),
            pad_token_id=tokenizer.pad_token_id,
        )

    for text in tokenizer.batch_decode(outputs, skip_special_tokens=True):
        print(text)
        print("=" * 40)


if __name__ == "__main__":
    main()
