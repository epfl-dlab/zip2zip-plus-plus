"""Quick inference test for exported zip2zip models on HF Hub.

Usage:
    python scripts/zip2zip_hf/run.py --repo epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1
    python scripts/zip2zip_hf/run.py --repo epfl-dlab/Llaza-3.2-1B-MS2F-4K-v0.1 --prompt "Hello world"
    python scripts/zip2zip_hf/run.py --repo epfl-dlab/Llaza-3.2-1B-MS2-instruct --instruct --prompt "Write a sort function"
"""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer


CHAT_TURN_END_TOKENS = ("<|end|>", "<|eot_id|>", "<|im_end|>")


def resolve_base_model_name(repo: str, revision: str, cli_base_model: str | None) -> str:
    if cli_base_model:
        return str(cli_base_model)
    cfg_path = hf_hub_download(repo_id=repo, filename="zip2zip_config.json", revision=revision)
    with open(cfg_path, "r", encoding="utf-8") as file:
        cfg = json.load(file)
    return str(cfg["base_model_name_or_path"])


def get_base_tokenizer(tokenizer):
    return getattr(tokenizer, "hf_tokenizer", getattr(tokenizer, "tokenizer", tokenizer))


def get_generation_stop_token_ids(tokenizer: Zip2ZipTokenizer, *, instruct: bool) -> int | list[int]:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=str, default="epfl-dlab/candidate-Llaza-3.2-1B-MS3F-4K-FT-1BT-v0.1")
    p.add_argument("--base_model", type=str, default=None, help="Optional override; defaults to the repo's zip2zip_config base model")
    p.add_argument("--revision", type=str, default="hf")
    p.add_argument("--prompt", type=str, default="Please write a MultiHeadAttention layer in PyTorch.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--instruct", action="store_true", help="Apply chat template (for Instruct models)")
    args = p.parse_args()

    resolved_base_model = resolve_base_model_name(args.repo, args.revision, args.base_model)

    # Load base model without revision (base_model repo doesn't have an "hf" branch)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        torch_dtype=torch.bfloat16,
    )

    model = Zip2ZipModel.from_pretrained(
        args.repo,
        revision=args.revision,
        base_model=base_model,
        dtype=torch.bfloat16,
    ).to("cuda").eval()

    tokenizer = Zip2ZipTokenizer.from_pretrained(args.repo, revision=args.revision)

    prompt = args.prompt
    if args.instruct:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # apply_chat_template already inserts <|begin_of_text|>; don't let the tokenizer add a
    # second BOS for instruct prompts (training used a single BOS). Plain-text prompts still
    # get their single BOS from the tokenizer.
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
            eos_token_id=get_generation_stop_token_ids(tokenizer, instruct=args.instruct),
            pad_token_id=tokenizer.pad_token_id,
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
