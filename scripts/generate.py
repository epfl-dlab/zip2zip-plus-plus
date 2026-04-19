"""Zip2Zip greedy generation.

Usage:
    python scripts/generate.py --prompt "The capital of France is"
    python scripts/generate.py --prompt "1 + 1 =" --max-new-tokens 32 --ckpt-dir /path/to/ckpt
"""

import argparse
import dataclasses
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import torch
from transformers import AutoTokenizer

from zip2zip_core.codebook import CodebookManager
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.model import Zip2ZipLlama3Model
from zip2zip_core.tokenizer import Zip2ZipTokenizer


# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT = "/mnt/scratch/checkpoints/zip2zip_1b_finemath_10bt_ms3/step_1000"
DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
EOS_ID = 128001
DISABLED_IDS = {128000, 128001, 128002, 128003}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device: str) -> tuple[Zip2ZipLlama3Model, dict]:
    meta = torch.load(f"{ckpt_dir}/meta.pt", map_location="cpu", weights_only=False)
    args = meta["args"]
    cfg = zip2zip_llama_configs[args["model_config"]]
    cfg = dataclasses.replace(
        cfg,
        max_subtokens=args["max_subtokens"],
        max_codebook_size=args["max_codebook_size"],
    )
    model = Zip2ZipLlama3Model(cfg).to(device)
    sd = torch.load(f"{ckpt_dir}/model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args


def _scatter_updates(
    codebook_tensor: torch.Tensor,
    codebook_dict: dict[int, list[int]],
    updates: torch.Tensor,
    updates_indices: list[list[int]],
    vocab_size: int,
    pad_id: int,
) -> None:
    """Write new codebook entries into codebook_tensor and codebook_dict in-place."""
    for i, ui in enumerate(updates_indices):
        for j, idx in enumerate(ui):
            codebook_tensor[i, idx] = updates[i, j]
            codebook_dict[vocab_size + idx] = [
                t for t in updates[i, j].tolist() if t != pad_id
            ]


# ── generation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    prompt: str,
    model: Zip2ZipLlama3Model,
    codebook_manager: CodebookManager,
    z_tokenizer: Zip2ZipTokenizer,
    hf_tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    cfg = model.zip2zip_config
    vocab_size = cfg.vocab_size
    max_sub = cfg.max_subtokens
    pad_id = cfg.pad_token_id
    max_cb = cfg.max_codebook_size

    # --- Encode prompt ---
    # LZW-compress the prompt to get the compressed token sequence
    encoding = z_tokenizer(
        prompt, return_tensors="pt", return_codebook=True, add_special_tokens=False
    )
    compressed_ids = encoding["input_ids"].to(device)  # (1, T_compressed)

    # Get base token IDs to initialize the codebook manager's LZW state
    base_ids = hf_tokenizer.encode(prompt, add_special_tokens=False)
    base_tensor = torch.tensor([base_ids], dtype=torch.long, device=device)

    # --- Reset state ---
    model.reset_inference_cache()
    codebook_manager.reset()

    # Process prompt base tokens through codebook_manager to build initial LZW state
    codebook_manager.update_codebooks(base_tensor)
    updates, updates_indices = codebook_manager.get_new_codes()

    # Pre-allocate full-size codebook tensor (pad-filled); grows in-place each step
    codebook_tensor = torch.full(
        (1, max_cb, max_sub), pad_id, dtype=torch.long, device=device
    )
    # codebook_dict: hyper token ID -> base token ID list (for decoding generated tokens)
    codebook_dict: dict[int, list[int]] = {}
    _scatter_updates(codebook_tensor, codebook_dict, updates, updates_indices, vocab_size, pad_id)

    # --- Prefill ---
    logits = model(
        compressed_ids,
        codebook=codebook_tensor,
        codebook_updates=updates,
        codebook_updates_indices=updates_indices,
    )  # (1, T_compressed, vocab_size + max_cb)

    context = compressed_ids  # grows by one compressed token per step
    generated_base_ids: list[int] = []

    # --- Decode loop ---
    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :]  # (vocab_size + max_cb,)
        if temperature != 1.0:
            last_logits = last_logits / temperature
        next_id = int(last_logits.argmax())

        if next_id == EOS_ID:
            break

        # Expand compressed token → base tokens for LZW update and final decoding
        if next_id >= vocab_size:
            new_base = codebook_dict.get(next_id, [])
        else:
            new_base = [next_id]
        generated_base_ids.extend(new_base)

        # Advance LZW state; may produce new codebook entries
        if new_base:
            new_base_tensor = torch.tensor([new_base], dtype=torch.long, device=device)
            codebook_manager.update_codebooks(new_base_tensor)
            updates, updates_indices = codebook_manager.get_new_codes()
            _scatter_updates(
                codebook_tensor, codebook_dict, updates, updates_indices, vocab_size, pad_id
            )
        else:
            # No base tokens to feed (degenerate case); pass empty updates
            updates = torch.zeros(1, 0, max_sub, dtype=torch.long, device=device)
            updates_indices = [[]]

        # Append generated token to context and run model
        context = torch.cat(
            [context, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1
        )
        logits = model(
            context,
            codebook=codebook_tensor,
            codebook_updates=updates,
            codebook_updates_indices=updates_indices,
        )

    return hf_tokenizer.decode(generated_base_ids, skip_special_tokens=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The Eiffel Tower is located in")
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    cli = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model, train_args = load_model(cli.ckpt_dir, device)
    print(
        f"  max_subtokens={model.zip2zip_config.max_subtokens}"
        f"  max_codebook_size={model.zip2zip_config.max_codebook_size}"
    )

    print(f"Loading tokenizer from {cli.tokenizer}...")
    hf_tok = AutoTokenizer.from_pretrained(cli.tokenizer)
    z_tok = Zip2ZipTokenizer(
        max_codebook_size=model.zip2zip_config.max_codebook_size,
        max_subtokens=model.zip2zip_config.max_subtokens,
        tokenizer=hf_tok,
    )

    codebook_manager = CodebookManager(
        initial_vocab_size=model.zip2zip_config.vocab_size,
        max_codebook_size=model.zip2zip_config.max_codebook_size,
        max_subtokens=model.zip2zip_config.max_subtokens,
        embedding_dim=model.zip2zip_config.dim,
        pad_token_id=model.zip2zip_config.pad_token_id,
        disabled_ids=list(DISABLED_IDS),
    )

    print(f"\nPrompt: {repr(cli.prompt)}\n")
    output = generate(
        cli.prompt,
        model,
        codebook_manager,
        z_tok,
        hf_tok,
        max_new_tokens=cli.max_new_tokens,
        temperature=cli.temperature,
        device=device,
    )
    print(f"\n{'='*60}")
    print(f"PROMPT:    {cli.prompt}")
    print(f"GENERATED: {output}")


if __name__ == "__main__":
    main()
