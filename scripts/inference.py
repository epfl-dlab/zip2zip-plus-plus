"""Zip2Zip inference — CodebookManager-driven generation.

The CodebookManager is the single source of truth for LZW state.
No separate LZWCompressor/Zip2ZipTokenizer is used during inference.

Usage:
    python scripts/inference.py --prompt "The capital of France is"
    python scripts/inference.py --prompt "1 + 1 =" --max-new-tokens 32 --ckpt-dir /path/to/ckpt
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


# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT = "/mnt/scratch/checkpoints/zip2zip_1b_finemath_10bt_ms3/step_1000"
DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
EOS_ID = 128001
PAD_ID = 128001
DISABLED_IDS = [128000, 128001, 128002, 128003]


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


# def _update_codebook_dict(
#     codebook_dict: dict[int, list[int]],
#     updates: torch.Tensor,
#     updates_indices: list[list[int]],
#     vocab_size: int,
#     pad_id: int,
# ) -> None:
#     """Register new codebook entries for hyper-token → base-token expansion."""
#     for i, ui in enumerate(updates_indices):
#         for j, idx in enumerate(ui):
#             codebook_dict[vocab_size + idx] = [
#                 t for t in updates[i, j].tolist() if t != pad_id
#             ]


# ── generation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    prompt: str,
    model: Zip2ZipLlama3Model,
    codebook_manager: CodebookManager,
    hf_tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    device: str = "cuda",
) -> str:
    cfg = model.zip2zip_config
    vocab_size = cfg.vocab_size
    pad_id = cfg.pad_token_id

    # ── Encode prompt (base tokens only — no LZW compression) ──
    base_ids = hf_tokenizer.encode(prompt, add_special_tokens=False)
    context = torch.tensor([base_ids], dtype=torch.long, device=device)  # (1, T)

    # ── Reset state ──
    model.reset_inference_cache()
    codebook_manager.reset()

    # ── Prefill: build initial codebook from prompt base tokens ──
    codebook_manager.update_codebooks(context)
    updates, updates_indices = codebook_manager.get_new_codes()

    # hyper-token → base-token mapping (for expanding generated hyper tokens)
    codebook_dict: dict[int, list[int]] = {}
    # _update_codebook_dict(codebook_dict, updates, updates_indices, vocab_size, pad_id)

    # Run model on the prompt. No codebook tensor — the model uses its internal
    # _hyper_embeds_buf and _hyper_embeds_used for hyper-token embedding + masking.
    logits = model(
        context,
        codebook_updates=updates,
        codebook_updates_indices=updates_indices,
    )
    # handle optional token_type_logits return
    if isinstance(logits, tuple):
        logits = logits[0]

    generated_base_ids: list[int] = []

    # ── Decode loop ──
    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :]  # (vocab_size + max_codebook_size,)
        if temperature != 1.0:
            last_logits = last_logits / temperature
        next_id = int(last_logits.argmax())

        generated_base_ids.append(next_id)

        if next_id == EOS_ID:
            break

        # # Expand compressed token → base tokens
        # if next_id >= vocab_size:
        #     new_base = codebook_dict.get(next_id)
        #     if new_base is None:
        #         print(f"WARNING: generated unknown hyper-token {next_id}, skipping")
        #         break
        # else:
        #     new_base = [next_id]
        # generated_base_ids.extend(new_base)

        # Advance LZW state with the new base tokens
        new_base_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        codebook_manager.update_codebooks(new_base_tensor)
        updates, updates_indices = codebook_manager.get_new_codes()
        # _update_codebook_dict(codebook_dict, updates, updates_indices, vocab_size, pad_id)

        # Append the *compressed* token (not base expansion) to context
        context = torch.cat(
            [context, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

        logits = model.forward(
            context,
            codebook_updates=updates,
            codebook_updates_indices=updates_indices,
        )
        if isinstance(logits, tuple):
            logits = logits[0]

        if step % 10 == 0:
            partial = hf_tokenizer.decode(generated_base_ids[-60:])
            print(f"  step {step:3d} | ctx_len={context.shape[1]} | ...{repr(partial)}")

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
    # to bf16 for faster inference (if not already in that dtype)
    if device == "cuda":
        model = model.half()
    cfg = model.zip2zip_config
    print(f"  max_subtokens={cfg.max_subtokens}  max_codebook_size={cfg.max_codebook_size}")

    print(f"Loading tokenizer from {cli.tokenizer}...")
    hf_tok = AutoTokenizer.from_pretrained(cli.tokenizer)

    codebook_manager = CodebookManager(
        initial_vocab_size=cfg.vocab_size,
        max_codebook_size=cfg.max_codebook_size,
        max_subtokens=cfg.max_subtokens,
        embedding_dim=cfg.dim,
        pad_token_id=cfg.pad_token_id,
        disabled_ids=DISABLED_IDS,
    )

    print(f"\nPrompt: {repr(cli.prompt)}\n")
    output = generate(
        cli.prompt,
        model,
        codebook_manager,
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
