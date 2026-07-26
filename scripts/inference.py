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
from zip2zip_core.model import Zip2ZipLlama3Model, restore_encoder_residual
from zip2zip_core.viz import colorize_by_ngram, render_colored_tokens


# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT = "/mnt/scratch/checkpoints/zip2zip_150m_finemath_10bt_ms4/step_6000"
DEFAULT_TOKENIZER = "bofenghuang/Meta-Llama-3-8B"
EOS_ID = 128001
PAD_ID = 128001
DISABLED_IDS = [128000, 128001, 128002, 128003]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device: str) -> tuple[Zip2ZipLlama3Model, dict]:
    meta = torch.load(f"{ckpt_dir}/meta.pt", map_location="cpu", weights_only=False)
    args = meta["args"]
    cfg = zip2zip_llama_configs[args["model_config"]]
    # Encoder architecture overrides recorded in meta.pt (None in legacy metas
    # means the config default; deeper/wider checkpoints crash the strict load
    # without these, e.g. the v0.6.1 4-layer encoder).
    enc_overrides = {
        k: args[k]
        for k in ("encoder_dim", "encoder_n_layers", "encoder_n_heads",
                  "encoder_intermediate_size")
        if args.get(k) is not None
    }
    cfg = dataclasses.replace(
        cfg,
        max_subtokens=args["max_subtokens"],
        max_codebook_size=args["max_codebook_size"],
        hyper_encoder_type=args.get("hyper_encoder_type", "flat"),
        # untied checkpoints carry a second hyper_output encoder; without this the
        # strict load below fails on unexpected hyper_output.* keys.
        tie_hyper_encoder=not args.get("untied_hyper_encoder", False),
        share_hyper_encoder_weights=bool(args.get("share_hyper_encoder_weights", False)),
        # behavior flag (no weights): compressed generation must position tokens
        # in base space exactly as trained.
        base_token_positions=bool(args.get("base_token_positions", False)),
        # behavior-only two-axis geometry must also be restored from meta.pt.
        two_axis_rope=bool(args.get("two_axis_rope", False)),
        # builds the (eval-unused) token_type_head so its checkpoint weights
        # have a home and the strict load below does not fail on them.
        token_type_loss_weight=float(args.get("token_type_loss_weight") or 0.0),
        **enc_overrides,
    )
    model = Zip2ZipLlama3Model(cfg)
    print(
        "[inference] decoder RoPE: "
        f"base_positions={cfg.base_token_positions} "
        f"two_axis={cfg.two_axis_rope}"
    )
    if not restore_encoder_residual(model, args):
        print("[inference] hyper-encoder residual: disabled from meta.pt")
    model = model.to(device)
    sd = torch.load(f"{ckpt_dir}/model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args



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

    def _color_decode_ids(ids: list[int]) -> str:
        codebooks = codebook_manager.internal_codebook_manager.get_codebooks()
        codebook_dict = codebooks[0].to_dict() if codebooks else {}
        special_ids = set(hf_tokenizer.get_added_vocab().values())
        colored_tokens = colorize_by_ngram(ids, codebook_dict, special_ids)
        return render_colored_tokens(colored_tokens, hf_tokenizer)

    # ── Decode loop ──
    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :]  # (vocab_size + max_codebook_size,)
        if temperature != 1.0:
            last_logits = last_logits / temperature
        next_id = int(last_logits.argmax())

        generated_base_ids.append(next_id)

        if next_id == EOS_ID:
            break

        # Advance LZW state with the new base tokens
        new_base_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
        codebook_manager.update_codebooks(new_base_tensor)
        updates, updates_indices = codebook_manager.get_new_codes()

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

        if step % 2 == 0:
            partial_colored = _color_decode_ids(generated_base_ids[-20:])
            ids_str = str(generated_base_ids[-10:])
            top10_vals, top10_ids = torch.topk(last_logits, 10)
            top10_tokens = [
                f"{tid}({repr(hf_tokenizer.decode([tid]))})" if tid < vocab_size
                else f"{tid}(hyper)"
                for tid in top10_ids.tolist()
            ]
            print(f"  step {step:3d} | ctx_len={context.shape[1]} | ids={ids_str} | ...{partial_colored}")
            print(f"           top10: {', '.join(top10_tokens)}")

    colored_output = _color_decode_ids(generated_base_ids)
    return hf_tokenizer.decode(generated_base_ids, skip_special_tokens=True), colored_output


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
    # Cast to bf16 for faster inference, but parameters and REAL buffers only.
    # A blanket .half()/.to(dtype) also converts the complex64 RoPE cache
    # (freqs_cis) to a real dtype, silently discarding the imaginary part and
    # destroying position encoding. Mirrors the eval adapter's cast.
    if device == "cuda":
        model._apply(
            lambda t: t.to(torch.bfloat16) if t.is_floating_point() else t
        )
    cfg = model.zip2zip_config
    print(f"  max_subtokens={cfg.max_subtokens}  max_codebook_size={cfg.max_codebook_size}")

    print(f"Loading tokenizer from {cli.tokenizer}...")
    hf_tok = AutoTokenizer.from_pretrained(cli.tokenizer)

    # Derive disabled_ids from the tokenizer (matches training) instead of the
    # hardcoded Llama constant — correct for Phi (specials 0,1,2,32000..32010).
    derived_disabled = sorted(
        i for i in (set(hf_tok.all_special_ids or []) | set(hf_tok.get_added_vocab().values()))
        if 0 <= i < cfg.vocab_size
    )
    print(f"  disabled_ids={derived_disabled}")

    codebook_manager = CodebookManager(
        initial_vocab_size=cfg.vocab_size,
        max_codebook_size=cfg.max_codebook_size,
        max_subtokens=cfg.max_subtokens,
        embedding_dim=cfg.dim,
        pad_token_id=cfg.pad_token_id,
        disabled_ids=derived_disabled,
    )

    print(f"\nPrompt: {repr(cli.prompt)}\n")
    output, colored_output = generate(
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
    print(f"\nCOLORED (blue=base, yellow=2-gram, orange=3-gram, red=4-gram):")
    print(colored_output)


if __name__ == "__main__":
    main()
