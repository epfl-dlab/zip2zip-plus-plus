"""Zip2Zip inference — CodebookManager-driven generation.

The CodebookManager is the single source of truth for LZW state.
No separate LZWCompressor/Zip2ZipTokenizer is used during inference.

Usage:
    python scripts/inference.py --prompt "The capital of France is"
    python scripts/inference.py --prompt "1 + 1 =" --max-new-tokens 32 --ckpt-dir /path/to/ckpt
    python scripts/inference.py --prompt-file prompts.json --output-file generations.jsonl \
        --instruct --ckpt-dir /path/to/ckpt
"""

import argparse
import dataclasses
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ext", "torchtitan"))

import torch
from transformers import AutoTokenizer, GenerationConfig
from zip2zip_compression import LZWCompressor

from zip2zip_core.checkpoint import (
    prepare_inference_state_dict,
    resolve_gated_rope_start_pair,
)
from zip2zip_core.codebook import CodebookManager
from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.disabled_ids import compute_disabled_ids
from zip2zip_core.model import Zip2ZipLlama3Model, restore_encoder_residual
from zip2zip_core.viz import colorize_by_ngram, render_colored_tokens


# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CKPT = "/mnt/scratch/checkpoints/zip2zip_150m_finemath_10bt_ms4/step_6000"
DEFAULT_TOKENIZER = "bofenghuang/Meta-Llama-3-8B"

DEFAULT_PROMPT = "The Eiffel Tower is located in"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_dir: str, device: str) -> tuple[Zip2ZipLlama3Model, dict]:
    meta = torch.load(f"{ckpt_dir}/meta.pt", map_location="cpu", weights_only=False)
    args = meta["args"]
    cfg = zip2zip_llama_configs[args["model_config"]]
    sd = torch.load(
        f"{ckpt_dir}/model.pt", map_location=device, weights_only=True
    )
    sd = prepare_inference_state_dict(sd, args)
    # Encoder architecture overrides recorded in meta.pt (None in legacy metas
    # means the config default; deeper/wider checkpoints crash the strict load
    # without these, e.g. the v0.6.1 4-layer encoder).
    enc_overrides = {
        k: args[k]
        for k in ("encoder_dim", "encoder_n_layers", "encoder_n_heads",
                  "encoder_intermediate_size")
        if args.get(k) is not None
    }
    gated_start_layer = args.get("gated_rope_start_layer")
    if gated_start_layer is None:
        gated_start_layer = cfg.gated_rope_start_layer
    head_dim = (
        getattr(cfg.layer.attention, "head_dim", None)
        or cfg.dim // cfg.layer.attention.n_heads
    )
    if args.get("gated_compressed_rope"):
        gated_start_pair = resolve_gated_rope_start_pair(
            args,
            sd,
            n_complex_pairs=head_dim // 2,
            default=cfg.gated_rope_start_pair,
        )
    else:
        gated_start_pair = cfg.gated_rope_start_pair
    overrides = dict(
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
        # v0.7.1 keeps ordinary RoPE and learns a gated compressed-coordinate
        # delta in configurable layer and low-frequency pair suffixes.
        gated_compressed_rope=bool(args.get("gated_compressed_rope", False)),
        gated_rope_start_layer=int(gated_start_layer),
        gated_rope_start_pair=int(gated_start_pair),
        # builds the (eval-unused) token_type_head so its checkpoint weights
        # have a home and the strict load below does not fail on them.
        token_type_loss_weight=float(args.get("token_type_loss_weight") or 0.0),
        **enc_overrides,
    )
    cfg = dataclasses.replace(cfg, **overrides)
    model = Zip2ZipLlama3Model(cfg)
    print(
        "[inference] decoder RoPE: "
        f"base_positions={cfg.base_token_positions} "
        f"two_axis={cfg.two_axis_rope} "
        f"gated_compressed={cfg.gated_compressed_rope} "
        f"gated_start_layer={cfg.gated_rope_start_layer} "
        f"gated_start_pair={cfg.gated_rope_start_pair}"
    )
    if not restore_encoder_residual(model, args):
        print("[inference] hyper-encoder residual: disabled from meta.pt")
    model = model.to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, args



# ── generation loop ───────────────────────────────────────────────────────────

def _update_codebook_dict(
    codebook_dict: dict[int, list[int]],
    updates: torch.Tensor,
    updates_indices: list[list[int]],
    vocab_size: int,
    pad_id: int,
) -> None:
    """Register newly installed hyper-token entries for output expansion."""
    for batch_idx, indices in enumerate(updates_indices):
        for row, slot in enumerate(indices):
            codebook_dict[vocab_size + int(slot)] = [
                int(token)
                for token in updates[batch_idx, row].tolist()
                if int(token) != pad_id
            ]


def _sample_next(last_logits: torch.Tensor, temperature: float) -> int:
    """Greedy decode at zero temperature; categorical sampling otherwise."""
    if temperature <= 0:
        return int(last_logits.argmax().item())
    probs = torch.softmax(last_logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def format_prompt(tokenizer, prompt: str, *, instruct: bool) -> str:
    """Optionally wrap a raw user prompt with the tokenizer's chat template."""
    if not instruct:
        return prompt
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "--instruct requires a tokenizer with a chat_template; "
            "drop --instruct or select the checkpoint's instruct tokenizer"
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_prompts(prompt: str | None, prompt_file: str | None) -> list[str]:
    """Load one CLI prompt or a JSON file containing a list of prompts."""
    if prompt_file is None:
        return [DEFAULT_PROMPT if prompt is None else prompt]

    with open(prompt_file, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    if not isinstance(prompts, list):
        raise ValueError("--prompt-file must contain a JSON list of strings")
    if not prompts:
        raise ValueError("--prompt-file must contain at least one prompt")
    for index, item in enumerate(prompts):
        if not isinstance(item, str):
            raise ValueError(
                f"--prompt-file item {index} must be a string, got "
                f"{type(item).__name__}"
            )
    return prompts


@torch.no_grad()
def generate(
    prompt: str,
    model: Zip2ZipLlama3Model,
    codebook_manager: CodebookManager,
    lzw_compressor: LZWCompressor,
    hf_tokenizer,
    stop_token_ids: set[int],
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str = "cuda",
) -> tuple[str, str]:
    """Generate compressed tokens and decode them back to base-token text.

    ``max_new_tokens`` counts sampled compressed tokens. A sampled hypertoken
    can expand to more than one base token in the returned text.
    """
    cfg = model.zip2zip_config
    vocab_size = cfg.vocab_size
    pad_id = cfg.pad_token_id

    # Compress the prompt for the decoder, while separately replaying the base
    # prompt through the online manager to seed the same LZW dictionary.
    base_ids = hf_tokenizer.encode(prompt, add_special_tokens=False)
    if not base_ids:
        raise ValueError("prompt must encode to at least one base token")
    compressed_ids, _, _ = lzw_compressor.encode(
        base_ids, padding="do_not_pad", truncation=False
    )
    context = torch.tensor(
        [compressed_ids], dtype=torch.long, device=device
    )

    model.reset_inference_cache()
    codebook_manager.reset()
    prompt_base_tensor = torch.tensor(
        [base_ids], dtype=torch.long, device=device
    )
    codebook_manager.update_codebooks(prompt_base_tensor)
    updates, updates_indices = codebook_manager.get_new_codes()

    codebook_dict: dict[int, list[int]] = {}
    _update_codebook_dict(
        codebook_dict, updates, updates_indices, vocab_size, pad_id
    )

    logits = model(
        context,
        codebook_updates=updates,
        codebook_updates_indices=updates_indices,
    )
    # handle optional token_type_logits return
    if isinstance(logits, tuple):
        logits = logits[0]

    generated_base_ids: list[int] = []
    generated_display_ids: list[int] = []

    def _color_decode_ids(ids: list[int]) -> str:
        codebooks = codebook_manager.internal_codebook_manager.get_codebooks()
        codebook_dict = codebooks[0].to_dict() if codebooks else {}
        special_ids = set(hf_tokenizer.get_added_vocab().values())
        colored_tokens = colorize_by_ngram(ids, codebook_dict, special_ids)
        return render_colored_tokens(colored_tokens, hf_tokenizer)

    # ── Decode loop ──
    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :]  # (vocab_size + max_codebook_size,)
        next_id = _sample_next(last_logits, temperature)
        if next_id < vocab_size:
            expansion = [next_id]
        else:
            expansion = list(codebook_dict.get(next_id, []))
            if not expansion:
                print(
                    f"[inference] sampled unavailable hyper-token {next_id}; "
                    "stopping"
                )
                break

        stop_at = [
            idx for idx, token in enumerate(expansion)
            if token in stop_token_ids
        ]
        if stop_at:
            cut = stop_at[0]
            generated_base_ids.extend(expansion[:cut])
            # A partially emitted hyper-token cannot be represented by its
            # compressed id in the visualization, so show the surviving bases.
            generated_display_ids.extend(expansion[:cut])
            break

        generated_base_ids.extend(expansion)
        generated_display_ids.append(next_id)

        # Advance online LZW with the emitted BASE expansion. The model context
        # still receives the single sampled compressed token below.
        new_base_tensor = torch.tensor(
            [expansion], dtype=torch.long, device=device
        )
        codebook_manager.update_codebooks(new_base_tensor)
        updates, updates_indices = codebook_manager.get_new_codes()
        _update_codebook_dict(
            codebook_dict, updates, updates_indices, vocab_size, pad_id
        )

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
            partial_colored = _color_decode_ids(generated_display_ids[-20:])
            ids_str = str(generated_base_ids[-10:])
            _, top10_ids = torch.topk(last_logits, 10)
            top10_tokens = [
                f"{tid}({repr(hf_tokenizer.decode([tid]))})" if tid < vocab_size
                else f"{tid}(hyper)"
                for tid in top10_ids.tolist()
            ]
            print(f"  step {step:3d} | ctx_len={context.shape[1]} | ids={ids_str} | ...{partial_colored}")
            print(f"           top10: {', '.join(top10_tokens)}")

    colored_output = _color_decode_ids(generated_display_ids)
    return hf_tokenizer.decode(generated_base_ids, skip_special_tokens=True), colored_output


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=f"Single prompt (default: {DEFAULT_PROMPT!r}).",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="JSON file containing a list of prompt strings.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional JSONL output path; one record is flushed per prompt.",
    )
    parser.add_argument("--ckpt-dir", type=str, default=DEFAULT_CKPT)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer override. Defaults to meta.pt, then the legacy fallback.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Apply the tokenizer's chat template to every prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy; positive values enable categorical sampling.",
    )
    cli = parser.parse_args()
    if cli.temperature < 0:
        parser.error("--temperature must be non-negative")
    if cli.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    try:
        prompts = load_prompts(cli.prompt, cli.prompt_file)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

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

    tokenizer_name = (
        cli.tokenizer or train_args.get("tokenizer") or DEFAULT_TOKENIZER
    )
    print(f"Loading tokenizer from {tokenizer_name}...")
    hf_tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if len(hf_tok) > cfg.vocab_size:
        raise ValueError(
            f"tokenizer {tokenizer_name!r} has {len(hf_tok)} tokens, but the "
            f"checkpoint model vocabulary has {cfg.vocab_size}"
        )
    trained_tokenizer = train_args.get("tokenizer")
    if trained_tokenizer and trained_tokenizer != tokenizer_name:
        print(
            f"[inference] WARNING: tokenizer override {tokenizer_name!r} differs "
            f"from training tokenizer {trained_tokenizer!r}"
        )

    prompt_mode = "instruct chat template" if cli.instruct else "raw text"
    print(f"[inference] prompt mode: {prompt_mode}")
    print(f"[inference] prompts: {len(prompts)}")
    disabled_ids = compute_disabled_ids(
        hf_tok,
        cfg.vocab_size,
        disable_digit_ids=bool(train_args.get("disable_digit_ids")),
    )
    print(f"  disabled_ids ({len(disabled_ids)}): {disabled_ids}")

    stop_token_ids = {
        int(token)
        for token in (hf_tok.eos_token_id, cfg.pad_token_id)
        if token is not None
    }
    try:
        generation_cfg = GenerationConfig.from_pretrained(tokenizer_name)
        eos = generation_cfg.eos_token_id
        if eos is not None:
            stop_token_ids.update(
                int(token)
                for token in (
                    eos if isinstance(eos, (list, tuple)) else [eos]
                )
            )
    except Exception as exc:
        print(
            "[inference] generation config unavailable; using tokenizer EOS "
            f"only ({type(exc).__name__})"
        )
    print(f"  stop_token_ids={sorted(stop_token_ids)}")

    compression_kwargs = dict(
        initial_vocab_size=cfg.vocab_size,
        max_codebook_size=cfg.max_codebook_size,
        max_subtokens=cfg.max_subtokens,
        pad_token_id=cfg.pad_token_id,
        disabled_ids=disabled_ids,
    )

    codebook_manager = CodebookManager(
        embedding_dim=cfg.dim,
        **compression_kwargs,
    )
    lzw_compressor = LZWCompressor(**compression_kwargs)

    output_file = None
    if cli.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(cli.output_file)), exist_ok=True)
        output_file = open(cli.output_file, "w", encoding="utf-8")
        print(f"[inference] writing JSONL results to {cli.output_file}")

    try:
        for index, raw_prompt in enumerate(prompts):
            prompt = format_prompt(hf_tok, raw_prompt, instruct=cli.instruct)
            print(f"\nPrompt {index + 1}/{len(prompts)}: {repr(prompt)}\n")
            output, colored_output = generate(
                prompt,
                model,
                codebook_manager,
                lzw_compressor,
                hf_tok,
                stop_token_ids,
                max_new_tokens=cli.max_new_tokens,
                temperature=cli.temperature,
                device=device,
            )
            print(f"\n{'='*60}")
            print(f"PROMPT:    {raw_prompt}")
            print(f"GENERATED: {output}")
            print(
                "\nCOLORED (blue=base, yellow=2-gram, "
                "orange=3-gram, red=4-gram):"
            )
            print(colored_output)

            if output_file is not None:
                json.dump(
                    {
                        "index": index,
                        "prompt": raw_prompt,
                        "generated": output,
                        "instruct": cli.instruct,
                    },
                    output_file,
                    ensure_ascii=False,
                )
                output_file.write("\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
