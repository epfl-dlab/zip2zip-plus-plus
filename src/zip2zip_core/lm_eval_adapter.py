"""lm-eval-harness adapter for Zip2Zip checkpoints.

Exposes a `Zip2ZipLM` class (registered under `"zip2zip"`) that lm-eval-harness
can use to score `(context, continuation)` pairs and to compute rolling
perplexity over long texts.

Two scoring modes:

* "compressed" (default): apply LZW compression over (context + continuation),
  then sum the logprobs of compressed tokens whose base-position span is at or
  after the continuation boundary. v0.1-v0.6.4 and v0.7 checkpoints use the
  historical `k <= t` codebook mask. The branched v0.6.5 checkpoint records the
  exact decoder-time mask in `meta.pt`, and this adapter restores it automatically.

* "base": skip compression, feed raw base tokens with no codebook (vanilla LM
  path of the model), and read logprobs from the base-vocab logits. This is OOD
  for a model trained on compressed sequences but provides a cheap baseline.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from zip2zip_compression import (
    CodebookManager as RustCodebookManager,
    CompressionConfig,
    LZWCompressor,
)

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.checkpoint import (
    prepare_inference_state_dict,
    resolve_gated_rope_start_pair,
    strip_wrapper_prefixes,
)
from zip2zip_core.data import online_codebook_counts, online_unavailable_targets
from zip2zip_core.disabled_ids import base_disabled_ids, digit_ids
from zip2zip_core.model import Zip2ZipLlama3Model, restore_encoder_residual
from zip2zip_core.multi_view import (
    MultiViewAccumulator,
    build_expansion_index,
    exact_segmentation_logprob,
    multi_view_candidates,
    multi_view_segmentations,
    segmentation_proper_prefixes,
)

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


_DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"

# Chat templates that inject the current date (Llama 3.x) otherwise render the
# eval-day date, making chat-templated evals non-reproducible and mismatched
# with the training data. Pin it to the same constant the SFT dataset scripts
# use (scripts/create_sft_dataset_zip2zip1B.py CHAT_TEMPLATE_DATE). Templates
# without date logic (e.g. Phi-3.5) never see this and are unaffected.
_CHAT_TEMPLATE_DATE = "26 Jul 2024"

# The LZW disabled-id set is always derived from the tokenizer through
# zip2zip_core.disabled_ids, exactly as train.py derives it, so train and eval
# cannot drift. A hardcoded 4-id Llama list [128000..128003] used to live here:
# training only ever used that set between 9201a4a and 83a252d (Mar-Apr 2026,
# raw-pretrain checkpoints where ids >= 128004 never occur, so the derived set
# is behaviorally identical for them), and for every later checkpoint it
# under-protects the chat specials the model was trained with.


def _strip_wrapper_prefixes(state_dict: dict) -> dict:
    """Backward-compatible wrapper around the shared checkpoint helper."""
    return strip_wrapper_prefixes(state_dict)


def _fold_lora_weights(sd: dict, train_args: dict) -> dict:
    """Merge LoRA-format weights into plain linear weights.

    Checkpoints trained with --lora_rank store each wrapped linear as
    X.base_layer.weight + X.lora_A.weight + X.lora_B.weight. The eval model is
    a vanilla (non-LoRA) module, so fold W' = W + (B @ A) * (alpha/rank) and
    rename to X.weight — otherwise load_state_dict(strict=False) silently skips
    every decoder weight and the model scores with random layers.
    """
    bases = [
        k[: -len(".base_layer.weight")]
        for k in sd
        if k.endswith(".base_layer.weight")
    ]
    out = prepare_inference_state_dict(sd, train_args)
    if bases:
        rank = int(train_args["lora_rank"])
        scaling = float(train_args["lora_alpha"]) / rank
        print(
            f"[zip2zip-lm-eval] folded LoRA into {len(bases)} linear layers "
            f"(scaling={scaling:g})"
        )
    return out


def _load_zip2zip_checkpoint(
    ckpt_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    eval_max_subtokens: int | None = None,
):
    """Load a checkpoint produced by zip2zip_core.train.

    Requires meta.pt to recover the training arguments and rebuilds
    the matching model config; strictly loads normalized model weights so an
    incomplete or structurally mismatched checkpoint cannot score silently.
    """
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"required checkpoint metadata is missing: {meta_path}; evaluation "
            "cannot safely restore behavior-only model settings"
        )
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if not isinstance(meta, dict):
        raise ValueError(f"checkpoint metadata {meta_path} must be a dictionary")
    train_args = meta.get("args")
    if not isinstance(train_args, dict) or not train_args:
        raise ValueError(
            f"checkpoint args in {meta_path} must be a non-empty dictionary"
        )

    cfg_key = train_args.get("model_config", "1B")
    cfg = zip2zip_llama_configs[cfg_key]
    sd = torch.load(
        os.path.join(ckpt_dir, "model.pt"),
        map_location=str(device),
        weights_only=True,
    )
    sd = _strip_wrapper_prefixes(sd)
    sd = _fold_lora_weights(sd, train_args)

    overrides: dict = {}
    for key in (
        "max_subtokens",
        "max_codebook_size",
        "hyper_encoder_type",
        "encoder_dim",
        "encoder_n_layers",
        "encoder_n_heads",
        "encoder_intermediate_size",
        "token_type_loss_weight",
    ):
        v = train_args.get(key)
        if v is not None:
            overrides[key] = v
    # Architecture flag with inverted polarity: the checkpoint records the arg
    # --untied_hyper_encoder; the config field is tie_hyper_encoder. Rebuild the
    # untied model structurally so the second encoder's weights have a home
    # (miss this and the model is tied while the checkpoint is untied).
    if train_args.get("untied_hyper_encoder"):
        overrides["tie_hyper_encoder"] = False
        print("[zip2zip-lm-eval] untied hyper-encoder: building separate output encoder")
    if train_args.get("share_hyper_encoder_weights"):
        overrides["share_hyper_encoder_weights"] = True
        print("[zip2zip-lm-eval] shared hyper-encoder weights: no hyper_output module")
    if train_args.get("base_token_positions"):
        # Behavior flag, no weights: compressed-mode evals must position tokens
        # in base space exactly as trained; base-mode evals are unaffected
        # (uncompressed stream, positions == arange either way).
        overrides["base_token_positions"] = True
        print("[zip2zip-lm-eval] base-token RoPE positions: enabled from meta.pt")
    if train_args.get("two_axis_rope"):
        overrides["two_axis_rope"] = True
        print("[zip2zip-lm-eval] two-axis RoPE: enabled from meta.pt")
    if train_args.get("gated_compressed_rope"):
        overrides["gated_compressed_rope"] = True
        if train_args.get("gated_rope_start_layer") is not None:
            overrides["gated_rope_start_layer"] = int(
                train_args["gated_rope_start_layer"]
            )
        head_dim = (
            getattr(cfg.layer.attention, "head_dim", None)
            or cfg.dim // cfg.layer.attention.n_heads
        )
        overrides["gated_rope_start_pair"] = resolve_gated_rope_start_pair(
            train_args,
            sd,
            n_complex_pairs=head_dim // 2,
            default=cfg.gated_rope_start_pair,
        )
        print(
            "[zip2zip-lm-eval] gated compressed-coordinate RoPE: enabled "
            f"from layer {overrides.get('gated_rope_start_layer', cfg.gated_rope_start_layer)}, "
            f"pair {overrides.get('gated_rope_start_pair', cfg.gated_rope_start_pair)} "
            "from meta.pt"
        )
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    checkpoint_max_subtokens = cfg.max_subtokens
    if eval_max_subtokens is not None:
        if eval_max_subtokens <= 0:
            raise ValueError(
                f"eval_max_subtokens must be positive, got {eval_max_subtokens}"
            )
        if eval_max_subtokens != checkpoint_max_subtokens:
            if cfg.hyper_encoder_type not in ("hierarchical", "fast_hierarchical"):
                raise ValueError(
                    "eval_max_subtokens can only change checkpoint max_subtokens "
                    "for hierarchical encoders. Flat hyper-encoders have "
                    "length-shaped positional embeddings, so this override would "
                    "not load the checkpoint safely."
                )
            cfg = dataclasses.replace(cfg, max_subtokens=eval_max_subtokens)
            print(
                "[zip2zip-lm-eval] eval max_subtokens override: "
                f"checkpoint={checkpoint_max_subtokens} eval={eval_max_subtokens}"
            )

    model = Zip2ZipLlama3Model(cfg)
    if not restore_encoder_residual(model, train_args):
        # Behavior-only flag: there is no state-dict key that can restore it.
        # Missing this silently evaluates a no-residual checkpoint with the
        # residual enabled.
        print("[zip2zip-lm-eval] hyper-encoder residual: disabled from meta.pt")
    model = model.to(device=device)
    if dtype is not None and dtype != torch.float32:
        # Cast parameters and real-valued buffers ONLY. A blanket .to(dtype)
        # also converts the complex64 RoPE cache (freqs_cis and rope.cache) to
        # a real dtype, silently discarding the imaginary part — which destroys
        # position encoding and caps any checkpoint at ~5 nats/token no matter
        # how good its weights are (the "Casting complex values to real
        # discards the imaginary part" UserWarning in earlier eval logs).
        model._apply(lambda t: t.to(dtype) if t.is_floating_point() else t)

    # Canonical training checkpoints are complete after wrapper normalization and
    # LoRA folding.  A permissive load can otherwise evaluate random decoder or
    # hyper-encoder weights while merely printing a warning.
    model.load_state_dict(sd, strict=True)

    model.eval()
    return model, cfg, train_args


@register_model("zip2zip")
class Zip2ZipLM(LM):
    """lm-eval-harness adapter for a single Zip2Zip checkpoint."""

    def __init__(
        self,
        pretrained: str,
        tokenizer: str = _DEFAULT_TOKENIZER,
        max_length: int = 4096,
        device: str = "cuda",
        dtype: str = "bfloat16",
        eval_mode: str = "compressed",
        batch_size: int | str = 1,
        hyper_causal_mask: bool = True,
        online_codebook_mask: bool | None = None,
        disable_digit_ids: bool = False,
        disable_mathsym_ids: bool = False,
        eval_max_subtokens: int | None = None,
        trim_stop_strings: bool = True,
        multi_view: bool = True,
        preserve_leading_space: bool = True,
    ):
        super().__init__()
        if not torch.cuda.is_available() and device.startswith("cuda"):
            device = "cpu"
        self._device = torch.device(device)
        self._dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }[dtype]
        if eval_mode not in ("compressed", "base"):
            raise ValueError(f"eval_mode must be 'compressed' or 'base', got {eval_mode!r}")

        self.model, self.cfg, self.train_args = _load_zip2zip_checkpoint(
            pretrained, self._device, self._dtype, eval_max_subtokens=eval_max_subtokens
        )
        self.checkpoint_max_subtokens = int(
            (self.train_args or {}).get("max_subtokens", self.cfg.max_subtokens)
        )
        self.eval_max_subtokens = int(self.cfg.max_subtokens)
        if self.cfg.max_codebook_size == 0 and eval_mode == "compressed":
            # With max_codebook_size=0 the compressor is an exact identity, so
            # compressed scoring equals base scoring numerically — but slower
            # and mislabeled in the results JSON. Control checkpoints are
            # base-mode-only: switch automatically.
            print("[zip2zip-lm-eval] checkpoint was trained with "
                  "max_codebook_size=0 (uncompressed control) — auto-switching "
                  "eval_mode to 'base'.")
            eval_mode = "base"
        # The harness/launchers default the tokenizer to Llama-3. If the
        # checkpoint's meta.pt recorded the training tokenizer, prefer it over
        # that default so a forgotten TOKENIZER env var cannot silently
        # evaluate e.g. a Phi checkpoint with Llama special-token rules.
        meta_tok = (self.train_args or {}).get("tokenizer")
        if meta_tok and tokenizer == _DEFAULT_TOKENIZER and meta_tok != tokenizer:
            print(f"[zip2zip-lm-eval] tokenizer left at the Llama default but "
                  f"meta.pt records {meta_tok!r} — using the checkpoint's tokenizer")
            tokenizer = meta_tok
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if len(self.tokenizer) > self.cfg.vocab_size:
            raise ValueError(
                f"tokenizer {tokenizer!r} has {len(self.tokenizer)} tokens but the "
                f"model vocab is {self.cfg.vocab_size}: wrong tokenizer for this "
                f"checkpoint. Pass the training tokenizer"
                + (f" ({meta_tok!r} per meta.pt)." if meta_tok else ".")
            )
        if meta_tok and meta_tok != tokenizer:
            msg = (f"eval tokenizer {tokenizer!r} != training tokenizer "
                   f"{meta_tok!r} (meta.pt). Evaluating a checkpoint with the "
                   f"wrong tokenizer silently corrupts every metric. Pass "
                   f"TOKENIZER={meta_tok!r}, or set ALLOW_TOKENIZER_MISMATCH=1 "
                   f"to override deliberately.")
            if os.environ.get("ALLOW_TOKENIZER_MISMATCH") != "1":
                raise ValueError(msg)
            print(f"[zip2zip-lm-eval] WARNING (ALLOW_TOKENIZER_MISMATCH=1): {msg}")
        # Must mirror train.py's derivation (all_special_ids | added vocab):
        # for Phi-3.5, all_special_ids alone is {unk,bos,eos} and misses the
        # chat tokens <|user|>/<|assistant|>/<|end|> (32001-32010), which the
        # compressor would then merge into hyper-tokens the model never saw
        # in training — collapsing every chat-templated eval. Sourced from
        # zip2zip_core.disabled_ids so train/eval/export can't drift apart.
        disabled_ids = sorted(base_disabled_ids(self.tokenizer, self.cfg.vocab_size))
        if (
            eval_mode == "compressed"
            and (self.train_args or {}).get("disable_digit_ids")
            and not disable_digit_ids
        ):
            # Canonical since the digitsafe run validated: eval follows the
            # checkpoint's training distribution automatically. Irrelevant in
            # base mode, where the compressor is never invoked.
            print("[zip2zip-lm-eval] checkpoint was trained with digit-protected "
                  "LZW (meta.pt) — auto-enabling digit protection for this eval.")
            disable_digit_ids = True
        if disable_digit_ids and not (self.train_args or {}).get("disable_digit_ids"):
            print("[zip2zip-lm-eval] note: digit-protected eval of a checkpoint "
                  "trained WITHOUT digit protection — fine as a diagnostic, but the "
                  "numbers are not comparable to this checkpoint's as-trained evals.")
        if disable_digit_ids:
            # Diagnostic: keep digits out of LZW merges so multi-digit numbers
            # stay digit-by-digit base tokens instead of composite hypertokens.
            digit_id_set = digit_ids(self.tokenizer, self.cfg.vocab_size)
            print(f"[zip2zip-lm-eval] digit ids disabled for LZW "
                  f"({len(digit_id_set)}): {sorted(digit_id_set)}")
            disabled_ids = sorted(set(disabled_ids) | digit_id_set)
        self.disable_digit_ids = bool(disable_digit_ids)
        if disable_mathsym_ids:
            # Diagnostic (triage for extending the protected set beyond digits):
            # keep math operators/symbols out of LZW merges. "." is deliberately
            # absent — decimals are already protected transitively when digits
            # are disabled (a merge needs an adjacent pair).
            sym_pieces = set("=+-*/%$^<>") | {f"▁{c}" for c in "=+-*/%$^<>"}
            sym_ids = sorted(
                i for piece, i in self.tokenizer.get_vocab().items()
                if piece in sym_pieces and 0 <= i < self.cfg.vocab_size
            )
            print(f"[zip2zip-lm-eval] math-symbol ids disabled for LZW "
                  f"({len(sym_ids)}): {sym_ids}")
            disabled_ids = sorted(set(disabled_ids) | set(sym_ids))
        self.disable_mathsym_ids = bool(disable_mathsym_ids)
        self._disabled_ids = disabled_ids
        print(f"[zip2zip-lm-eval] disabled_ids ({len(disabled_ids)}): {disabled_ids[:16]}"
              f"{'...' if len(disabled_ids) > 16 else ''}")
        self._max_length = int(max_length)
        self.eval_mode = eval_mode
        self._batch_size = int(batch_size)
        self.hyper_causal_mask = bool(hyper_causal_mask)
        self.trim_stop_strings = bool(trim_stop_strings)
        self.preserve_leading_space = bool(preserve_leading_space)
        # One fixed newline token to decode generations behind (see
        # _decode_generation). The LAST id of the tokenizer's encoding of "\n" is
        # the newline piece itself: SentencePiece tokenizers put a word-boundary
        # piece in front of it, BPE tokenizers encode it alone.
        self._decode_prefix_ids: List[int] = []
        self._decode_prefix_text = ""
        try:
            _nl = self.tokenizer.encode("\n", add_special_tokens=False)[-1:]
            if _nl and self.tokenizer.decode(_nl, skip_special_tokens=True) == "\n":
                self._decode_prefix_ids, self._decode_prefix_text = list(_nl), "\n"
        except Exception:
            pass
        if not self.preserve_leading_space:
            print("[zip2zip-lm-eval] legacy standalone decode of generations "
                  "(a leading space is lost on SentencePiece tokenizers)")
        elif not self._decode_prefix_ids:
            print("[zip2zip-lm-eval] WARNING: no usable newline token — "
                  "generations decode standalone, leading whitespace may be lost")
        # What the CHECKPOINT was trained with (provenance, never overridden).
        self.online_codebook_mask = bool(
            (self.train_args or {}).get("online_codebook_mask")
        )
        # What THIS eval scores with. None = follow the checkpoint (the default,
        # so a v0.6.5 model is scored in its own regime). Setting it explicitly
        # is for cross-version comparison: v0.1-v0.6.4 were all scored with the
        # legacy k<=t mask, so reading a v0.6.5 number against that table needs
        # online_codebook_mask=False. The two flags stay separate on purpose —
        # the results JSON reports both, so a comparison can never silently mix
        # regimes.
        self.online_codebook_mask_requested = (
            self.online_codebook_mask
            if online_codebook_mask is None
            else bool(online_codebook_mask)
        )
        if self.online_codebook_mask_requested and not self.online_codebook_mask:
            raise ValueError(
                "online_codebook_mask=True was requested but this checkpoint was "
                "not trained with it (meta.pt says online_codebook_mask=False); "
                "scoring it with the exact mask would not match any training "
                "regime. Drop the override."
            )
        self.online_codebook_mask_active = bool(
            self.online_codebook_mask_requested
            and self.eval_mode == "compressed"
            and self.hyper_causal_mask
        )
        if self.online_codebook_mask:
            if self.online_codebook_mask_active:
                state = "active"
            elif not self.online_codebook_mask_requested:
                state = "OVERRIDDEN OFF — scoring with the legacy k<=t mask"
            else:
                state = "inactive"
            print(
                "[zip2zip-lm-eval] decoder-time online codebook mask: "
                f"enabled from meta.pt ({state} in this eval)"
            )

        # Generation must stop on every eos the base model declares, not just
        # tokenizer.eos_token_id: Phi-3.5's generation_config lists
        # [<|end|>, <|assistant|>, <|endoftext|>] and chat turns end with <|end|>.
        stop_ids = {self.eot_token_id}
        try:
            from transformers import GenerationConfig
            gen_cfg = GenerationConfig.from_pretrained(tokenizer)
            eos = gen_cfg.eos_token_id
            if eos is not None:
                stop_ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        except Exception:
            pass
        self._stop_token_ids = {i for i in stop_ids if i is not None}
        print(f"[zip2zip-lm-eval] generation stop token ids: {sorted(self._stop_token_ids)}")

        # Aggregate compression counters over the whole run, read by callers
        # (e.g. scripts/eval_harness.py) after simple_evaluate returns.
        # in_*: full (context + continuation) sequences fed for scoring.
        # gen_*: tokens emitted by generate_until.
        self.compression_stats = {"in_comp": 0, "in_base": 0, "gen_comp": 0, "gen_base": 0}
        # Multi-view (segmentation-marginalized) perplexity, accumulated per
        # rolling-perplexity task and read back via multi_view_summary(). Exact
        # complete-segmentation scoring is the primary metric; the historical
        # first-token upper bound is reported as a companion metric at no
        # additional model-forward cost.
        self.multi_view = bool(multi_view)
        self._exact_forest_max_nodes = 512
        self._multi_view_accum = MultiViewAccumulator()
        if self.online_codebook_mask_active:
            self.compression_stats.update(
                online_skipped_targets=0,
                online_replay_requests=0,
                online_replay_tokens=0,
                online_replay_seconds=0.0,
            )

        self._compressor_kwargs = dict(
            initial_vocab_size=self.cfg.vocab_size,
            max_codebook_size=self.cfg.max_codebook_size,
            max_subtokens=self.cfg.max_subtokens,
            pad_token_id=self.cfg.pad_token_id,
            disabled_ids=self._disabled_ids,
        )

    # ───────────────────── lm-eval registry hooks ─────────────────────────

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs: dict = {}
        if arg_string:
            for kv in arg_string.split(","):
                if not kv.strip():
                    continue
                k, _, v = kv.partition("=")
                kwargs[k.strip()] = v.strip()
        # Accept both "max_length" (canonical lm-eval name) and "max_seq_len"
        # (pre-rename alias) on the model_args string.
        if "max_seq_len" in kwargs and "max_length" not in kwargs:
            kwargs["max_length"] = kwargs.pop("max_seq_len")
        for int_key in ("max_length", "batch_size"):
            if int_key in kwargs:
                kwargs[int_key] = int(kwargs[int_key])
        if "hyper_causal_mask" in kwargs:
            kwargs["hyper_causal_mask"] = kwargs["hyper_causal_mask"].lower() in ("1", "true", "yes")
        if "online_codebook_mask" in kwargs:
            kwargs["online_codebook_mask"] = kwargs["online_codebook_mask"].lower() in ("1", "true", "yes")
        if "trim_stop_strings" in kwargs:
            kwargs["trim_stop_strings"] = kwargs["trim_stop_strings"].lower() in ("1", "true", "yes")
        if "multi_view" in kwargs:
            kwargs["multi_view"] = kwargs["multi_view"].lower() in ("1", "true", "yes")
        if "preserve_leading_space" in kwargs:
            kwargs["preserve_leading_space"] = kwargs["preserve_leading_space"].lower() in ("1", "true", "yes")
        if additional_config:
            for k in ("batch_size", "device"):
                if k not in kwargs and additional_config.get(k) is not None:
                    kwargs[k] = additional_config[k]
        return cls(**kwargs)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id or self.cfg.pad_token_id

    @property
    def rolling_prefix_token_id(self) -> int:
        # Rolling perplexity conditions the first window on one synthetic
        # document-start token. Historically this was eos_token_id, which for
        # Phi (<|endoftext|>) is exactly the document separator seen in
        # training. Llama-3 instruct tokenizers set eos to <|eot_id|> (a
        # turn marker that never precedes fresh document text), so use BOS
        # instead — every training document begins with it. Gated on the
        # <|eot_id|> eos so Phi and base-Llama tokenizers keep their exact
        # historical prefix and the v0.6.x wikitext series stays comparable.
        if (
            self.tokenizer.eos_token == "<|eot_id|>"
            and self.tokenizer.bos_token_id is not None
        ):
            return self.tokenizer.bos_token_id
        return self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, s: str) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=False)

    def tok_decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        # Pin the template date only for templates that actually use it (Llama
        # 3.x), so date-free templates (Phi-3.5) render byte-identically to
        # before and their historical evals stay reproducible.
        extra = {}
        template = self.tokenizer.chat_template
        if isinstance(template, str) and "date_string" in template:
            extra["date_string"] = _CHAT_TEMPLATE_DATE
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            **extra,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    # ───────────────────────── codebook helpers ───────────────────────────

    def _codebook_to_tensor(self, codebook) -> torch.LongTensor:
        """Build the (max_codebook_size, max_subtokens) tensor expected by the model.

        Mirrors `Zip2ZipDataset._codebook_to_tensor` so that unused slots are
        filled with pad_token_id and the per-position pad mask in the model lines
        up correctly.
        """
        cb_dict = codebook.to_dict()
        pad = self.cfg.pad_token_id
        S = self.cfg.max_subtokens
        V = self.cfg.vocab_size
        rows = []
        for i in range(self.cfg.max_codebook_size):
            entry = cb_dict.get(V + i)
            if entry is None:
                rows.append([pad] * S)
                continue
            entry = list(entry)
            if len(entry) < S:
                entry = entry + [pad] * (S - len(entry))
            rows.append(entry[:S])
        return torch.tensor(rows, dtype=torch.long)

    # ─────────────────────────── scoring ──────────────────────────────────

    def _exact_multi_view_target_logprob_sequential(
        self,
        *,
        target_id: int,
        canonical_context: List[int],
        codebook,
        cb_tensor: torch.LongTensor,
        cb_dict: Dict[int, Sequence[int]],
        expansion_index: Dict[Tuple[int, ...], List[int]],
        root_log_probs: torch.Tensor,
    ) -> float:
        """Exact formula-(12) marginal for one canonical hypertoken target.

        S(x_i) and the output codebook are frozen to the hyper-token ids
        available at the start of the target. For every non-empty proper
        segmentation prefix, the model is re-run on canonical_context + prefix
        so subsequent factors use the actual non-canonical autoregressive
        history without exposing later codebook rows.
        """
        V = self.cfg.vocab_size
        finite_slots = torch.nonzero(
            torch.isfinite(root_log_probs[V:]), as_tuple=False
        ).flatten()
        available_hyper_ids = {V + int(slot) for slot in finite_slots.tolist()}
        if target_id not in available_hyper_ids:
            raise RuntimeError(
                f"exact multi-view target {target_id} is unavailable at the "
                "start of its canonical context"
            )
        root_codebook_count = int(finite_slots[-1]) + 1

        segmentations = multi_view_segmentations(
            target_id,
            cb_dict,
            expansion_index,
            V,
            available_hyper_ids=available_hyper_ids,
        )
        prefix_log_probs: Dict[Tuple[int, ...], torch.Tensor] = {
            (): root_log_probs
        }

        def token_logprob(prefix: Tuple[int, ...], token: int) -> float:
            branch_log_probs = prefix_log_probs.get(prefix)
            if branch_log_probs is None:
                branch_ids = canonical_context + list(prefix)
                if len(branch_ids) > self._max_length:
                    raise RuntimeError(
                        "exact multi-view branch exceeded max_length: "
                        f"{len(branch_ids)} > {self._max_length}"
                    )
                x = torch.tensor(
                    branch_ids, dtype=torch.long, device=self._device
                ).unsqueeze(0)
                # The complete-segmentation marginal is defined against the
                # target-start codebook. Passing an explicit constant count
                # prevents the legacy k<=t mask (or online replay) from
                # exposing additional rows as this branch prefix grows.
                branch_counts = torch.full_like(x, root_codebook_count)

                with torch.autocast(
                    device_type=self._device.type, dtype=self._dtype
                ):
                    out = self.model(
                        x,
                        codebook=cb_tensor,
                        hyper_causal_mask=self.hyper_causal_mask,
                        codebook_counts=branch_counts,
                    )
                logits = out[0] if isinstance(out, tuple) else out
                branch_log_probs = F.log_softmax(
                    logits[0, -1].float(), dim=-1
                )
                prefix_log_probs[prefix] = branch_log_probs
            return branch_log_probs[token].item()

        return exact_segmentation_logprob(segmentations, token_logprob)

    def _exact_multi_view_targets_logprobs_sequential(
        self,
        *,
        canonical_tokens: Sequence[int],
        targets: Sequence[Tuple[int, int, torch.Tensor]],
        codebook,
        cb_tensor: torch.LongTensor,
        cb_dict: Dict[int, Sequence[int]],
        expansion_index: Dict[Tuple[int, ...], List[int]],
        canonical_codebook_counts: torch.Tensor | None = None,
    ) -> Dict[int, float]:
        """Correctness-first fallback that scores every target sequentially."""
        del canonical_codebook_counts
        return {
            position: self._exact_multi_view_target_logprob_sequential(
                target_id=target_id,
                canonical_context=list(canonical_tokens[: position + 1]),
                codebook=codebook,
                cb_tensor=cb_tensor,
                cb_dict=cb_dict,
                expansion_index=expansion_index,
                root_log_probs=root_log_probs,
            )
            for position, target_id, root_log_probs in targets
        }

    def _exact_multi_view_targets_logprobs(self, **kwargs) -> Dict[int, float]:
        """Score all target marginals with a shared packed tree-attention batch."""
        cfg = self.model.zip2zip_config
        first_layer = next(iter(self.model.layers.values()))
        forest_supported = (
            first_layer.attention.attn_backend == "sdpa"
            and not cfg.two_axis_rope
            and not cfg.gated_compressed_rope
        )
        if not forest_supported:
            return self._exact_multi_view_targets_logprobs_sequential(**kwargs)
        return self._exact_multi_view_targets_logprobs_forest(**kwargs)

    def _exact_multi_view_targets_logprobs_forest(
        self,
        *,
        canonical_tokens: Sequence[int],
        targets: Sequence[Tuple[int, int, torch.Tensor]],
        codebook,
        cb_tensor: torch.LongTensor,
        cb_dict: Dict[int, Sequence[int]],
        expansion_index: Dict[Tuple[int, ...], List[int]],
        canonical_codebook_counts: torch.Tensor | None = None,
    ) -> Dict[int, float]:
        """Score many independent target marginals in packed attention forests.

        The canonical tokens form a shared causal backbone. A branch node for
        target position ``t`` can attend to canonical rows ``[:t + 1]`` and to
        nodes on its own segmentation-prefix ancestry, but never to another
        target or sibling branch. Packing therefore changes only the execution
        schedule: every node retains the per-target scorer's dependency graph,
        RoPE position, and codebook mask.

        Forests are split by node count to bound dense-mask and activation
        memory. Tests can override the conservative default through
        ``_exact_forest_max_nodes`` on the adapter.
        """
        if not targets:
            return {}

        V = self.cfg.vocab_size
        prepared = []
        results: Dict[int, float] = {}
        for position, target_id, root_log_probs in targets:
            finite_slots = torch.nonzero(
                torch.isfinite(root_log_probs[V:]), as_tuple=False
            ).flatten()
            available_hyper_ids = {
                V + int(slot) for slot in finite_slots.tolist()
            }
            if target_id not in available_hyper_ids:
                raise RuntimeError(
                    f"exact multi-view target {target_id} is unavailable at "
                    f"canonical position {position}"
                )
            segmentations = multi_view_segmentations(
                target_id,
                cb_dict,
                expansion_index,
                V,
                available_hyper_ids=available_hyper_ids,
            )
            prefixes = segmentation_proper_prefixes(segmentations)
            if not prefixes:
                results[position] = exact_segmentation_logprob(
                    segmentations,
                    lambda prefix, token, root=root_log_probs: root[token].item(),
                )
                continue
            context_len = position + 1
            for prefix in prefixes:
                branch_len = context_len + len(prefix)
                if branch_len > self._max_length:
                    raise RuntimeError(
                        "exact multi-view branch exceeded max_length: "
                        f"{branch_len} > {self._max_length}"
                    )
            root_codebook_count = int(finite_slots[-1]) + 1
            prepared.append(
                (
                    position,
                    target_id,
                    root_log_probs,
                    segmentations,
                    prefixes,
                    root_codebook_count,
                )
            )

        max_nodes = int(getattr(self, "_exact_forest_max_nodes", 512))
        if max_nodes <= 0:
            raise ValueError(
                f"_exact_forest_max_nodes must be positive, got {max_nodes}"
            )
        chunks = []
        chunk = []
        chunk_nodes = 0
        for target in prepared:
            target_nodes = len(target[4])
            if chunk and chunk_nodes + target_nodes > max_nodes:
                chunks.append(chunk)
                chunk = []
                chunk_nodes = 0
            chunk.append(target)
            chunk_nodes += target_nodes
        if chunk:
            chunks.append(chunk)

        def token_span(token: int) -> int:
            return len(cb_dict[token]) if token >= V else 1

        for forest_targets in chunks:
            backbone_len = max(target[0] + 1 for target in forest_targets)
            backbone = list(canonical_tokens[:backbone_len])
            base_ends = []
            base_cursor = 0
            for token in backbone:
                base_cursor += token_span(token)
                base_ends.append(base_cursor)

            node_records = []
            node_lookup = {}
            for target_number, target in enumerate(forest_targets):
                for prefix in target[4]:
                    node_lookup[(target_number, prefix)] = len(node_records)
                    node_records.append((target_number, prefix))

            targets_by_position: Dict[int, List[int]] = {}
            for target_number, target in enumerate(forest_targets):
                targets_by_position.setdefault(target[0], []).append(target_number)

            # Interleave every target tree immediately after its canonical root.
            # Physical order is presentation-only: the explicit row maps below
            # preserve the same canonical and branch dependency graph.
            packed_ids = []
            canonical_rows = {}
            node_rows = {}
            for position, token in enumerate(backbone):
                canonical_rows[position] = len(packed_ids)
                packed_ids.append(token)
                for target_number in targets_by_position.get(position, []):
                    for prefix in forest_targets[target_number][4]:
                        node_rows[(target_number, prefix)] = len(packed_ids)
                        packed_ids.append(prefix[-1])

            total_len = len(packed_ids)
            x = torch.tensor(
                packed_ids, dtype=torch.long, device=self._device
            ).unsqueeze(0)

            forest_mask = torch.zeros(
                (total_len, total_len),
                dtype=torch.bool,
            )
            canonical_row_indices = torch.tensor(
                [canonical_rows[position] for position in range(backbone_len)],
                dtype=torch.long,
            )
            forest_mask[
                canonical_row_indices[:, None], canonical_row_indices[None, :]
            ] = torch.ones((backbone_len, backbone_len), dtype=torch.bool).tril_()
            for target_number, prefix in node_records:
                row = node_rows[(target_number, prefix)]
                position = forest_targets[target_number][0]
                forest_mask[row, canonical_row_indices[: position + 1]] = True
                for depth in range(1, len(prefix) + 1):
                    ancestor = prefix[:depth]
                    forest_mask[row, node_rows[(target_number, ancestor)]] = True
            forest_mask = forest_mask.to(self._device)

            packed_positions = torch.empty(total_len, dtype=torch.long)
            if self.cfg.base_token_positions:
                for position, row in canonical_rows.items():
                    packed_positions[row] = base_ends[position] - 1
                for target_number, prefix in node_records:
                    position = forest_targets[target_number][0]
                    packed_positions[node_rows[(target_number, prefix)]] = (
                        base_ends[position]
                        + sum(token_span(token) for token in prefix)
                        - 1
                    )
            else:
                for position, row in canonical_rows.items():
                    packed_positions[row] = position
                for target_number, prefix in node_records:
                    packed_positions[node_rows[(target_number, prefix)]] = (
                        forest_targets[target_number][0] + len(prefix)
                    )
            positions = packed_positions.unsqueeze(0).to(self._device)

            # Keep every branch on its target-start output codebook. The
            # backbone values only make this a full-shaped tensor;
            # logit_positions selects branch nodes before the output heads.
            if self.online_codebook_mask_active:
                if canonical_codebook_counts is None:
                    raise RuntimeError(
                        "online forest scoring requires canonical codebook counts"
                    )
                counts = torch.empty(
                    total_len, dtype=torch.long, device=self._device
                )
                for position, row in canonical_rows.items():
                    counts[row] = canonical_codebook_counts[position]
            elif self.hyper_causal_mask:
                counts = torch.empty(
                    total_len, dtype=torch.long, device=self._device
                )
                for position, row in canonical_rows.items():
                    counts[row] = position + 1
            else:
                counts = torch.full(
                    (total_len,),
                    cb_tensor.shape[1],
                    dtype=torch.long,
                    device=self._device,
                )
            for target_number, prefix in node_records:
                counts[node_rows[(target_number, prefix)]] = forest_targets[
                    target_number
                ][5]
            packed_counts = counts.unsqueeze(0)

            logit_positions = torch.tensor(
                [node_rows[record] for record in node_records],
                dtype=torch.long,
                device=self._device,
            )
            with torch.autocast(
                device_type=self._device.type, dtype=self._dtype
            ):
                out = self.model(
                    x,
                    codebook=cb_tensor,
                    attention_masks=forest_mask,
                    positions=positions,
                    hyper_causal_mask=self.hyper_causal_mask,
                    codebook_counts=packed_counts,
                    logit_positions=logit_positions,
                )
            logits = out[0] if isinstance(out, tuple) else out
            node_log_probs = F.log_softmax(logits[0].float(), dim=-1)

            for target_number, target in enumerate(forest_targets):
                position, _, root_log_probs, segmentations, prefixes, _ = target
                prefix_log_probs = {
                    prefix: node_log_probs[
                        node_lookup[(target_number, prefix)]
                    ]
                    for prefix in prefixes
                }

                def token_logprob(
                    prefix: Tuple[int, ...],
                    token: int,
                    *,
                    root=root_log_probs,
                    branches=prefix_log_probs,
                ) -> float:
                    if not prefix:
                        return root[token].item()
                    return branches[prefix][token].item()

                results[position] = exact_segmentation_logprob(
                    segmentations, token_logprob
                )

            self.compression_stats.setdefault("exact_forest_forwards", 0)
            self.compression_stats.setdefault("exact_forest_targets", 0)
            self.compression_stats.setdefault("exact_forest_nodes", 0)
            self.compression_stats.setdefault("exact_forest_backbone_tokens", 0)
            self.compression_stats.setdefault("exact_forest_packed_tokens", 0)
            self.compression_stats.setdefault("exact_forest_max_packed_tokens", 0)
            self.compression_stats["exact_forest_forwards"] += 1
            self.compression_stats["exact_forest_targets"] += len(
                forest_targets
            )
            self.compression_stats["exact_forest_nodes"] += len(node_records)
            self.compression_stats["exact_forest_backbone_tokens"] += backbone_len
            self.compression_stats["exact_forest_packed_tokens"] += total_len
            self.compression_stats["exact_forest_max_packed_tokens"] = max(
                self.compression_stats["exact_forest_max_packed_tokens"],
                total_len,
            )

        return results

    @torch.no_grad()
    def _score_compressed(
        self,
        full_ids: List[int],
        cont_start_base: int,
        *,
        reject_unavailable_targets: bool = False,
        compute_multi_view: bool = False,
    ) -> Tuple[float, bool, int, int, float, int, float]:
        """Compress full_ids, run the model, sum logprobs of compressed tokens
        whose base span starts at or after `cont_start_base`.

        Returns (logprob_sum, is_greedy, n_compressed_scored, n_base_scored,
        first_token_logprob_sum, n_hyper_scored, exact_logprob_sum).
        ``compute_multi_view`` computes both the exact complete-segmentation
        marginal and its first-token upper bound. It never affects the strict
        score or ``is_greedy``.
        """
        if len(full_ids) < 2:
            return 0.0, True, 0, 0, 0.0, 0, 0.0

        compressor = LZWCompressor(**self._compressor_kwargs)
        compressed, _, codebook = compressor.encode(
            full_ids, padding="do_not_pad", truncation=False
        )
        if len(compressed) < 2:
            return 0.0, True, 0, 0, 0.0, 0, 0.0

        self.compression_stats["in_base"] += len(full_ids)
        self.compression_stats["in_comp"] += len(compressed)

        cb_dict = codebook.to_dict()
        V = self.cfg.vocab_size

        # Base-position span [start, end) for each compressed token.
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for tok in compressed:
            length = len(cb_dict[tok]) if tok >= V else 1
            spans.append((cursor, cursor + length))
            cursor += length

        x = torch.tensor(compressed[:-1], dtype=torch.long, device=self._device).unsqueeze(0)
        cb = self._codebook_to_tensor(codebook).to(self._device).unsqueeze(0)
        codebook_counts = None
        unavailable = None
        if self.online_codebook_mask_active:
            replay_start = time.perf_counter()
            counts_cpu = online_codebook_counts(
                compressed[:-1], codebook, self._compressor_kwargs
            )
            self.compression_stats["online_replay_seconds"] += (
                time.perf_counter() - replay_start
            )
            self.compression_stats["online_replay_requests"] += 1
            self.compression_stats["online_replay_tokens"] += len(compressed) - 1
            targets_cpu = torch.tensor(compressed[1:], dtype=torch.long)
            unavailable = online_unavailable_targets(
                targets_cpu, counts_cpu, V, cb.shape[1]
            ).tolist()
            codebook_counts = counts_cpu.to(self._device).unsqueeze(0)

        with torch.autocast(device_type=self._device.type, dtype=self._dtype):
            if codebook_counts is None:
                # Preserve the historical scorer exactly for v0.1-v0.6.4 and
                # for explicit --no_hyper_causal_mask diagnostics.
                out = self.model(
                    x, codebook=cb, hyper_causal_mask=self.hyper_causal_mask
                )
            else:
                out = self.model(
                    x,
                    codebook=cb,
                    hyper_causal_mask=self.hyper_causal_mask,
                    codebook_counts=codebook_counts,
                )
        logits = out[0] if isinstance(out, tuple) else out
        log_probs = F.log_softmax(logits[0].float(), dim=-1)

        # Ordered so the MC/generation paths (compute_multi_view=False) never
        # even read self.multi_view and pay zero multi-view overhead.
        mv_index = (
            build_expansion_index(cb_dict)
            if compute_multi_view and self.multi_view
            else None
        )

        total = 0.0
        mv_total = 0.0
        exact_total = 0.0
        is_greedy = True
        n_comp = 0
        n_base = 0
        n_hyper = 0
        exact_targets: List[Tuple[int, int, torch.Tensor]] = []
        exact_bounds: Dict[int, Tuple[float, float]] = {}
        for t in range(len(compressed) - 1):
            target = compressed[t + 1]
            bstart, bend = spans[t + 1]
            if bstart < cont_start_base:
                continue
            if unavailable is not None and unavailable[t]:
                self.compression_stats["online_skipped_targets"] += 1
                if reject_unavailable_targets:
                    raise RuntimeError(
                        "online codebook target is unavailable inside a "
                        "loglikelihood continuation; skipping it would bias "
                        "the request score and is_greedy result "
                        f"(compressed_position={t + 1}, target_id={target}). "
                        "The evaluation was stopped instead."
                    )
                continue
            lp_target = log_probs[t, target].item()
            total += lp_target
            if mv_index is None or target < V:
                # Base-token targets have a single view; both variants are strict.
                mv_total += lp_target
                exact_total += lp_target
            else:
                cands = multi_view_candidates(target, cb_dict, mv_index, V)
                first_token_lp = torch.logsumexp(
                    log_probs[t, cands], dim=0
                ).item()
                mv_total += first_token_lp
                exact_targets.append((t, target, log_probs[t]))
                exact_bounds[t] = (lp_target, first_token_lp)
                n_hyper += 1
            if int(log_probs[t].argmax().item()) != target:
                is_greedy = False
            n_comp += 1
            n_base += bend - bstart

        if exact_targets:
            if mv_index is None:
                raise RuntimeError("exact multi-view targets require an index")
            exact_logprobs = self._exact_multi_view_targets_logprobs(
                canonical_tokens=compressed[:-1],
                targets=exact_targets,
                codebook=codebook,
                cb_tensor=cb,
                cb_dict=cb_dict,
                expansion_index=mv_index,
                canonical_codebook_counts=(
                    codebook_counts[0] if codebook_counts is not None else None
                ),
            )
            tolerance = 5e-5
            for t, target, _ in exact_targets:
                exact_lp = exact_logprobs[t]
                lp_target, first_token_lp = exact_bounds[t]
                if (
                    exact_lp < lp_target - tolerance
                    or exact_lp > first_token_lp + tolerance
                ):
                    raise RuntimeError(
                        "exact multi-view probability violated "
                        "strict <= exact <= first-token bound: "
                        f"target={target} strict={lp_target} "
                        f"exact={exact_lp} first_token={first_token_lp}"
                    )
                exact_total += exact_lp
        return total, is_greedy, n_comp, n_base, mv_total, n_hyper, exact_total

    @torch.no_grad()
    def _score_base(
        self, full_ids: List[int], cont_start_base: int
    ) -> Tuple[float, bool, int, int, float, int, float]:
        """Vanilla LM scoring: feed base tokens, no codebook, base-vocab logits only.

        Base tokens have a single view, so the multi-view sum equals the
        strict sum by definition (returned for a uniform _score signature).
        """
        if len(full_ids) < 2:
            return 0.0, True, 0, 0, 0.0, 0, 0.0
        self.compression_stats["in_base"] += len(full_ids)
        self.compression_stats["in_comp"] += len(full_ids)
        x = torch.tensor(full_ids[:-1], dtype=torch.long, device=self._device).unsqueeze(0)
        with torch.autocast(device_type=self._device.type, dtype=self._dtype):
            out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        base_logits = logits[0, :, : self.cfg.vocab_size].float()
        log_probs = F.log_softmax(base_logits, dim=-1)

        total = 0.0
        is_greedy = True
        n = 0
        for t in range(len(full_ids) - 1):
            if (t + 1) < cont_start_base:
                continue
            target = full_ids[t + 1]
            total += log_probs[t, target].item()
            if int(log_probs[t].argmax().item()) != target:
                is_greedy = False
            n += 1
        return total, is_greedy, n, n, total, 0, total

    def _score(
        self,
        full_ids: List[int],
        cont_start_base: int,
        *,
        reject_unavailable_targets: bool = False,
        compute_multi_view: bool = False,
    ):
        if self.eval_mode == "base":
            return self._score_base(full_ids, cont_start_base)
        return self._score_compressed(
            full_ids,
            cont_start_base,
            reject_unavailable_targets=reject_unavailable_targets,
            compute_multi_view=compute_multi_view,
        )

    def compression_summary(self) -> dict:
        """Raw counters plus derived ratios (base tokens per compressed token,
        > 1 = more compression; same orientation as train.py's eval/compression).
        """
        s = dict(self.compression_stats)
        if s["in_comp"]:
            s["input_compression_ratio"] = s["in_base"] / s["in_comp"]
        if s["gen_comp"]:
            s["gen_compression_ratio"] = s["gen_base"] / s["gen_comp"]
        if s.get("online_replay_requests"):
            s["online_replay_ms_per_request"] = (
                1000 * s["online_replay_seconds"]
                / s["online_replay_requests"]
            )
        if s.get("online_replay_tokens"):
            s["online_replay_us_per_token"] = (
                1_000_000 * s["online_replay_seconds"]
                / s["online_replay_tokens"]
            )
        return s

    def multi_view_summary(self) -> dict:
        """Per-task strict/exact multi-view loglikelihood sums from rolling PPL.

        The historical first-token upper bound is included as
        ``first_token_multi_view_loglik_sum``. Empty when multi_view is off or
        no loglikelihood_rolling task ran. Callers turn the sums into metrics
        with ``derive_multi_view_metrics`` so task denominators cancel exactly.
        """
        if not self.multi_view:
            return {}
        return self._multi_view_accum.summary()

    # ──────────────────────── lm-eval interface ───────────────────────────

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Score (context, continuation) pairs.

        Truncates the joint sequence from the left if it exceeds max_length,
        keeping the continuation intact.
        """
        out: List[Tuple[float, bool]] = []
        for req in tqdm(requests, desc="loglikelihood", disable=len(requests) < 8):
            ctx, cont = req.args[0], req.args[1]
            if ctx:
                ctx_ids = self.tok_encode(ctx)
                full_ids = self.tok_encode(ctx + cont)
            else:
                ctx_ids = []
                full_ids = self.tok_encode(cont)
            n_cont = max(0, len(full_ids) - len(ctx_ids))
            if n_cont == 0:
                out.append((0.0, True))
                continue

            # Left-truncate, never drop continuation tokens.
            if len(full_ids) > self._max_length:
                drop = len(full_ids) - self._max_length
                drop = min(drop, max(0, len(ctx_ids) - 1))  # leave at least 1 ctx token
                if drop > 0:
                    full_ids = full_ids[drop:]
                if len(full_ids) > self._max_length:
                    full_ids = full_ids[-self._max_length:]
                cont_start_base = max(0, len(full_ids) - n_cont)
            else:
                cont_start_base = len(ctx_ids)

            lp, greedy, _, _, _, _, _ = self._score(
                full_ids,
                cont_start_base,
                reject_unavailable_targets=True,
            )
            out.append((lp, greedy))
        return out

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Sum logprobs over a long text using rolling windows with context overlap."""
        out: List[float] = []
        for req in tqdm(requests, desc="loglikelihood_rolling", disable=len(requests) < 4):
            text = req.args[0]
            task_name = getattr(req, "task_name", None)
            ids = self.tok_encode(text)
            if len(ids) < 2:
                if self.multi_view:
                    self._multi_view_accum.add_document(
                        task_name,
                        0.0,
                        0.0,
                        0,
                        0,
                        first_token_multi_view_logprob=0.0,
                    )
                out.append(0.0)
                continue
            total = 0.0
            mv_total = 0.0
            exact_total = 0.0
            n_targets = 0
            n_hyper = 0
            for prefix_tokens, pred_tokens in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    ids,
                    prefix_token=self.rolling_prefix_token_id,
                    max_seq_len=self._max_length,
                    context_len=1,
                ),
            ):
                full_ids = list(prefix_tokens) + list(pred_tokens)
                if len(full_ids) < 2:
                    continue
                lp, _, n_comp, _, mv_lp, nh, exact_lp = self._score(
                    full_ids,
                    cont_start_base=len(prefix_tokens),
                    compute_multi_view=True,
                )
                total += lp
                mv_total += mv_lp
                exact_total += exact_lp
                n_targets += n_comp
                n_hyper += nh
            if self.multi_view:
                self._multi_view_accum.add_document(
                    task_name,
                    total,
                    exact_total,
                    n_hyper,
                    n_targets,
                    first_token_multi_view_logprob=mv_total,
                )
            out.append(total)
        return out

    # ───────────────────────── generation ─────────────────────────────────

    def _manager_updates_to_tensor(
        self, flat_updates: List[int], indices: List[int]
    ) -> torch.LongTensor:
        """Reshape the manager's flat (num_entries * S,) update list into the
        (1, len(indices), max_subtokens) tensor the model expects.

        The rust manager pads each entry to `max_subtokens` and emits them in
        slot order; only the first `len(indices)` rows are real, the rest are
        padding from the unused tail of the codebook buffer.
        """
        S = self.cfg.max_subtokens
        n = len(indices)
        if n == 0:
            return torch.zeros(
                (1, 0, S), dtype=torch.long, device=self._device
            )
        rows = [flat_updates[i * S : (i + 1) * S] for i in range(n)]
        return torch.tensor([rows], dtype=torch.long, device=self._device)

    def _strip_pad(self, entry: List[int]) -> List[int]:
        return [t for t in entry if t != self.cfg.pad_token_id]

    def _sample_next(
        self,
        last_logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """Sample (or argmax) a single next token id from a (vocab,) logits tensor."""
        if not do_sample or temperature <= 0:
            return int(last_logits.argmax().item())
        logits = last_logits / max(temperature, 1e-5)
        probs = torch.softmax(logits.float(), dim=-1)
        if top_k and top_k > 0:
            topk = torch.topk(probs, k=min(top_k, probs.numel()))
            idx = torch.multinomial(topk.values, num_samples=1)
            return int(topk.indices[idx].item())
        if top_p and 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= top_p
            mask[..., 0] = True
            filtered = sorted_probs * mask
            filtered = filtered / filtered.sum()
            idx = torch.multinomial(filtered, num_samples=1)
            return int(sorted_idx[idx].item())
        idx = torch.multinomial(probs, num_samples=1)
        return int(idx.item())

    def _trim_stops(self, text: str, until: List[str]) -> str:
        """Cut the decoded continuation at the first stop-string occurrence.

        The generation loops only break AFTER a stop string has landed in the
        decoded text (and a hyper-token expansion can overshoot it by up to
        max_subtokens-1 base tokens), so the raw decode still carries the stop
        string and whatever followed it. lm-eval's contract is that the
        continuation ends before the stop. Regex-extracted tasks (gsm8k) never
        noticed the tail; text-scored tasks (triviaqa exact_match, code pass@1)
        need the cut. trim_stop_strings=False restores the raw return of all
        evals before 2026-08, for bit-exact legacy comparisons.
        """
        if not self.trim_stop_strings:
            return text
        for term in until:
            if term:
                text = text.split(term)[0]
        return text

    def _decode_generation(self, gen_ids: List[int]) -> str:
        """Decode generated ids as a CONTINUATION of the prompt.

        tokenizer.decode() on a standalone id list treats its first piece as the
        start of a text, and SentencePiece tokenizers (Phi-3.5, Llama-2) drop that
        piece's word-boundary marker: a completion that begins with four spaces of
        Python indentation comes back with three, so HumanEval raises
        IndentationError on every otherwise-correct function (measured 2026-09-06:
        4 of 5 smoke generations). Decoding behind a fixed newline token and
        cutting it back off yields exactly the text the model produced after the
        prompt. Byte-identical for pieces without a leading marker; on BPE
        tokenizers (Llama-3) decode never stripped anything, so nothing changes.
        preserve_leading_space=False keeps the pre-2026-09 standalone decode.
        """
        if not self.preserve_leading_space or not gen_ids or not self._decode_prefix_ids:
            return self.tok_decode(gen_ids)
        full = self.tok_decode(self._decode_prefix_ids + list(gen_ids))
        head = self._decode_prefix_text
        if full.startswith(head):
            return full[len(head):]
        return self.tok_decode(gen_ids)

    @torch.no_grad()
    def _generate_base(
        self,
        ctx_ids: List[int],
        until: List[str],
        max_gen_toks: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        """Vanilla LM generation: feed base tokens, no codebook, base-vocab logits."""
        max_ctx = max(1, self._max_length - max_gen_toks)
        if len(ctx_ids) > max_ctx:
            ctx_ids = ctx_ids[-max_ctx:]

        seq = list(ctx_ids)
        gen_ids: List[int] = []
        for _ in range(max_gen_toks):
            x = torch.tensor(seq, dtype=torch.long, device=self._device).unsqueeze(0)
            with torch.autocast(device_type=self._device.type, dtype=self._dtype):
                out = self.model(x)
            logits = out[0] if isinstance(out, tuple) else out
            last = logits[0, -1, : self.cfg.vocab_size].float()
            tok = self._sample_next(last, do_sample, temperature, top_p, top_k)
            if tok in self._stop_token_ids:
                break
            gen_ids.append(tok)
            seq.append(tok)
            if until:
                text = self.tok_decode(gen_ids)
                if any(s in text for s in until):
                    break
            if len(seq) >= self._max_length:
                break
        self.compression_stats["gen_comp"] += len(gen_ids)
        self.compression_stats["gen_base"] += len(gen_ids)
        return self._trim_stops(self._decode_generation(gen_ids), until)

    @torch.no_grad()
    def _generate_compressed(
        self,
        ctx_ids: List[int],
        until: List[str],
        max_gen_toks: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        """Compress prompt with LZW, then autoregressively sample compressed tokens.

        For each step we expand the sampled compressed token into its base-token
        span, advance an LZW manager with those base tokens to get the new codebook
        entries the model needs, and append the (compressed) token to the model
        input. No KV cache: the model is re-run on the full prefix each step.
        """
        cfg = self.cfg
        V = cfg.vocab_size
        max_ctx = max(1, self._max_length - max_gen_toks)

        compressor = LZWCompressor(**self._compressor_kwargs)
        compressed, _, _ = compressor.encode(
            ctx_ids, padding="do_not_pad", truncation=False
        )

        # If the compressed prompt is too long, drop base tokens from the left
        # and recompress until we have room for max_gen_toks new tokens.
        while len(compressed) > max_ctx and len(ctx_ids) > 1:
            drop = max(1, len(ctx_ids) // 8)
            ctx_ids = ctx_ids[drop:]
            compressor = LZWCompressor(**self._compressor_kwargs)
            compressed, _, _ = compressor.encode(
                ctx_ids, padding="do_not_pad", truncation=False
            )

        if len(compressed) == 0:
            return ""

        # LZW manager seeded by replaying the prompt's base tokens. The manager
        # returns the same codebook the compressor produced (same algorithm),
        # plus delta-only updates on subsequent calls.
        manager = RustCodebookManager(
            CompressionConfig(**self._compressor_kwargs)
        )
        flat, indices = manager.update_codebooks([list(ctx_ids)])
        seed_indices = list(indices[0])
        # Build hyper→base dict from the seeded entries (used to expand
        # generated hyper tokens back into base tokens for stop-string checks).
        S = cfg.max_subtokens
        hyper_to_base: dict[int, List[int]] = {}
        for row, slot in enumerate(seed_indices):
            entry = flat[0][row * S : (row + 1) * S]
            hyper_to_base[V + slot] = self._strip_pad(entry)

        self.model.reset_inference_cache()

        updates_t = self._manager_updates_to_tensor(flat[0], seed_indices)
        updates_indices = [seed_indices]

        x = torch.tensor(compressed, dtype=torch.long, device=self._device).unsqueeze(0)
        with torch.autocast(device_type=self._device.type, dtype=self._dtype):
            out = self.model(
                x,
                codebook_updates=updates_t,
                codebook_updates_indices=updates_indices,
            )
        logits = out[0] if isinstance(out, tuple) else out

        gen_base_ids: List[int] = []
        n_gen_comp = 0  # sampled compressed tokens that emitted >= 1 base token
        for _ in range(max_gen_toks):
            last = logits[0, -1].float()  # (V + max_codebook_size,)
            next_tok = self._sample_next(last, do_sample, temperature, top_p, top_k)

            if next_tok < V:
                expansion = [next_tok]
            else:
                expansion = list(hyper_to_base.get(next_tok, []))
                if not expansion:
                    # Model produced an unused hyper slot; treat as stop.
                    break

            stop_at = [i for i, t in enumerate(expansion) if t in self._stop_token_ids]
            if stop_at:
                cut = stop_at[0]
                gen_base_ids.extend(expansion[:cut])
                if cut > 0:
                    n_gen_comp += 1
                break

            gen_base_ids.extend(expansion)
            n_gen_comp += 1

            if until:
                text = self.tok_decode(gen_base_ids)
                if any(s in text for s in until):
                    break

            if len(compressed) + 1 > self._max_length:
                break

            # Advance LZW state with the newly-emitted base tokens.
            new_flat, new_indices = manager.update_codebooks([expansion])
            new_idx_list = list(new_indices[0])
            for row, slot in enumerate(new_idx_list):
                entry = new_flat[0][row * S : (row + 1) * S]
                hyper_to_base[V + slot] = self._strip_pad(entry)

            updates_t = self._manager_updates_to_tensor(new_flat[0], new_idx_list)
            updates_indices = [new_idx_list]

            compressed.append(next_tok)
            x = torch.tensor(
                compressed, dtype=torch.long, device=self._device
            ).unsqueeze(0)
            with torch.autocast(device_type=self._device.type, dtype=self._dtype):
                out = self.model(
                    x,
                    codebook_updates=updates_t,
                    codebook_updates_indices=updates_indices,
                )
            logits = out[0] if isinstance(out, tuple) else out

        self.compression_stats["gen_comp"] += n_gen_comp
        self.compression_stats["gen_base"] += len(gen_base_ids)
        return self._trim_stops(self._decode_generation(gen_base_ids), until)

    def generate_until(self, requests) -> List[str]:
        out: List[str] = []
        for req in tqdm(requests, desc="generate_until", disable=len(requests) < 4):
            ctx = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 and req.args[1] else {}
            until = gen_kwargs.get("until", []) or []
            if isinstance(until, str):
                until = [until]
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            do_sample = bool(gen_kwargs.get("do_sample", False))
            temperature = float(gen_kwargs.get("temperature", 0.0) or 0.0)
            top_p = float(gen_kwargs.get("top_p", 1.0) or 1.0)
            top_k = int(gen_kwargs.get("top_k", 0) or 0)

            ctx_ids = self.tok_encode(ctx) if ctx else []
            if self.eval_mode == "compressed":
                text = self._generate_compressed(
                    ctx_ids, until, max_gen_toks, do_sample, temperature, top_p, top_k
                )
            else:
                text = self._generate_base(
                    ctx_ids, until, max_gen_toks, do_sample, temperature, top_p, top_k
                )
            out.append(text)
        return out
