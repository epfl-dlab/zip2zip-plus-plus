"""lm-eval-harness adapter for Zip2Zip checkpoints.

Exposes a `Zip2ZipLM` class (registered under `"zip2zip"`) that lm-eval-harness
can use to score `(context, continuation)` pairs and to compute rolling
perplexity over long texts.

Two scoring modes:

* "compressed" (default): apply LZW compression over (context + continuation),
  then sum the logprobs of compressed tokens whose base-position span is at or
  after the continuation boundary. v0.1-v0.6.4 checkpoints use the historical
  `k <= t` codebook mask. A v0.6.5+ checkpoint records the exact decoder-time
  mask in `meta.pt`, and this adapter restores it automatically.

* "base": skip compression, feed raw base tokens with no codebook (vanilla LM
  path of the model), and read logprobs from the base-vocab logits. This is OOD
  for a model trained on compressed sequences but provides a cheap baseline.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import List, Tuple

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
from zip2zip_core.data import online_codebook_counts, online_unavailable_targets
from zip2zip_core.disabled_ids import base_disabled_ids, digit_ids
from zip2zip_core.model import Zip2ZipLlama3Model, restore_encoder_residual

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


_LLAMA3_DISABLED_IDS = [128000, 128001, 128002, 128003]
_DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"

# Special-token ids that must never be merged into the codebook. The Llama
# tokenizers historically disabled a fixed set; for other tokenizers (e.g.
# Phi-3.5-mini) we fall back to the tokenizer's own special-token ids.
_DISABLED_IDS_BY_TOKENIZER = {
    "meta-llama/Meta-Llama-3-8B": _LLAMA3_DISABLED_IDS,
    "meta-llama/Llama-3.1-8B": _LLAMA3_DISABLED_IDS,
    "meta-llama/Llama-3.2-1B-Instruct": _LLAMA3_DISABLED_IDS,
}


def _strip_wrapper_prefixes(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        k = k.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        out[k] = v
    return out


def _fold_lora_weights(sd: dict, train_args: dict) -> dict:
    """Merge LoRA-format weights into plain linear weights.

    Checkpoints trained with --lora_rank store each wrapped linear as
    X.base_layer.weight + X.lora_A.weight + X.lora_B.weight. The eval model is
    a vanilla (non-LoRA) module, so fold W' = W + (B @ A) * (alpha/rank) and
    rename to X.weight — otherwise load_state_dict(strict=False) silently skips
    every decoder weight and the model scores with random layers.
    """
    bases = [k[: -len(".base_layer.weight")] for k in sd if k.endswith(".base_layer.weight")]
    if not bases:
        return sd
    rank = train_args.get("lora_rank") or 0
    alpha = train_args.get("lora_alpha") or 1.0
    scaling = (alpha / rank) if rank else 1.0
    out = dict(sd)
    for base in bases:
        w = out.pop(f"{base}.base_layer.weight")
        a = out.pop(f"{base}.lora_A.weight", None)
        b = out.pop(f"{base}.lora_B.weight", None)
        if a is not None and b is not None:
            w = w.float() + (b.float() @ a.float()) * scaling
        out[f"{base}.weight"] = w
        bias = out.pop(f"{base}.base_layer.bias", None)
        if bias is not None:
            out[f"{base}.bias"] = bias
    print(f"[zip2zip-lm-eval] folded LoRA into {len(bases)} linear layers (scaling={scaling:g})")
    return out


def _load_zip2zip_checkpoint(ckpt_dir: str, device: torch.device, dtype: torch.dtype):
    """Load a checkpoint produced by zip2zip_core.train.

    Reads meta.pt (if present) to recover the training arguments and rebuilds
    the matching model config; loads model.pt non-strictly so older B/C/recon
    experiment checkpoints still load.
    """
    meta_path = os.path.join(ckpt_dir, "meta.pt")
    train_args: dict = {}
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        train_args = meta.get("args", {}) or {}

    cfg_key = train_args.get("model_config", "1B")
    cfg = zip2zip_llama_configs[cfg_key]

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
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

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

    sd = torch.load(
        os.path.join(ckpt_dir, "model.pt"),
        map_location=str(device),
        weights_only=True,
    )
    sd = _strip_wrapper_prefixes(sd)
    sd = _fold_lora_weights(sd, train_args)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[zip2zip-lm-eval] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[zip2zip-lm-eval] unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    # The load is strict=False (LoRA folding leaves benign gaps), so a config that
    # is tied while the checkpoint is untied would SILENTLY leave the output
    # encoder at random init. Hard-fail if the output encoder never got weights.
    if (
        not cfg.tie_hyper_encoder
        and not cfg.share_hyper_encoder_weights
        and any(k.startswith("hyper_output") for k in missing)
    ):
        raise RuntimeError(
            "untied checkpoint is missing hyper_output.* weights after load — the "
            "output encoder would score with random weights. Check the checkpoint "
            "and the untied_hyper_encoder flag in its meta.pt."
        )

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
            pretrained, self._device, self._dtype
        )
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
            print(f"[zip2zip-lm-eval] WARNING: eval tokenizer {tokenizer!r} != "
                  f"training tokenizer {meta_tok!r} (meta.pt)")
        disabled_ids = _DISABLED_IDS_BY_TOKENIZER.get(tokenizer)
        if disabled_ids is None:
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
        if additional_config:
            for k in ("batch_size", "device"):
                if k not in kwargs and additional_config.get(k) is not None:
                    kwargs[k] = additional_config[k]
        return cls(**kwargs)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id or self.cfg.pad_token_id

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
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
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

    @torch.no_grad()
    def _score_compressed(
        self,
        full_ids: List[int],
        cont_start_base: int,
        *,
        reject_unavailable_targets: bool = False,
    ) -> Tuple[float, bool, int, int]:
        """Compress full_ids, run the model, sum logprobs of compressed tokens
        whose base span starts at or after `cont_start_base`.

        Returns (logprob_sum, is_greedy, n_compressed_scored, n_base_scored).
        """
        if len(full_ids) < 2:
            return 0.0, True, 0, 0

        compressor = LZWCompressor(**self._compressor_kwargs)
        compressed, _, codebook = compressor.encode(
            full_ids, padding="do_not_pad", truncation=False
        )
        if len(compressed) < 2:
            return 0.0, True, 0, 0

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

        total = 0.0
        is_greedy = True
        n_comp = 0
        n_base = 0
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
            total += log_probs[t, target].item()
            if int(log_probs[t].argmax().item()) != target:
                is_greedy = False
            n_comp += 1
            n_base += bend - bstart
        return total, is_greedy, n_comp, n_base

    @torch.no_grad()
    def _score_base(
        self, full_ids: List[int], cont_start_base: int
    ) -> Tuple[float, bool, int, int]:
        """Vanilla LM scoring: feed base tokens, no codebook, base-vocab logits only."""
        if len(full_ids) < 2:
            return 0.0, True, 0, 0
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
        return total, is_greedy, n, n

    def _score(
        self,
        full_ids: List[int],
        cont_start_base: int,
        *,
        reject_unavailable_targets: bool = False,
    ):
        if self.eval_mode == "base":
            return self._score_base(full_ids, cont_start_base)
        return self._score_compressed(
            full_ids,
            cont_start_base,
            reject_unavailable_targets=reject_unavailable_targets,
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

            lp, greedy, _, _ = self._score(
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
            ids = self.tok_encode(text)
            if len(ids) < 2:
                out.append(0.0)
                continue
            total = 0.0
            for prefix_tokens, pred_tokens in map(
                make_disjoint_window,
                get_rolling_token_windows(
                    ids,
                    prefix_token=self.eot_token_id,
                    max_seq_len=self._max_length,
                    context_len=1,
                ),
            ):
                full_ids = list(prefix_tokens) + list(pred_tokens)
                if len(full_ids) < 2:
                    continue
                lp, _, _, _ = self._score(full_ids, cont_start_base=len(prefix_tokens))
                total += lp
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
        return self.tok_decode(gen_ids)

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
        return self.tok_decode(gen_base_ids)

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
