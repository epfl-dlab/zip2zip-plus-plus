"""Dependency-light checkpoint normalization for inference and export.

Training checkpoints can contain wrapper-qualified parameter names and LoRA
modules.  Runtime callers build a plain decoder, so they must normalize names
and fold every LoRA adapter before loading.  Keep that logic here rather than
duplicating it in evaluation, export, and standalone inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import torch


_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_orig_mod.")
_BASE_WEIGHT_SUFFIX = ".base_layer.weight"
_BASE_BIAS_SUFFIX = ".base_layer.bias"
_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_LORA_SUFFIXES = (
    _BASE_WEIGHT_SUFFIX,
    _BASE_BIAS_SUFFIX,
    _LORA_A_SUFFIX,
    _LORA_B_SUFFIX,
)


def strip_wrapper_prefixes(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strip FSDP/compile wrapper components and reject key collisions."""
    normalized: dict[str, torch.Tensor] = {}
    original_for_key: dict[str, str] = {}
    for original, value in state_dict.items():
        key = original
        for prefix in _WRAPPER_PREFIXES:
            key = key.replace(prefix, "")
        if key in normalized:
            raise ValueError(
                "checkpoint key collision after wrapper normalization: "
                f"{original_for_key[key]!r} and {original!r} both map to {key!r}"
            )
        normalized[key] = value
        original_for_key[key] = original
    return normalized


def merge_lora_weights(
    state_dict: Mapping[str, torch.Tensor],
    *,
    scaling: float,
    expected_rank: int | None = None,
) -> dict[str, torch.Tensor]:
    """Fold a complete LoRA state dict into plain linear weights.

    ``scaling`` is the training-time ``alpha / rank``.  Malformed or partial
    adapter state is rejected rather than silently falling back to the frozen
    base weight.
    """
    if not isinstance(scaling, Real):
        raise TypeError(f"LoRA scaling must be numeric, got {scaling!r}")
    scaling = float(scaling)
    if not torch.isfinite(torch.tensor(scaling)):
        raise ValueError(f"LoRA scaling must be finite, got {scaling!r}")
    if expected_rank is not None and (
        not isinstance(expected_rank, int) or isinstance(expected_rank, bool)
        or expected_rank <= 0
    ):
        raise ValueError(f"LoRA rank must be a positive integer, got {expected_rank!r}")

    bases = sorted(
        key[: -len(_BASE_WEIGHT_SUFFIX)]
        for key in state_dict
        if key.endswith(_BASE_WEIGHT_SUFFIX)
    )
    stray = []
    for key in state_dict:
        for suffix in (_LORA_A_SUFFIX, _LORA_B_SUFFIX):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                if f"{base}{_BASE_WEIGHT_SUFFIX}" not in state_dict:
                    stray.append(key)
                break
    if not bases:
        orphaned = [
            key
            for key in state_dict
            if key.endswith((_LORA_A_SUFFIX, _LORA_B_SUFFIX, _BASE_BIAS_SUFFIX))
        ]
        if orphaned:
            raise ValueError(
                "checkpoint contains LoRA tensors without matching base weights: "
                f"{orphaned[:5]}"
            )
        return dict(state_dict)
    if stray:
        raise ValueError(
            "checkpoint contains LoRA tensors without matching base weights: "
            f"{stray[:5]}"
        )

    out = dict(state_dict)
    for base in bases:
        weight_key = f"{base}{_BASE_WEIGHT_SUFFIX}"
        a_key = f"{base}{_LORA_A_SUFFIX}"
        b_key = f"{base}{_LORA_B_SUFFIX}"
        weight = out.pop(weight_key)
        a = out.pop(a_key, None)
        b = out.pop(b_key, None)
        if a is None or b is None:
            missing = [key for key, value in ((a_key, a), (b_key, b)) if value is None]
            raise ValueError(
                f"incomplete LoRA state for {base!r}; missing {missing}"
            )
        if not all(isinstance(tensor, torch.Tensor) for tensor in (weight, a, b)):
            raise TypeError(f"LoRA tensors for {base!r} must be torch tensors")
        if weight.ndim != 2 or a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {base!r} must all be matrices; got "
                f"base={tuple(weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        rank = int(a.shape[0])
        if rank <= 0:
            raise ValueError(f"LoRA rank for {base!r} must be positive, got {rank}")
        shapes_match = (
            b.shape[1] == rank
            and a.shape[1] == weight.shape[1]
            and b.shape[0] == weight.shape[0]
        )
        if not shapes_match:
            raise ValueError(
                f"incompatible LoRA shapes for {base!r}: "
                f"base={tuple(weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        if expected_rank is not None and rank != expected_rank:
            raise ValueError(
                f"LoRA rank mismatch for {base!r}: metadata says {expected_rank}, "
                f"but A has rank {rank}"
            )

        # Fold in fp32, matching the established evaluation path and avoiding a
        # low-precision B@A during checkpoint conversion.
        merged = weight.float() + (b.float() @ a.float()) * scaling
        merged_weight_key = f"{base}.weight"
        if merged_weight_key in out:
            raise ValueError(
                "checkpoint contains both folded and LoRA-format weights for "
                f"{base!r}: {merged_weight_key!r} and {weight_key!r}"
            )
        out[merged_weight_key] = merged

        bias = out.pop(f"{base}{_BASE_BIAS_SUFFIX}", None)
        if bias is not None:
            if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
                raise ValueError(
                    f"LoRA base bias for {base!r} must be a vector, got "
                    f"{type(bias).__name__} shape={getattr(bias, 'shape', None)}"
                )
            if bias.shape[0] != weight.shape[0]:
                raise ValueError(
                    f"LoRA base bias for {base!r} has shape {tuple(bias.shape)}, "
                    f"expected ({weight.shape[0]},)"
                )
            merged_bias_key = f"{base}.bias"
            if merged_bias_key in out:
                raise ValueError(
                    "checkpoint contains both folded and LoRA-format biases for "
                    f"{base!r}: {merged_bias_key!r}"
                )
            out[merged_bias_key] = bias

    leftovers = [key for key in out if key.endswith(_LORA_SUFFIXES)]
    if leftovers:
        raise ValueError(f"unconsumed LoRA checkpoint tensors: {leftovers[:5]}")
    return out


def prepare_inference_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    train_args: Mapping[str, object] | None,
) -> dict[str, torch.Tensor]:
    """Normalize a training checkpoint for loading into a plain runtime model."""
    normalized = strip_wrapper_prefixes(state_dict)
    has_lora = any(key.endswith(_BASE_WEIGHT_SUFFIX) for key in normalized)
    has_adapter_tensor = any(
        key.endswith(_LORA_SUFFIXES) for key in normalized
    )
    if not has_lora:
        if has_adapter_tensor:
            # ``merge_lora_weights`` provides the detailed orphan diagnostic.
            return merge_lora_weights(normalized, scaling=1.0)
        return normalized

    args = train_args or {}
    rank = args.get("lora_rank")
    alpha = args.get("lora_alpha")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
        raise ValueError(
            "LoRA-format checkpoint requires a positive lora_rank in meta.pt; "
            f"got {rank!r}"
        )
    if not isinstance(alpha, Real):
        raise ValueError(
            "LoRA-format checkpoint requires numeric lora_alpha in meta.pt; "
            f"got {alpha!r}"
        )
    return merge_lora_weights(
        normalized,
        scaling=float(alpha) / rank,
        expected_rank=rank,
    )
