"""Incremental key/value cache for Zip2Zip generation.

Long-context RULER evaluation decodes after prompts as long as 128K tokens.
Without a cache every decode step re-runs the whole prefix,
so one sample costs ``max_gen_toks`` full prefills instead of one — the
measured reason the long-context presets were unusable.

The cache is deliberately a plain preallocated buffer rather than a growing
``torch.cat``: concatenating a 32K x 32-layer cache on every step moves more
bytes than the attention it saves.

Two Zip2Zip-specific invariants ride along with the keys and values, because
getting either wrong changes the numbers rather than only the speed:

``base_offset``
    Zip2Zip RoPE positions are base-space (``cumsum(spans) - 1``), so a
    hypertoken advances the position by its span, not by one. A cached step
    only sees the new token, so the running base-space offset has to be
    carried here for ``_base_token_positions`` to continue the cumsum.

``rope_regime`` / ``two_axis_mixing``
    Phi-3 LongRoPE picks short or long factors from the maximum position in
    the sequence, and two-axis RoPE switches on as soon as the sequence holds
    a hypertoken. Both decisions are re-made from the whole prefix on every
    uncached step, so flipping one retroactively re-rotates every key. Cached
    keys were rotated once and cannot follow, so both are recorded at prefill
    and a flip raises RopeRegimeChanged, letting the caller drop the cache and
    re-prefill instead of silently mixing two rotations.
"""

from __future__ import annotations

import torch


class RopeRegimeChanged(RuntimeError):
    """The LongRoPE factor regime changed under a populated cache.

    Raised before any cached key is reused with the wrong factors. The caller
    is expected to discard the cache and re-run the full prefix, which is what
    the uncached path does implicitly on every step.
    """


class KVCache:
    """Preallocated per-layer key/value cache for one generation stream.

    Args:
        n_layers: number of decoder layers to hold buffers for.
        max_seq_len: capacity in compressed tokens. Size it as
            ``len(prompt) + max_gen_toks``; the cache grows if exceeded, at the
            cost of one reallocation.
    """

    def __init__(self, n_layers: int, max_seq_len: int):
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.n_layers = int(n_layers)
        self.capacity = int(max_seq_len)
        self.reset()

    # ───────────────────────────── lifecycle ──────────────────────────────

    def reset(self) -> None:
        """Drop every cached key/value and all carried position state."""
        self._k: list[torch.Tensor | None] = [None] * self.n_layers
        self._v: list[torch.Tensor | None] = [None] * self.n_layers
        self.seq_len = 0
        self.base_offset: torch.Tensor | None = None
        self.rope_regime: str | None = None
        self.two_axis_mixing: bool | None = None

    def __len__(self) -> int:
        return self.seq_len

    @property
    def is_empty(self) -> bool:
        return self.seq_len == 0

    # ─────────────────────────────── writes ───────────────────────────────

    def update(
        self, layer_id: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append one layer's new keys/values and return the full history.

        Args:
            layer_id: which layer's buffer to write.
            k, v: ``(B, n_kv_heads, T, head_dim)`` for the NEW tokens only.

        Returns:
            ``(k_all, v_all)`` views of shape ``(B, n_kv_heads, seq_len + T,
            head_dim)`` — the cached prefix followed by the new tokens.

        ``seq_len`` is not advanced here: every layer writes at the same
        offset within one forward, and :meth:`advance` moves the cursor once
        the whole forward is done.
        """
        if not 0 <= layer_id < self.n_layers:
            raise ValueError(
                f"layer_id {layer_id} outside [0, {self.n_layers})"
            )
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "kv cache expects (B, n_kv_heads, T, head_dim) tensors, got "
                f"k={tuple(k.shape)} v={tuple(v.shape)}"
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}"
            )

        new = k.shape[2]
        end = self.seq_len + new
        if end > self.capacity:
            self._grow(end)

        buf_k, buf_v = self._k[layer_id], self._v[layer_id]
        if buf_k is None:
            buf_k, buf_v = self._allocate(layer_id, k)
        elif buf_k.dtype != k.dtype:
            # Silently casting here would make cached and fresh keys disagree
            # in precision, which shows up as a small logit drift rather than
            # an error. An inference loop has no reason to change dtype.
            raise ValueError(
                f"layer {layer_id} kv cache holds {buf_k.dtype} but received "
                f"{k.dtype}; keep one autocast regime for the whole stream"
            )
        elif buf_k.shape[0] != k.shape[0]:
            raise ValueError(
                f"layer {layer_id} kv cache holds batch {buf_k.shape[0]} but "
                f"received {k.shape[0]}; call reset() between sequences"
            )

        buf_k[:, :, self.seq_len : end] = k
        buf_v[:, :, self.seq_len : end] = v
        return buf_k[:, :, :end], buf_v[:, :, :end]

    def advance(self, n_tokens: int, base_offset: torch.Tensor | None) -> None:
        """Commit one forward: move the cursor and carry the position state.

        Args:
            n_tokens: compressed tokens just written by every layer.
            base_offset: ``(B, 1)`` next base-space RoPE position, or None when
                the model is not using base-space positions.
        """
        self.seq_len += int(n_tokens)
        self.base_offset = base_offset

    # ──────────────────────────── allocation ──────────────────────────────

    def _allocate(
        self, layer_id: int, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, _, head_dim = ref.shape
        shape = (batch, heads, self.capacity, head_dim)
        self._k[layer_id] = torch.empty(
            shape, dtype=ref.dtype, device=ref.device
        )
        self._v[layer_id] = torch.empty(
            shape, dtype=ref.dtype, device=ref.device
        )
        return self._k[layer_id], self._v[layer_id]

    def _grow(self, needed: int) -> None:
        """Reallocate every populated buffer to hold at least ``needed``.

        A safety net, not the expected path: the generation loops size the
        cache from ``prompt + max_gen_toks`` up front.
        """
        capacity = max(int(needed), self.capacity * 2)
        for layer_id, buf_k in enumerate(self._k):
            if buf_k is None:
                continue
            buf_v = self._v[layer_id]
            batch, heads, _, head_dim = buf_k.shape
            shape = (batch, heads, capacity, head_dim)
            grown_k = torch.empty(shape, dtype=buf_k.dtype, device=buf_k.device)
            grown_v = torch.empty(shape, dtype=buf_v.dtype, device=buf_v.device)
            grown_k[:, :, : self.seq_len] = buf_k[:, :, : self.seq_len]
            grown_v[:, :, : self.seq_len] = buf_v[:, :, : self.seq_len]
            self._k[layer_id] = grown_k
            self._v[layer_id] = grown_v
        self.capacity = capacity

    # ───────────────────────────── diagnostics ────────────────────────────

    def memory_bytes(self) -> int:
        """Bytes currently held by the allocated buffers."""
        return sum(
            buf.numel() * buf.element_size()
            for buf in (*self._k, *self._v)
            if buf is not None
        )
