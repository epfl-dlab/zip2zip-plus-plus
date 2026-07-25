"""Zip2Zip Llama model - Llama with adaptive tokenization via LZW compression.

Extends the standard Llama decoder with:
- A hyper-encoder that computes embeddings for hypertokens from their constituent base tokens
- Dynamic output logits over the expanded vocabulary (base + hypertokens)
"""

from dataclasses import dataclass, field

import torch
import torch.utils.checkpoint
from torch import nn

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func as varlen_attn
    except ImportError:
        varlen_attn = None

from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.rope import RoPE
from torchtitan.models.common.utils import trunc_normal_
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.tools.logging import logger


def rope_cache_len(seq_len: int, max_subtokens: int, base_token_positions: bool) -> int:
    """Rows the freqs_cis cache needs: base-space positions can reach
    seq_len * max_subtokens - 1 in a fully merged window; compressed-index
    positions never exceed seq_len - 1."""
    return seq_len * (max_subtokens if base_token_positions else 1)


def restore_encoder_residual(model: nn.Module, train_args: dict) -> bool:
    """Restore the checkpoint's behavior-only encoder residual setting."""
    enabled = not bool(train_args.get("no_encoder_residual", False))
    model.encoder_residual = enabled
    return enabled


class HyperEncoderLayer(nn.Module):
    """Single transformer layer for the hyper-encoder."""

    def __init__(self, dim: int, intermediate_size: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        # x: (N, S, dim), mask: (N, S) bool
        B, S, D = x.shape

        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)

        q = self.wq(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_norm).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Padding mask: (N, 1, 1, S)
        attn_mask = mask.unsqueeze(1).unsqueeze(2).float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

        if causal:
            # Combine with causal mask: (1, 1, S, S)
            causal_mask = (
                torch.triu(
                    torch.full((S, S), float("-inf"), device=x.device), diagonal=1
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            attn_mask = (
                attn_mask + causal_mask
            )  # broadcasts (N,1,1,S) + (1,1,S,S) -> (N,1,S,S)

        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = residual + self.wo(attn_out)

        # FFN with pre-norm
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.w2(nn.functional.gelu(self.w1(x_norm)))

        return x

    def forward_varlen(
        self,
        x_packed: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        window_size: tuple[int, int] = (-1, -1),
    ) -> torch.Tensor:
        """Forward pass on packed (variable-length) inputs using varlen_attn.

        Args:
            x_packed: (total_tokens, dim) packed valid tokens (no padding)
            cu_seqlens: (num_seqs + 1,) int32 cumulative sequence lengths
            max_seqlen: maximum sequence length (upper bound)
            window_size: (-1, -1) for full attention, (-1, 0) for causal
        """
        T, D = x_packed.shape

        residual = x_packed
        x_norm = self.norm1(x_packed)

        with torch.profiler.record_function("he.forward_varlen.qkv_proj"):
            q = self.wq(x_norm).view(T, self.n_heads, self.head_dim)
            k = self.wk(x_norm).view(T, self.n_heads, self.head_dim)
            v = self.wv(x_norm).view(T, self.n_heads, self.head_dim)

        with torch.profiler.record_function("he.forward_varlen.varlen_attn"):
            attn_out = varlen_attn(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                window_size=window_size,
            )

        # attn_out: (total_tokens, n_heads, head_dim)
        attn_out = attn_out.reshape(T, D)
        x = residual + self.wo(attn_out)

        residual = x
        x_norm = self.norm2(x)
        with torch.profiler.record_function("he.forward_varlen.ffn"):
            x = residual + self.w2(nn.functional.gelu(self.w1(x_norm)))

        return x


class HyperEncoder(nn.Module):
    """Encoder that maps codebook entries (sequences of base tokens) to embeddings.

    Supports two modes:
    - Bidirectional (causal=False): full attention + mean pooling (v2)
    - Causal (causal=True): causal attention + last-token pooling (v3)

    When encoder_dim != model_dim, projection layers are used to map between
    the model's embedding space and the encoder's internal dimension.
    """

    def __init__(
        self,
        dim: int,
        max_subtokens: int,
        n_layers: int = 2,
        n_heads: int = 8,
        intermediate_size: int | None = None,
        causal: bool = False,
        model_dim: int | None = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * dim

        self.causal = causal
        self.model_dim = model_dim or dim
        self.dim = dim

        # Projection layers when encoder dim != model dim
        if self.model_dim != dim:
            self.proj_in = nn.Linear(self.model_dim, dim, bias=False)
            self.proj_out = nn.Linear(dim, self.model_dim, bias=False)
        else:
            self.proj_in = None
            self.proj_out = None

        self.pos_embed = nn.Embedding(max_subtokens, dim)
        self.layers = nn.ModuleList(
            [
                HyperEncoderLayer(dim, intermediate_size, n_heads)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (N, S, model_dim) embedded base tokens from codebook entries
            mask: (N, S) bool mask, True for valid tokens
        Returns:
            (N, model_dim) pooled embeddings for each codebook entry
        """
        N, S, _ = token_embeddings.shape

        with torch.profiler.record_function("he.proj_in"):
            # Project down to encoder dim if needed
            if self.proj_in is not None:
                token_embeddings = self.proj_in(token_embeddings)

            D = self.dim
            pos = torch.arange(S, device=token_embeddings.device)
            x = token_embeddings + self.pos_embed(pos)

        if varlen_attn is not None and not getattr(self, 'disable_varlen', False):
            result = self._forward_varlen(x, mask, N, S, D)
            with torch.profiler.record_function("he.proj_out"):
                if self.proj_out is not None:
                    result = self.proj_out(result)
            return result

        with torch.profiler.record_function("he.novarlen_layers"):
            for layer in self.layers:
                x = layer(x, mask, causal=self.causal)

        with torch.profiler.record_function("he.pool"):
            x = self.norm(x)

            if self.causal:
                lengths = mask.sum(dim=-1).clamp(min=1)
                last_idx = (lengths - 1).long()
                result = x[torch.arange(x.shape[0], device=x.device), last_idx]
            else:
                masked_x = x * mask.unsqueeze(-1)
                count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                result = masked_x.sum(dim=1) / count

        with torch.profiler.record_function("he.proj_out"):
            if self.proj_out is not None:
                result = self.proj_out(result)
        return result

    def _forward_varlen(
        self, x: torch.Tensor, mask: torch.Tensor, N: int, S: int, D: int
    ) -> torch.Tensor:
        """Varlen forward path: pack valid tokens, run layers, pool results."""
        with torch.profiler.record_function("he.forward_varlen.pack"):
            lengths = mask.sum(dim=-1)  # (N,)
            cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=x.device)
            torch.cumsum(lengths.to(torch.int32), dim=0, out=cu_seqlens[1:])
            max_seqlen = S
            x_packed = x[mask]

        window_size = (-1, 0) if self.causal else (-1, -1)

        with torch.profiler.record_function("he.forward_varlen.layers"):
            for layer in self.layers:
                x_packed = layer.forward_varlen(
                    x_packed, cu_seqlens, max_seqlen, window_size=window_size
                )

        with torch.profiler.record_function("he.forward_varlen.pool"):
            x_packed = self.norm(x_packed)

            if self.causal:
                last_indices = cu_seqlens[1:].long() - 1
                return x_packed[last_indices]
            else:
                entry_ids = torch.arange(N, device=x.device).repeat_interleave(lengths)
                result = torch.zeros(N, D, device=x_packed.device, dtype=x_packed.dtype)
                result.scatter_add_(
                    0, entry_ids.unsqueeze(-1).expand_as(x_packed), x_packed
                )
                result = result / lengths.unsqueeze(-1).clamp(min=1).to(result.dtype)
                return result


class PairwiseHyperEncoder(nn.Module):
    """Compose two embeddings into one with a small transformer encoder."""

    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        n_heads: int = 8,
        intermediate_size: int | None = None,
        causal: bool = False,
        model_dim: int | None = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * dim

        self.causal = causal
        self.model_dim = model_dim or dim
        self.dim = dim

        if self.model_dim != dim:
            self.proj_in = nn.Linear(self.model_dim, dim, bias=False)
            self.proj_out = nn.Linear(dim, self.model_dim, bias=False)
        else:
            self.proj_in = None
            self.proj_out = None

        self.pos_embed = nn.Embedding(2, dim)
        self.layers = nn.ModuleList(
            [
                HyperEncoderLayer(dim, intermediate_size, n_heads)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, pair_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode a pair of embeddings into a single composed representation."""
        _, seq_len, _ = pair_embeddings.shape
        if seq_len != 2:
            raise ValueError(
                f"PairwiseHyperEncoder expects sequence length 2, got {seq_len}"
            )

        x = pair_embeddings
        if self.proj_in is not None:
            x = self.proj_in(x)

        pos = torch.arange(2, device=x.device)
        x = x + self.pos_embed(pos)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)

        for layer in self.layers:
            x = layer(x, mask, causal=self.causal)

        x = self.norm(x)
        result = x[:, 1] if self.causal else x.mean(dim=1)

        if self.proj_out is not None:
            result = self.proj_out(result)
        return result


class HierarchicalHyperEncoder(nn.Module):
    """Compose each valid subtoken sequence left-to-right."""

    def __init__(
        self,
        dim: int,
        max_subtokens: int,
        n_layers: int = 2,
        n_heads: int = 8,
        intermediate_size: int | None = None,
        causal: bool = False,
        model_dim: int | None = None,
    ):
        super().__init__()
        self.max_subtokens = max_subtokens
        self.model_dim = model_dim or dim
        self.pair_encoder = PairwiseHyperEncoder(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            intermediate_size=intermediate_size,
            causal=causal,
            model_dim=model_dim,
        )

    def forward(
            self, token_embeddings: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            """Left-fold valid subtokens into a single composed embedding."""
            num_entries = token_embeddings.shape[0]
            lengths = mask.sum(dim=1).long()
            outputs = torch.zeros(
                num_entries,
                self.model_dim,
                dtype=token_embeddings.dtype,
                device=token_embeddings.device,
            )

            active_indices = torch.nonzero(lengths >= 2, as_tuple=False).flatten()
            if active_indices.numel() == 0:
                return outputs

            states = self.pair_encoder(token_embeddings[active_indices, :2, :])

            for pos in range(2, self.max_subtokens):
                still_active = (pos < lengths[active_indices]).unsqueeze(-1)
                pair_inputs = torch.stack(
                    [states, token_embeddings[active_indices, pos]],
                    dim=1,
                )
                updated = self.pair_encoder(pair_inputs)
                states = torch.where(still_active, updated, states)

            outputs[active_indices] = states
            return outputs


class FastHierarchicalHyperEncoder(nn.Module):
    """Fast left-to-right composition using gated MLP instead of attention.

    Same left-fold semantics as HierarchicalHyperEncoder but replaces the
    PairwiseHyperEncoder (transformer layers on seq_len=2) with a single
    gated MLP: gate * proj(a) + (1-gate) * proj(b).
    """

    def __init__(
        self,
        dim: int,
        max_subtokens: int,
        n_layers: int = 2,
        n_heads: int = 8,
        intermediate_size: int | None = None,
        causal: bool = False,
        model_dim: int | None = None,
    ):
        super().__init__()
        self.max_subtokens = max_subtokens
        self.model_dim = model_dim or dim
        D = self.model_dim

        self.gate = nn.Linear(2 * D, D)
        self.proj_a = nn.Linear(D, D, bias=False)
        self.proj_b = nn.Linear(D, D, bias=False)

        # Init: gate bias to 0 → sigmoid(0) = 0.5 → even mix of a and b
        # proj_a/proj_b near identity → compose ≈ mean(a, b) at init
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.eye_(self.proj_a.weight)
        nn.init.eye_(self.proj_b.weight)

    def init_weights(self):
        """Re-apply identity-like init (called after model-level xavier init)."""
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.eye_(self.proj_a.weight)
        nn.init.eye_(self.proj_b.weight)

    def _compose(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Gated composition: (N, D) x (N, D) -> (N, D)."""
        g = self.gate(torch.cat([a, b], dim=-1)).sigmoid()
        return (g * self.proj_a(a) + (1 - g) * self.proj_b(b)).to(a.dtype)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Left-fold valid subtokens into a single composed embedding."""
        num_entries = token_embeddings.shape[0]
        lengths = mask.sum(dim=1).long()
        outputs = torch.zeros(
            num_entries,
            self.model_dim,
            dtype=token_embeddings.dtype,
            device=token_embeddings.device,
        )

        active_mask = lengths >= 2
        if not active_mask.any():
            return outputs

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        states = self._compose(
            token_embeddings[active_indices, 0],
            token_embeddings[active_indices, 1],
        )

        for pos in range(2, self.max_subtokens):
            still_active = pos < lengths[active_indices]
            if not still_active.any():
                break
            idx = active_indices[still_active]
            new_states = states.clone()
            new_states[still_active] = self._compose(
                states[still_active],
                token_embeddings[idx, pos],
            )
            states = new_states
            

        outputs[active_indices] = states
        return outputs


class Zip2ZipTransformerBlock(TransformerBlock):
    """Llama3 TransformerBlock for Zip2Zip (same as Llama3TransformerBlock)."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.attention = config.attention.build(dim=dim)
        assert config.feed_forward is not None
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = nn.RMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=config.norm_eps)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, **kwargs):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Zip2ZipLlama3Model(Decoder):
    """Llama3 model with zip2zip adaptive tokenization.

    The model processes compressed token sequences that contain both base tokens
    and hypertokens. Hypertokens are composed from multiple base tokens via LZW
    compression, and their embeddings are computed dynamically by the hyper-encoder.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        n_layers: int = 16
        vocab_size: int = 128256
        layer: TransformerBlock.Config

        # zip2zip specific
        max_codebook_size: int = 2048
        max_subtokens: int = 4
        encoder_dim: int | None = None  # None = same as decoder dim
        encoder_n_layers: int = 2
        encoder_n_heads: int = 8
        encoder_intermediate_size: int | None = None
        encoder_causal: bool = False
        tie_word_embeddings: bool = True
        # Tie the hyper-encoder across the input-embedding and output-logit roles.
        # True (default) = one shared hyper-encoder (legacy behavior); False =
        # a second output-role encoder reading lm_head rows (matches the released
        # model). Checkpoints saved without this field load as tied.
        tie_hyper_encoder: bool = True
        hyper_encoder_type: str = "flat"
        # RoPE positions follow the UNCOMPRESSED stream: every token sits at the
        # base-space index of its last constituent (base tokens: their own index;
        # hypertokens: span end), so relative distances mean the same thing they
        # meant in pretraining regardless of local compression ratio. Off = one
        # position per compressed token (v0.5 and released-model behavior).
        base_token_positions: bool = False
        # Guarantee the hyper-encoder emits EXACTLY zero at init, so the first
        # hypertoken embedding equals its first base token's embedding (the
        # documented intent of the encoder_residual design). The existing
        # zero-init only touches proj_out, which does not exist when
        # encoder_dim == model_dim — see _init_hyper_encoder. Off = the
        # v0.1-v0.6.3 behavior (encoder starts at ~54x the embedding norm).
        zero_init_encoder_output: bool = False
        reconstruction_loss_weight: float = 0.0
        token_type_loss_weight: float = (
            0.0  # weight for base/hyper token type prediction head
        )
        pad_token_id: int = 128001

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            import dataclasses as _dc

            self.rope = _dc.replace(
                self.rope,
                max_seq_len=rope_cache_len(
                    seq_len, self.max_subtokens, self.base_token_positions
                ),
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * (self.dim // self.layer.attention.n_heads),
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.zip2zip_config = config

        # Tie input and output embeddings (Llama-style). Phi-3.5-mini uses
        # untied embeddings, so keep self.output as a separate Linear in that case.
        if config.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

        # Optional token type prediction head (base vs hyper)
        if config.token_type_loss_weight > 0:
            self.token_type_head = nn.Linear(config.dim, 1, bias=True)
        else:
            self.token_type_head = None

        # Pre-allocated hyper embedding buffers for incremental inference.
        # Reset to None between sequences via reset_inference_cache().
        self._hyper_embeds_buf: torch.Tensor | None = None  # (B, max_codebook_size, dim)
        self._hyper_embeds_used: torch.Tensor | None = None  # (B, max_codebook_size) bool
        # Output-role buffer, only used when untied (mirrors _hyper_embeds_buf,
        # encoded from lm_head rows instead of tok_embeddings rows).
        self._hyper_out_embeds_buf: torch.Tensor | None = None
        # Per-entry base-token span counts (B, max_codebook_size), maintained
        # alongside the embed buffers so base_token_positions can be computed
        # in incremental inference where only codebook UPDATES are visible.
        self._hyper_span_buf: torch.Tensor | None = None
        # Which module(s) init_weights() zeroed per encoder role, so a zero-init
        # that silently matches nothing is visible. Populated by
        # _init_hyper_encoder; plain dict, so it never reaches the state dict or
        # FSDP.
        self.zero_init_report: dict[str, list[str]] = {}

        # Hyper-encoder(s) for computing hypertoken embeddings. Tied: one encoder
        # serves both the input-embedding and output-logit roles. Untied: a second
        # encoder (self.hyper_output) produces the output-logit vectors from
        # lm_head rows. The 'hyper_output' prefix is intentional — freeze/LoRA,
        # the hyper-LR optimizer group, and HF missing-key classification already
        # key on it.
        encoder_dim = config.encoder_dim or config.dim
        hyper_encoder_kwargs = dict(
            dim=encoder_dim,
            max_subtokens=config.max_subtokens,
            n_layers=config.encoder_n_layers,
            n_heads=config.encoder_n_heads,
            intermediate_size=config.encoder_intermediate_size,
            causal=config.encoder_causal,
            model_dim=config.dim,
        )
        self.hyper_encoder = self._build_hyper_encoder(hyper_encoder_kwargs)
        if config.tie_hyper_encoder:
            self.hyper_output = None
        else:
            self.hyper_output = self._build_hyper_encoder(hyper_encoder_kwargs)

    def _build_hyper_encoder(self, hyper_encoder_kwargs: dict):
        htype = self.zip2zip_config.hyper_encoder_type
        if htype == "flat":
            return HyperEncoder(**hyper_encoder_kwargs)
        elif htype == "hierarchical":
            return HierarchicalHyperEncoder(**hyper_encoder_kwargs)
        elif htype == "fast_hierarchical":
            return FastHierarchicalHyperEncoder(**hyper_encoder_kwargs)
        else:
            raise ValueError(f"Unsupported hyper_encoder_type: {htype!r}")

    def init_weights(self, **kwargs):
        super().init_weights(**kwargs)

        # Initialize token type head
        if self.token_type_head is not None:
            for m in self.token_type_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize hyper-encoder(s) — same treatment for the tied encoder and,
        # when untied, the output-role encoder (single-variable: untied differs
        # from tied only in having a second encoder, not in init scheme).
        self._init_hyper_encoder(self.hyper_encoder, role="hyper_encoder")
        if self.hyper_output is not None:
            self._init_hyper_encoder(self.hyper_output, role="hyper_output")

    def _init_hyper_encoder(self, encoder, role: str = "hyper_encoder") -> None:
        if hasattr(encoder, 'init_weights'):
            encoder.init_weights()
        else:
            for name, p in encoder.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)

            for module in encoder.modules():
                if isinstance(module, (nn.LayerNorm,)):
                    module.reset_parameters()

        # Zero-init encoder output so initial hyper embedding ≈ first base token
        zeroed: list[str] = []
        if getattr(self, 'encoder_residual', True):
            for name, module in encoder.named_modules():
                if name.endswith('proj_out') and isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    zeroed.append(name)
            # proj_out exists ONLY when encoder_dim != model_dim, so with the
            # released Phi recipe (encoder_dim == dim == 3072) the loop above
            # matches nothing and silently leaves the intent unimplemented: the
            # encoder then starts emitting a LayerNorm-scaled random vector with
            # ~54x the norm of the embedding it is supposed to nudge. That is the
            # initialization every v0.1-v0.6.3 run used and is consistent with
            # their large early-loss transient.
            # Zeroing the final LayerNorm (weight AND bias, so the output is
            # exactly 0 through both the padded and the varlen pooling paths)
            # restores the intended identity start with no new parameters and no
            # state-dict change.
            if not zeroed and self.zip2zip_config.zero_init_encoder_output:
                tail, prefix = encoder, ""
                while getattr(tail, "pair_encoder", None) is not None:
                    tail = tail.pair_encoder
                    prefix += "pair_encoder."
                norm = getattr(tail, "norm", None)
                if not isinstance(norm, nn.LayerNorm):
                    raise ValueError(
                        f"zero_init_encoder_output=True but {type(encoder).__name__} "
                        f"has no proj_out and no final LayerNorm to zero, so a zero "
                        f"initial encoder output cannot be guaranteed. Use a "
                        f"hyper_encoder_type with a zeroable final output gate, or "
                        f"drop the flag."
                    )
                nn.init.zeros_(norm.weight)
                nn.init.zeros_(norm.bias)
                zeroed.append(prefix + "norm")
        self.zero_init_report[role] = zeroed

    def reset_inference_cache(self) -> None:
        """Reset hyper embedding buffers. Call between sequences during inference."""
        self._hyper_embeds_buf = None
        self._hyper_embeds_used = None
        self._hyper_out_embeds_buf = None
        self._hyper_span_buf = None

    def _base_token_positions(
        self, tokens: torch.Tensor, codebook: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Base-space (uncompressed-stream) RoPE positions for compressed tokens.

        Each token's span = how many base tokens it stands for (1 for base ids,
        non-pad subtoken count for hypertokens). A token sits at the base-space
        index of its LAST constituent: cumsum(spans) - 1. All-base rows reduce
        exactly to arange(T). Span source: the full codebook (training) or the
        persistent _hyper_span_buf (incremental inference).
        """
        vocab_size = self.zip2zip_config.vocab_size
        is_hyper = tokens >= vocab_size
        spans = torch.ones_like(tokens)
        if is_hyper.any():
            entry_ids = (tokens - vocab_size).clamp(min=0)
            if codebook is not None:
                pad_id = self.zip2zip_config.pad_token_id
                entry_spans = (codebook != pad_id).sum(dim=-1)  # (B, K)
            else:
                entry_spans = self._hyper_span_buf  # (B, K)
            # clamp_min(1): a degenerate reference to an empty/unwritten entry
            # must not collapse positions (the compressor never produces one).
            spans = torch.where(
                is_hyper, entry_spans.gather(1, entry_ids).clamp(min=1), spans
            )
        pos = spans.cumsum(dim=-1) - 1
        cache_rows = self.freqs_cis.shape[0]
        if int(pos.max()) >= cache_rows:
            # Fail actionably instead of a CUDA device-side assert in the
            # freqs_cis gather. Training sizes the cache exactly (rope_cache_len);
            # this is reachable only when a small-config rope cache meets a long
            # compressed-space-truncated generation prompt.
            raise ValueError(
                f"base_token_positions: max base-space position {int(pos.max())} "
                f"exceeds the rope cache ({cache_rows} rows). Truncate the prompt "
                f"in base space or enlarge rope.max_seq_len."
            )
        return pos

    def _embed_tokens_train(
        self, tokens: torch.Tensor, codebook: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed tokens during training: encode full codebook, mix base + hyper embeddings.

        Args:
            tokens: (B, T) token IDs (mix of base + hyper)
            codebook: (B, K, max_subtokens) full codebook

        Returns:
            h: (B, T, dim) token embeddings
            hyper_embeds: (B, K, dim) input-role codebook embeddings (mixed into h)
            hyper_out_embeds: (B, K, dim) output-role embeddings scored in the
                logits (same tensor as hyper_embeds when tied)
        """
        vocab_size = self.zip2zip_config.vocab_size

        hyper_embeds = self._encode_codebook_with_weights(
            codebook, self.tok_embeddings.weight
        )  # (B, K, dim)
        # Output role: untied re-encodes the same entries from lm_head rows with
        # the second encoder; tied reuses the input tensor. Runs unconditionally
        # inside the rank-uniform gate at forward() — never behind hyper_mask.any().
        if self.hyper_output is not None:
            hyper_out_embeds = self._encode_codebook_with_weights(
                codebook, self.output.weight, encoder=self.hyper_output
            )
        else:
            hyper_out_embeds = hyper_embeds

        base_ids = tokens.clamp(max=vocab_size - 1)
        h = self.tok_embeddings(base_ids)  # (B, T, dim)

        hyper_mask = tokens >= vocab_size
        if hyper_mask.any():
            hyper_ids = (tokens - vocab_size).clamp(min=0)  # (B, T)
            B, T = tokens.shape
            batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, T)
            h = torch.where(
                hyper_mask.unsqueeze(-1), hyper_embeds[batch_idx, hyper_ids], h
            )

        return h, hyper_embeds, hyper_out_embeds

    def _embed_tokens_inference(
        self,
        tokens: torch.Tensor,
        codebook_updates: torch.Tensor,
        codebook_updates_indices: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed tokens during inference: incrementally update embedding buffer.

        Args:
            tokens: (B, T) token IDs (mix of base + hyper)
            codebook_updates: (B, max_updates, max_subtokens) new codebook entries
            codebook_updates_indices: per-batch list of buffer indices to write into

        Returns:
            h: (B, T, dim) token embeddings
            hyper_embeds: (B, max_codebook_size, dim) input-role buffer (mixed into h)
            hyper_out_embeds: (B, max_codebook_size, dim) output-role buffer scored
                in the logits (same buffer as hyper_embeds when tied)
        """
        vocab_size = self.zip2zip_config.vocab_size
        bs = tokens.size(0)
        max_cb = self.zip2zip_config.max_codebook_size
        untied = self.hyper_output is not None

        # Lazy-allocate the full-size buffer(s) on first inference step
        if self._hyper_embeds_buf is None:
            weight = self.tok_embeddings.weight
            self._hyper_embeds_buf = torch.zeros(
                bs, max_cb, self.zip2zip_config.dim,
                device=weight.device, dtype=weight.dtype,
            )
            self._hyper_embeds_used = torch.zeros(
                bs, max_cb, dtype=torch.bool, device=weight.device,
            )
            self._hyper_span_buf = torch.zeros(
                bs, max_cb, dtype=torch.long, device=weight.device,
            )
            if untied:
                self._hyper_out_embeds_buf = torch.zeros_like(self._hyper_embeds_buf)

        # Scatter new entries into their exact buffer positions (per batch item)
        if any(len(ui) > 0 for ui in codebook_updates_indices):
            new_embeds = self._encode_codebook_with_weights(
                codebook_updates, self.tok_embeddings.weight
            )  # (B, max_updates, dim)
            new_out_embeds = (
                self._encode_codebook_with_weights(
                    codebook_updates, self.output.weight, encoder=self.hyper_output
                )
                if untied
                else None
            )
            # The hyper-encoder's padded attention path can upcast to fp32;
            # index_put refuses mismatched dtypes, so cast to the buffer's.
            pad_id = self.zip2zip_config.pad_token_id
            update_spans = (codebook_updates != pad_id).sum(dim=-1)  # (B, max_updates)
            for i, ui in enumerate(codebook_updates_indices):
                if ui:
                    self._hyper_embeds_buf[i, ui] = new_embeds[i, : len(ui)].to(
                        self._hyper_embeds_buf.dtype
                    )
                    self._hyper_embeds_used[i, ui] = True
                    self._hyper_span_buf[i, ui] = update_spans[i, : len(ui)]
                    if untied:
                        self._hyper_out_embeds_buf[i, ui] = new_out_embeds[
                            i, : len(ui)
                        ].to(self._hyper_out_embeds_buf.dtype)

        hyper_embeds = self._hyper_embeds_buf  # (B, max_codebook_size, dim)
        hyper_out_embeds = self._hyper_out_embeds_buf if untied else hyper_embeds

        base_ids = tokens.clamp(max=vocab_size - 1)
        h = self.tok_embeddings(base_ids)  # (B, T, dim)

        hyper_mask = tokens >= vocab_size
        if hyper_mask.any():
            hyper_ids = (tokens - vocab_size).clamp(min=0)  # (B, T)
            B, T = tokens.shape
            batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, T)
            h = torch.where(
                hyper_mask.unsqueeze(-1), hyper_embeds[batch_idx, hyper_ids], h
            )

        return h, hyper_embeds, hyper_out_embeds

    def _encode_codebook_with_weights(
        self, codebook: torch.Tensor, weight_matrix: torch.Tensor, encoder=None
    ) -> torch.Tensor:
        """Compute encoded representations for codebook entries using given weight matrix.

        Args:
            codebook: (B, max_codebook_size, max_subtokens) base token IDs
            weight_matrix: (vocab_size, dim) weight matrix to look up base tokens from
            encoder: hyper-encoder module to run (default: self.hyper_encoder).
                The untied output role passes self.hyper_output.

        Returns:
            (B, max_codebook_size, dim) encoded representations
        """
        if encoder is None:
            encoder = self.hyper_encoder
        B, H, S = codebook.shape

        # Create mask for valid (non-pad) tokens
        pad_id = self.zip2zip_config.pad_token_id
        mask = codebook != pad_id  # (B, H, S)

        # Clamp IDs to valid range
        cb_clamped = codebook.clamp(min=0, max=self.zip2zip_config.vocab_size - 1)

        # Under FSDP, weight_matrix (tok_embeddings.weight) is a sharded DTensor.
        # nn.functional.embedding can't mix a DTensor weight with a plain-tensor
        # index, so gather it to a full local tensor first (the embedding is small).
        if hasattr(weight_matrix, "full_tensor"):
            weight_matrix = weight_matrix.full_tensor()

        # Look up from given weight matrix
        cb_embeds = nn.functional.embedding(cb_clamped, weight_matrix)  # (B, H, S, dim)

        # Reshape for encoder: (B*H, S, dim)
        cb_embeds_flat = cb_embeds.view(B * H, S, -1)
        mask_flat = mask.view(B * H, S)

        # Encode: residual from first token + learned delta
        with torch.profiler.record_function("hyper_encoder.core"):
            encoder_out = encoder(cb_embeds_flat, mask_flat)
            if getattr(self, 'encoder_residual', True):
                first_token_embed = cb_embeds_flat[:, 0, :]  # (B*H, dim)
                encoded = first_token_embed + encoder_out
            else:
                encoded = encoder_out

        return encoded.view(B, H, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        codebook: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        codebook_updates: torch.Tensor | None = None,
        codebook_updates_indices: list[list[int]] | None = None,
        hyper_causal_mask: bool = False,
    ):
        """Forward pass with zip2zip compressed tokens.

        Training mode (codebook is not None):
            Encodes full codebook, applies pad mask + optional hyper causal mask.

        Inference mode (codebook_updates provided):
            Incrementally updates embedding buffer, applies used-entry mask.
            Causal constraint is inherent from incremental building.

        Args:
            tokens: (B, T) compressed token IDs (mix of base + hyper)
            codebook: (B, K, max_subtokens) base token compositions (training)
            attention_masks: optional attention masks
            positions: optional position IDs
            codebook_updates: (B, max_updates, max_subtokens) new codebook entries (inference)
            codebook_updates_indices: per-batch buffer write indices (inference)
            hyper_causal_mask: if True, mask future codebook entries at each position (training only)
        """

        # === Embedding ===
        with torch.profiler.record_function("hyper_encoder"):
            if codebook is not None and (codebook != self.zip2zip_config.pad_token_id).any():
                h, hyper_embeds, hyper_out_embeds = self._embed_tokens_train(tokens, codebook)
                if self.zip2zip_config.base_token_positions and positions is None:
                    positions = self._base_token_positions(tokens, codebook)
            elif codebook_updates is not None and codebook_updates_indices is not None:
                h, hyper_embeds, hyper_out_embeds = self._embed_tokens_inference(
                    tokens, codebook_updates, codebook_updates_indices
                )
                if self.zip2zip_config.base_token_positions and positions is None:
                    positions = self._base_token_positions(tokens, codebook=None)
            else:
                # Plain/base-mode path: tokens ARE the uncompressed stream, so
                # base-space positions == arange — the default (positions=None)
                # is already correct with or without base_token_positions.
                h = self.tok_embeddings(tokens)
                hyper_embeds = None
                hyper_out_embeds = None

        # === Transformer layers ===
        use_ac = getattr(self, "gradient_checkpointing", False) and self.training
        with torch.profiler.record_function("Main LM"):
            for layer_id, layer in enumerate(self.layers.values()):
                with torch.profiler.record_function(f"layer_{layer_id}"):
                    if use_ac:
                        # Recompute layer activations in backward to save memory.
                        h = torch.utils.checkpoint.checkpoint(
                            layer, h, self.freqs_cis, attention_masks, positions,
                            use_reentrant=False,
                        )
                    else:
                        h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h

        # === Token type prediction (base vs hyper) ===
        token_type_logits = None
        if self.token_type_head is not None:
            token_type_logits = self.token_type_head(h).squeeze(-1)  # (B, T)

        with torch.profiler.record_function("lm_head"):
            # === Output logits ===
            base_logits = self.output(h)  # (B, T, vocab_size)

        with torch.profiler.record_function("hyper_lm_head"):
            if hyper_embeds is not None:
                # Untied: score against the output-role embeddings (lm_head rows);
                # tied: hyper_out_embeds is the same tensor as hyper_embeds.
                hyper_logits = torch.bmm(h, hyper_out_embeds.transpose(1, 2))  # (B, T, K)

                if codebook is not None:
                    # --- Training: pad mask + optional hyper causal mask ---
                    pad_id = self.zip2zip_config.pad_token_id
                    codebook_used = (codebook != pad_id).any(dim=-1)  # (B, K)
                    hyper_logits = hyper_logits.masked_fill(
                        ~codebook_used.unsqueeze(1), float("-inf")
                    )
                    # Hyper causal mask: entry k is created at LZW step k,
                    # so at position t only entries with k <= t are available.
                    if hyper_causal_mask:
                        T = h.shape[1]
                        K = hyper_embeds.shape[1]
                        pos = torch.arange(T, device=h.device).view(1, T, 1)
                        entry_idx = torch.arange(K, device=h.device).view(1, 1, K)
                        hyper_logits = hyper_logits.masked_fill(
                            entry_idx > pos, float("-inf")
                        )
                else:
                    # --- Inference: used-entry mask (causality is inherent) ---
                    codebook_used = self._hyper_embeds_used[:, : hyper_embeds.shape[1]]
                    hyper_logits = hyper_logits.masked_fill(
                        ~codebook_used.unsqueeze(1), float("-inf")
                    )

                with torch.profiler.record_function("logit_cat"):
                    logits = torch.cat([base_logits, hyper_logits], dim=-1)
            else:
                logits = base_logits

        if token_type_logits is not None:
            return logits, token_type_logits

        return logits
