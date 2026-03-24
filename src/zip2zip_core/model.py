"""Zip2Zip Llama model - Llama with adaptive tokenization via LZW compression.

Extends the standard Llama decoder with:
- A hyper-encoder that computes embeddings for hypertokens from their constituent base tokens
- Dynamic output logits over the expanded vocabulary (base + hypertokens)
"""

from dataclasses import dataclass, field

import torch
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

        q = self.wq(x_norm).view(T, self.n_heads, self.head_dim)
        k = self.wk(x_norm).view(T, self.n_heads, self.head_dim)
        v = self.wv(x_norm).view(T, self.n_heads, self.head_dim)

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

        # Project down to encoder dim if needed
        if self.proj_in is not None:
            token_embeddings = self.proj_in(token_embeddings)

        D = self.dim
        pos = torch.arange(S, device=token_embeddings.device)
        x = token_embeddings + self.pos_embed(pos)

        if varlen_attn is not None:
            result = self._forward_varlen(x, mask, N, S, D)
            if self.proj_out is not None:
                result = self.proj_out(result)
            return result

        for layer in self.layers:
            x = layer(x, mask, causal=self.causal)

        x = self.norm(x)

        if self.causal:
            lengths = mask.sum(dim=-1).clamp(min=1)
            last_idx = (lengths - 1).long()
            result = x[torch.arange(x.shape[0], device=x.device), last_idx]
        else:
            masked_x = x * mask.unsqueeze(-1)
            count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            result = masked_x.sum(dim=1) / count

        if self.proj_out is not None:
            result = self.proj_out(result)
        return result

    def _forward_varlen(
        self, x: torch.Tensor, mask: torch.Tensor, N: int, S: int, D: int
    ) -> torch.Tensor:
        """Varlen forward path: pack valid tokens, run layers, pool results."""
        # Compute per-entry lengths and cumulative offsets
        lengths = mask.sum(dim=-1)  # (N,)
        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=x.device)
        torch.cumsum(lengths.to(torch.int32), dim=0, out=cu_seqlens[1:])
        # Use max_subtokens as upper bound to avoid device-to-host sync
        max_seqlen = S

        # Pack valid tokens: (N, S, D) -> (total_valid_tokens, D)
        x_packed = x[mask]

        # window_size for varlen_attn: (-1, 0) = causal, (-1, -1) = bidirectional
        window_size = (-1, 0) if self.causal else (-1, -1)

        # Run transformer layers on packed tokens
        for layer in self.layers:
            x_packed = layer.forward_varlen(
                x_packed, cu_seqlens, max_seqlen, window_size=window_size
            )

        x_packed = self.norm(x_packed)

        if self.causal:
            # Last token pooling: take the last valid token of each sequence
            last_indices = cu_seqlens[1:].long() - 1
            return x_packed[last_indices]
        else:
            # Mean pooling via scatter_add
            entry_ids = torch.arange(N, device=x.device).repeat_interleave(lengths)
            result = torch.zeros(N, D, device=x_packed.device, dtype=x_packed.dtype)
            result.scatter_add_(
                0, entry_ids.unsqueeze(-1).expand_as(x_packed), x_packed
            )
            result = result / lengths.unsqueeze(-1).clamp(min=1).to(result.dtype)
            return result


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

            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

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

        # Tie input and output embeddings
        self.output.weight = self.tok_embeddings.weight

        # Optional token type prediction head (base vs hyper)
        if config.token_type_loss_weight > 0:
            self.token_type_head = nn.Linear(config.dim, 1, bias=True)
        else:
            self.token_type_head = None

        # Hyper-encoder for computing hypertoken embeddings
        encoder_dim = config.encoder_dim or config.dim
        self.hyper_encoder = HyperEncoder(
            dim=encoder_dim,
            max_subtokens=config.max_subtokens,
            n_layers=config.encoder_n_layers,
            n_heads=config.encoder_n_heads,
            intermediate_size=config.encoder_intermediate_size,
            causal=config.encoder_causal,
            model_dim=config.dim,
        )

    def init_weights(self, **kwargs):
        super().init_weights(**kwargs)

        # Initialize token type head
        if self.token_type_head is not None:
            for m in self.token_type_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initialize hyper-encoder
        for name, p in self.hyper_encoder.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

        for module in self.hyper_encoder.modules():
            if isinstance(module, (nn.LayerNorm,)):
                module.reset_parameters()

    def _encode_codebook_with_weights(
        self, codebook: torch.Tensor, weight_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Compute encoded representations for codebook entries using given weight matrix.

        Args:
            codebook: (B, max_codebook_size, max_subtokens) base token IDs
            weight_matrix: (vocab_size, dim) weight matrix to look up base tokens from

        Returns:
            (B, max_codebook_size, dim) encoded representations
        """
        B, H, S = codebook.shape

        # Create mask for valid (non-pad) tokens
        pad_id = self.zip2zip_config.pad_token_id
        mask = codebook != pad_id  # (B, H, S)

        # Clamp IDs to valid range
        cb_clamped = codebook.clamp(min=0, max=self.zip2zip_config.vocab_size - 1)

        # Look up from given weight matrix
        cb_embeds = nn.functional.embedding(cb_clamped, weight_matrix)  # (B, H, S, dim)

        # Reshape for encoder: (B*H, S, dim)
        cb_embeds_flat = cb_embeds.view(B * H, S, -1)
        mask_flat = mask.view(B * H, S)

        # Encode
        encoded = self.hyper_encoder(cb_embeds_flat, mask_flat)  # (B*H, dim)

        return encoded.view(B, H, -1)

    def forward(
        self,
        tokens: torch.Tensor,
        codebook: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """Forward pass with zip2zip compressed tokens.

        Args:
            tokens: (B, T) compressed token IDs (mix of base + hyper)
            codebook: (B, max_codebook_size, max_subtokens) base token compositions
            attention_masks: optional attention masks
            positions: optional position IDs
        """
        vocab_size = self.zip2zip_config.vocab_size

        # === Embedding ===
        if codebook is not None:
            # Compute hyper INPUT embeddings (from tok_embeddings weight)
            hyper_input_embeds = self._encode_codebook_with_weights(
                codebook, self.tok_embeddings.weight
            )  # (B, max_codebook_size, dim)

            # Separate base and hyper tokens
            base_mask = tokens < vocab_size
            hyper_mask = ~base_mask

            # Base token embeddings
            base_ids = tokens.clamp(max=vocab_size - 1)
            h = self.tok_embeddings(base_ids)  # (B, T, dim)

            if hyper_mask.any():
                # Hyper token embeddings via gather from encoded codebook
                hyper_ids = (tokens - vocab_size).clamp(min=0)  # (B, T)
                B, T = tokens.shape

                # Efficient gather using advanced indexing
                batch_idx = (
                    torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, T)
                )
                selected = hyper_input_embeds[batch_idx, hyper_ids]  # (B, T, dim)

                # Use hyper embeddings where tokens are hypertokens
                h = torch.where(hyper_mask.unsqueeze(-1), selected, h)
        else:
            # No compression - standard forward
            h = (
                self.tok_embeddings(tokens)
                if self.tok_embeddings is not None
                else tokens
            )

        # === Transformer layers ===
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h

        # === Token type prediction (base vs hyper) ===
        token_type_logits = None
        if self.token_type_head is not None:
            token_type_logits = self.token_type_head(h).squeeze(-1)  # (B, T)

        # === Output logits ===
        base_logits = self.output(h)  # (B, T, vocab_size)

        if codebook is not None:
            # With tied weights, encoded embeddings are already in output space
            # hyper_logits = h @ hyper_embeds^T
            hyper_logits = torch.bmm(h, hyper_input_embeds.transpose(1, 2))  # (B, T, K)

            # Mask padded codebook entries (all pad tokens) to -inf for exact softmax
            pad_id = self.zip2zip_config.pad_token_id
            codebook_used = (codebook != pad_id).any(dim=-1)  # (B, K)
            hyper_logits = hyper_logits.masked_fill(
                ~codebook_used.unsqueeze(1), float("-inf")
            )

            logits = torch.cat([base_logits, hyper_logits], dim=-1)
        else:
            logits = base_logits

        if token_type_logits is not None:
            return logits, token_type_logits

        return logits
