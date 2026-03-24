"""Model configurations for zip2zip-llama."""

from torchtitan.models.common import (
    Embedding,
    FeedForward,
    GQAttention,
    RoPE,
    compute_ffn_hidden_dim,
)

from zip2zip_core.model import Zip2ZipLlama3Model, Zip2ZipTransformerBlock

zip2zip_llama_configs = {
    "debugmodel": Zip2ZipLlama3Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        max_codebook_size=256,
        max_subtokens=4,
        encoder_n_layers=1,
        encoder_n_heads=4,
        pad_token_id=0,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16, attn_backend="sdpa", rope_backend="complex"
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    # ~50M params (dim=256, 12 layers, embed ~33M + backbone ~10M + encoder ~0.3M)
    "50M": Zip2ZipLlama3Model.Config(
        dim=256,
        n_layers=12,
        vocab_size=128256,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=128,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=512,
        pad_token_id=128001,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 8,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    # ~150M params (dim=640, 12 layers, embed ~82M + backbone ~61M + encoder ~1M)
    "150M": Zip2ZipLlama3Model.Config(
        dim=640,
        n_layers=12,
        vocab_size=128256,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=256,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=1024,
        pad_token_id=128001,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(640, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=10,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=640 // 10,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "1B": Zip2ZipLlama3Model.Config(
        dim=2048,
        n_layers=16,
        vocab_size=128256,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=512,
        encoder_n_layers=2,
        encoder_n_heads=8,
        encoder_intermediate_size=2048,
        pad_token_id=128001,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(2048, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=2048 // 32,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
}
