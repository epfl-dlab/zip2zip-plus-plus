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
    # ~20M params (dim=128, 6 layers, embed ~16M + backbone ~2M + encoder ~0.1M)
    "20M": Zip2ZipLlama3Model.Config(
        dim=128,
        n_layers=6,
        vocab_size=128256,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=2,
        encoder_intermediate_size=256,
        pad_token_id=128001,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(128, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=4,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=128 // 4,
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
    # ~400M params (dim=1024, 24 layers, embed ~131M + backbone ~307M + encoder ~3M)
    "400M": Zip2ZipLlama3Model.Config(
        dim=1024,
        n_layers=24,
        vocab_size=128256,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=256,
        encoder_n_layers=2,
        encoder_n_heads=4,
        encoder_intermediate_size=1024,
        pad_token_id=128001,
        tok_embeddings=Embedding.Config(),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(1024, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=16,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=1024 // 16,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    "1B_legacy": Zip2ZipLlama3Model.Config(
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
    # ~1B params — matches official Llama 3.2 1B architecture
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
                hidden_dim=8192,
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
            max_seq_len=131072,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    # ~3.8B params — matches the official Phi-3.5-mini-instruct tensor shapes
    # (dim=3072, 32 layers, 32 heads MHA, head_dim=96, ffn=8192, vocab=32064,
    #  rope theta=10000, untied input/output embeddings). Important: the
    #  official checkpoint uses LongRoPE short/long factor vectors; this
    #  inherited core config uses unscaled RoPE. That pre-existing baseline
    #  discrepancy must not be silently bundled into a candidate recipe.
    # zip2zip components follow the from-scratch core defaults (flat encoder,
    # encoder_dim=512, intermediate=2048, max_codebook_size=4096), max_subtokens=3.
    "Phi3.5-mini": Zip2ZipLlama3Model.Config(
        dim=3072,
        n_layers=32,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=3,
        encoder_dim=512,
        encoder_n_layers=2,
        encoder_n_heads=8,
        encoder_intermediate_size=2048,
        pad_token_id=32000,
        # Untied embeddings: init the input embedding with a small std (~dim^-0.5),
        # matching the output head's init. The default Embedding init_std=1.0 makes
        # the hyper-token embeddings (derived from tok_embeddings) ~55x too large,
        # which blows up the hyper logits and gives a ~1500 initial loss. In the
        # tied configs this was masked because the output init overwrote the shared
        # weight; with untied embeddings we must set it explicitly.
        tok_embeddings=Embedding.Config(init_std=3072 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=8192,
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=32,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=3072 // 32,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
    ),
    # ~3B params (dim=3072, 28 layers, head_dim=128)
    "3B": Zip2ZipLlama3Model.Config(
        dim=3072,
        n_layers=28,
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
                hidden_dim=compute_ffn_hidden_dim(3072, multiple_of=256)
            ),
            attention=GQAttention.Config(
                n_heads=24,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=3072 // 24,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
    # ~8B params — Llama 3.1 8B architecture
    # ⚠️ Official Llama 3.1 8B uses tie_word_embeddings=False, but our model ties them.
    # Must add untied embedding support before loading official 8B weights.
    "8B": Zip2ZipLlama3Model.Config(
        dim=4096,
        n_layers=32,
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
                hidden_dim=14336,
            ),
            attention=GQAttention.Config(
                n_heads=32,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=8192,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    ),
}
