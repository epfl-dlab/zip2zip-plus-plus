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
            # Llama 3.2 ships rope_scaling factor 32.0; the field's default
            # (8.0) is the Llama 3.1 value and silently detunes pretrained
            # Llama 3.2 weights at every position.
            scaling_factor=32.0,
        ),
    ),
    # ── Phi-shaped from-scratch family ─────────────────────────────────────
    # Scale sweep for the LZW transducer task (--mode compress) on a Phi-3.5
    # tokenized corpus. Same role as the 20M/50M/150M/1B Llama configs above,
    # but every entry follows Phi-3.5-mini's aspect ratios so the sweep and the
    # pretrained 3.8B point below sit on one architectural family:
    #   head_dim=96, MHA (n_kv_heads == n_heads), ffn = 8/3 * dim,
    #   RoPE theta=10000 unscaled, vocab 32064, untied embeddings.
    # Like Phi3.5-mini these use unscaled RoPE, not the official LongRoPE
    # short/long factors -- consistent within the family, and these are trained
    # from scratch so no pretrained checkpoint disagrees with it.
    #
    # DEPTHS DIFFER from the Llama configs of the same name on purpose: Phi's
    # 32k vocab makes the embedding table ~4x smaller (e.g. 12.3M vs 49M at
    # dim=192), so matching the Llama shape would land far below the label.
    # Depth is chosen instead so the TOTAL lands within ~12% of the name, which
    # is what the figure's x-axis claims. Verify with count_encoder_params.py
    # after any edit rather than trusting these comments.
    #
    # 6.7M params (dim=96, 4 layers, embed 6.2M + backbone 0.44M + encoder 0.07M)
    # The bottom rung. Phi-20M's 5.3M backbone is 3.4x Llama "20M"'s 1.57M -- the
    # size at which the published sweep shows the task is not learnable -- so the
    # ladder had no point small enough to reproduce that failure. This one sits
    # BELOW it (0.44M), on purpose: if even this learns the task, capacity is not
    # what separates the two families and the explanation has to be the tokenizer
    # or the single-direction objective.
    # Half of Phi-20M's width and a third of its depth. At dim=96 a head_dim of 96
    # leaves a single attention head -- the one place this family departs from
    # Phi-3.5-mini's shape, and unavoidable without changing head_dim. There is no
    # ~10M rung available: the embedding table alone steps 6.2M -> 12.3M between
    # dim=96 and dim=192, so totals jump from ~7M straight to ~14M.
    "Phi-7M": Zip2ZipLlama3Model.Config(
        dim=96,
        n_layers=4,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=2,
        encoder_intermediate_size=256,
        pad_token_id=32000,
        tok_embeddings=Embedding.Config(init_std=96 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=256,
            ),
            attention=GQAttention.Config(
                n_heads=1,
                n_kv_heads=1,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=96 // 1,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
    ),
    # 17.7M params (dim=192, 12 layers, embed 12.3M + backbone 5.3M + encoder 0.07M)
    # -- Llama "20M" for reference: 18.1M (16.4M of it embeddings)
    "Phi-20M": Zip2ZipLlama3Model.Config(
        dim=192,
        n_layers=12,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=64,
        encoder_n_layers=1,
        encoder_n_heads=2,
        encoder_intermediate_size=256,
        pad_token_id=32000,
        # Untied embeddings need the explicit small init (see Phi3.5-mini below):
        # the default init_std=1.0 makes the hyper-token embeddings ~55x too large
        # and the run starts at a ~1500 loss.
        tok_embeddings=Embedding.Config(init_std=192 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=512,
            ),
            attention=GQAttention.Config(
                n_heads=2,
                n_kv_heads=2,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=192 // 2,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
    ),
    # 46.2M params (dim=384, 12 layers, embed 24.6M + backbone 21.2M + encoder 0.30M)
    # -- Llama "50M" for reference: 43.3M
    "Phi-50M": Zip2ZipLlama3Model.Config(
        dim=384,
        n_layers=12,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=128,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=512,
        pad_token_id=32000,
        tok_embeddings=Embedding.Config(init_std=384 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=1024,
            ),
            attention=GQAttention.Config(
                n_heads=4,
                n_kv_heads=4,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=384 // 4,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
    ),
    # 149.5M params (dim=768, 14 layers, embed 49.3M + backbone 99.1M + encoder 1.18M)
    # -- Llama "150M" for reference: 144.2M
    "Phi-150M": Zip2ZipLlama3Model.Config(
        dim=768,
        n_layers=14,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=256,
        encoder_n_layers=1,
        encoder_n_heads=4,
        encoder_intermediate_size=1024,
        pad_token_id=32000,
        tok_embeddings=Embedding.Config(init_std=768 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=2048,
            ),
            attention=GQAttention.Config(
                n_heads=8,
                n_kv_heads=8,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=768 // 8,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
        ),
    ),
    # 978.9M params (dim=1728, 24 layers, embed 110.8M + backbone 860.0M + encoder 8.07M)
    # -- Llama "1B" for reference: 1244.2M, inflated by its 262.7M untied
    # embeddings; the two backbones are within 12% of each other.
    "Phi-1B": Zip2ZipLlama3Model.Config(
        dim=1728,
        n_layers=24,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=4,
        encoder_dim=512,
        encoder_n_layers=2,
        encoder_n_heads=8,
        encoder_intermediate_size=2048,
        pad_token_id=32000,
        tok_embeddings=Embedding.Config(init_std=1728 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=4608,
            ),
            attention=GQAttention.Config(
                n_heads=18,
                n_kv_heads=18,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=1728 // 18,
            max_seq_len=131072,
            theta=10000,
            backend="complex",
            scaling="none",
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
    # ~14.0B params: microsoft/Phi-3-medium-4k-instruct tensor shapes (dim=5120,
    # 40 layers, GQA 40 q-heads / 10 kv-heads, head_dim=128, ffn=17920,
    # vocab=32064, untied embeddings). RoPE theta=10000 unscaled is exact for the
    # 4k checkpoint (rope_scaling is null there). Not modeled: the checkpoint's
    # sliding_window=2047, inert for every training window and eval context this
    # project uses (<= 2047 attended tokens); positions stay within its trained
    # 4096 range. The launcher overrides the encoder geometry to 5120/40/20480
    # (the released 14B model's encoder shape).
    "Phi3-medium": Zip2ZipLlama3Model.Config(
        dim=5120,
        n_layers=40,
        vocab_size=32064,
        tie_word_embeddings=False,
        max_codebook_size=4096,
        max_subtokens=3,
        encoder_dim=512,
        encoder_n_layers=2,
        encoder_n_heads=8,
        encoder_intermediate_size=2048,
        pad_token_id=32000,
        tok_embeddings=Embedding.Config(init_std=5120 ** -0.5),
        layer=Zip2ZipTransformerBlock.Config(
            feed_forward=FeedForward.Config(
                hidden_dim=17920,
            ),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=10,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=5120 // 40,
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
            # Llama 3.2 ships rope_scaling factor 32.0; the field's default
            # (8.0) is the Llama 3.1 value and silently detunes pretrained
            # Llama 3.2 weights at every position.
            scaling_factor=32.0,
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
