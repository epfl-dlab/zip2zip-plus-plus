"""Correctness invariants for the generation KV cache (2026-09).

Long-context generation used to re-run the model on the entire prefix for
every sampled token: a 32K-token RULER prompt with 128 generated tokens cost
128 full prefills. The cache makes a decode step cost one token — but only if
every piece of Zip2Zip state that the uncached path silently recomputed from
the whole prefix is carried forward explicitly. These tests pin that state.

Runs on CPU with the shrunken config (decoder dim 256 / vocab 512 / 2 layers,
encoder dim 128), fp32, varlen disabled. Runnable as a script
(python tests/test_kv_cache.py) and under pytest. Shared mechanics live in
tests/_hyper_common.py.

Invariants proven:
  K1  plain/base path: prefill + one-token steps == one whole-sequence forward.
  K2  compressed path with hypertokens, base_token_positions off AND on.
  K3  two_axis_rope: the odd (compressed-index) axis keeps counting across
      cached steps instead of restarting at 0.
  K4  gated_compressed_rope: same, through the per-layer gated cache.
  K5  chunked prefill (several new tokens onto a populated cache) — the case
      an is_causal=True shortcut gets silently wrong.
  K6  base-space positions advance by a hypertoken's SPAN, not by one: the
      carried offset equals the uncached cumsum, and generated tokens land
      after the prompt rather than on top of it.
  K7  a decode step attends to the whole cached prefix. Pinned directly,
      because is_causal=True on a (q_len=1, k_len=N) call is top-left aligned
      — it exposes key 0 only, and still returns plausible text.
  K8  LongRoPE short->long mid-generation raises RopeRegimeChanged rather than
      mixing factor sets, and the documented recovery (drop cache, re-prefill)
      reproduces the uncached logits exactly.
  K9  two-axis RoPE turning on mid-stream raises the same way.
  K10 the adapter's real generate loops return identical text with the cache
      on and off, for both eval modes.
  K11 cache bookkeeping: capacity growth preserves history, and a dtype or
      batch change between steps raises instead of corrupting the buffer.
"""

import dataclasses
import os
import sys

from _hyper_common import build_model, make_harness, summarize, DEV

import torch

from zip2zip_core.configs import zip2zip_llama_configs
from zip2zip_core.kv_cache import KVCache, RopeRegimeChanged
from zip2zip_core.rope_adaptation import phi3_longrope_config

MS = 4
T = 24
ENTRIES = [[11, 12, 13], [21, 22], [31, 32, 33, 34]]  # spans 3, 2, 4


def small_cfg(**over):
    base = zip2zip_llama_configs["Phi3.5-mini"]
    kw = dict(
        dim=256,
        n_layers=2,
        vocab_size=512,
        pad_token_id=0,
        max_codebook_size=8,
        max_subtokens=MS,
        tie_hyper_encoder=False,
        encoder_dim=128,
        encoder_n_layers=2,
        encoder_n_heads=4,
        encoder_intermediate_size=256,
        tok_embeddings=dataclasses.replace(base.tok_embeddings, init_std=256 ** -0.5),
        layer=dataclasses.replace(
            base.layer,
            feed_forward=dataclasses.replace(base.layer.feed_forward, hidden_dim=512),
            attention=dataclasses.replace(
                base.layer.attention, n_heads=8, n_kv_heads=4
            ),
        ),
        rope=dataclasses.replace(base.rope, dim=256 // 8, max_seq_len=T * MS * 2),
    )
    kw.update(over)
    return dataclasses.replace(base, **kw)


def build(cfg, seed=3):
    """Tiny model with a NON-degenerate hyper-encoder.

    init_weights() zero-inits each encoder's proj_out so an untrained model
    starts with hyper embedding == first sub-token embedding. That makes every
    hypertoken's logit EXACTLY equal to its first base token's logit, so greedy
    decoding is decided by argmax tie-breaking rather than by the model — the
    cached and uncached paths then diverge on float noise of 1e-7 for reasons
    that have nothing to do with the cache. Any trained checkpoint has a
    non-zero proj_out; give the toy one the same property.
    """
    model = build_model(cfg, seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for encoder in (model.hyper_encoder, getattr(model, "hyper_output", None)):
            if encoder is None:
                continue
            for name, module in encoder.named_modules():
                if name.endswith("proj_out") and isinstance(module, torch.nn.Linear):
                    module.weight.copy_(
                        torch.randn(
                            module.weight.shape, generator=generator
                        ).to(module.weight.device)
                        * 0.05
                    )
    return model


def spanned_input(cfg, length=T):
    """tokens with three hypertokens of spans 3, 2 and 4, plus their updates."""
    V = cfg.vocab_size
    torch.manual_seed(11)
    toks = torch.randint(1, V, (1, length), device=DEV)
    toks[0, 3] = V + 0
    toks[0, 7] = V + 1
    toks[0, 15] = V + 2
    upd = torch.full((1, len(ENTRIES), cfg.max_subtokens), cfg.pad_token_id, device=DEV)
    for i, entry in enumerate(ENTRIES):
        upd[0, i, : len(entry)] = torch.tensor(entry, device=DEV)
    return toks, upd, [list(range(len(ENTRIES)))]


def empty_updates(cfg):
    return torch.empty((1, 0, cfg.max_subtokens), dtype=torch.long, device=DEV)


def logits_of(out):
    return (out[0] if isinstance(out, tuple) else out).float()


def uncached(model, toks, upd=None, idx=None):
    model.reset_inference_cache()
    kw = {} if upd is None else dict(codebook_updates=upd, codebook_updates_indices=idx)
    with torch.no_grad():
        return logits_of(model(toks, **kw))


def cached(model, cfg, toks, upd=None, idx=None, splits=None):
    """Feed `toks` through a cache in chunks, returning the concatenated logits."""
    compressed = upd is not None
    splits = splits or [10] + list(range(11, toks.shape[1] + 1))
    model.reset_inference_cache()
    cache = KVCache(cfg.n_layers, toks.shape[1] + 4)
    outs = []
    start = 0
    with torch.no_grad():
        for stop in splits:
            kw = {}
            if compressed:
                kw = dict(
                    codebook_updates=upd if start == 0 else empty_updates(cfg),
                    codebook_updates_indices=idx if start == 0 else [[]],
                )
            outs.append(
                logits_of(model(toks[:, start:stop], kv_cache=cache, **kw))
            )
            start = stop
    return torch.cat(outs, dim=1), cache


def agree(got, ref, tol=1e-4):
    """Equal where finite, identical -inf mask, identical argmax.

    -inf columns (unused codebook slots) would make a plain diff NaN, and they
    are themselves part of the contract: the cached path must mask exactly the
    same slots.
    """
    if torch.isinf(got).ne(torch.isinf(ref)).any():
        return False, "masked (-inf) columns differ"
    finite = torch.isfinite(ref)
    delta = (got[finite] - ref[finite]).abs().max().item()
    if not torch.equal(got.argmax(-1), ref.argmax(-1)):
        return False, f"argmax differs (max|diff|={delta:.2e})"
    return delta < tol, f"max|diff|={delta:.2e}"


def main():
    results, check = make_harness()

    # ---- K1: plain / base path ----
    cfg = small_cfg(base_token_positions=True)
    model = build(cfg)
    torch.manual_seed(5)
    btoks = torch.randint(1, cfg.vocab_size, (1, T), device=DEV)
    got, _ = cached(model, cfg, btoks)
    check("K1_base_path", *agree(got, uncached(model, btoks)))

    # ---- K2: compressed path, both position regimes ----
    for flag in (False, True):
        cfg = small_cfg(base_token_positions=flag)
        model = build(cfg)
        toks, upd, idx = spanned_input(cfg)
        got, cache = cached(model, cfg, toks, upd, idx)
        check(
            f"K2_compressed_base_positions_{flag}",
            *agree(got, uncached(model, toks, upd, idx)),
        )

    # ---- K3: two-axis RoPE ----
    cfg = small_cfg(base_token_positions=True, two_axis_rope=True)
    model = build(cfg)
    toks, upd, idx = spanned_input(cfg)
    got, _ = cached(model, cfg, toks, upd, idx)
    check("K3_two_axis_rope", *agree(got, uncached(model, toks, upd, idx)))

    # ---- K4: gated compressed RoPE ----
    cfg = small_cfg(
        base_token_positions=True,
        gated_compressed_rope=True,
        gated_rope_start_layer=0,
        gated_rope_start_pair=0,
    )
    model = build(cfg)
    with torch.no_grad():
        # A non-trivial gate: all-zero would reduce to plain base positions and
        # the compressed-axis offset would go untested.
        model.compressed_rope_gate.uniform_(0.1, 0.9)
    toks, upd, idx = spanned_input(cfg)
    got, _ = cached(model, cfg, toks, upd, idx)
    check("K4_gated_compressed_rope", *agree(got, uncached(model, toks, upd, idx)))

    # ---- K5: chunked prefill onto a populated cache ----
    cfg = small_cfg(base_token_positions=True)
    model = build(cfg)
    toks, upd, idx = spanned_input(cfg)
    got, _ = cached(model, cfg, toks, upd, idx, splits=[8, 14, 19, T])
    check("K5_chunked_prefill", *agree(got, uncached(model, toks, upd, idx)))

    # ---- K6: base-space offset advances by span ----
    cfg = small_cfg(base_token_positions=True)
    model = build(cfg)
    toks, upd, idx = spanned_input(cfg)
    _, cache = cached(model, cfg, toks, upd, idx)
    model.reset_inference_cache()
    with torch.no_grad():
        model(toks, codebook_updates=upd, codebook_updates_indices=idx)
        want = model._base_token_positions(toks, codebook=None)[:, -1:] + 1
    # 21 single-span tokens + spans 3 + 2 + 4 = 30
    check(
        "K6_offset_counts_spans",
        cache.base_offset is not None
        and torch.equal(cache.base_offset, want)
        and int(want) == 30,
        f"carried {None if cache.base_offset is None else cache.base_offset.tolist()}, "
        f"uncached cumsum says {want.tolist()}",
    )

    # ---- K7: a decode step sees the whole prefix ----
    # is_causal=True on a (q_len=1, k_len=N) SDPA call is top-left aligned: the
    # single query would attend to key 0 only. That returns fluent-looking
    # garbage rather than an error, so pin it by construction — perturbing a
    # key in the middle of the prefix MUST move the next-token logits.
    cfg = small_cfg(base_token_positions=True)
    model = build(cfg)
    torch.manual_seed(5)
    btoks = torch.randint(1, cfg.vocab_size, (1, T), device=DEV)
    cache_a = KVCache(cfg.n_layers, T + 4)
    with torch.no_grad():
        model(btoks[:, :T], kv_cache=cache_a)
        step_a = logits_of(model(btoks[:, T - 1 : T], kv_cache=cache_a))
    altered = btoks.clone()
    altered[0, T // 2] = (int(altered[0, T // 2]) + 17) % cfg.vocab_size
    cache_b = KVCache(cfg.n_layers, T + 4)
    with torch.no_grad():
        model(altered[:, :T], kv_cache=cache_b)
        step_b = logits_of(model(btoks[:, T - 1 : T], kv_cache=cache_b))
    check(
        "K7_decode_attends_full_prefix",
        not torch.allclose(step_a, step_b, atol=1e-5),
        "changing a mid-prefix token must move the decode-step logits "
        f"(max|diff|={(step_a - step_b).abs().max().item():.2e})",
    )

    # ---- K8: LongRoPE regime flip is refused, and re-prefill recovers ----
    from transformers import Phi3Config

    head_dim = 256 // 8
    hf_config = Phi3Config(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=T * MS * 2,
        original_max_position_embeddings=8,
        rope_theta=10_000.0,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0 + 0.1 * i for i in range(head_dim // 2)],
            "long_factor": [2.0 + 0.5 * i for i in range(head_dim // 2)],
        },
    )
    base_cfg = small_cfg(base_token_positions=True)
    cfg = dataclasses.replace(
        base_cfg, rope=phi3_longrope_config(base_cfg.rope, hf_config)
    )
    model = build(cfg)
    torch.manual_seed(5)
    btoks = torch.randint(1, cfg.vocab_size, (1, T), device=DEV)
    cache = KVCache(cfg.n_layers, T + 4)
    raised = False
    with torch.no_grad():
        model(btoks[:, :6], kv_cache=cache)  # max position 5 -> short factors
        try:
            model(btoks[:, 6:12], kv_cache=cache)  # crosses 8 -> long factors
        except RopeRegimeChanged:
            raised = True
    check(
        "K8a_longrope_flip_refused",
        raised and cache.rope_regime == "short",
        f"regime pinned at {cache.rope_regime!r}, raised={raised}",
    )
    # Documented recovery: drop the cache, replay the prefix.
    cache.reset()
    with torch.no_grad():
        replay = logits_of(model(btoks[:, :12], kv_cache=cache))
        ref = logits_of(model(btoks[:, :12]))
    check("K8b_reprefill_recovers", *agree(replay, ref))

    # ---- K9: two-axis mixing turning on mid-stream is refused ----
    cfg = small_cfg(base_token_positions=True, two_axis_rope=True)
    model = build(cfg)
    V = cfg.vocab_size
    torch.manual_seed(5)
    hyper_late = torch.randint(1, V, (1, T), device=DEV)
    hyper_late[0, T - 1] = V + 0  # first hypertoken arrives after prefill
    _, upd, idx = spanned_input(cfg)
    model.reset_inference_cache()
    cache = KVCache(cfg.n_layers, T + 4)
    raised = False
    with torch.no_grad():
        model(
            hyper_late[:, : T - 1],
            codebook_updates=upd,
            codebook_updates_indices=idx,
            kv_cache=cache,
        )
        try:
            model(
                hyper_late[:, T - 1 :],
                codebook_updates=empty_updates(cfg),
                codebook_updates_indices=[[]],
                kv_cache=cache,
            )
        except RopeRegimeChanged:
            raised = True
    check(
        "K9_two_axis_flip_refused",
        raised and cache.two_axis_mixing is False,
        f"latched mixing={cache.two_axis_mixing}, raised={raised}",
    )

    # ---- K10: the adapter's generate loops, cache on vs off ----
    check("K10_adapter_generation", *adapter_generation_matches())

    # ---- K11: cache bookkeeping ----
    cache = KVCache(2, 4)
    k = torch.randn(1, 2, 3, 8)
    cache.update(0, k, k)
    cache.update(1, k, k)
    cache.advance(3, None)
    k2 = torch.randn(1, 2, 3, 8)
    grown_k, _ = cache.update(0, k2, k2)
    preserved = torch.equal(grown_k[:, :, :3], k) and torch.equal(
        grown_k[:, :, 3:6], k2
    )
    check(
        "K11a_growth_preserves_history",
        preserved and cache.capacity >= 6,
        f"capacity now {cache.capacity}, history preserved={preserved}",
    )
    cache = KVCache(1, 8)
    cache.update(0, k, k)
    cache.advance(3, None)
    rejects = []
    for bad in (k.to(torch.float16), torch.randn(2, 2, 1, 8)):
        try:
            cache.update(0, bad, bad)
            rejects.append(False)
        except ValueError:
            rejects.append(True)
    check(
        "K11b_rejects_dtype_and_batch_change",
        all(rejects),
        f"dtype/batch guards fired: {rejects}",
    )

    return not summarize(results)


class _IdTokenizer:
    """decode() is injective on id lists, so comparing text compares ids."""

    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return "|".join(str(int(i)) for i in ids)


def _make_adapter(model, cfg, use_kv_cache, eval_mode):
    from zip2zip_core.lm_eval_adapter import Zip2ZipLM

    lm = object.__new__(Zip2ZipLM)
    lm.model = model
    lm.cfg = cfg
    lm._device = torch.device(DEV)
    lm._dtype = torch.float32
    lm._max_length = 4096
    lm.eval_mode = eval_mode
    lm.tokenizer = _IdTokenizer()
    lm._stop_token_ids = set()
    lm.fail_on_truncation = False
    lm.trim_stop_strings = False
    lm.preserve_leading_space = False
    lm._decode_prefix_ids = []
    lm._decode_prefix_text = ""
    lm.use_kv_cache = use_kv_cache
    lm.compression_stats = {
        "in_comp": 0,
        "in_base": 0,
        "gen_comp": 0,
        "gen_base": 0,
        "kv_cache_reprefills": 0,
    }
    lm._disabled_ids = []
    lm._compressor_kwargs = dict(
        initial_vocab_size=cfg.vocab_size,
        max_codebook_size=cfg.max_codebook_size,
        max_subtokens=cfg.max_subtokens,
        pad_token_id=cfg.pad_token_id,
        disabled_ids=[],
    )
    return lm


def adapter_generation_matches():
    """Run the real generate loops both ways and compare the returned text."""
    cfg = small_cfg(base_token_positions=True, max_codebook_size=64)
    model = build(cfg)
    torch.manual_seed(5)
    # A repetitive prompt so LZW actually builds a codebook and the compressed
    # loop exercises hypertoken expansion rather than degenerating to base ids.
    prompt = ([7, 8, 9, 10, 11, 12] * 12)[:64]
    notes = []
    for mode, method in (
        ("base", "_generate_base"),
        ("compressed", "_generate_compressed"),
    ):
        texts = []
        for use_cache in (False, True):
            lm = _make_adapter(model, cfg, use_cache, mode)
            texts.append(
                getattr(lm, method)(list(prompt), [], 24, False, 0.0, 1.0, 0)
            )
        if texts[0] != texts[1]:
            return False, f"{mode}: {texts[0][:60]!r} != {texts[1][:60]!r}"
        notes.append(f"{mode}={len(texts[1].split('|'))} toks")
    return True, "identical greedy output (" + ", ".join(notes) + ")"


def test_kv_cache_invariants():
    assert main()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
