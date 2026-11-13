"""Likelihood behavior for codebook availability and continuation boundaries."""
from collections import defaultdict
from types import SimpleNamespace
import math
import pytest
import torch
import zip2zip_core.lm_eval_adapter as adapter
from test_online_codebook_mask import small_model, codec_args


def make_lm(online):
    model, cfg = small_model()
    model.eval()
    lm = object.__new__(adapter.Zip2ZipLM)
    lm.cfg, lm.model = cfg, model
    lm._device, lm._dtype = torch.device('cpu'), torch.bfloat16
    lm._max_length = 64
    lm._compressor_kwargs = dict(codec_args(disabled_ids=[0]), max_codebook_size=cfg.max_codebook_size)
    lm.hyper_causal_mask = True
    lm.online_codebook_mask = lm.online_codebook_mask_active = online
    lm.multi_view = False
    lm.eval_mode = 'compressed'
    lm.tokenizer = SimpleNamespace(eos_token_id=0, bos_token_id=0, eos_token='</s>')
    lm.tok_encode = lambda text: [1, 2, 1, 2, 1, 2, 1]
    lm.compression_stats = defaultdict(int)
    return lm


def test_online_rolling_ppl_rejects_unavailable_canonical_target():
    lm = make_lm(True)
    with pytest.raises(RuntimeError, match='target is unavailable'):
        lm.loglikelihood_rolling([SimpleNamespace(args=('kwkwk',), task_name='wikitext')])


def test_legacy_rolling_ppl_keeps_its_original_scoring_protocol():
    lm = make_lm(False)
    result = lm.loglikelihood_rolling([SimpleNamespace(args=('kwkwk',), task_name='wikitext')])
    assert len(result) == 1 and math.isfinite(result[0]) and result[0] < 0
    assert lm.compression_stats['online_skipped_targets'] == 0


def test_cross_boundary_continuation_requires_explicit_override():
    lm = make_lm(False)
    lm.tok_encode = lambda text: [1, 2, 1] if text == 'ctx' else [1, 2, 1, 2]
    with pytest.raises(RuntimeError, match='crosses the context/continuation boundary'):
        lm.loglikelihood([SimpleNamespace(args=('ctx', 'cont'))])


def test_explicit_legacy_boundary_reproduction_records_omitted_tokens():
    lm = make_lm(False)
    lm.allow_legacy_boundary_merges = True
    lm.tok_encode = lambda text: [1, 2, 1] if text == 'ctx' else [1, 2, 1, 2]
    assert lm.loglikelihood([SimpleNamespace(args=('ctx', 'cont'))])[0][0] == 0.0
    assert lm.compression_stats['boundary_omitted_base_tokens'] == 1
    assert lm.compression_stats['boundary_crossing_requests'] == 1


def test_aligned_hypertoken_continuation_remains_scored():
    lm = make_lm(False)
    lm.tok_encode = lambda text: [1, 2] if text == 'ctx' else [1, 2, 1, 2]
    score = lm.loglikelihood([SimpleNamespace(args=('ctx', 'cont'))])[0][0]
    assert math.isfinite(score) and score < 0
    assert lm.compression_stats['boundary_omitted_base_tokens'] == 0
