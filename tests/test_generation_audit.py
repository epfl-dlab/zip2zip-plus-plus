from types import SimpleNamespace
import pytest
import transformers
import zip2zip_core.lm_eval_adapter as adapter
from test_shared_windows_and_pareto import _MockPhiTokenizer, _save_tiny_flat_checkpoint, DEV

@pytest.mark.parametrize('ms,cb',[(1,16),(2,4),(4,16),(4,0)])
def test_audit_preserves_generation_and_accounts_for_expansion(tmp_path,monkeypatch,ms,cb):
 tok=_MockPhiTokenizer(128)
 monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',classmethod(lambda cls,*a,**k:tok))
 key,cfg=_save_tiny_flat_checkpoint(tmp_path/'ckpt')
 adapter.zip2zip_llama_configs[key]=cfg
 try:
  lm=adapter.Zip2ZipLM(pretrained=str(tmp_path/'ckpt'),tokenizer='mock/phi-test',device=DEV,dtype='float32',max_length=128,eval_max_subtokens=ms,eval_max_codebook_size=cb)
  lm.model.hyper_encoder.disable_varlen=True
  if getattr(lm.model,'hyper_output',None) is not None:lm.model.hyper_output.disable_varlen=True
  req=SimpleNamespace(args=('abababababababababababababababab',{'until':[],'max_gen_toks':8,'do_sample':False}),task_name='gsm8k',doc_id=17)
  before=lm.generate_until([req]); comp0=lm.compression_stats.copy()
  assert not lm.generation_audit_records
  lm.audit_generation=True
  after=lm.generate_until([req]);r=lm.generation_audit_records[0]
  assert before==after
  assert r['task']=='gsm8k' and r['doc_id']==17 and r['response']==after[0]
  assert r['generated_base_tokens']==len(r['generated_base_ids'])==sum(r['generated_token_spans'])
  assert r['generated_model_tokens']==len(r['generated_token_spans'])
  assert r['generated_model_tokens']==lm.compression_stats['gen_comp']-comp0['gen_comp']
  assert r['generated_base_tokens']==lm.compression_stats['gen_base']-comp0['gen_base']
  assert 0<r['prompt_compressed_tokens']<=r['prompt_base_tokens']
  assert r['prompt_original_base_tokens']-r['prompt_base_tokens']==r['prompt_tokens_dropped']
 finally:adapter.zip2zip_llama_configs.pop(key,None)
