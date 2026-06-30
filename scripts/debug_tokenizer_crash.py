"""Find which Pile document crashes the Phi-3.5 fast tokenizer.

The eval-phi35-ppl-v2 job crashed at ~30,355/309,461 in loglikelihood_rolling.
Pile is processed first (214,670 docs), so the bad document is around index 30,355.

Run on the cluster:
    python scripts/debug_tokenizer_crash.py
"""
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
tok = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

print(f"Dataset size: {len(ds)}")
print(f"Tokenizer: {type(tok).__name__}, is_fast={tok.is_fast}")
print(f"Scanning all documents...\n")

failures = []
for i in range(len(ds)):
    text = ds[i]["text"]
    if not isinstance(text, str):
        print(f"[{i}] NOT A STRING: type={type(text)}")
        failures.append((i, "not_string", type(text)))
        continue
    try:
        tok.encode(text)
    except Exception as e:
        print(f"[{i}] CRASH: {type(e).__name__}: {e}")
        print(f"  len(text)={len(text)}, has_null={'chr(0)' in repr(text[:1000])}")
        print(f"  first 200 chars: {repr(text[:200])}")
        failures.append((i, str(e), len(text)))

print(f"\nDone. {len(failures)} failures out of {len(ds)} documents.")
for idx, err, info in failures:
    print(f"  doc {idx}: {err} ({info})")
