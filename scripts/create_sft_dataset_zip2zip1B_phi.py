"""Process epfl-dlab/zip2zip-1B for SFT training — Phi-3.5-mini tokenizer variant.

Same pipeline as create_sft_dataset_zip2zip1B.py but tokenized with the
Phi-3.5-mini-instruct tokenizer (vocab 32064) and its chat template, so it
matches a from-scratch Phi3.5-mini model trained on Phi-tokenized data.

Dataset has 'text' column with mixed formats:
- HuggingFaceH4/ultrachat_200k: Zephyr chat format (<|user|>/<|end|>/<|assistant|>)
- AI-MO/NuminaMath-1.5: packed "problem\n\nsolution" pairs joined by single "\n",
  no chat tags, ~1-5 problems per doc
- Other sources: plain text

Processing:
- Chat examples (ultrachat): parse Zephyr → re-apply apply_chat_template (Phi-3.5 format)
  Loss mask = 1 only on assistant turns (including the trailing <|end|>)
- Math (NuminaMath): split at ground-truth problem boundaries (exact match against
  the original AI-MO/NuminaMath-1.5 problems) → multi-turn Phi-3.5 chat, same loss
  masking as ultrachat. Docs that cannot be fully segmented stay plain text.
- Plain text: tokenize as-is, loss mask = 1 on all tokens
"""

import json
import os
import re
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/phi-1B-sft-8shards")
TOKENS_PER_SHARD = 125_000_000
TOKENIZE_BATCH_SIZE = 5_000
TOKENIZE_NUM_PROC = 8
SHUFFLE_SEED = 42
ADD_BOS = True
ADD_EOS = True

# Phi-3.5-mini tokenizer provides its own chat template, so base == chat.
CHAT_TOKENIZER_NAME = "microsoft/Phi-3.5-mini-instruct"
BASE_TOKENIZER_NAME = "microsoft/Phi-3.5-mini-instruct"

# Phi-3.5 chat format markers (real single special tokens):
#   <|assistant|> (32001) / <|end|> (32007) / <|user|> (32010)
# Assistant turns render as: "<|assistant|>\n{content}<|end|>\n"
ASSISTANT_HEADER = "<|assistant|>\n"
TURN_END = "<|end|>"

# Sources that have chat format in Zephyr style
CHAT_SOURCES = {"HuggingFaceH4/ultrachat_200k"}

# Math source: packed plain-text "problem\n\nsolution" pairs, re-rendered as chat.
# Revision pinned so the problem index always matches the packed docs.
MATH_SOURCE = "AI-MO/NuminaMath-1.5"
MATH_SOURCE_REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
MATH_PROBLEM_PREFIX = 80

user_cache = os.environ.get("USER_HF_CACHE", None)
tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME, cache_dir=user_cache)
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_TOKENIZER_NAME, cache_dir=user_cache)


def parse_zephyr_to_messages(text):
    """Parse <|user|>...<|end|>\n<|assistant|>...<|end|> text into messages list."""
    messages = []
    # Split on role markers, keeping delimiters
    parts = re.split(r'(<\|user\|>|<\|assistant\|>)', text)
    current_role = None
    for part in parts:
        if part == '<|user|>':
            current_role = 'user'
        elif part == '<|assistant|>':
            current_role = 'assistant'
        elif current_role is not None:
            content = part.strip()
            if content.endswith('<|end|>'):
                content = content[:-len('<|end|>')].strip()
            if content:
                messages.append({'role': current_role, 'content': content})
            current_role = None
    return messages


math_problem_index = None


def init_math_problem_index():
    """Build {80-char prefix -> [full problem texts]} from AI-MO/NuminaMath-1.5.

    Packed math docs are "problem\n\nsolution" pairs joined by a single "\n".
    Both separators also occur inside solutions, so the only reliable split
    points are exact matches of the original problem texts (verified 2026-07-15
    on 1000 sampled docs: 92.2% segment fully, 0 false boundaries in a strict
    solution-level check, 0.6% end in a truncated last solution).
    """
    global math_problem_index
    if math_problem_index is not None:
        return math_problem_index
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        f for f in HfApi().list_repo_files(
            MATH_SOURCE, repo_type="dataset", revision=MATH_SOURCE_REVISION
        )
        if f.startswith("data/") and f.endswith(".parquet")
    )
    index = {}
    n_problems = 0
    for rf in files:
        local = hf_hub_download(
            MATH_SOURCE, rf, repo_type="dataset",
            revision=MATH_SOURCE_REVISION, cache_dir=user_cache,
        )
        for p in pq.read_table(local, columns=["problem"])["problem"].to_pylist():
            if not p or not p.strip():
                continue
            n_problems += 1
            index.setdefault(p[:MATH_PROBLEM_PREFIX], []).append(p)
    for key, cands in index.items():
        if len(cands) > 1:
            index[key] = list(dict.fromkeys(cands))
    print(f"Math problem index: {n_problems:,} problems, {len(index):,} prefixes")
    math_problem_index = index
    return index


def split_math_doc(text):
    """Split a packed NuminaMath doc into [(problem, solution), ...].

    A boundary is position 0, or a position right after "\n" where a known
    problem text matches exactly and is followed by "\n\n". Solutions are taken
    verbatim from the doc (packed solutions may differ from the originals).
    Returns None when the doc cannot be fully segmented from position 0.
    """
    index = math_problem_index if math_problem_index is not None else init_math_problem_index()
    starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    bounds = []  # (position, matched problem)
    prev_end = 0
    for pos in starts:
        if pos < prev_end:  # inside the previous problem's own text
            continue
        cands = index.get(text[pos:pos + MATH_PROBLEM_PREFIX])
        if not cands:
            continue
        best = None
        for p in cands:
            if text.startswith(p, pos) and text.startswith("\n\n", pos + len(p)):
                if best is None or len(p) > len(best):
                    best = p
        if best is None:
            continue
        bounds.append((pos, best))
        prev_end = pos + len(best) + 2
    if not bounds or bounds[0][0] != 0:
        return None
    pairs = []
    for j, (pos, problem) in enumerate(bounds):
        sol_start = pos + len(problem) + 2
        sol_end = bounds[j + 1][0] - 1 if j + 1 < len(bounds) else len(text)
        problem = problem.strip()
        solution = text[sol_start:sol_end].strip()
        if not problem or not solution:
            return None
        pairs.append((problem, solution))
    return pairs


def convert_to_chatml(sample):
    """Convert text to ChatML format. Returns {'text': ..., 'is_chat': bool}."""
    source = sample.get('source', '')
    text = sample['text']

    if source in CHAT_SOURCES and '<|user|>' in text and '<|assistant|>' in text:
        try:
            messages = parse_zephyr_to_messages(text)
            if messages and any(m['role'] == 'assistant' for m in messages):
                # Use Phi-3.5 tokenizer → proper <|end|> at each turn end
                phi_fmt = chat_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                return {'text': phi_fmt, 'is_chat': True}
        except Exception:
            pass

    if source == MATH_SOURCE:
        pairs = split_math_doc(text)
        if pairs:
            messages = []
            for problem, solution in pairs:
                messages.append({'role': 'user', 'content': problem})
                messages.append({'role': 'assistant', 'content': solution})
            phi_fmt = chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {'text': phi_fmt, 'is_chat': True}

    return {'text': text, 'is_chat': False}


def assistant_mask_from_offsets(text, offsets, is_chat):
    """Build token-level loss mask.

    For chat text (Phi format): 1 only on assistant turn content + <|end|>.
    For plain text: 1 on all tokens.
    """
    if not is_chat:
        return [1 if (s != e) else 0 for s, e in offsets]

    spans = []
    search_pos = 0
    while True:
        header_start = text.find(ASSISTANT_HEADER, search_pos)
        if header_start == -1:
            break
        content_start = header_start + len(ASSISTANT_HEADER)
        content_end = text.find(TURN_END, content_start)
        if content_end == -1:
            content_end = len(text)
        else:
            content_end += len(TURN_END)  # include <|end|> in loss
        spans.append((content_start, content_end))
        search_pos = content_end

    if not spans:
        # Fallback: mask everything if no assistant turn found
        return [1 if (s != e) else 0 for s, e in offsets]

    mask = []
    for token_start, token_end in offsets:
        if token_start == token_end:
            mask.append(0)
            continue
        mask.append(int(any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in spans
        )))
    return mask


def maybe_add_boundary_tokens(tokens, mask):
    if ADD_BOS and tokenizer.bos_token_id is not None:
        if not tokens or tokens[0] != tokenizer.bos_token_id:
            tokens = [tokenizer.bos_token_id] + tokens
            mask = [0] + mask
    if ADD_EOS and tokenizer.eos_token_id is not None:
        if not tokens or tokens[-1] != tokenizer.eos_token_id:
            tokens = tokens + [tokenizer.eos_token_id]
            # EOS must be IN the loss (mask=1): with mask=0 the model gets no
            # gradient to ever emit end-of-text after a completed plain-text
            # document, so at inference it runs past its answer into a
            # fabricated next document (observed on GSM8K: the step_8000 repro
            # answers correctly, then generates a new invented math problem
            # whose numbers poison lm-eval's flexible-extract scoring).
            # Chat docs already learn stopping via <|end|> in the assistant
            # span; this fixes plain-text (NuminaMath/fineweb/stack) docs.
            mask = mask + [1]
    return tokens, mask


def tokenize_dataset(dataset):
    if not tokenizer.is_fast:
        raise ValueError("Requires a fast tokenizer with offset mappings")

    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    dataset = dataset.filter(
        lambda s: s['text'] is not None and len(s['text']) > 0,
        num_proc=TOKENIZE_NUM_PROC,
    )

    def tokenize(batch):
        encoding = tokenizer(
            batch['text'],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        all_tokens = []
        all_masks = []
        for text, is_chat, tokens, offsets in zip(
            batch['text'],
            batch['is_chat'],
            encoding['input_ids'],
            encoding['offset_mapping'],
        ):
            mask = assistant_mask_from_offsets(text, offsets, is_chat)
            tokens, mask = maybe_add_boundary_tokens(list(tokens), mask)
            all_tokens.append(tokens)
            all_masks.append(mask)
        return {'tokens': all_tokens, 'mask': all_masks}

    return dataset.map(
        tokenize,
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=TOKENIZE_NUM_PROC,
        remove_columns=dataset.column_names,
    )


def save_token_shards(tokenized_dataset):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy'))
    if existing:
        raise FileExistsError(
            f"{OUTPUT_DIR} already has .npy files. Remove them first: {existing[:3]}"
        )

    shard_idx = 0
    token_buffer = []
    mask_buffer = []
    total_tokens = 0
    docs_processed = 0
    start_time = time.time()

    for sample in tokenized_dataset:
        if len(sample['tokens']) != len(sample['mask']):
            raise ValueError("Token/mask length mismatch")
        token_buffer.extend(sample['tokens'])
        mask_buffer.extend(sample['mask'])
        docs_processed += 1

        while len(token_buffer) >= TOKENS_PER_SHARD:
            shard_tokens = np.array(token_buffer[:TOKENS_PER_SHARD], dtype=np.uint32)
            shard_mask = np.array(mask_buffer[:TOKENS_PER_SHARD], dtype=np.uint8)
            np.save(os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}.npy"), shard_tokens)
            np.save(os.path.join(OUTPUT_DIR, f"mask_{shard_idx:05d}.npy"), shard_mask)

            total_tokens += TOKENS_PER_SHARD
            token_buffer = token_buffer[TOKENS_PER_SHARD:]
            mask_buffer = mask_buffer[TOKENS_PER_SHARD:]
            elapsed = time.time() - start_time
            print(
                f"Shard {shard_idx}: {total_tokens/1e9:.2f}B tokens, "
                f"{docs_processed} docs, {total_tokens/elapsed/1e6:.2f}M tok/s"
            )
            shard_idx += 1

    if token_buffer:
        np.save(os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}.npy"),
                np.array(token_buffer, dtype=np.uint32))
        np.save(os.path.join(OUTPUT_DIR, f"mask_{shard_idx:05d}.npy"),
                np.array(mask_buffer, dtype=np.uint8))
        total_tokens += len(token_buffer)
        print(f"Final shard {shard_idx}: {total_tokens/1e9:.2f}B tokens")
        shard_idx += 1

    elapsed = time.time() - start_time
    manifest = {
        "total_tokens": total_tokens,
        "documents_processed": docs_processed,
        "num_shards": shard_idx,
        "tokens_per_shard": TOKENS_PER_SHARD,
        "has_loss_masks": True,
        "mask_dtype": "uint8",
        "token_dtype": "uint32",
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone! {total_tokens/1e9:.2f}B tokens in {shard_idx} shards, {elapsed:.1f}s")
    print(f"Documents: {docs_processed}")


# ---- main ----

def main():
    print("Loading epfl-dlab/zip2zip-1B (train split)...")
    ds = load_dataset("epfl-dlab/zip2zip-1B", split="train")
    print(ds)

    # Check chat stats
    n_chat = sum(1 for s in ds['source'] if s in CHAT_SOURCES)
    print(f"Chat examples (ultrachat): {n_chat} / {len(ds)}")

    # Built before ds.map so forked workers inherit it (workers on spawn
    # platforms rebuild it lazily from the shared HF cache).
    print("Building NuminaMath problem index for math doc splitting...")
    init_math_problem_index()

    print("Converting chat + math → Phi-3.5 ChatML, plain text passthrough...")
    ds = ds.map(
        convert_to_chatml,
        batched=False,
        num_proc=TOKENIZE_NUM_PROC,
        remove_columns=[c for c in ds.column_names if c not in ('text', 'is_chat', 'source')],
    )

    n_math = sum(1 for s in ds['source'] if s == MATH_SOURCE)
    n_math_chat = sum(
        1 for s, c in zip(ds['source'], ds['is_chat']) if s == MATH_SOURCE and c
    )
    rate = n_math_chat / max(n_math, 1)
    print(f"Math docs chat-formatted: {n_math_chat:,} / {n_math:,} ({100*rate:.1f}%) "
          f"— ~92% expected, rest stay plain text")
    if rate < 0.85:
        raise RuntimeError(
            f"Math chat-format rate {rate:.3f} < 0.85 — problem index missing or "
            f"revision mismatch; refusing to tokenize"
        )

    print("Tokenizing...")
    tokenized = tokenize_dataset(ds)
    print(tokenized)

    print("Saving shards...")
    save_token_shards(tokenized)


if __name__ == "__main__":
    main()
