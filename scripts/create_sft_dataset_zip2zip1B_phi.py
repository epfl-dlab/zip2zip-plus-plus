"""Process epfl-dlab/zip2zip-1B for SFT training — Phi-3.5-mini tokenizer variant.

Same pipeline as create_sft_dataset_zip2zip1B.py but tokenized with the
Phi-3.5-mini-instruct tokenizer (vocab 32064) and its chat template, so it
matches a from-scratch Phi3.5-mini model trained on Phi-tokenized data.

Dataset has 'text' column with mixed formats:
- HuggingFaceH4/ultrachat_200k: Zephyr chat format (<|user|>/<|end|>/<|assistant|>)
- Other sources: plain text

Processing:
- Chat examples (ultrachat): parse Zephyr → re-apply apply_chat_template (Phi-3.5 format)
  Loss mask = 1 only on assistant turns (including the trailing <|end|>)
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
            mask = mask + [0]
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

print("Loading epfl-dlab/zip2zip-1B (train split)...")
ds = load_dataset("epfl-dlab/zip2zip-1B", split="train")
print(ds)

# Check chat stats
n_chat = sum(1 for s in ds['source'] if s in CHAT_SOURCES)
print(f"Chat examples (ultrachat): {n_chat} / {len(ds)}")

print("Converting Zephyr chat → Phi-3.5 ChatML, plain text passthrough...")
ds = ds.map(
    convert_to_chatml,
    batched=False,
    num_proc=TOKENIZE_NUM_PROC,
    remove_columns=[c for c in ds.column_names if c not in ('text', 'is_chat', 'source')],
)

print("Tokenizing...")
tokenized = tokenize_dataset(ds)
print(tokenized)

print("Saving shards...")
save_token_shards(tokenized)
