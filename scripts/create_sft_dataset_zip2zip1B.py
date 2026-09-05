"""Process epfl-dlab/zip2zip-1B for SFT training — Llama-3.2 tokenizer variant.

Same pipeline as create_sft_dataset_zip2zip1B_phi.py at the eosfix stage
(commit 15d7968), tokenized with the official meta-llama/Llama-3.2-1B-Instruct
tokenizer and its chat template. Deliberately does NOT include the NuminaMath
mathchat re-rendering: the production Phi dataset behind the v0.4+ lineage is
phi-1B-sft-8shards-eosfix (eosfix only), and the Llama dataset must differ from
it on no axis other than the tokenizer. The one formatting difference that the
tokenizer brings with it: the Llama chat template always prepends its dated
system block (~20 tokens, loss mask 0) to chat documents; Phi's template has no
such block.

Dataset has 'text' column with mixed formats:
- HuggingFaceH4/ultrachat_200k: Zephyr chat format (<|user|>/<|end|>/<|assistant|>)
- Other sources (NuminaMath included): plain text

Processing:
- Chat examples (ultrachat): parse Zephyr → re-apply apply_chat_template (Llama format)
  Loss mask = 1 only on assistant turns (including <|eot_id|>)
- Plain text: tokenize as-is, loss mask = 1 on all tokens
"""

import json
import os
import re
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/capstor/store/cscs/swissai/a0101/mxx/zip2zip-data/llama32-1B-sft-8shards")
# data.py hands shard files out round-robin per DP rank (shard_files[rank::world_size])
# and a rank that runs out of tokens wraps into a second epoch silently, so the
# shard count must be a multiple of the training world size and the shards
# equal-sized. 125M gives the Phi set 8 full shards; this corpus tokenizes to
# ~846M Llama tokens, so the launcher passes ~105.8M for 8 shards / 4 GPUs.
TOKENS_PER_SHARD = int(os.environ.get("TOKENS_PER_SHARD", 125_000_000))
TOKENIZE_BATCH_SIZE = 5_000
TOKENIZE_NUM_PROC = 8
SHUFFLE_SEED = 42
ADD_BOS = True
ADD_EOS = True

# Official tokenizer for both base tokenization and chat templating (the Phi
# variant does the same). The earlier version of this script tokenized with
# nathanrchn/zip2zip-tokenizer relying on its vocab being identical to
# Llama 3.2's; using the official tokenizer end-to-end removes that assumption
# and matches what the eval side (TOKENIZER=meta-llama/Llama-3.2-1B-Instruct)
# will use. Gated repo: the job needs HF credentials or a warm cache.
CHAT_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"
BASE_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Pin the date the Llama chat template renders into its default system block,
# so the emitted text (and therefore every token id and mask) is independent
# of the day the script runs. "26 Jul 2024" is the template's own fallback.
CHAT_TEMPLATE_DATE = "26 Jul 2024"

# Llama 3.2 format markers (these are real single special tokens):
#   <|start_header_id|> (128006) / <|end_header_id|> (128007) / <|eot_id|> (128009)
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
TURN_END = "<|eot_id|>"

# Sources that have chat format in Zephyr style
CHAT_SOURCES = {"HuggingFaceH4/ultrachat_200k"}

user_cache = os.environ.get("USER_HF_CACHE", None)
tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME, cache_dir=user_cache)
chat_tokenizer = AutoTokenizer.from_pretrained(CHAT_TOKENIZER_NAME, cache_dir=user_cache)

# Document separator appended after each doc. The instruct tokenizer's
# eos_token is <|eot_id|> (turn end), but the Phi dataset's equivalent role
# (<|endoftext|> 32000) is the document-end token, which for Llama 3 is
# <|end_of_text|> (128001) — generation_config stops on it too. Resolve it
# explicitly instead of trusting tokenizer.eos_token_id.
DOC_EOS_TOKEN = "<|end_of_text|>"
DOC_EOS_ID = tokenizer.convert_tokens_to_ids(DOC_EOS_TOKEN)
if DOC_EOS_ID is None or DOC_EOS_ID == tokenizer.unk_token_id:
    raise ValueError(f"{DOC_EOS_TOKEN} not found in {BASE_TOKENIZER_NAME} vocab")


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
                # Use Llama 3.2 tokenizer → proper <|eot_id|> at each turn end
                llama_fmt = chat_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    date_string=CHAT_TEMPLATE_DATE,
                )
                return {'text': llama_fmt, 'is_chat': True}
        except Exception:
            pass

    return {'text': text, 'is_chat': False}


def assistant_mask_from_offsets(text, offsets, is_chat):
    """Build token-level loss mask.

    For chat text (Llama format): 1 only on assistant turn content + <|eot_id|>.
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
            content_end += len(TURN_END)  # include <|eot_id|> in loss
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
    if ADD_EOS:
        if not tokens or tokens[-1] != DOC_EOS_ID:
            tokens = tokens + [DOC_EOS_ID]
            # EOS must be IN the loss (mask=1): with mask=0 the model gets no
            # gradient to ever emit end-of-text after a completed plain-text
            # document, so at inference it runs past its answer into a
            # fabricated next document (observed on GSM8K: the step_8000 repro
            # answers correctly, then generates a new invented math problem
            # whose numbers poison lm-eval's flexible-extract scoring).
            # Chat docs already learn stopping via <|eot_id|> in the assistant
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

    print("Converting Zephyr chat → Llama 3.2 ChatML, plain text passthrough...")
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


if __name__ == "__main__":
    main()
