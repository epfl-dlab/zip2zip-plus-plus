# - [ ] nvidia/Nemotron-Science-v1 (only the content)
# - [ ] nvidia/Nemotron-Instruction-Following-Chat-v1 (reasoning off)
# - [ ] nvidia/Nemotron-Math-v2 (low subset)
# - [ ] nvidia/OpenCodeReasoning-2

import json
import os
import time

import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

OUTPUT_DIR = "/capstor/scratch/cscs/nathanrchn/zip2zip-data/sft_dataset"
TOKENS_PER_SHARD = 100_000_000
TOKENIZE_BATCH_SIZE = 10_000
TOKENIZE_NUM_PROC = 8
SHUFFLE_SEED = 42
ADD_BOS = True
ADD_EOS = True
ASSISTANT_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"

tokenizer = AutoTokenizer.from_pretrained("nathanrchn/zip2zip-tokenizer")


def assistant_mask_from_offsets(text, offsets):
    """Build a token-level assistant loss mask from formatted ChatML text."""
    spans = []
    search_pos = 0
    while True:
        header_start = text.find(ASSISTANT_HEADER, search_pos)
        if header_start == -1:
            break

        content_start = header_start + len(ASSISTANT_HEADER)
        content_end = text.find(IM_END, content_start)
        if content_end == -1:
            content_end = len(text)
        else:
            content_end += len(IM_END)
            if content_end < len(text) and text[content_end] == "\n":
                content_end += 1

        spans.append((content_start, content_end))
        search_pos = content_end

    mask = []
    for token_start, token_end in offsets:
        if token_start == token_end:
            mask.append(0)
            continue
        mask.append(int(any(token_start < span_end and token_end > span_start for span_start, span_end in spans)))
    return mask


def maybe_add_boundary_tokens(tokens, mask):
    if ADD_BOS and tokenizer.bos_token_id is not None and (not tokens or tokens[0] != tokenizer.bos_token_id):
        tokens = [tokenizer.bos_token_id] + tokens
        mask = [0] + mask
    if ADD_EOS and tokenizer.eos_token_id is not None and (not tokens or tokens[-1] != tokenizer.eos_token_id):
        tokens = tokens + [tokenizer.eos_token_id]
        mask = mask + [0]
    return tokens, mask


def tokenize_sft_dataset(dataset):
    if not tokenizer.is_fast:
        raise ValueError("SFT mask creation requires a fast tokenizer with offset mappings")

    dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    dataset = dataset.filter(
        lambda sample: sample["text"] is not None and len(sample["text"]) > 0,
        num_proc=TOKENIZE_NUM_PROC,
    )

    def tokenize(batch):
        encoding = tokenizer(
            batch["text"],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        all_tokens = []
        all_masks = []
        for text, tokens, offsets in zip(
            batch["text"],
            encoding["input_ids"],
            encoding["offset_mapping"],
        ):
            mask = assistant_mask_from_offsets(text, offsets)
            tokens, mask = maybe_add_boundary_tokens(list(tokens), mask)
            all_tokens.append(tokens)
            all_masks.append(mask)

        return {"tokens": all_tokens, "mask": all_masks}

    return dataset.map(
        tokenize,
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=TOKENIZE_NUM_PROC,
        remove_columns=dataset.column_names,
    )


def save_token_shards(tokenized_dataset):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_arrays = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy"))
    if existing_arrays:
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains token or mask shards. "
            f"Remove them before regenerating the SFT dataset: {existing_arrays[:3]}"
        )

    shard_idx = 0
    token_buffer = []
    mask_buffer = []
    total_tokens = 0
    docs_processed = 0
    start_time = time.time()

    for sample in tokenized_dataset:
        if len(sample["tokens"]) != len(sample["mask"]):
            raise ValueError("Token and mask lengths are misaligned")

        token_buffer.extend(sample["tokens"])
        mask_buffer.extend(sample["mask"])
        docs_processed += 1

        while len(token_buffer) >= TOKENS_PER_SHARD:
            shard_tokens = np.array(token_buffer[:TOKENS_PER_SHARD], dtype=np.uint32)
            shard_mask = np.array(mask_buffer[:TOKENS_PER_SHARD], dtype=np.uint8)
            shard_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}.npy")
            mask_path = os.path.join(OUTPUT_DIR, f"mask_{shard_idx:05d}.npy")
            np.save(shard_path, shard_tokens)
            np.save(mask_path, shard_mask)

            total_tokens += TOKENS_PER_SHARD
            token_buffer = token_buffer[TOKENS_PER_SHARD:]
            mask_buffer = mask_buffer[TOKENS_PER_SHARD:]
            elapsed = time.time() - start_time
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            print(
                f"Saved shard {shard_idx}: {shard_path} "
                f"({total_tokens/1e9:.2f}B tokens, {docs_processed} docs, {tok_per_sec/1e6:.2f}M tok/s)"
            )
            shard_idx += 1

    if token_buffer:
        shard_tokens = np.array(token_buffer, dtype=np.uint32)
        shard_mask = np.array(mask_buffer, dtype=np.uint8)
        shard_path = os.path.join(OUTPUT_DIR, f"shard_{shard_idx:05d}.npy")
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{shard_idx:05d}.npy")
        np.save(shard_path, shard_tokens)
        np.save(mask_path, shard_mask)
        total_tokens += len(token_buffer)
        print(f"Saved final shard {shard_idx}: {shard_path} ({total_tokens/1e9:.2f}B tokens)")
        shard_idx += 1

    elapsed = time.time() - start_time
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
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

    print(f"\nDone! Total: {total_tokens/1e9:.2f}B tokens in {shard_idx} shards")
    print(f"Documents processed: {docs_processed}")
    print(f"Elapsed: {elapsed:.1f}s, Average: {tok_per_sec/1e6:.2f}M tok/s")

nemotron_science = load_dataset("nvidia/Nemotron-Science-v1", split="MCQ")

def to_text_nemotron_science(sample):
    return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

nemotron_science_text = nemotron_science.map(
    to_text_nemotron_science,
    batched=True,
    batch_size=1000,
    num_proc=8,
    remove_columns=nemotron_science.column_names,
)

print(nemotron_science_text)

nemotron_instruction_following_chat = load_dataset("nvidia/Nemotron-Instruction-Following-Chat-v1", split="chat_if")

def to_text_nemotron_instruction_following_chat(sample):
    return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

nemotron_instruction_following_chat_text = nemotron_instruction_following_chat.map(
    to_text_nemotron_instruction_following_chat,
    batched=True,
    batch_size=1000,
    num_proc=8,
    remove_columns=nemotron_instruction_following_chat.column_names,
)

print(nemotron_instruction_following_chat_text)

nemotron_math = load_dataset("nvidia/Nemotron-Math-v2", split="low")

def to_text_nemotron_math(sample):
    for conversation in sample["messages"]:
        for message in conversation:
            for tool_call in message.get("tool_calls") or []:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    tool_call["function"]["arguments"] = json.loads(args)
    return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

nemotron_math_text = nemotron_math.map(
    to_text_nemotron_math,
    batched=True,
    batch_size=1000,
    num_proc=8,
    remove_columns=nemotron_math.column_names,
)

print(nemotron_math_text)

hf_datasets = {
    "taco": load_dataset("ReactiveAI/BAAI-TACO-reupload"),
    "apps": load_dataset("ReactiveAI/codeparrot-apps-reupload"),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces"),
}


def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        if benchmark["input_format"]:
            question += "\n\nInput\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += "\n\nOutput\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                if "input" in example:
                    question += "\n\nInput\n\n" + example["input"]
                if "output" in example:
                    question += "\n\nOutput\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None


ocr2_dataset = load_dataset("nvidia/OpenCodeReasoning-2")

def strip_thinking(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()

def to_text_open_code_reasoning(sample):
    texts = []
    for ds_name, ds_split, ds_index, blank_question, r1_generation in zip(
        sample["dataset"],
        sample["split"],
        sample["index"],
        sample["question"],
        sample["r1_generation"],
    ):
        assert ds_name in ["taco", "apps", "code_contests", "open-r1/codeforces"]
        question = get_question(ds_name, ds_split, int(ds_index))
        assert question is not None
        assert blank_question == "-"
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": strip_thinking(r1_generation)},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

open_code_reasoning_text = concatenate_datasets([
    ocr2_dataset["python"].map(
        to_text_open_code_reasoning,
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=ocr2_dataset["python"].column_names,
    ),
    ocr2_dataset["cpp"].map(
        to_text_open_code_reasoning,
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=ocr2_dataset["cpp"].column_names,
    ),
])

print(open_code_reasoning_text)

dataset = concatenate_datasets([nemotron_science_text, nemotron_instruction_following_chat_text, nemotron_math_text, open_code_reasoning_text])

print(dataset)

tokenized_dataset = tokenize_sft_dataset(dataset)
print(tokenized_dataset)

save_token_shards(tokenized_dataset)
