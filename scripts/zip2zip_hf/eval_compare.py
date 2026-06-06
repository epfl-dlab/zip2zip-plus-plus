"""Generate and evaluate zip2zip HF models on local Spec-Bench prompts.

Examples:
  python scripts/zip2zip_hf/eval_compare.py generate \
      --repo epfl-dlab/candidate-Llaza-MS3-flat-20BT-v1 \
      --question-file /home/xinxian/Semester_Project/question.jsonl

  python scripts/zip2zip_hf/eval_compare.py aggregate \
      --run-dir outputs/eval_compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from zip2zip.model import Zip2ZipModel
from zip2zip.tokenizer import Zip2ZipTokenizer
from zip2zip_compression import LZWCompressor


DEFAULT_REPOS = [
    # "epfl-dlab/candidate-Llaza-MS2-FT-1BT-v1",
    # "epfl-dlab/candidate-Llaza-MS3-FT-1BT-v1",
    # "epfl-dlab/candidate-Llaza-MS4-FT-1BT-v1",
    "epfl-dlab/candidate-Llaza-MS2-FT-1BT-base-v1",
    "epfl-dlab/candidate-Llaza-MS3-FT-1BT-base-v1",
    "epfl-dlab/candidate-Llaza-MS4-FT-1BT-base-v1",
    "epfl-dlab/candidate-Llaza-MS2-flat-20BT-v1",
    "epfl-dlab/candidate-Llaza-MS3-flat-20BT-v1",
    "epfl-dlab/candidate-Llaza-MS4-flat-20BT-v1",
]

METRIC_FIELDS = [
    "actual_bytes_per_zip_token",
    "theory_bytes_per_zip_token",
    "compression_efficiency",
    "base_token_saving",
    "gpt2_mean_nll",
    "gpt2_bits_per_byte",
    "empty_or_too_short",
    "repeat_4gram_rate",
    "max_4gram_count",
    "degenerate_repetition",
]


def repo_to_slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo).strip("_")


def infer_model_label(repo: str) -> str:
    slug = repo.split("/")[-1]
    label = slug.replace("candidate-", "").replace("-v1", "").replace("-", " ")
    return re.sub(r"\s+", " ", label).strip()


def infer_model_meta(repo: str) -> dict[str, str]:
    slug = repo.split("/")[-1]
    ms_match = re.search(r"MS(\d+)", slug, flags=re.IGNORECASE)
    if re.search(r"(^|[-_])FT([-_]|$)", slug, flags=re.IGNORECASE):
        model_group = "ft"
    elif re.search(r"flat|scratch", slug, flags=re.IGNORECASE):
        model_group = "scratch"
    else:
        model_group = "unknown"
    return {
        "model": repo,
        "model_slug": repo_to_slug(repo),
        "model_group": model_group,
        "ms": f"MS{ms_match.group(1)}" if ms_match else "unknown",
    }



def load_questions(question_file: Path, limit: int | None, start: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with question_file.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if line_index < start:
                continue
            if limit is not None and len(rows) >= limit:
                break
            obj = json.loads(line)
            turns = obj.get("turns") or []
            if not turns:
                continue
            rows.append(
                {
                    "question_id": obj.get("question_id", line_index),
                    "category": obj.get("category", "unknown"),
                    "prompt": str(turns[0]),
                }
            )
    return rows


def make_lzw(tokenizer: Zip2ZipTokenizer) -> LZWCompressor:
    return LZWCompressor(
        initial_vocab_size=int(tokenizer.initial_vocab_size),
        max_codebook_size=int(tokenizer.max_codebook_size),
        max_subtokens=int(tokenizer.max_subtokens),
        pad_token_id=tokenizer.pad_token_id,
        disabled_ids=list(tokenizer.disabled_ids or []),
    )


def lzw_token_count(token_ids: list[int], tokenizer: Zip2ZipTokenizer) -> int:
    if not token_ids:
        return 0
    encoded, _, _ = make_lzw(tokenizer).encode([int(token_id) for token_id in token_ids])
    return len(encoded)


def theory_zip_token_count_from_base_ids(
    prompt_base_ids: list[int],
    continuation_base_ids: list[int],
    zip_tokenizer: Zip2ZipTokenizer,
) -> int:
    if not continuation_base_ids:
        return 0
    full_count = lzw_token_count(prompt_base_ids + continuation_base_ids, zip_tokenizer)
    prompt_count = lzw_token_count(prompt_base_ids, zip_tokenizer)
    return max(0, full_count - prompt_count)


def base_token_count_from_ids(
    continuation_base_ids: list[int],
    zip_tokenizer: Zip2ZipTokenizer,
) -> int:
    specials = special_token_ids(zip_tokenizer)
    return sum(1 for token_id in continuation_base_ids if int(token_id) not in specials)


def special_token_ids(tokenizer: Zip2ZipTokenizer) -> set[int]:
    ids = set()
    for token_id in [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.unk_token_id,
    ]:
        if token_id is not None:
            ids.add(int(token_id))
    try:
        ids.update(int(token_id) for token_id in tokenizer.get_added_vocab().values())
    except Exception:
        pass
    return ids


def filter_special_ids(token_ids: list[int], tokenizer: Zip2ZipTokenizer) -> list[int]:
    specials = special_token_ids(tokenizer)
    return [int(token_id) for token_id in token_ids if int(token_id) not in specials]


def actual_new_token_count(new_token_ids: list[int], tokenizer: Zip2ZipTokenizer) -> int:
    return len(filter_special_ids(new_token_ids, tokenizer))


def trim_response(prompt: str, decoded_text: str) -> str:
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt) :]
    prompt_stripped = prompt.strip()
    decoded_stripped = decoded_text.strip()
    if decoded_stripped.startswith(prompt_stripped):
        return decoded_stripped[len(prompt_stripped) :]
    return decoded_text


def decode_continuation_from_lzw_output(
    *,
    output_ids: list[int],
    prompt_zip_ids: list[int],
    prompt: str,
    decoded_text: str,
    tokenizer: Zip2ZipTokenizer,
) -> tuple[str, list[int], list[int], int]:
    """Decode only the generated continuation from a full LZW output sequence.

    String-prefix trimming is not reliable because decoded text can normalize spaces
    or repeat parts of the prompt.  Instead, decode full/prompt LZW ids to base
    tokenizer ids and remove the prompt base-token prefix.
    """
    pad_id = tokenizer.pad_token_id
    output_no_pad = list(output_ids)
    while output_no_pad and pad_id is not None and int(output_no_pad[0]) == int(pad_id):
        output_no_pad.pop(0)

    full_base_ids, _ = tokenizer._lzw_decode([output_no_pad])[0]
    prompt_base_ids, _ = tokenizer._lzw_decode([prompt_zip_ids])[0]

    if full_base_ids[: len(prompt_base_ids)] == prompt_base_ids:
        continuation_base_ids = filter_special_ids(
            full_base_ids[len(prompt_base_ids) :],
            tokenizer,
        )
        response = tokenizer.old_decode(
            continuation_base_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return response, prompt_base_ids, continuation_base_ids, 1

    response = trim_response(prompt, decoded_text)
    return response, prompt_base_ids, [], 0


def fourgram_counts(text: str) -> Counter[tuple[str, ...]]:
    words = re.findall(r"\S+", text)
    if len(words) < 4:
        return Counter()
    return Counter(tuple(words[index : index + 4]) for index in range(len(words) - 3))


def repetition_metrics(
    *,
    response: str,
    valid_output: bool,
    rate_threshold: float,
    count_threshold: int,
) -> dict[str, float | int]:
    if not valid_output:
        return {
            "repeat_4gram_rate": float("nan"),
            "max_4gram_count": float("nan"),
            "degenerate_repetition": float("nan"),
        }

    counts = fourgram_counts(response)
    total_4grams = sum(counts.values())
    if total_4grams == 0:
        return {
            "repeat_4gram_rate": float("nan"),
            "max_4gram_count": float("nan"),
            "degenerate_repetition": float("nan"),
        }

    repeated = sum(count - 1 for count in counts.values() if count > 1)
    repeat_rate = repeated / total_4grams
    max_count = max(counts.values())
    degenerate = int(repeat_rate >= rate_threshold or max_count >= count_threshold)
    return {
        "repeat_4gram_rate": repeat_rate,
        "max_4gram_count": max_count,
        "degenerate_repetition": degenerate,
    }


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def compute_compression_metrics(
    *,
    response: str,
    new_token_ids: list[int],
    prompt_base_ids: list[int],
    continuation_base_ids: list[int],
    zip_tokenizer: Zip2ZipTokenizer,
    too_short_bytes: int,
    repetition_rate_threshold: float,
    repetition_count_threshold: int,
) -> dict[str, float | int]:
    response_bytes = len(response.encode("utf-8"))
    actual_tokens = actual_new_token_count(new_token_ids, zip_tokenizer)
    theory_tokens = theory_zip_token_count_from_base_ids(
        prompt_base_ids,
        continuation_base_ids,
        zip_tokenizer,
    )
    base_tokens = base_token_count_from_ids(continuation_base_ids, zip_tokenizer)
    empty_or_too_short = int(response_bytes < too_short_bytes)
    valid_output = int(
        not empty_or_too_short
        and actual_tokens > 0
        and theory_tokens > 0
        and base_tokens > 0
    )

    if valid_output:
        actual_bpt = safe_div(response_bytes, actual_tokens)
        theory_bpt = safe_div(response_bytes, theory_tokens)
        compression_efficiency = safe_div(actual_bpt, theory_bpt)
        base_token_saving = 1.0 - safe_div(actual_tokens, base_tokens)
    else:
        actual_bpt = float("nan")
        theory_bpt = float("nan")
        compression_efficiency = float("nan")
        base_token_saving = float("nan")

    repeat_metrics = repetition_metrics(
        response=response,
        valid_output=bool(valid_output),
        rate_threshold=repetition_rate_threshold,
        count_threshold=repetition_count_threshold,
    )

    return {
        "response_bytes": response_bytes,
        "actual_new_zip_tokens": actual_tokens,
        "actual_new_zip_tokens_raw": len(new_token_ids),
        "theory_new_zip_tokens": theory_tokens,
        "base_new_tokens": base_tokens,
        "valid_output": valid_output,
        "actual_bytes_per_zip_token": actual_bpt,
        "theory_bytes_per_zip_token": theory_bpt,
        "compression_efficiency": compression_efficiency,
        "base_token_saving": base_token_saving,
        "empty_or_too_short": empty_or_too_short,
        **repeat_metrics,
    }


class GPT2Scorer:
    def __init__(self, model_name: str, device: str, stride: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.max_length = int(getattr(self.model.config, "n_positions", 1024))
        self.stride = max(1, min(int(stride), self.max_length))

    def score(self, prompt: str, response: str) -> dict[str, float | int]:
        response_bytes = len(response.encode("utf-8"))
        if not response:
            return {
                "gpt2_nll_sum": float("nan"),
                "gpt2_mean_nll": float("nan"),
                "gpt2_bits_per_byte": float("nan"),
                "gpt2_scored_tokens": 0,
                "gpt2_scored_bytes": 0,
            }

        full_text = prompt + response
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0].tolist()
        sequence_length = int(input_ids.numel())
        prompt_chars = len(prompt)
        continuation_mask = torch.tensor(
            [end > prompt_chars for _start, end in offsets],
            dtype=torch.bool,
        )
        if sequence_length <= 1:
            return {
                "gpt2_nll_sum": float("nan"),
                "gpt2_mean_nll": float("nan"),
                "gpt2_bits_per_byte": float("nan"),
                "gpt2_scored_tokens": 0,
                "gpt2_scored_bytes": 0,
            }

        nll_sum = 0.0
        scored_positions: set[int] = set()
        previous_end = 0

        for begin in range(0, sequence_length - 1, self.stride):
            end = min(begin + self.max_length, sequence_length)
            input_slice = input_ids[begin:end].to(self.device)
            labels = input_slice.clone()
            global_positions = torch.arange(begin, end)
            score_mask = continuation_mask[begin:end].clone()
            score_mask[0] = False
            score_mask &= global_positions >= previous_end
            labels[~score_mask.to(self.device)] = -100

            if int((labels[1:] != -100).sum().item()) > 0:
                with torch.no_grad():
                    logits = self.model(input_ids=input_slice.unsqueeze(0)).logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[1:].unsqueeze(0).contiguous()
                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                valid = shift_labels.view(-1) != -100
                nll_sum += float(token_losses[valid].sum().item())
                valid_positions = global_positions[1:][valid.cpu()].tolist()
                scored_positions.update(int(position) for position in valid_positions)

            previous_end = end
            if end >= sequence_length:
                break

        effective_tokens = len(scored_positions)
        if effective_tokens == 0:
            return {
                "gpt2_nll_sum": float("nan"),
                "gpt2_mean_nll": float("nan"),
                "gpt2_bits_per_byte": float("nan"),
                "gpt2_scored_tokens": 0,
                "gpt2_scored_bytes": 0,
            }

        scored_start = min(max(offsets[position][0], prompt_chars) for position in scored_positions)
        scored_end = max(offsets[position][1] for position in scored_positions)
        scored_bytes = len(full_text[scored_start:scored_end].encode("utf-8"))
        mean_nll = safe_div(nll_sum, effective_tokens)
        return {
            "gpt2_nll_sum": nll_sum,
            "gpt2_mean_nll": mean_nll,
            "gpt2_bits_per_byte": safe_div(nll_sum, math.log(2) * scored_bytes),
            "gpt2_scored_tokens": effective_tokens,
            "gpt2_scored_bytes": scored_bytes,
        }


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_arg]


def set_generation_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_generation_model(args):
    dtype = resolve_dtype(args.torch_dtype)
    resolved_base_model = resolve_base_model_name(args.repo, args.revision, args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        dtype=dtype,
    )
    model = Zip2ZipModel.from_pretrained(
        args.repo,
        revision=args.revision,
        base_model=base_model,
        dtype=dtype,
        max_codebook_size=args.max_codebook_size,
    ).to(args.device).eval()
    tokenizer = Zip2ZipTokenizer.from_pretrained(
        args.repo,
        revision=args.revision,
        max_codebook_size=args.max_codebook_size,
    )
    tokenizer.padding_side = "left"
    tokenizer.tokenizer.padding_side = "left"
    return model, tokenizer, resolved_base_model


def get_generation_stop_token_ids(tokenizer: Zip2ZipTokenizer) -> int | list[int]:
    stop_ids: list[int] = []
    for token_id in (getattr(tokenizer, "eos_token_id", None), 128001):
        if token_id is None:
            continue
        token_int = int(token_id)
        if token_int not in stop_ids:
            stop_ids.append(token_int)
    if not stop_ids:
        raise ValueError("No stop token ids available for generation")
    if len(stop_ids) == 1:
        return stop_ids[0]
    return stop_ids


def build_generation_kwargs(args, tokenizer: Zip2ZipTokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": bool(args.do_sample),
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": get_generation_stop_token_ids(tokenizer),
    }
    if args.do_sample:
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            }
        )
    if args.repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = args.repetition_penalty
    return kwargs


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: sanitize_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    return value


def dumps_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        sanitize_json_value(value),
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
    )


def write_jsonl_row(file, row: dict[str, Any]) -> None:
    file.write(dumps_json(row) + "\n")
    file.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_base_tokenizer(zip_tokenizer: Zip2ZipTokenizer):
    tokenizer = getattr(zip_tokenizer, "hf_tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    tokenizer = getattr(zip_tokenizer, "tokenizer", zip_tokenizer)
    return getattr(tokenizer, "tokenizer", tokenizer)


def get_tokenizer_name(zip_tokenizer: Zip2ZipTokenizer, run_config: dict[str, Any] | None = None) -> str:
    config = getattr(zip_tokenizer, "zip2zip_config", None)
    config_name = getattr(config, "base_model_name_or_path", None)
    if config_name:
        return str(config_name)
    if run_config is not None and run_config.get("base_model"):
        return str(run_config["base_model"])
    base_tokenizer = get_base_tokenizer(zip_tokenizer)
    return getattr(base_tokenizer, "name_or_path", "unknown")


class DemoZipTokenizer:
    def __init__(
        self,
        *,
        base_tokenizer,
        tokenizer_name: str,
        initial_vocab_size: int,
        max_codebook_size: int,
        max_subtokens: int,
        disabled_ids: list[int],
    ) -> None:
        self.tokenizer = base_tokenizer
        self.hf_bpe_tokenizer = base_tokenizer
        self.initial_vocab_size = int(initial_vocab_size)
        self.max_codebook_size = int(max_codebook_size)
        self.max_subtokens = int(max_subtokens)
        self.disabled_ids = [int(token_id) for token_id in disabled_ids]
        self.pad_token_id = base_tokenizer.pad_token_id
        self.eos_token_id = base_tokenizer.eos_token_id
        self.bos_token_id = base_tokenizer.bos_token_id
        self.unk_token_id = base_tokenizer.unk_token_id
        self.name_or_path = tokenizer_name
        self.compressor = LZWCompressor(
            initial_vocab_size=self.initial_vocab_size,
            max_codebook_size=self.max_codebook_size,
            max_subtokens=self.max_subtokens,
            pad_token_id=self.pad_token_id,
            disabled_ids=self.disabled_ids,
        )

    def get_added_vocab(self):
        return self.tokenizer.get_added_vocab()

    def _lzw_decode(self, token_ids: list[list[int]]):
        return self.compressor.batch_decode(token_ids)


def load_zip2zip_config_dict(repo: str, revision: str, *, local_files_only: bool = True) -> dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=repo,
        filename="zip2zip_config.json",
        revision=revision,
        local_files_only=local_files_only,
    )
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def resolve_base_model_name(repo: str, revision: str, cli_base_model: str | None) -> str:
    if cli_base_model:
        return str(cli_base_model)
    zip_config = load_zip2zip_config_dict(repo, revision, local_files_only=False)
    return str(zip_config["base_model_name_or_path"])


def load_export_tokenizer(run_config: dict[str, Any]):
    zip_config = load_zip2zip_config_dict(
        str(run_config["repo"]),
        str(run_config.get("revision", "hf")),
    )
    tokenizer_name = str(
        zip_config.get("base_model_name_or_path")
        or run_config.get("base_model")
        or "unknown"
    )
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    zip_tokenizer = DemoZipTokenizer(
        base_tokenizer=base_tokenizer,
        tokenizer_name=tokenizer_name,
        initial_vocab_size=int(zip_config["compression"]["initial_vocab_size"]),
        max_codebook_size=int(zip_config["compression"]["max_codebook_size"]),
        max_subtokens=int(zip_config["compression"]["max_subtokens"]),
        disabled_ids=[int(token_id) for token_id in zip_config["compression"]["disabled_ids"]],
    )
    return zip_tokenizer, zip_config


def extract_tokenizer_metadata(
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
    *,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_tokenizer = get_base_tokenizer(zip_tokenizer)
    special_ids = sorted(special_token_ids(zip_tokenizer))
    zip_token_start = int(zip_tokenizer.initial_vocab_size)
    zip_vocab_size = int(zip_tokenizer.max_codebook_size)
    return {
        "name": get_tokenizer_name(zip_tokenizer, run_config),
        "base_vocab_size": int(zip_tokenizer.initial_vocab_size),
        "zip_token_start": zip_token_start,
        "zip_vocab_size": zip_vocab_size,
        "pad_token_id": (
            None if base_tokenizer.pad_token_id is None else int(base_tokenizer.pad_token_id)
        ),
        "eos_token_id": (
            None if base_tokenizer.eos_token_id is None else int(base_tokenizer.eos_token_id)
        ),
        "special_token_ids": special_ids,
    }


def decode_base_token_text(base_tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return base_tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def classify_token_kind(
    token_id: int,
    *,
    tokenizer_meta: dict[str, Any],
    special_ids: set[int],
) -> str:
    if token_id in special_ids:
        if tokenizer_meta["pad_token_id"] is not None and token_id == tokenizer_meta["pad_token_id"]:
            return "pad"
        return "special"
    if token_id >= int(tokenizer_meta["zip_token_start"]):
        return "zip"
    return "base"


def build_token_entries(
    token_ids: list[int],
    *,
    codebook_map: dict[int, list[int]],
    tokenizer_meta: dict[str, Any],
    base_tokenizer,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    special_ids = set(int(token_id) for token_id in tokenizer_meta["special_token_ids"])
    source_token_cursor = 0
    for index, token_id in enumerate(token_ids):
        kind = classify_token_kind(
            int(token_id),
            tokenizer_meta=tokenizer_meta,
            special_ids=special_ids,
        )
        if int(token_id) in codebook_map:
            expansion = [int(value) for value in codebook_map[int(token_id)]]
        elif kind in {"special", "pad", "base"}:
            expansion = [int(token_id)]
        else:
            raise ValueError(f"Missing codebook expansion for zip token {token_id}")
        span_len = len(expansion)
        entries.append(
            {
                "index": index,
                "id": int(token_id),
                "text": decode_base_token_text(base_tokenizer, expansion),
                "kind": kind,
                "source_token_start": source_token_cursor,
                "source_token_end": source_token_cursor + span_len,
                "expands_to_token_ids": expansion,
            }
        )
        source_token_cursor += span_len
    return entries


def build_codebook_entries(
    codebook_map: dict[int, list[int]],
    *,
    base_tokenizer,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for token_id in sorted(int(key) for key in codebook_map):
        base_token_ids = [int(value) for value in codebook_map[token_id]]
        entries.append(
            {
                "id": token_id,
                "base_token_ids": base_token_ids,
                "base_token_texts": [
                    decode_base_token_text(base_tokenizer, [base_token_id])
                    for base_token_id in base_token_ids
                ],
                "decoded_text": decode_base_token_text(base_tokenizer, base_token_ids),
                "length": len(base_token_ids),
            }
        )
    return entries


def build_original_view(
    response: str,
    *,
    base_tokenizer,
) -> dict[str, Any]:
    token_ids = [
        int(token_id)
        for token_id in base_tokenizer.encode(response, add_special_tokens=False)
    ]
    decoded_text = decode_base_token_text(base_tokenizer, token_ids)
    return {
        "token_ids": token_ids,
        "tokens": [
            {
                "index": index,
                "id": token_id,
                "text": decode_base_token_text(base_tokenizer, [token_id]),
            }
            for index, token_id in enumerate(token_ids)
        ],
        "decoded_text": decoded_text,
    }


def trim_leading_pad(token_ids: list[int], pad_token_id: int | None) -> list[int]:
    trimmed = list(token_ids)
    while trimmed and pad_token_id is not None and int(trimmed[0]) == int(pad_token_id):
        trimmed.pop(0)
    return trimmed


def codebook_to_map(codebook) -> dict[int, list[int]]:
    return {
        int(token_id): [int(value) for value in values]
        for token_id, values in codebook.to_dict().items()
    }


def decode_full_sequence(
    full_zip_token_ids: list[int],
    *,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
) -> tuple[list[int], dict[int, list[int]]]:
    trimmed_ids = trim_leading_pad(full_zip_token_ids, zip_tokenizer.pad_token_id)
    if not trimmed_ids:
        return [], {}
    base_token_ids, codebook = zip_tokenizer._lzw_decode([trimmed_ids])[0]
    return [int(token_id) for token_id in base_token_ids], codebook_to_map(codebook)


def split_continuation_from_full_decode(
    full_base_token_ids: list[int],
    prompt_base_ids: list[int],
    *,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
) -> tuple[list[int], int]:
    if full_base_token_ids[: len(prompt_base_ids)] != prompt_base_ids:
        return [], 0
    continuation = filter_special_ids(full_base_token_ids[len(prompt_base_ids) :], zip_tokenizer)
    return continuation, 1


def build_optimal_view(
    prompt_base_ids: list[int],
    continuation_base_ids: list[int],
    *,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
    tokenizer_meta: dict[str, Any],
    base_tokenizer,
) -> dict[str, Any]:
    full_base_ids = [int(token_id) for token_id in [*prompt_base_ids, *continuation_base_ids]]
    full_compressed_ids, _, _ = make_lzw(zip_tokenizer).encode(full_base_ids)
    prompt_compressed_ids, _, _ = make_lzw(zip_tokenizer).encode(prompt_base_ids)
    compressed_token_ids = [int(token_id) for token_id in full_compressed_ids[len(prompt_compressed_ids) :]]
    full_reconstructed_ids, codebook_map = decode_full_sequence(
        [int(token_id) for token_id in full_compressed_ids],
        zip_tokenizer=zip_tokenizer,
    )
    reconstructed_token_ids, _ = split_continuation_from_full_decode(
        full_reconstructed_ids,
        [int(token_id) for token_id in prompt_base_ids],
        zip_tokenizer=zip_tokenizer,
    )
    return {
        "compressed_token_ids": compressed_token_ids,
        "tokens": build_token_entries(
            compressed_token_ids,
            codebook_map=codebook_map,
            tokenizer_meta=tokenizer_meta,
            base_tokenizer=base_tokenizer,
        ),
        "codebook": build_codebook_entries(codebook_map, base_tokenizer=base_tokenizer),
        "reconstructed_token_ids": [int(token_id) for token_id in reconstructed_token_ids],
        "reconstructed_text": decode_base_token_text(base_tokenizer, reconstructed_token_ids),
    }


def build_model_result_view(
    raw_generated_token_ids: list[int],
    compressed_token_ids: list[int],
    *,
    prompt_zip_ids: list[int],
    prompt_base_ids: list[int],
    response_text: str,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
    tokenizer_meta: dict[str, Any],
    base_tokenizer,
) -> dict[str, Any]:
    full_output_ids = [int(token_id) for token_id in [*prompt_zip_ids, *raw_generated_token_ids]]
    full_reconstructed_ids, codebook_map = decode_full_sequence(
        full_output_ids,
        zip_tokenizer=zip_tokenizer,
    )
    reconstructed_token_ids, prefix_match = split_continuation_from_full_decode(
        full_reconstructed_ids,
        [int(token_id) for token_id in prompt_base_ids],
        zip_tokenizer=zip_tokenizer,
    )
    reconstructed_text = (
        decode_base_token_text(base_tokenizer, reconstructed_token_ids)
        if prefix_match
        else response_text
    )
    return {
        "raw_generated_token_ids": [int(token_id) for token_id in raw_generated_token_ids],
        "compressed_token_ids": [int(token_id) for token_id in compressed_token_ids],
        "tokens": build_token_entries(
            compressed_token_ids,
            codebook_map=codebook_map,
            tokenizer_meta=tokenizer_meta,
            base_tokenizer=base_tokenizer,
        ),
        "codebook": build_codebook_entries(codebook_map, base_tokenizer=base_tokenizer),
        "reconstructed_token_ids": [int(token_id) for token_id in reconstructed_token_ids],
        "reconstructed_text": reconstructed_text,
    }


def parse_metric_value(value: str) -> int | float | str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def load_metrics_by_question_id(metrics_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows[str(row["question_id"])] = {
                key: parse_metric_value(value)
                for key, value in row.items()
            }
    return rows


def load_run_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))


def synthesize_trace_rows(
    model_dir: Path,
    *,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
    tokenizer_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    generations_path = model_dir / "generations.jsonl"
    rows = load_jsonl(generations_path)
    base_tokenizer = get_base_tokenizer(zip_tokenizer)
    traces: list[dict[str, Any]] = []
    for row in rows:
        new_token_ids = [int(token_id) for token_id in row.get("new_token_ids", [])]
        traces.append(
            {
                "model": row.get("model"),
                "model_slug": row.get("model_slug"),
                "model_group": row.get("model_group"),
                "ms": row.get("ms"),
                "question_id": row["question_id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "response": row["response"],
                "decoded_text": row.get("decoded_text", row["prompt"] + row["response"]),
                "tokenizer": tokenizer_meta,
                "raw_generated_token_ids": new_token_ids,
                "compressed_token_ids": filter_special_ids(new_token_ids, zip_tokenizer),
                "prompt_zip_ids": [],
                "prompt_base_ids": [
                    int(token_id)
                    for token_id in base_tokenizer.encode(row["prompt"], add_special_tokens=False)
                ],
                "continuation_base_ids": [
                    int(token_id)
                    for token_id in base_tokenizer.encode(row["response"], add_special_tokens=False)
                ],
                "lzw_prefix_match": 1,
            }
        )
    return traces


def load_trace_rows(
    model_dir: Path,
    *,
    zip_tokenizer: Zip2ZipTokenizer | DemoZipTokenizer,
    tokenizer_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_path = model_dir / "generation_traces.jsonl"
    if trace_path.exists():
        return load_jsonl(trace_path)
    return synthesize_trace_rows(
        model_dir,
        zip_tokenizer=zip_tokenizer,
        tokenizer_meta=tokenizer_meta,
    )


def metric_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float):
        if math.isnan(value):
            return False
        return value != 0.0
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "nan", "none", "null"}:
            return False
        try:
            number = float(lowered)
        except ValueError:
            return lowered in {"true", "1", "yes"}
        return not math.isnan(number) and number != 0.0
    return bool(value)


def build_status(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "valid_output": metric_flag(metrics.get("valid_output", 0)),
        "empty_or_too_short": metric_flag(metrics.get("empty_or_too_short", 0)),
        "degenerate_repetition": metric_flag(metrics.get("degenerate_repetition", 0)),
    }


def validate_example(
    example: dict[str, Any],
    *,
    tokenizer_meta: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    base_new_tokens = metrics.get("base_new_tokens")
    if isinstance(base_new_tokens, int) and len(example["original"]["token_ids"]) != base_new_tokens:
        issues.append(
            {
                "level": "warning",
                "check": "original_length_matches_metrics",
                "detail": (
                    f"original token count {len(example['original']['token_ids'])} "
                    f"!= metrics.base_new_tokens {base_new_tokens}"
                ),
            }
        )
    zip_token_upper_bound = int(tokenizer_meta["zip_token_start"]) + int(tokenizer_meta["zip_vocab_size"])
    for group_name in ("optimal", "model_result"):
        for token_id in example[group_name]["compressed_token_ids"]:
            if token_id >= int(tokenizer_meta["zip_token_start"]) and token_id >= zip_token_upper_bound:
                issues.append(
                    {
                        "level": "error",
                        "check": "zip_token_within_vocab",
                        "detail": f"{group_name} token id {token_id} exceeds configured zip vocab range",
                    }
                )
                break
    if example["optimal"]["reconstructed_text"] != example["response"]:
        issues.append(
            {
                "level": "warning",
                "check": "optimal_reconstruction_matches_response",
                "detail": "optimal reconstructed text differs from response text",
            }
        )
    if example["model_result"]["reconstructed_text"] != example["response"]:
        issues.append(
            {
                "level": "warning",
                "check": "model_reconstruction_matches_response",
                "detail": "model reconstructed text differs from response text",
            }
        )
    return issues


def export_demo_command(args) -> int:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = run_dir / out_dir
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = sorted(run_dir.glob("*/generation_traces.jsonl"))
    generation_paths = sorted(run_dir.glob("*/generations.jsonl"))
    model_dirs = sorted({path.parent for path in [*trace_paths, *generation_paths]})
    if not model_dirs:
        raise FileNotFoundError(
            f"No generations.jsonl or generation_traces.jsonl files found under {run_dir}."
        )

    manifest_models: list[dict[str, Any]] = []
    manifest_examples: dict[str, dict[str, Any]] = {}
    validation_examples: list[dict[str, Any]] = []

    for model_dir in model_dirs:
        run_config = load_run_config(model_dir)
        metrics_by_question_id = load_metrics_by_question_id(model_dir / "metrics.csv")
        meta = infer_model_meta(str(run_config["repo"]))
        zip_tokenizer, _zip_config = load_export_tokenizer(run_config)
        tokenizer_meta = extract_tokenizer_metadata(zip_tokenizer, run_config=run_config)
        traces = load_trace_rows(
            model_dir,
            zip_tokenizer=zip_tokenizer,
            tokenizer_meta=tokenizer_meta,
        )
        base_tokenizer = get_base_tokenizer(zip_tokenizer)

        model_examples: list[dict[str, Any]] = []
        for trace in traces:
            question_id = str(trace["question_id"])
            metrics = dict(metrics_by_question_id[question_id])
            original = build_original_view(trace["response"], base_tokenizer=base_tokenizer)
            optimal = build_optimal_view(
                [int(token_id) for token_id in trace["prompt_base_ids"]],
                [int(token_id) for token_id in trace["continuation_base_ids"]],
                zip_tokenizer=zip_tokenizer,
                tokenizer_meta=tokenizer_meta,
                base_tokenizer=base_tokenizer,
            )
            model_result = build_model_result_view(
                [int(token_id) for token_id in trace["raw_generated_token_ids"]],
                [int(token_id) for token_id in trace["compressed_token_ids"]],
                prompt_zip_ids=[int(token_id) for token_id in trace.get("prompt_zip_ids", [])],
                prompt_base_ids=[int(token_id) for token_id in trace.get("prompt_base_ids", [])],
                response_text=trace["response"],
                zip_tokenizer=zip_tokenizer,
                tokenizer_meta=tokenizer_meta,
                base_tokenizer=base_tokenizer,
            )
            example = {
                "example_id": question_id,
                "category": trace["category"],
                "prompt": trace["prompt"],
                "response": trace["response"],
                "status": build_status(metrics),
                "original": original,
                "optimal": optimal,
                "model_result": model_result,
                "metrics": metrics,
            }
            issues = validate_example(example, tokenizer_meta=tokenizer_meta, metrics=metrics)
            validation_examples.append(
                {
                    "model_slug": meta["model_slug"],
                    "example_id": question_id,
                    "issues": issues,
                }
            )
            model_examples.append(example)
            manifest_examples.setdefault(
                question_id,
                {
                    "id": question_id,
                    "category": trace["category"],
                    "display_label": f"{question_id} · {trace['category']}",
                    "prompt": trace["prompt"],
                },
            )

        model_payload = {
            "model_slug": meta["model_slug"],
            "model": meta["model"],
            "model_group": meta["model_group"],
            "ms": meta["ms"],
            "tokenizer": tokenizer_meta,
            "examples": model_examples,
        }
        model_out_path = models_dir / f"{meta['model_slug']}.json"
        model_out_path.write_text(
            dumps_json(model_payload, indent=2),
            encoding="utf-8",
        )
        manifest_models.append(
            {
                "slug": meta["model_slug"],
                "label": infer_model_label(meta["model"]),
                "repo": meta["model"],
                "model_group": meta["model_group"],
                "ms": meta["ms"],
                "tokenizer": tokenizer_meta,
                "results_file": f"models/{meta['model_slug']}.json",
            }
        )

    manifest = {
        "schema_version": 1,
        "models": manifest_models,
        "examples": sorted(manifest_examples.values(), key=lambda row: int(row["id"])),
    }
    validation = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "num_models": len(manifest_models),
        "num_examples": len(validation_examples),
        "examples": validation_examples,
    }
    (out_dir / "manifest.json").write_text(
        dumps_json(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "validation.json").write_text(
        dumps_json(validation, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {out_dir / 'manifest.json'}")
    print(f"[saved] {out_dir / 'validation.json'}")
    print(f"[saved] {models_dir}")
    return 0


def generate_command(args) -> int:
    if args.question_file is None:
        raise ValueError("Pass --question-file /path/to/question.jsonl explicitly.")
    args.question_file = Path(args.question_file)

    args.device = resolve_device(args.device)
    gpt2_device = resolve_device(args.gpt2_device)
    set_generation_seed(args.seed)
    questions = load_questions(args.question_file, args.limit, args.start)
    if not questions:
        raise ValueError(f"No questions loaded from {args.question_file}")

    meta = infer_model_meta(args.repo)
    model_out_dir = Path(args.out_dir) / meta["model_slug"]
    model_out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = model_out_dir / "generations.jsonl"
    traces_path = model_out_dir / "generation_traces.jsonl"
    metrics_path = model_out_dir / "metrics.csv"
    config_path = model_out_dir / "run_config.json"

    if not args.overwrite and (generations_path.exists() or traces_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Output already exists in {model_out_dir}. Pass --overwrite to replace it."
        )

    print(f"[load] repo={args.repo} device={args.device}")
    model, zip_tokenizer, resolved_base_model = load_generation_model(args)
    scorer = None if args.skip_gpt2 else GPT2Scorer(args.gpt2_model, gpt2_device, args.gpt2_stride)

    run_config_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }
    run_config = {
        **run_config_args,
        "question_file": str(args.question_file),
        "base_model": resolved_base_model,
        "device": args.device,
        "gpt2_device": gpt2_device,
        **meta,
    }
    config_path.write_text(dumps_json(run_config, indent=2), encoding="utf-8")

    metric_fields = [
        "model",
        "model_slug",
        "model_group",
        "ms",
        "question_id",
        "category",
        "prompt_bytes",
        "response_bytes",
        "actual_new_zip_tokens",
        "actual_new_zip_tokens_raw",
        "theory_new_zip_tokens",
        "base_new_tokens",
        "valid_output",
        "gpt2_nll_sum",
        "gpt2_scored_tokens",
        "gpt2_scored_bytes",
        "lzw_prefix_match",
        *METRIC_FIELDS,
    ]

    generation_kwargs = build_generation_kwargs(args, zip_tokenizer)
    tokenizer_meta = extract_tokenizer_metadata(zip_tokenizer, run_config=run_config)
    with (
        generations_path.open("w", encoding="utf-8") as generation_file,
        traces_path.open("w", encoding="utf-8") as trace_file,
        metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
    ):
        writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        writer.writeheader()

        for batch_start in range(0, len(questions), args.batch_size):
            batch = questions[batch_start : batch_start + args.batch_size]
            if args.instruct:
                prompts = [
                    zip_tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in batch
                ]
            else:
                prompts = [row["prompt"] for row in batch]
            if hasattr(model, "codebook_manager"):
                model.codebook_manager.reset()
            # In instruct mode the chat template already emits <|begin_of_text|>, so letting
            # the tokenizer add special tokens again would produce a double BOS (training used
            # a single BOS). For non-instruct/plain-text prompts we still want the tokenizer to
            # add the single BOS, matching the from-scratch/base training setup.
            inputs = zip_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=not args.instruct,
            ).to(model.device)
            # LZWCompressor always right-pads; flip to left-pad for decoder-only generation
            pad_id = int(zip_tokenizer.pad_token_id)
            _ids, _mask = inputs["input_ids"], inputs["attention_mask"]
            _seq_len = _ids.shape[1]
            for _i in range(_ids.shape[0]):
                _n = int(_mask[_i].sum())
                _ids[_i] = torch.cat([_ids[_i].new_full((_seq_len - _n,), pad_id), _ids[_i][_mask[_i].bool()]])
                _mask[_i] = torch.cat([_mask[_i].new_zeros(_seq_len - _n), _mask[_i].new_ones(_n)])
            prompt_width = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            decoded = zip_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_ids = outputs.detach().cpu().tolist()
            input_ids_cpu = inputs["input_ids"].detach().cpu().tolist()
            attention_mask_cpu = inputs["attention_mask"].detach().cpu().tolist()

            for index, row in enumerate(batch):
                new_token_ids = [int(token_id) for token_id in output_ids[index][prompt_width:]]
                prompt_zip_ids = [
                    int(token_id)
                    for token_id, keep in zip(input_ids_cpu[index], attention_mask_cpu[index])
                    if int(keep) == 1
                ]
                response, prompt_base_ids, continuation_base_ids, lzw_prefix_match = decode_continuation_from_lzw_output(
                    output_ids=[int(token_id) for token_id in output_ids[index]],
                    prompt_zip_ids=prompt_zip_ids,
                    prompt=prompts[index],
                    decoded_text=decoded[index],
                    tokenizer=zip_tokenizer,
                )
                compression = compute_compression_metrics(
                    response=response,
                    new_token_ids=new_token_ids,
                    prompt_base_ids=prompt_base_ids,
                    continuation_base_ids=continuation_base_ids,
                    zip_tokenizer=zip_tokenizer,
                    too_short_bytes=args.too_short_bytes,
                    repetition_rate_threshold=args.repetition_rate_threshold,
                    repetition_count_threshold=args.repetition_count_threshold,
                )
                gpt2_metrics = (
                    {
                        "gpt2_nll_sum": float("nan"),
                        "gpt2_mean_nll": float("nan"),
                        "gpt2_bits_per_byte": float("nan"),
                        "gpt2_scored_tokens": 0,
                        "gpt2_scored_bytes": 0,
                    }
                    if scorer is None or not int(compression["valid_output"])
                    else scorer.score(row["prompt"], response)
                )
                metric_row = {
                    **meta,
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "prompt_bytes": len(row["prompt"].encode("utf-8")),
                    "lzw_prefix_match": lzw_prefix_match,
                    **compression,
                    **gpt2_metrics,
                }
                writer.writerow(metric_row)
                metrics_file.flush()
                write_jsonl_row(
                    generation_file,
                    {
                        **meta,
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "prompt": row["prompt"],
                        "response": response,
                        "decoded_text": decoded[index],
                        "new_token_ids": new_token_ids,
                        "metrics": {
                            key: metric_row[key]
                            for key in METRIC_FIELDS
                            if key in metric_row
                        },
                    },
                )
                write_jsonl_row(
                    trace_file,
                    {
                        **meta,
                        "question_id": row["question_id"],
                        "category": row["category"],
                        "prompt": row["prompt"],
                        "response": response,
                        "decoded_text": decoded[index],
                        "tokenizer": tokenizer_meta,
                        "raw_generated_token_ids": [int(token_id) for token_id in new_token_ids],
                        "compressed_token_ids": [
                            int(token_id)
                            for token_id in filter_special_ids(new_token_ids, zip_tokenizer)
                        ],
                        "prompt_zip_ids": [int(token_id) for token_id in prompt_zip_ids],
                        "prompt_base_ids": [int(token_id) for token_id in prompt_base_ids],
                        "continuation_base_ids": [
                            int(token_id) for token_id in continuation_base_ids
                        ],
                        "lzw_prefix_match": lzw_prefix_match,
                    },
                )

            print(f"[progress] {min(batch_start + len(batch), len(questions))}/{len(questions)}")

    print(f"[saved] {generations_path}")
    print(f"[saved] {traces_path}")
    print(f"[saved] {metrics_path}")
    return 0


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def aggregate_command(args) -> int:
    run_dir = Path(args.run_dir)
    metrics_paths = sorted(run_dir.glob("*/metrics.csv"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics.csv files found under {run_dir}")

    rows: list[dict[str, str]] = []
    for path in metrics_paths:
        with path.open("r", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    group_fields = ["model", "model_slug", "model_group", "ms", "category"]
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "unknown") for field in group_fields)].append(row)
        overall_key = tuple(
            "__all__" if field == "category" else row.get(field, "unknown")
            for field in group_fields
        )
        grouped[overall_key].append(row)

    summary_fields = [*group_fields, "num_samples", *[f"mean_{field}" for field in METRIC_FIELDS]]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = run_dir / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        for key, group_rows in sorted(grouped.items()):
            summary = dict(zip(group_fields, key, strict=True))
            summary["num_samples"] = len(group_rows)
            for field in METRIC_FIELDS:
                values = [
                    parse_float(row.get(field, "nan"))
                    for row in group_rows
                    if not math.isnan(parse_float(row.get(field, "nan")))
                ]
                summary[f"mean_{field}"] = mean(values) if values else float("nan")
            writer.writerow(summary)

    print(f"[saved] {out_path}")
    return 0


def add_generate_args(subparsers) -> None:
    parser = subparsers.add_parser("generate", help="Generate and score one model")
    parser.add_argument("--repo", required=True, help="HF repo id")
    parser.add_argument("--revision", default="hf")
    parser.add_argument("--base-model", default=None, help="Optional override; defaults to the repo's zip2zip_config base model")
    parser.add_argument("--question-file", type=Path, required=True)
    parser.add_argument("--out-dir", default="outputs/eval_compare")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-codebook-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--gpt2-model", default="gpt2")
    parser.add_argument("--gpt2-device", default="auto")
    parser.add_argument("--gpt2-stride", type=int, default=512)
    parser.add_argument("--skip-gpt2", action="store_true")
    parser.add_argument("--too-short-bytes", type=int, default=20)
    parser.add_argument("--repetition-rate-threshold", type=float, default=0.3)
    parser.add_argument("--repetition-count-threshold", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--instruct", action="store_true", help="Apply chat template before tokenization (for instruct/chat models)")
    parser.set_defaults(func=generate_command)


def add_aggregate_args(subparsers) -> None:
    parser = subparsers.add_parser("aggregate", help="Aggregate per-model metrics")
    parser.add_argument("--run-dir", default="outputs/eval_compare")
    parser.add_argument("--out", default="summary.csv")
    parser.set_defaults(func=aggregate_command)


def add_export_demo_args(subparsers) -> None:
    parser = subparsers.add_parser("export-demo", help="Export demo-ready zip2zip artifacts")
    parser.add_argument("--run-dir", default="outputs/eval_compare")
    parser.add_argument("--out-dir", default="demo-data")
    parser.set_defaults(func=export_demo_command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    add_generate_args(subparsers)
    add_aggregate_args(subparsers)
    add_export_demo_args(subparsers)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
