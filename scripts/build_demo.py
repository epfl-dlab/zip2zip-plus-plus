"""Build frontend demo data directly from a zip2zip-core checkpoint.

No Hugging Face model export is involved.  The checkpoint is loaded through the
same core runtime as ``scripts/inference.py``, including behavior-only settings
recorded in ``meta.pt`` such as base-token positions and two-axis/gated RoPE.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig
from zip2zip_compression import LZWCompressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inference import (  # noqa: E402
    DEFAULT_TOKENIZER,
    format_prompt,
    generate_trace,
    load_model,
)
from zip2zip_core.codebook import CodebookManager  # noqa: E402
from zip2zip_core.disabled_ids import compute_disabled_ids  # noqa: E402


DEFAULT_QUESTION_FILE = Path("/dlabscratch1/xinma/demo_question.jsonl")
@dataclasses.dataclass
class CoreRuntime:
    model: Any
    train_args: dict[str, Any]
    tokenizer: Any
    tokenizer_name: str
    codebook_manager: CodebookManager
    compressor: LZWCompressor
    compression_kwargs: dict[str, Any]
    stop_token_ids: set[int]
    device: str


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            if limit is not None and len(rows) >= limit:
                break
            item = json.loads(line)
            turns = item.get("turns") or []
            if not turns:
                continue
            prompt = turns[0]
            if not isinstance(prompt, str):
                raise ValueError(f"question line {line_index + 1} has a non-string first turn")
            rows.append(
                {
                    "question_id": item.get("question_id", line_index),
                    "category": item.get("category", "unknown"),
                    "prompt": prompt,
                }
            )
    if not rows:
        raise ValueError(f"No questions loaded from {path}")
    return rows


def checkpoint_slug(ckpt_dir: Path) -> str:
    raw = f"{ckpt_dir.parent.name}_{ckpt_dir.name}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def model_meta(ckpt_dir: Path) -> dict[str, str]:
    slug = checkpoint_slug(ckpt_dir)
    ms_match = re.search(r"(?:^|[-_])MS(\d+)(?:[-_]|$)", slug, re.IGNORECASE)
    return {
        "model": str(ckpt_dir),
        "model_slug": slug,
        "model_group": "core",
        "ms": f"MS{ms_match.group(1)}" if ms_match else "unknown",
    }


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def generation_stop_ids(tokenizer, tokenizer_name: str, pad_token_id: int) -> set[int]:
    stop_ids = {
        int(token)
        for token in (tokenizer.eos_token_id, pad_token_id)
        if token is not None
    }
    try:
        generation_cfg = GenerationConfig.from_pretrained(tokenizer_name)
        eos = generation_cfg.eos_token_id
        if eos is not None:
            stop_ids.update(
                int(token)
                for token in (eos if isinstance(eos, (list, tuple)) else [eos])
            )
    except Exception as exc:
        print(
            "[build_demo] generation config unavailable; using tokenizer EOS only "
            f"({type(exc).__name__})"
        )
    return stop_ids


def load_core_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    tokenizer_override: str | None,
) -> CoreRuntime:
    print(f"[load] checkpoint={ckpt_dir} device={device}")
    model, train_args = load_model(
        str(ckpt_dir),
        device,
        dtype=torch.bfloat16 if device.startswith("cuda") else None,
    )

    cfg = model.zip2zip_config
    tokenizer_name = tokenizer_override or train_args.get("tokenizer") or DEFAULT_TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if len(tokenizer) > cfg.vocab_size:
        raise ValueError(
            f"tokenizer {tokenizer_name!r} has {len(tokenizer)} tokens, but checkpoint "
            f"vocabulary has {cfg.vocab_size}"
        )
    trained_tokenizer = train_args.get("tokenizer")
    if tokenizer_override and trained_tokenizer and tokenizer_override != trained_tokenizer:
        raise ValueError(
            f"tokenizer override {tokenizer_override!r} differs from checkpoint tokenizer "
            f"{trained_tokenizer!r}"
        )

    disabled_ids = compute_disabled_ids(
        tokenizer,
        cfg.vocab_size,
        disable_digit_ids=bool(train_args.get("disable_digit_ids")),
    )
    compression_kwargs = {
        "initial_vocab_size": cfg.vocab_size,
        "max_codebook_size": cfg.max_codebook_size,
        "max_subtokens": cfg.max_subtokens,
        "pad_token_id": cfg.pad_token_id,
        "disabled_ids": disabled_ids,
    }
    codebook_manager = CodebookManager(
        embedding_dim=cfg.dim,
        **compression_kwargs,
    )
    compressor = LZWCompressor(**compression_kwargs)
    stop_ids = generation_stop_ids(tokenizer, tokenizer_name, cfg.pad_token_id)

    print(
        "[runtime] "
        f"base_positions={cfg.base_token_positions} "
        f"two_axis={cfg.two_axis_rope} "
        f"gated_compressed={cfg.gated_compressed_rope} "
        f"untied_encoder={not cfg.tie_hyper_encoder} "
        f"max_subtokens={cfg.max_subtokens}"
    )
    return CoreRuntime(
        model=model,
        train_args=train_args,
        tokenizer=tokenizer,
        tokenizer_name=str(tokenizer_name),
        codebook_manager=codebook_manager,
        compressor=compressor,
        compression_kwargs=compression_kwargs,
        stop_token_ids=stop_ids,
        device=device,
    )


def special_token_ids(tokenizer) -> set[int]:
    ids = {
        int(token_id)
        for token_id in (
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        )
        if token_id is not None
    }
    ids.update(int(token_id) for token_id in tokenizer.get_added_vocab().values())
    return ids


def tokenizer_metadata(runtime: CoreRuntime) -> dict[str, Any]:
    cfg = runtime.model.zip2zip_config
    return {
        "name": runtime.tokenizer_name,
        "base_vocab_size": int(cfg.vocab_size),
        "zip_token_start": int(cfg.vocab_size),
        "zip_vocab_size": int(cfg.max_codebook_size),
        "pad_token_id": None if runtime.tokenizer.pad_token_id is None else int(runtime.tokenizer.pad_token_id),
        "eos_token_id": None if runtime.tokenizer.eos_token_id is None else int(runtime.tokenizer.eos_token_id),
        "special_token_ids": sorted(special_token_ids(runtime.tokenizer)),
    }


def decode_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def normalized_codebook(raw: dict[Any, Any]) -> dict[int, list[int]]:
    return {
        int(token_id): [int(value) for value in expansion]
        for token_id, expansion in raw.items()
    }


def token_entries(
    token_ids: list[int],
    *,
    codebook: dict[int, list[int]],
    tokenizer,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = []
    cursor = 0
    specials = set(metadata["special_token_ids"])
    zip_start = int(metadata["zip_token_start"])
    for index, token_id in enumerate(token_ids):
        token_id = int(token_id)
        if token_id in codebook:
            expansion = codebook[token_id]
            kind = "zip"
        elif token_id in specials:
            expansion = [token_id]
            kind = "pad" if token_id == metadata["pad_token_id"] else "special"
        elif token_id < zip_start:
            expansion = [token_id]
            kind = "base"
        else:
            raise ValueError(f"Missing codebook expansion for generated zip token {token_id}")
        entries.append(
            {
                "index": index,
                "id": token_id,
                "text": decode_ids(tokenizer, expansion),
                "kind": kind,
                "source_token_start": cursor,
                "source_token_end": cursor + len(expansion),
                "expands_to_token_ids": expansion,
            }
        )
        cursor += len(expansion)
    return entries


def codebook_entries(codebook: dict[int, list[int]], tokenizer) -> list[dict[str, Any]]:
    return [
        {
            "id": token_id,
            "base_token_ids": expansion,
            "base_token_texts": [decode_ids(tokenizer, [value]) for value in expansion],
            "decoded_text": decode_ids(tokenizer, expansion),
            "length": len(expansion),
        }
        for token_id, expansion in sorted(codebook.items())
    ]


def original_view(base_ids: list[int], tokenizer) -> dict[str, Any]:
    return {
        "token_ids": base_ids,
        "tokens": [
            {"index": index, "id": token_id, "text": decode_ids(tokenizer, [token_id])}
            for index, token_id in enumerate(base_ids)
        ],
        "decoded_text": decode_ids(tokenizer, base_ids),
    }


def new_compressor(runtime: CoreRuntime) -> LZWCompressor:
    return LZWCompressor(**runtime.compression_kwargs)


def optimal_view(
    prompt_base_ids: list[int],
    continuation_base_ids: list[int],
    *,
    runtime: CoreRuntime,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    compressor = new_compressor(runtime)
    full_ids = [*prompt_base_ids, *continuation_base_ids]
    full_compressed, _, _ = compressor.encode(full_ids)
    prompt_compressed, _, _ = compressor.encode(prompt_base_ids)
    compressed = [int(token_id) for token_id in full_compressed[len(prompt_compressed) :]]
    reconstructed, codebook = compressor.batch_decode(
        [[int(token_id) for token_id in full_compressed]]
    )[0]
    reconstructed = [int(token_id) for token_id in reconstructed]
    continuation = (
        reconstructed[len(prompt_base_ids) :]
        if reconstructed[: len(prompt_base_ids)] == prompt_base_ids
        else continuation_base_ids
    )
    mapping = normalized_codebook(codebook.to_dict())
    return {
        "compressed_token_ids": compressed,
        "tokens": token_entries(
            compressed,
            codebook=mapping,
            tokenizer=runtime.tokenizer,
            metadata=metadata,
        ),
        "codebook": codebook_entries(mapping, runtime.tokenizer),
        "reconstructed_token_ids": continuation,
        "reconstructed_text": decode_ids(runtime.tokenizer, continuation),
    }


def model_result_view(
    trace: dict[str, Any],
    *,
    runtime: CoreRuntime,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    generated_ids = [int(token_id) for token_id in trace["generated_compressed_ids"]]
    raw_generated_ids = [int(token_id) for token_id in trace["raw_generated_ids"]]
    generated_base_ids = [int(token_id) for token_id in trace["generated_base_ids"]]
    mapping = normalized_codebook(trace["codebook"])
    return {
        "raw_generated_token_ids": raw_generated_ids,
        "compressed_token_ids": generated_ids,
        "tokens": token_entries(
            generated_ids,
            codebook=mapping,
            tokenizer=runtime.tokenizer,
            metadata=metadata,
        ),
        "codebook": codebook_entries(mapping, runtime.tokenizer),
        "reconstructed_token_ids": generated_base_ids,
        "reconstructed_text": decode_ids(runtime.tokenizer, generated_base_ids),
    }


def fourgram_metrics(text: str) -> dict[str, float | int | None]:
    words = re.findall(r"\S+", text)
    if len(words) < 4:
        return {
            "repeat_4gram_rate": None,
            "max_4gram_count": None,
            "degenerate_repetition": None,
        }
    counts = Counter(tuple(words[index : index + 4]) for index in range(len(words) - 3))
    total = sum(counts.values())
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    rate = repeated / total
    maximum = max(counts.values())
    return {
        "repeat_4gram_rate": rate,
        "max_4gram_count": maximum,
        "degenerate_repetition": int(rate >= 0.3 or maximum >= 8),
    }


def metrics_for(trace: dict[str, Any], response: str, optimal: dict[str, Any]) -> dict[str, Any]:
    response_bytes = len(response.encode("utf-8"))
    actual = len(trace["generated_compressed_ids"])
    theory = len(optimal["compressed_token_ids"])
    base = len(trace["generated_base_ids"])
    valid = int(response_bytes >= 20 and actual > 0 and theory > 0 and base > 0)
    actual_bpt = response_bytes / actual if valid else None
    theory_bpt = response_bytes / theory if valid else None
    repetition = fourgram_metrics(response) if valid else {
        "repeat_4gram_rate": None,
        "max_4gram_count": None,
        "degenerate_repetition": None,
    }
    return {
        "response_bytes": response_bytes,
        "actual_new_zip_tokens": actual,
        "actual_new_zip_tokens_raw": actual,
        "theory_new_zip_tokens": theory,
        "base_new_tokens": base,
        "valid_output": valid,
        "actual_bytes_per_zip_token": actual_bpt,
        "theory_bytes_per_zip_token": theory_bpt,
        "compression_efficiency": actual_bpt / theory_bpt if valid else None,
        "base_token_saving": 1.0 - actual / base if valid else None,
        "gpt2_nll_sum": None,
        "gpt2_mean_nll": None,
        "gpt2_bits_per_byte": None,
        "gpt2_scored_tokens": 0,
        "gpt2_scored_bytes": 0,
        "empty_or_too_short": int(response_bytes < 20),
        **repetition,
    }


def build_example(trace: dict[str, Any], runtime: CoreRuntime, metadata: dict[str, Any]) -> dict[str, Any]:
    prompt_base_ids = [int(token_id) for token_id in trace["prompt_base_ids"]]
    generated_base_ids = [int(token_id) for token_id in trace["generated_base_ids"]]
    response = trace["response"]
    original = original_view(generated_base_ids, runtime.tokenizer)
    optimal = optimal_view(
        prompt_base_ids,
        generated_base_ids,
        runtime=runtime,
        metadata=metadata,
    )
    model_result = model_result_view(trace, runtime=runtime, metadata=metadata)
    metrics = metrics_for(trace, response, optimal)
    return {
        "example_id": str(trace["question_id"]),
        "category": trace["category"],
        "prompt": trace["prompt"],
        "response": response,
        "status": {
            "valid_output": bool(metrics["valid_output"]),
            "empty_or_too_short": bool(metrics["empty_or_too_short"]),
            "degenerate_repetition": bool(metrics.get("degenerate_repetition")),
        },
        "original": original,
        "optimal": optimal,
        "model_result": model_result,
        "metrics": metrics,
    }


def validate_example(example: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, str]]:
    issues = []
    upper = int(metadata["zip_token_start"]) + int(metadata["zip_vocab_size"])
    for view_name in ("optimal", "model_result"):
        if any(int(token_id) >= upper for token_id in example[view_name]["compressed_token_ids"]):
            issues.append(
                {
                    "level": "error",
                    "check": "zip_token_within_vocab",
                    "detail": f"{view_name} contains a token outside the configured vocabulary",
                }
            )
    for view_name in ("original", "optimal", "model_result"):
        text_field = "decoded_text" if view_name == "original" else "reconstructed_text"
        if example[view_name][text_field] != example["response"]:
            issues.append(
                {
                    "level": "warning",
                    "check": f"{view_name}_reconstruction_matches_response",
                    "detail": f"{view_name} reconstructed text differs from response text",
                }
            )
    return issues


def write_metrics_csv(path: Path, examples: list[dict[str, Any]], meta: dict[str, str]) -> None:
    fields = [
        "model",
        "model_slug",
        "model_group",
        "ms",
        "question_id",
        "category",
        "prompt_bytes",
        *list(examples[0]["metrics"].keys()),
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    **meta,
                    "question_id": example["example_id"],
                    "category": example["category"],
                    "prompt_bytes": len(example["prompt"].encode("utf-8")),
                    **example["metrics"],
                }
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume_config(path: Path, requested: dict[str, Any]) -> None:
    if not path.is_file():
        raise ValueError(f"existing generation trace has no {path.name}; use --overwrite")
    previous = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "checkpoint",
        "question_file",
        "question_sha256",
        "seed",
        "max_new_tokens",
        "temperature",
        "instruct",
        "tokenizer",
        "checkpoint_runtime",
    )
    mismatches = [key for key in keys if previous.get(key) != requested.get(key)]
    if mismatches:
        raise ValueError(
            "refusing to mix resumed generations with changed settings "
            f"{mismatches}; use --overwrite"
        )


def build_demo(args: argparse.Namespace) -> int:
    ckpt_dir = args.ckpt_dir.resolve()
    if not (ckpt_dir / "model.pt").is_file() or not (ckpt_dir / "meta.pt").is_file():
        raise FileNotFoundError(f"checkpoint must contain model.pt and meta.pt: {ckpt_dir}")
    questions = load_questions(args.question_file, args.limit)
    meta = model_meta(ckpt_dir)
    out_dir = args.out_dir or Path("outputs") / f"{meta['model_slug']}_demo_data"
    out_dir = out_dir.resolve()
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "generation_traces.jsonl"
    config_path = out_dir / "run_config.json"

    if args.overwrite and trace_path.exists():
        trace_path.unlink()
    existing = load_jsonl(trace_path)
    traces_by_id = {str(row["question_id"]): row for row in existing}
    expected_ids = {str(row["question_id"]) for row in questions}
    unexpected = set(traces_by_id) - expected_ids
    if unexpected:
        raise ValueError(
            f"existing trace contains questions not in {args.question_file}: {sorted(unexpected)}; "
            "use --overwrite"
        )

    device = resolve_device(args.device)
    runtime = load_core_runtime(
        ckpt_dir,
        device=device,
        tokenizer_override=args.tokenizer,
    )
    metadata = tokenizer_metadata(runtime)

    run_config = {
        "checkpoint": str(ckpt_dir),
        "question_file": str(args.question_file.resolve()),
        "question_sha256": file_sha256(args.question_file),
        "out_dir": str(out_dir),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "instruct": args.instruct,
        "device": device,
        "tokenizer": metadata,
        "checkpoint_runtime": {
            "base_token_positions": bool(runtime.model.zip2zip_config.base_token_positions),
            "two_axis_rope": bool(runtime.model.zip2zip_config.two_axis_rope),
            "gated_compressed_rope": bool(runtime.model.zip2zip_config.gated_compressed_rope),
            "untied_hyper_encoder": bool(runtime.train_args.get("untied_hyper_encoder")),
            "max_subtokens": int(runtime.model.zip2zip_config.max_subtokens),
            "max_codebook_size": int(runtime.model.zip2zip_config.max_codebook_size),
        },
        **meta,
    }
    if trace_path.exists():
        validate_resume_config(config_path, run_config)
    config_path.write_text(dumps_json(run_config, indent=2), encoding="utf-8")

    mode = "a" if trace_path.exists() else "w"
    with trace_path.open(mode, encoding="utf-8") as trace_file:
        for index, question in enumerate(questions):
            question_id = str(question["question_id"])
            if question_id in traces_by_id:
                print(f"[resume] {index + 1}/{len(questions)} question_id={question_id}")
                continue
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            formatted = format_prompt(
                runtime.tokenizer,
                question["prompt"],
                instruct=args.instruct,
            )
            result = generate_trace(
                formatted,
                runtime.model,
                runtime.codebook_manager,
                runtime.compressor,
                runtime.tokenizer,
                runtime.stop_token_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=runtime.device,
                verbose=False,
            )
            row = {
                **meta,
                "question_id": question["question_id"],
                "category": question["category"],
                "prompt": question["prompt"],
                "formatted_prompt": formatted,
                "response": result.text,
                "colored_response": result.colored_text,
                "prompt_base_ids": result.prompt_base_ids,
                "prompt_compressed_ids": result.prompt_compressed_ids,
                "generated_base_ids": result.generated_base_ids,
                "raw_generated_ids": result.raw_generated_ids,
                "generated_compressed_ids": result.generated_compressed_ids,
                "codebook": result.codebook,
                "stop_reason": result.stop_reason,
            }
            write_jsonl_row(trace_file, row)
            traces_by_id[question_id] = row
            print(
                f"[progress] {index + 1}/{len(questions)} question_id={question_id} "
                f"compressed_tokens={len(result.generated_compressed_ids)} "
                f"stop={result.stop_reason}"
            )

    ordered_traces = [traces_by_id[str(question["question_id"])] for question in questions]
    examples = [build_example(trace, runtime, metadata) for trace in ordered_traces]
    model_filename = f"{meta['model_slug']}_demo_data.json"
    model_payload = {
        **meta,
        "tokenizer": metadata,
        "examples": examples,
    }
    (models_dir / model_filename).write_text(
        dumps_json(model_payload, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "models": [
            {
                "slug": meta["model_slug"],
                "label": f"{ckpt_dir.parent.name} / {ckpt_dir.name}",
                "repo": str(ckpt_dir),
                "model_group": meta["model_group"],
                "ms": meta["ms"],
                "tokenizer": metadata,
                "results_file": f"models/{model_filename}",
            }
        ],
        "examples": [
            {
                "id": example["example_id"],
                "category": example["category"],
                "display_label": f"{example['example_id']} · {example['category']}",
                "prompt": example["prompt"],
            }
            for example in examples
        ],
    }
    validation_rows = [
        {
            "model_slug": meta["model_slug"],
            "example_id": example["example_id"],
            "issues": validate_example(example, metadata),
        }
        for example in examples
    ]
    validation = {
        "schema_version": 1,
        "run_dir": str(out_dir),
        "num_models": 1,
        "num_examples": len(examples),
        "examples": validation_rows,
    }
    (out_dir / "manifest.json").write_text(dumps_json(manifest, indent=2), encoding="utf-8")
    (out_dir / "validation.json").write_text(dumps_json(validation, indent=2), encoding="utf-8")
    write_metrics_csv(out_dir / "metrics.csv", examples, meta)

    errors = sum(
        issue["level"] == "error"
        for row in validation_rows
        for issue in row["issues"]
    )
    print(f"[saved] {out_dir}")
    print(f"[saved] {models_dir / model_filename}")
    print(f"[validation] examples={len(examples)} errors={errors}")
    return 1 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--question-file", type=Path, default=DEFAULT_QUESTION_FILE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: outputs/<checkpoint-name>_<step>_demo_data",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--instruct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the checkpoint tokenizer's chat template (default: true)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(build_demo(parse_args()))
