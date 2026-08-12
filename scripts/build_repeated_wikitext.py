#!/usr/bin/env python3
"""Build a token-aware repeated-window WikiText evaluation corpus.

The generated corpus is an opt-in stress test for merge-size transfer. It is
not used by any default evaluation preset or pipeline. Each source block is
decoded, repeated N times as text, then re-tokenized and shrunk as necessary so
the final row fits in one evaluation window. All copies are scored normally;
the JSONL stores byte and word counts for the exact repeated text.

Build either checked-in task from the repository root::

    uv run python scripts/build_repeated_wikitext.py --repeat_n 4
    uv run python scripts/build_repeated_wikitext.py --repeat_n 8

The default output is the sibling ``datasets/`` directory next to the cloned
repository, independent of the caller's working directory. The repeat factor
remains configurable for compression-only exploration, but factors other than
four and eight need a matching task YAML so result provenance stays explicit::

    uv run python scripts/build_repeated_wikitext.py --repeat_n 16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence


DEFAULT_TOKENIZER = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_REPEAT_N = 4
DEFAULT_MAX_TOKENS = 1000
DEFAULT_SEPARATOR = "\n\n"
CHECKED_IN_REPEAT_FACTORS = frozenset({4, 8})
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "datasets"
SOURCE_DATASET = "EleutherAI/wikitext_document_level"
SOURCE_CONFIG = "wikitext-2-raw-v1"
SOURCE_SPLIT = "test"


def default_output_path(repeat_n: int) -> Path:
    """Return the canonical local corpus path for a repeat factor."""
    return DEFAULT_DATA_ROOT / f"wikitext-repeat{repeat_n}-phi35" / "test.jsonl"


def wikitext_detokenize(page: str) -> str:
    """Match lm-eval's built-in WikiText v2 detokenizer."""
    string = page
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r'"\s*([^\"]*?)\s*"', r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def repeated_rows_for_text(
    text: str,
    tokenizer,
    *,
    source_doc_id: int,
    repeat_n: int = DEFAULT_REPEAT_N,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    separator: str = DEFAULT_SEPARATOR,
) -> list[dict]:
    """Split one detokenized document into repeated, single-window rows."""
    if repeat_n < 1:
        raise ValueError(f"repeat_n must be positive, got {repeat_n}")
    if max_tokens < 2:
        raise ValueError(f"max_tokens must be at least 2, got {max_tokens}")

    source_ids = _encode(tokenizer, text)
    if not source_ids:
        return []

    # Start near the theoretical maximum. Text decode/re-encode and separators
    # can add boundary tokens, so every candidate is checked and shrunk below.
    nominal_source_tokens = max_tokens // repeat_n
    if nominal_source_tokens < 1:
        raise ValueError(
            f"max_tokens={max_tokens} is too small for repeat_n={repeat_n}"
        )

    rows: list[dict] = []
    source_start = 0
    block_id = 0
    while source_start < len(source_ids):
        source_end = min(
            len(source_ids), source_start + nominal_source_tokens
        )

        while source_end > source_start:
            source_text = _decode(
                tokenizer, source_ids[source_start:source_end]
            )
            repeated_text = separator.join([source_text] * repeat_n)
            repeated_ids = _encode(tokenizer, repeated_text)
            if source_text and len(repeated_ids) <= max_tokens:
                break
            source_end -= 1
        else:
            raise ValueError(
                "A single source token cannot be repeated within max_tokens; "
                "increase --max_tokens, reduce --repeat_n, or shorten the separator"
            )

        n_words = len(re.findall(r"\S+", repeated_text))
        rows.append(
            {
                "text": repeated_text,
                "n_bytes": len(repeated_text.encode("utf-8")),
                "n_words": n_words,
                "source_doc_id": source_doc_id,
                "source_block_id": block_id,
                "repeat_n": repeat_n,
                "source_base_tokens": source_end - source_start,
                "source_reencoded_tokens": len(_encode(tokenizer, source_text)),
                "repeated_base_tokens": len(repeated_ids),
            }
        )
        source_start = source_end
        block_id += 1

    return rows


def iter_repeated_rows(
    documents: Iterable[Mapping[str, object]],
    tokenizer,
    *,
    repeat_n: int = DEFAULT_REPEAT_N,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    separator: str = DEFAULT_SEPARATOR,
) -> Iterator[dict]:
    """Yield repeated rows for WikiText document mappings."""
    for source_doc_id, doc in enumerate(documents):
        if "page" not in doc:
            raise KeyError("WikiText document is missing the 'page' field")
        text = wikitext_detokenize(str(doc["page"]))
        if not text.strip():
            continue
        yield from repeated_rows_for_text(
            text,
            tokenizer,
            source_doc_id=source_doc_id,
            repeat_n=repeat_n,
            max_tokens=max_tokens,
            separator=separator,
        )


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def build_corpus(
    documents: Iterable[Mapping[str, object]],
    tokenizer,
    *,
    output: Path,
    tokenizer_name: str,
    repeat_n: int,
    max_tokens: int,
    separator: str,
    overwrite: bool = False,
) -> dict:
    """Write the JSONL corpus and adjacent manifest, returning the manifest."""
    manifest_path = output.parent / "manifest.json"
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite {names}; pass --overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    stats = {
        "num_rows": 0,
        "total_source_base_tokens": 0,
        "total_repeated_base_tokens": 0,
        "total_bytes": 0,
        "total_words": 0,
    }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in iter_repeated_rows(
                documents,
                tokenizer,
                repeat_n=repeat_n,
                max_tokens=max_tokens,
                separator=separator,
            ):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                stats["num_rows"] += 1
                stats["total_source_base_tokens"] += row["source_base_tokens"]
                stats["total_repeated_base_tokens"] += row[
                    "repeated_base_tokens"
                ]
                stats["total_bytes"] += row["n_bytes"]
                stats["total_words"] += row["n_words"]
        if stats["num_rows"] == 0:
            raise ValueError("Source split produced no non-empty repeated rows")
        os.replace(tmp_name, output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    manifest = {
        "format_version": 1,
        "source_dataset": SOURCE_DATASET,
        "source_config": SOURCE_CONFIG,
        "source_split": SOURCE_SPLIT,
        "tokenizer": tokenizer_name,
        "repeat_n": repeat_n,
        "max_base_tokens_per_row": max_tokens,
        "separator": separator,
        "output": str(output),
        **stats,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an opt-in repeated-window WikiText JSONL corpus."
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--repeat_n", type=int, default=DEFAULT_REPEAT_N)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--separator", default=DEFAULT_SEPARATOR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL. By default, writes under "
            f"{DEFAULT_DATA_ROOT}/wikitext-repeatN-phi35/test.jsonl"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.tokenizer != DEFAULT_TOKENIZER and args.output is None:
        parser.error(
            "--output is required with a non-default tokenizer so its corpus "
            "cannot be mistaken for the Phi-3.5 task data"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or default_output_path(args.repeat_n)
    output = output.expanduser().resolve()

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Building repeated WikiText requires the eval dependencies. "
            "Install the project with its 'eval' extra."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    documents = load_dataset(
        SOURCE_DATASET,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
    )
    manifest = build_corpus(
        documents,
        tokenizer,
        output=output,
        tokenizer_name=args.tokenizer,
        repeat_n=args.repeat_n,
        max_tokens=args.max_tokens,
        separator=args.separator,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nREADY: repeated-window corpus (repeat_n={args.repeat_n})")
    print(f"Dataset: {output}")
    print(f"Manifest: {output.parent / 'manifest.json'}")
    if args.repeat_n in CHECKED_IN_REPEAT_FACTORS:
        task_name = f"zip2zip_wikitext_repeat{args.repeat_n}"
        print("Select this opt-in task from the repository root with:")
        print(f"  --tasks {task_name}")
    else:
        print(
            "No checked-in lm-eval task matches this repeat factor; "
            "add a separate task YAML before evaluation."
        )


if __name__ == "__main__":
    main()
