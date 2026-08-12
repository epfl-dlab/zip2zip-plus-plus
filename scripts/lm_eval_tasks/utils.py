import hashlib
import json
import random
import re


def doc_to_target(doc):
    return str(doc["text"])


# ── Pinned perplexity subsets ───────────────────────────────────────────────
#
# The *_sub1k tasks score a fixed 1000-document subset of their parent corpus
# so a comparable perplexity lands in minutes instead of the ~15h full run.
# Comparability across checkpoints requires that every run scores EXACTLY the
# same documents, so nothing about the selection is left to runtime state:
#
#   - the indices are a seeded sample drawn from the pinned corpus size below,
#     never from len(dataset) — the subset is a pure function of the constants
#     in this file;
#   - len(dataset) must equal that pinned size, so a re-uploaded or re-sharded
#     corpus fails loudly instead of silently shifting which documents the
#     indices point at;
#   - the drawn indices must reproduce a pinned sha256, so a change in
#     random.Random's sampling across Python versions also fails loudly.
#
# Each document is scored independently in the adapter (fresh LZW state per
# rolling request), so a subset document's loglikelihood is identical to what
# the full run would assign it. Subset numbers are still NOT comparable with
# full-corpus numbers — log them under subset_ppl/, never final/.
_SUBSET_SIZE = 1000
_SUBSET_SEED = 1234

# task -> (pinned corpus doc count, sha256 of json.dumps(sorted indices)).
# Counts measured 2026-08-12: pile-val via the HF datasets-server API, the two
# pinned C4 validation shards by downloading and counting rows.
_SUBSET_SPECS = {
    "zip2zip_pile_sub1k": (
        214670,
        "df088ba49e1c7ba8ae5e011a7196794e60121f7170cd0a7916c2bbe33cd77ad1",
    ),
    "zip2zip_mc4_sub1k": (
        45576,
        "24f8625b76fe3a81faad03651d69401f108a839e278e00c50f7b896478a27586",
    ),
    "zip2zip_dc4_sub1k": (
        49153,
        "e80ec8e5108107d1ce3538e31b2f948bba4fc0b98ff3e6a7a6549931669f87d6",
    ),
}


def subset_indices(task):
    """The pinned, sorted document indices for a *_sub1k task."""
    expected_n, expected_sha = _SUBSET_SPECS[task]
    indices = sorted(
        random.Random(_SUBSET_SEED).sample(range(expected_n), _SUBSET_SIZE)
    )
    digest = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    if digest != expected_sha:
        raise ValueError(
            f"{task}: drawn subset indices hash to {digest}, pinned "
            f"{expected_sha}. random.Random no longer reproduces the pinned "
            f"sample on this Python — these results would not be comparable "
            f"with earlier subset_ppl numbers."
        )
    return indices


def _fixed_subset(dataset, task):
    expected_n, _ = _SUBSET_SPECS[task]
    n = len(dataset)
    if n != expected_n:
        raise ValueError(
            f"{task}: corpus has {n} docs, pinned {expected_n}. The underlying "
            f"dataset changed, so the pinned indices would select different "
            f"documents — these results would not be comparable with earlier "
            f"subset_ppl numbers."
        )
    indices = subset_indices(task)
    print(
        f"[subset_ppl] {task}: scoring the pinned {len(indices)}/{n} docs "
        f"(seed={_SUBSET_SEED})"
    )
    return dataset.select(indices)


def subset_pile_sub1k(dataset):
    return _fixed_subset(dataset, "zip2zip_pile_sub1k")


def subset_mc4_sub1k(dataset):
    return _fixed_subset(dataset, "zip2zip_mc4_sub1k")


def subset_dc4_sub1k(dataset):
    return _fixed_subset(dataset, "zip2zip_dc4_sub1k")


def repeated_wikitext_process_results(doc, results):
    """Perplexity denominators for a generated repeated-WikiText row.

    The generated corpus stores the counts for auditability, but the task
    recomputes them from the exact text handed to lm-eval. A stale or edited
    JSONL therefore fails loudly instead of silently reporting a wrong PPL.
    """
    (loglikelihood,) = results
    text = str(doc["text"])
    n_bytes = len(text.encode("utf-8"))
    n_words = len(re.findall(r"\S+", text))

    stored_bytes = int(doc["n_bytes"])
    stored_words = int(doc["n_words"])
    if (stored_bytes, stored_words) != (n_bytes, n_words):
        raise ValueError(
            "Repeated-WikiText denominator mismatch: "
            f"stored bytes/words=({stored_bytes}, {stored_words}), "
            f"actual=({n_bytes}, {n_words}). Rebuild the corpus."
        )

    return {
        "word_perplexity": (loglikelihood, n_words),
        "byte_perplexity": (loglikelihood, n_bytes),
        "bits_per_byte": (loglikelihood, n_bytes),
    }
