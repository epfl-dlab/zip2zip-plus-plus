import re


def doc_to_target(doc):
    return str(doc["text"])


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
