#!/usr/bin/env python3
"""Reproduce the four original-paper paired tests without optional packages."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

EXPECTED_FAMILY = {
    "4B gsm8k": 1319,
    "4B math500": 500,
    "14B gsm8k": 1319,
    "14B math500": 500,
}
COLUMNS = ["problem_sha256", "zip2zip", "zip2zippp"]


def exact_mcnemar(left_only, right_only):
    """Exact two-sided Binomial(b+c, .5) tail, without continuity correction."""
    if any(type(value) is not int or value < 0 for value in (left_only, right_only)):
        raise ValueError("Discordant counts must be nonnegative integers")
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, j) for j in range(min(left_only, right_only) + 1))
    return min(1.0, (2 * tail) / (1 << discordant))


def holm_adjust(p_values):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and between zero and one")
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def load_comparisons(input_dir):
    root = Path(input_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sources = manifest["comparisons"]
    names = [source["comparison"] for source in sources]
    if (manifest.get("family_size") != 4 or len(sources) != 4
            or len(set(names)) != 4 or set(names) != set(EXPECTED_FAMILY)):
        raise ValueError("The primary family must contain exactly the four original-paper comparisons")
    if len({source["file"] for source in sources}) != 4:
        raise ValueError("Comparisons must use distinct outcome files")

    results = []
    for source in sources:
        path = (root / source["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Outcome files must be inside the input directory")
        # Parse the exact bytes whose checksum was verified, not a second read.
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError(f"Input hash mismatch: {path.name}")
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected outcome columns: {path.name}")
        rows = list(reader)
        expected = EXPECTED_FAMILY[source["comparison"]]
        if source["N"] != expected or len(rows) != expected:
            raise ValueError(f"Incomplete benchmark coverage: {path.name}")
        ids = [row["problem_sha256"] for row in rows]
        if len(set(ids)) != expected or any(re.fullmatch(r"[0-9a-f]{64}", key) is None for key in ids):
            raise ValueError(f"Duplicate or invalid problem hashes: {path.name}")
        if any(row[name] not in ("0", "1") for row in rows for name in ("zip2zip", "zip2zippp")):
            raise ValueError(f"Outcomes must be binary: {path.name}")
        pairs = [(int(row["zip2zip"]), int(row["zip2zippp"])) for row in rows]
        left_only = sum(left == 1 and right == 0 for left, right in pairs)
        right_only = sum(left == 0 and right == 1 for left, right in pairs)
        left = sum(left for left, _ in pairs)
        right = sum(right for _, right in pairs)
        results.append({
            "comparison": source["comparison"], "N": expected,
            "correct_zip2zip": left, "correct_zip2zippp": right,
            "accuracy_zip2zip": left / expected, "accuracy_zip2zippp": right / expected,
            "delta_pp": 100 * (right - left) / expected,
            "zip2zip_only": left_only, "zip2zippp_only": right_only,
            "p_raw": exact_mcnemar(left_only, right_only),
        })
    for row, adjusted in zip(results, holm_adjust([row["p_raw"] for row in results])):
        row["p_holm"] = adjusted
    return {"family_size": 4, "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load_comparisons(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
