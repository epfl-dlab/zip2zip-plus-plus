"""CPU-only checks for the public frozen-outcome statistical entrypoint."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_significance", ROOT / "scripts/analysis/paired_significance.py")
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)
DATA = ROOT / "results/paired_significance"


class PairedSignificanceTests(unittest.TestCase):
    def test_exact_discordant_tail_and_symmetry(self):
        self.assertEqual(STATS.exact_mcnemar(0, 0), 1.0)
        self.assertEqual(STATS.exact_mcnemar(0, 5), 0.0625)
        self.assertEqual(STATS.exact_mcnemar(1, 1), 1.0)
        self.assertEqual(STATS.exact_mcnemar(2, 11), STATS.exact_mcnemar(11, 2))
        with self.assertRaises(ValueError):
            STATS.exact_mcnemar(-1, 2)

    def test_holm_preserves_order_and_step_down_monotonicity(self):
        self.assertEqual(STATS.holm_adjust([0.04, 0.001, 0.03, 0.01]), [0.06, 0.004, 0.06, 0.03])

    def test_all_four_certified_results(self):
        actual = STATS.load_comparisons(DATA)
        expected = json.loads((DATA / "verified_results.json").read_text())
        self.assertEqual(actual["family_size"], 4)
        for left, right in zip(actual["comparisons"], expected["comparisons"]):
            for key, value in left.items():
                self.assertEqual(value, right[key], (left["comparison"], key))

    def test_wrong_family_size_and_duplicate_comparison_rejected(self):
        for mutation in ("wrong_size", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                shutil.copytree(DATA, tmp, dirs_exist_ok=True)
                path = Path(tmp) / "manifest.json"
                manifest = json.loads(path.read_text())
                if mutation == "wrong_size":
                    manifest["family_size"] = 3
                else:
                    manifest["comparisons"][1] = manifest["comparisons"][0]
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "four original-paper"):
                    STATS.load_comparisons(tmp)

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(DATA, tmp, dirs_exist_ok=True)
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            path = Path(tmp) / manifest["comparisons"][0]["file"]
            with path.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                STATS.load_comparisons(tmp)

    def test_duplicate_problem_and_nonbinary_outcome_rejected(self):
        for mutation, message in (("duplicate", "Duplicate"), ("score", "binary")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                shutil.copytree(DATA, tmp, dirs_exist_ok=True)
                mp = Path(tmp) / "manifest.json"
                manifest = json.loads(mp.read_text())
                source = manifest["comparisons"][0]
                path = Path(tmp) / source["file"]
                with path.open() as stream:
                    reader = csv.DictReader(stream)
                    columns, rows = reader.fieldnames, list(reader)
                if mutation == "duplicate":
                    rows[1]["problem_sha256"] = rows[0]["problem_sha256"]
                else:
                    rows[0]["zip2zip"] = "0.5"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, columns, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
                source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                mp.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    STATS.load_comparisons(tmp)


if __name__ == "__main__":
    unittest.main()
