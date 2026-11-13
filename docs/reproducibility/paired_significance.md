# Original-paper paired significance tests

Run from the repository root, with the Python standard library only:

```sh
python scripts/analysis/paired_significance.py --input-dir results/paired_significance --output /tmp/paired-significance.json
```

The four primary comparisons are Phi-3.5-4B and Phi-3-14B, each on GSM8K (1319 questions) and MATH-500 (500). Each outcome file contains a question SHA256 and the two models' binary scores. The declared family must contain exactly the four named comparisons with the full benchmark sizes. Input hashes, unique 64-character question hashes and strictly binary scores are checked; the parser uses the same bytes whose hash was verified. Exact two-sided McNemar uses the symmetric binomial tail of the discordances; Holm covers the four tests. Reference values are in results/paired_significance/verified_results.csv and .json.

GSM8K uses the paper's combined rule: valid strict #### extraction, otherwise flexible last-number extraction, followed by the original exact-match normalization. This gives 469/1319 versus 894/1319 at 4B (plain flexible gives 893 for the latter), and 678/1319 versus 916/1319 at 14B. MATH-500 uses stored original math_verify judgments: 76/500 versus 119/500 and 112/500 versus 148/500. These outcomes come from the original paper runs. Raw samples were matched by exact question identity and checked for prompt/target agreement before binary export. Tests quantify question-level variation conditional on these trained runs, not training-seed variance. The frozen outcomes reproduce the statistical calculations from these runs; model inference and symbolic regrading are separate steps.

The evaluator writes strict and flexible GSM8K metrics separately; the frozen outcomes here apply the combined paper rule before pairing. Run `python -m unittest discover -s tests -p test_paired_significance.py` to validate the input checks and statistical calculations using the Python standard library.
