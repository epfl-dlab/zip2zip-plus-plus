"""MATH-500 task helpers: thin delegates to lm-eval's own minerva_math utilities.

The math500 task differs from lm-eval's minerva_math_* tasks only in the
dataset (HuggingFaceH4/MATH-500 instead of the seven EleutherAI/hendrycks_math
subjects). Prompt, fixed 4-shot exemplars, answer normalisation and both
scorers are the harness's code, re-exported here because lm-eval resolves
`!function module.name` against a .py file next to the task YAML.

This lives in its own module rather than utils.py on purpose: minerva_math.utils
requires sympy, math_verify and antlr4-python3-runtime==4.11 at import time and
raises otherwise, which must not take the perplexity tasks in this directory
down with it.
"""

from lm_eval.tasks.minerva_math import utils as _minerva

doc_to_text = _minerva.doc_to_text
process_docs = _minerva.process_docs
process_results = _minerva.process_results
list_fewshot_samples = _minerva.list_fewshot_samples
