"""HumanEval helpers for humaneval_instruct_fence: thin delegates to lm-eval's
own humaneval utilities (pass@k via HF evaluate's code_eval and the instruct
prediction builder), re-exported because lm-eval resolves `!function
module.name` against a .py file next to the task YAML.

Kept out of utils.py on purpose: importing lm_eval.tasks.humaneval.utils loads
HF evaluate's code_eval at import time, which refuses to run unless
HF_ALLOW_CODE_EVAL=1 — that must not take the perplexity tasks in this
directory down with it.
"""

from lm_eval.tasks.humaneval import utils as _humaneval

pass_at_k = _humaneval.pass_at_k
build_predictions_instruct = _humaneval.build_predictions_instruct
