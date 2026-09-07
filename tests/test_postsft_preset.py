"""Wiring tests for the post-SFT generation preset (paper Table 2).

What must hold so the numbers mean what the paper says:

  - math500 is minerva_math on a different, revision-pinned dataset: same
    prompt/scorer functions (delegated, not re-implemented), fixed 4-shot
    exemplars, both metrics (exact_match = strict, math_verify = flex);
  - the postsft preset sets NO global num_fewshot (it would override every
    task's own protocol) and every preset consumer defaults to per-task;
  - the code-execution switch is a single env var: eval_ckpt_rcp.sh turns it on
    only for postsft / humaneval and the harnesses forward it to lm-eval;
  - postsft extras are pinned and isolated on PYTHONPATH for those runs only;
    the shared venv and the pipeline's dependency list stay untouched (antlr4
    4.11 breaks omegaconf, and a second venv would lose the torch nightly);
  - generations are decoded as continuations (leading whitespace preserved),
    with a legacy switch plumbed end to end.
"""

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TASKS_DIR = SCRIPTS / "lm_eval_tasks"

POSTSFT_TASKS = ["math500", "humaneval_instruct_fence", "ifeval"]
EXTRA_EVAL_PINS = {
    "langdetect==1.0.9",
    "immutabledict==4.2.1",
    "math-verify==0.7.0",
    "antlr4-python3-runtime==4.11.0",
}


class _FunctionLoader(yaml.SafeLoader):
    """Read lm-eval task YAMLs without importing the referenced functions."""


_FunctionLoader.add_constructor(
    "!function", lambda loader, node: ("!function", node.value)
)


def _task_yaml(name):
    with open(TASKS_DIR / f"{name}.yaml") as f:
        return yaml.load(f, Loader=_FunctionLoader)


def _presets():
    with open(SCRIPTS / "eval_presets.yaml") as f:
        return yaml.safe_load(f)["presets"]


def _code_lines(text):
    return [l for l in text.splitlines() if not l.lstrip().startswith("#")]


# ───────────────────────────── math500 task ─────────────────────────────────


def test_math500_is_minerva_math_on_the_pinned_math500_dataset():
    cfg = _task_yaml("math500")
    assert cfg["task"] == "math500"
    assert cfg["dataset_path"] == "HuggingFaceH4/MATH-500"
    assert re.fullmatch(r"[0-9a-f]{40}", cfg["dataset_kwargs"]["revision"]), (
        "the dataset revision must be pinned to a commit sha"
    )
    assert cfg["test_split"] == "test"
    assert cfg["output_type"] == "generate_until"
    for key in ("doc_to_text", "process_docs", "process_results"):
        assert cfg[key] == ("!function", f"math500_utils.{key}"), key
    assert cfg["fewshot_config"] == {
        "sampler": "first_n",
        "samples": ("!function", "math500_utils.list_fewshot_samples"),
    }
    assert cfg["num_fewshot"] == 4
    assert cfg["doc_to_target"] == "{{answer if few_shot is undefined else solution}}"


def test_math500_reports_strict_and_flex_metrics():
    cfg = _task_yaml("math500")
    metrics = [m["metric"] for m in cfg["metric_list"]]
    assert metrics == ["exact_match", "math_verify"]
    assert all(m["aggregation"] == "mean" and m["higher_is_better"] for m in cfg["metric_list"])


def test_math500_generation_is_greedy_and_long_enough():
    gen = _task_yaml("math500")["generation_kwargs"]
    assert gen["until"] == ["Problem:"]
    assert gen["do_sample"] is False
    assert gen["max_gen_toks"] == 1024, "256 (the harness default) truncates MATH solutions"


def test_math500_utils_only_delegates_to_minerva_math():
    source = (TASKS_DIR / "math500_utils.py").read_text()
    assert "from lm_eval.tasks.minerva_math import utils as _minerva" in source
    for fn in ("doc_to_text", "process_docs", "process_results", "list_fewshot_samples"):
        assert f"{fn} = _minerva.{fn}" in source, fn
    assert "def " not in source, "no local re-implementation of the scorer"


def test_math500_utils_resolves_to_the_harness_functions():
    pytest.importorskip("lm_eval")
    # Missing math deps mean "wrong venv", not a wiring bug: skip only on those.
    pytest.importorskip("math_verify")
    try:
        from lm_eval.tasks.minerva_math import utils as minerva  # must import here
    except (ImportError, AssertionError) as exc:  # minerva asserts the antlr4 version at import
        pytest.skip(f"minerva_math utils not importable in this venv: {exc}")

    spec = importlib.util.spec_from_file_location(
        "_math500_utils", TASKS_DIR / "math500_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn in ("doc_to_text", "process_docs", "process_results", "list_fewshot_samples"):
        assert getattr(module, fn) is getattr(minerva, fn), fn
    assert len(module.list_fewshot_samples()) == 4
    assert module.doc_to_text({"problem": "1+1?"}) == "Problem:\n1+1?\n\nSolution:"


# ───────────────────────────── preset wiring ────────────────────────────────


def test_postsft_preset_tasks_and_no_global_fewshot():
    preset = _presets()["postsft"]
    assert preset["tasks"] == POSTSFT_TASKS
    assert "num_fewshot" not in preset, (
        "a preset num_fewshot is a global override that would flatten math500's "
        "4-shot and ifeval's 0-shot protocols"
    )
    assert preset["apply_chat_template"] is True
    assert preset["fewshot_as_multiturn"] is True
    assert preset["eval_mode"] == "compressed"
    assert preset["max_length"] == 4096
    assert preset["seed"] == 1234


def test_other_presets_still_pin_their_global_fewshot():
    presets = _presets()
    for name, expected in {"default": 2, "default_base": 2, "smoke": 2,
                           "perplexity": 0, "perplexity_subset": 0}.items():
        assert presets[name]["num_fewshot"] == expected, name


def test_every_preset_consumer_defaults_to_per_task_fewshot():
    for script in ("eval_harness.py", "eval_hf_model.py"):
        source = (SCRIPTS / script).read_text()
        assert re.search(
            r'add_argument\("--num_fewshot", type=int, default=None', source
        ), f"{script}: --num_fewshot must default to None (per-task protocol)"
        assert (
            'confirm_run_unsafe_code=os.environ.get("HF_ALLOW_CODE_EVAL") == "1"'
            in source
        ), script
    assert "(not wired up)" not in (SCRIPTS / "eval_harness.py").read_text()
    load_preset = (SCRIPTS / "load_preset.py").read_text()
    assert "print(f\"NUM_FEWSHOT={'' if nf is None else nf}\")" in load_preset
    validate = (SCRIPTS / "validate_phi35_rcp.sh").read_text()
    assert '${NUM_FEWSHOT:+--num_fewshot "$NUM_FEWSHOT"}' in validate


def test_eval_ckpt_enables_code_eval_only_for_postsft_or_humaneval():
    source = (SCRIPTS / "eval_ckpt_rcp.sh").read_text()
    blocks = re.findall(r'case "\$PRESET,\$\{TASKS:-\}" in\n(.*?)\nesac', source, re.S)
    assert len(blocks) == 2, "expected the dependency-selection and code-eval case blocks"
    deps_block, code_block = blocks
    assert "postsft,*|*humaneval*|*math500*|*ifeval*) POSTSFT_DEPS=1" in deps_block
    assert "postsft,*|*humaneval*) export HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1}" in code_block
    assert sum("export HF_ALLOW_CODE_EVAL" in l for l in _code_lines(source)) == 1


def test_postsft_extras_are_isolated_on_pythonpath_with_pinned_versions():
    source = (SCRIPTS / "eval_ckpt_rcp.sh").read_text()
    assert source.count("VENV_DIR=$Z2Z_SCRATCH/.venvs/lm-eval\n") == 1
    assert ".venvs/lm-eval-postsft\n" not in source, "no second venv: it would lose the torch nightly"
    branch = re.search(r'if \[ -n "\$POSTSFT_DEPS" \]; then\n(.*?)\nfi', source, re.S)
    assert branch, "POSTSFT_DEPS branch missing"
    body = branch.group(1)
    assert 'POSTSFT_EXTRAS_DIR=$Z2Z_SCRATCH/.venvs/lm-eval-postsft-extras' in body
    assert 'pip install --quiet --upgrade --target "$POSTSFT_EXTRAS_DIR"' in body
    for pin in EXTRA_EVAL_PINS:
        assert f'"{pin}"' in body, f"eval_ckpt_rcp.sh missing {pin}"
    assert 'export PYTHONPATH="$POSTSFT_EXTRAS_DIR${PYTHONPATH:+:$PYTHONPATH}"' in body
    # the shared venv's own install line is untouched
    shared = re.search(r'source "\$VENV_DIR/bin/activate"\npip install --quiet \\\n(.*?)\n    wandb\n', source, re.S)
    assert shared, "shared pip install block changed"
    for pin in EXTRA_EVAL_PINS:
        assert pin not in shared.group(1)
    assert '"lm-eval==0.4.9"' in shared.group(1), "the harness pin must not move"


def test_shared_venv_and_pipeline_do_not_get_the_postsft_extras():
    pipeline = (SCRIPTS / "pipeline_ft_eval_rcp.sh").read_text()
    for text in (pipeline, (SCRIPTS / "eval_z2z_rcp.sh").read_text(),
                 (SCRIPTS / "validate_phi35_rcp.sh").read_text()):
        for pin in EXTRA_EVAL_PINS:
            assert pin.split("==")[0] not in text, (
                f"{pin} must stay out of the shared .venvs/lm-eval: antlr4 4.11 "
                "breaks omegaconf there"
            )
    assert 'EVAL_DEPS=("lm-eval==0.4.9" "zip2zip-compression>=0.3.3")' in pipeline


# ───────────────────── continuation-preserving decode ───────────────────────


def test_legacy_stripped_generation_is_plumbed_end_to_end():
    harness = (SCRIPTS / "eval_harness.py").read_text()
    assert '"--legacy_stripped_generation", action="store_true"' in harness
    assert "preserve_leading_space=not args.legacy_stripped_generation" in harness
    assert "args.preserve_leading_space = lm.preserve_leading_space" in harness
    launcher = (SCRIPTS / "eval_ckpt_rcp.sh").read_text()
    assert launcher.count("${LEGACY_STRIPPED_GENERATION:+--legacy_stripped_generation}") == 2
    assert 'VARIANT_TAG="${VARIANT_TAG}-strippedgen"' in launcher
    adapter = (REPO / "src" / "zip2zip_core" / "lm_eval_adapter.py").read_text()
    assert "preserve_leading_space: bool = True" in adapter
    assert adapter.count("self._trim_stops(self._decode_generation(") == 2
    assert "self._trim_stops(self.tok_decode(" not in adapter


class _SentencePieceLikeTokenizer:
    """decode() drops the word-boundary marker of the FIRST piece, like SP."""

    pieces = {13: "\n", 100: "▁▁▁▁for", 101: "▁i", 102: "The", 103: "▁cat"}

    def encode(self, text, add_special_tokens=False):
        assert text == "\n"
        return [29871, 13]  # SP puts a boundary piece in front of the newline

    def decode(self, ids, skip_special_tokens=True):
        pieces = [self.pieces[i] for i in ids]
        if pieces and pieces[0].startswith("▁"):
            pieces[0] = pieces[0][1:]
        return "".join(pieces).replace("▁", " ")


def _adapter_stub(preserve):
    pytest.importorskip("torch")  # the adapter module imports torch and lm_eval
    pytest.importorskip("lm_eval")
    from zip2zip_core.lm_eval_adapter import Zip2ZipLM

    lm = object.__new__(Zip2ZipLM)
    lm.tokenizer = _SentencePieceLikeTokenizer()
    lm.preserve_leading_space = preserve
    ids = lm.tokenizer.encode("\n")[-1:]
    lm._decode_prefix_ids = ids
    lm._decode_prefix_text = lm.tokenizer.decode(ids)
    return lm


def test_decode_generation_keeps_python_indentation():
    lm = _adapter_stub(preserve=True)
    assert lm.tok_decode([100, 101]) == "   for i", "standalone decode loses a space"
    assert lm._decode_generation([100, 101]) == "    for i"
    assert lm._decode_generation([102, 103]) == "The cat"
    assert lm._decode_generation([103]) == " cat"
    assert lm._decode_generation([]) == ""


def test_decode_generation_legacy_switch_restores_standalone_decode():
    lm = _adapter_stub(preserve=False)
    assert lm._decode_generation([100, 101]) == "   for i"


def test_docs_describe_the_preset():
    docs = (REPO / "docs" / "evaluation.md").read_text()
    assert "### Post-SFT generation benchmarks (`postsft`, paper Table 2)" in docs
    for task in POSTSFT_TASKS:
        assert f"`{task}`" in docs
    assert "final/humaneval_instruct_fence/pass@1_create_test" in docs
    assert ".venvs/lm-eval-postsft-extras" in docs
    assert "LEGACY_STRIPPED_GENERATION" in docs
