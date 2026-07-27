"""Correctness invariants for the named recipe registry (scripts/recipes.py).

A recipe is what a launch command means, so a wrong recipe silently trains the
wrong thing for hours. These checks pin the properties the launchers depend on.

Pure stdlib + the repo's files: no torch, no cluster. Runnable as a script and
under pytest.

Invariants proven:
  R1  the registry is internally consistent: exactly one status=current, CURRENT
      points at it, every `extends` resolves, no inheritance cycles.
  R2  v0.6.4 resolves to the five validated levers; the archived experiments
      and v0.7.1 resolve to their exact declared deltas.
  R3  the ledger property: every recipe differs from its parent by exactly the
      keys in its own `env` block (so the file reads as "parent + one change").
  R4  every env key any recipe sets is ACTUALLY CONSUMED by both launchers — the
      typo guard, because a misspelled env var would otherwise do nothing at all.
  R5  the emitted shell has the semantics the launchers rely on, verified by
      running bash: unset gets the recipe value, an explicit value (including 0
      and empty) wins, and resolving twice is idempotent.
  R6  an unknown recipe name is a fatal non-zero exit, never a silent no-op.
  R7  identify() round-trips: the args a recipe implies map back to its name, and
      an off-recipe checkpoint is reported as custom with the right diff.
  R8  docs/finetuning.md documents CURRENT and lists exactly its resolved levers
      (anti-drift: the docs table cannot silently disagree with the code).
  R9  the RCP launcher forwards the complete v0.7.1 contract and training seed.
"""
import ast
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import recipes as R  # noqa: E402


# Local harness on purpose: this suite stays stdlib-only (no torch), like the
# module it tests, so it runs anywhere a launcher can — including a bare
# container with no venv.
def make_harness():
    results = {}

    def check(name, cond, detail=""):
        results[name] = bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    return results, check


def summarize(results):
    print("\n==== SUMMARY ====")
    n_pass = sum(results.values())
    print(f"{n_pass}/{len(results)} passed")
    failed = [k for k, v in results.items() if not v]
    for k in failed:
        print(f"  FAILED: {k}")
    return failed

RECIPES_PY = os.path.join(REPO, "scripts", "recipes.py")
LAUNCHERS = [os.path.join(REPO, "scripts", "finetune_phi35_rcp.sh"),
             os.path.join(REPO, "scripts", "pipeline_ft_eval_rcp.sh")]
DOCS = os.path.join(REPO, "docs", "finetuning.md")

EXPECTED_V064 = {
    "DISABLE_DIGIT_IDS": "1",
    "UNTIED_HYPER_ENCODER": "1",
    "BASE_TOKEN_POSITIONS": "1",
    "TOKEN_TYPE_LOSS_WEIGHT": "0.05",
    "ZERO_INIT_ENCODER_OUTPUT": "1",
}
EXPECTED_V065 = {
    **EXPECTED_V064,
    "ONLINE_CODEBOOK_MASK": "1",
}
EXPECTED_V07 = {
    **EXPECTED_V064,
    "TWO_AXIS_ROPE": "1",
}
EXPECTED_V071 = {
    **EXPECTED_V064,
    "GATED_COMPRESSED_ROPE": "1",
    "GATED_ROPE_START_LAYER": "16",
    "BASE_VIEW_REPLAY_PROB": "0.25",
}


def bash(script):
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, cwd=REPO)


def main():
    results, check = make_harness()

    # ---- R1: registry consistency (import already validated it) ----
    currents = [n for n, s in R.RECIPES.items() if s["status"] == "current"]
    check("R1_single_current_matches_CURRENT",
          currents == [R.CURRENT] == ["v0.6.4"],
          f"CURRENT={R.CURRENT}, status=current -> {currents}")
    ok_chains = all(R.lineage(n)[-1] == n for n in R.RECIPES)
    check("R1_every_lineage_resolves", ok_chains,
          f"{len(R.RECIPES)} recipes, all chains terminate without a cycle")

    # ---- R2: the current recipe is the five documented levers ----
    check("R2_current_resolves_to_five_levers",
          R.resolve("v0.6.4") == EXPECTED_V064,
          f"{sorted(R.resolve('v0.6.4').items())}")
    # v0.6.5 ran on 2026-07-26 and lost on 6 of 7 metrics: it is archived as a
    # measured negative, still selectable so the experiment stays reproducible.
    check("R2_v065_is_v064_plus_online_mask",
          R.resolve("v0.6.5") == EXPECTED_V065
          and R.RECIPES["v0.6.5"]["extends"] == "v0.6.4"
          and R.RECIPES["v0.6.5"]["status"] == "negative",
          f"{sorted(R.resolve('v0.6.5').items())}")
    check("R2_v07_is_v064_plus_two_axis_rope",
          R.resolve("v0.7") == EXPECTED_V07
          and R.RECIPES["v0.7"]["extends"] == "v0.6.4"
          and R.RECIPES["v0.7"]["status"] == "negative",
          f"{sorted(R.resolve('v0.7').items())}")
    check("R2_v071_is_v064_plus_gated_rope_and_replay",
          R.resolve("v0.7.1") == EXPECTED_V071
          and R.RECIPES["v0.7.1"]["extends"] == "v0.6.4"
          and R.RECIPES["v0.7.1"]["status"] == "candidate"
          and "TWO_AXIS_ROPE" not in R.resolve("v0.7.1"),
          f"{sorted(R.resolve('v0.7.1').items())}")

    # ---- R2b: resume-hard coverage for levers that live OUTSIDE ENV_KEYS ----
    # lora_alpha sets scaling = alpha/rank as a runtime attribute, so a resume
    # that changes it loads weights fine and silently changes the model.
    # Read the source instead of importing train.py: launch hosts may not have
    # torch installed yet, and this registry suite is intentionally stdlib-only.
    _train_src = open(
        os.path.join(REPO, "src", "zip2zip_core", "train.py")
    ).read()
    _train_ast = ast.parse(_train_src)
    _resume_node = next(
        node
        for node in _train_ast.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "validate_resume_args"
    )
    _hard_src = ast.get_source_segment(_train_src, _resume_node) or ""
    _missing = [
        k
        for k in (
            "lora_alpha",
            "hyper_causal_mask",
            "no_remap_codebook",
            "gated_compressed_rope",
            "gated_rope_start_layer",
            "base_view_replay_prob",
            "gradient_accumulation_steps",
        )
        if f'"{k}"' not in _hard_src
    ]
    check("R2b_runtime_only_levers_are_resume_hard", not _missing,
          "runtime geometry/replay/accumulation levers are compared on resume"
          if not _missing else f"NOT in the hard list: {_missing}")

    # ---- R3: ledger property — each version = parent + its own env block ----
    offenders = []
    for name, spec in R.RECIPES.items():
        parent = spec.get("extends")
        if parent is None:
            continue
        delta = {k: v for k, v in R.resolve(name).items()
                 if R.resolve(parent).get(k) != v}
        if delta != spec.get("env", {}):
            offenders.append((name, delta, spec.get("env", {})))
    check("R3_each_recipe_is_parent_plus_its_own_env", not offenders,
          "every recipe's diff vs its parent equals its declared env block"
          if not offenders else f"offenders={offenders}")

    # ---- R4: the typo guard — launchers really consume every key ----
    launcher_src = "\n".join(open(p).read() for p in LAUNCHERS)
    used = set()
    for spec in R.RECIPES.values():
        used |= set(spec.get("env", {}))
    unconsumed = sorted(k for k in used if k not in launcher_src)
    check("R4_every_recipe_key_is_read_by_a_launcher", not unconsumed,
          f"{len(used)} keys used by recipes, all present in the launchers"
          if not unconsumed else f"NOT consumed anywhere: {unconsumed}")
    # and every declarable key too, so ENV_KEYS cannot grow dead entries
    dead = sorted(k for k in R.ENV_KEYS if k not in launcher_src)
    check("R4_no_dead_keys_in_ENV_KEYS", not dead,
          f"{len(R.ENV_KEYS)} declarable keys, all consumed"
          if not dead else f"declared but unused: {dead}")

    # ---- R5: shell semantics, verified by actually running bash ----
    resolve_cmd = f'R=$({sys.executable} scripts/recipes.py v0.6.4 2>/dev/null)'
    cases = [
        ("unset takes the recipe value",
         f'{resolve_cmd}; eval "$R"; echo "$TOKEN_TYPE_LOSS_WEIGHT"', "0.05"),
        ("explicit value wins",
         f'{resolve_cmd}; TOKEN_TYPE_LOSS_WEIGHT=0.2; eval "$R"; echo "$TOKEN_TYPE_LOSS_WEIGHT"', "0.2"),
        ("explicit 0 (off) wins",
         f'{resolve_cmd}; ZERO_INIT_ENCODER_OUTPUT=0; eval "$R"; echo "$ZERO_INIT_ENCODER_OUTPUT"', "0"),
        ("explicit empty (off) wins",
         f'{resolve_cmd}; UNTIED_HYPER_ENCODER=; eval "$R"; echo "[$UNTIED_HYPER_ENCODER]"', "[]"),
        ("second resolve is idempotent",
         f'{resolve_cmd}; eval "$R"; TOKEN_TYPE_LOSS_WEIGHT=0.2; eval "$R"; echo "$TOKEN_TYPE_LOSS_WEIGHT"', "0.2"),
        ("values reach child processes",
         f'{resolve_cmd}; eval "$R"; bash -c \'echo "$DISABLE_DIGIT_IDS"\'', "1"),
    ]
    for label, script, want in cases:
        got = bash(script).stdout.strip()
        check(f"R5_{label.replace(' ', '_')}", got == want, f"got {got!r} want {want!r}")
    candidate_cmd = (
        f'R=$({sys.executable} scripts/recipes.py v0.6.5 2>/dev/null); '
        'ONLINE_CODEBOOK_MASK=0; eval "$R"; echo "$ONLINE_CODEBOOK_MASK"'
    )
    check("R5_candidate_explicit_off_wins",
          bash(candidate_cmd).stdout.strip() == "0",
          "ONLINE_CODEBOOK_MASK=0 overrides the v0.6.5 candidate")
    v07_cmd = (
        f'R=$({sys.executable} scripts/recipes.py v0.7 2>/dev/null); '
        'TWO_AXIS_ROPE=0; eval "$R"; echo "$TWO_AXIS_ROPE"'
    )
    check("R5_v07_explicit_off_wins",
          bash(v07_cmd).stdout.strip() == "0",
          "TWO_AXIS_ROPE=0 overrides the archived v0.7 recipe")
    v071_cmd = (
        f'R=$({sys.executable} scripts/recipes.py v0.7.1 2>/dev/null); '
        'GATED_COMPRESSED_ROPE=0; BASE_VIEW_REPLAY_PROB=0.10; '
        'eval "$R"; printf "%s|%s|%s|%s" "$GATED_COMPRESSED_ROPE" '
        '"$GATED_ROPE_START_LAYER" "$BASE_VIEW_REPLAY_PROB" "${TWO_AXIS_ROPE-unset}"'
    )
    check("R5_v071_explicit_overrides_win",
          bash(v071_cmd).stdout.strip() == "0|16|0.10|unset",
          "explicit gate-off/replay override wins and legacy two-axis stays unset")

    # ---- R6: unknown name is fatal ----
    bad = subprocess.run([sys.executable, RECIPES_PY, "v9.9"],
                         capture_output=True, text=True)
    check("R6_unknown_recipe_exits_nonzero",
          bad.returncode != 0 and not bad.stdout.strip(),
          f"exit={bad.returncode}, stdout empty={not bad.stdout.strip()}")
    neg = subprocess.run([sys.executable, RECIPES_PY, "v0.6"],
                         capture_output=True, text=True)
    check("R6_negative_recipe_warns_but_works",
          neg.returncode == 0 and "WARNING" in neg.stderr and "WARMSTART_STEPS" in neg.stdout,
          "a measured-negative recipe still resolves, loudly")
    archived = subprocess.run(
        [sys.executable, RECIPES_PY, "v0.6.5"],
        capture_output=True,
        text=True,
    )
    check("R6_archived_negative_warns_but_works",
          archived.returncode == 0
          and "MEASURED NEGATIVE" in archived.stderr
          and "ONLINE_CODEBOOK_MASK" in archived.stdout,
          "the archived v0.6.5 still resolves, but warns it lost to v0.6.4")
    candidate = subprocess.run(
        [sys.executable, RECIPES_PY, "v0.7"],
        capture_output=True,
        text=True,
    )
    check("R6_v07_negative_warns_but_works",
          candidate.returncode == 0
          and "MEASURED NEGATIVE" in candidate.stderr
          and "TWO_AXIS_ROPE" in candidate.stdout,
          "v0.7 still resolves, but warns it lost to v0.6.4")
    v071 = subprocess.run(
        [sys.executable, RECIPES_PY, "v0.7.1"],
        capture_output=True,
        text=True,
    )
    check("R6_v071_candidate_warns_but_works",
          v071.returncode == 0
          and "UNMEASURED CANDIDATE" in v071.stderr
          and "GATED_COMPRESSED_ROPE" in v071.stdout
          and "BASE_VIEW_REPLAY_PROB" in v071.stdout,
          "v0.7.1 resolves all levers and remains explicitly unmeasured")

    # ---- R7: identify() round-trip, for EVERY recipe ----
    wrong = []
    for name in R.RECIPES:
        args = dict(R.expected_args(name))
        hint = R.resolve_data(name)
        if hint:
            args["data_dir"] = hint
        matches, diffs = R.identify(args)
        if matches != [name] or diffs:
            wrong.append((name, matches, sorted(diffs)))
    check("R7_identify_roundtrips_every_recipe", not wrong,
          f"all {len(R.RECIPES)} recipes map back to themselves"
          if not wrong else f"wrong={wrong}")

    # absent keys must count as OFF: a v0.4-era checkpoint (recorded before the
    # untied/basepos/zeroinit flags existed) must NOT also match v0.5+.
    v04_era = {"disable_digit_ids": True, "token_type_loss_weight": 0.0,
               "data_dir": R.resolve_data("v0.4")}
    matches, _ = R.identify(v04_era)
    check("R7_absent_keys_count_as_off", matches == ["v0.4"],
          f"v0.4-era args (newer flags absent) -> {matches}")

    # v0.2 and v0.3 have identical levers; only the dataset separates them
    check("R7_data_breaks_the_v02_v03_tie",
          R.identify(dict(R.expected_args("v0.3"),
                          data_dir="/x/phi-1B-sft-8shards-mathchat"))[0] == ["v0.3"]
          and R.identify(dict(R.expected_args("v0.2"),
                              data_dir="/x/phi-1B-sft-8shards-eosfix"))[0] == ["v0.2"],
          "same levers, disambiguated by the recorded data_dir")
    ambiguous, _ = R.identify(R.expected_args("v0.2"))  # no data_dir recorded
    check("R7_ambiguity_is_reported_not_guessed",
          sorted(ambiguous) == ["v0.2", "v0.3"],
          f"without data_dir -> {sorted(ambiguous)} (both, honestly)")

    off = dict(R.expected_args("v0.6.4"), encoder_n_layers=4)
    _, diffs2 = R.identify(off)
    check("R7_identify_flags_off_recipe_checkpoint",
          "encoder_n_layers" in diffs2, f"diffs={sorted(diffs2)}")

    # ---- R8: the docs cannot silently disagree with the code ----
    docs = open(DOCS).read()
    section = docs.split("## Current standard recipe", 1)[-1].split("\n## ", 1)[0]
    check("R8_docs_name_the_current_recipe",
          f"recipe: {R.CURRENT}" in f"recipe: {R.CURRENT}" and R.CURRENT in section.split("\n")[0] + section[:400],
          f"docs section headline mentions {R.CURRENT}")
    documented = dict(re.findall(r"`([A-Z_]+)=([^`]+)`", section))
    resolved = R.resolve(R.CURRENT)
    listed = {k: v for k, v in documented.items() if k in R.ENV_KEYS}
    check("R8_docs_table_matches_resolved_recipe", listed == resolved,
          f"docs lists {sorted(listed.items())}"
          if listed == resolved else
          f"MISMATCH docs={sorted(listed.items())} code={sorted(resolved.items())}")

    # ---- R9: launcher forwards the exact low-level contract ----
    finetune_src = open(LAUNCHERS[0]).read()
    required_cli = {
        "--gated_compressed_rope",
        '--gated_rope_start_layer "$GATED_ROPE_START_LAYER"',
        '--base_view_replay_prob "$BASE_VIEW_REPLAY_PROB"',
        '--seed "$SEED"',
    }
    missing_cli = sorted(fragment for fragment in required_cli
                         if fragment not in finetune_src)
    check("R9_finetune_forwards_v071_and_seed", not missing_cli,
          "gated flag, layer, replay, and seed reach train.py"
          if not missing_cli else f"missing launcher fragments: {missing_cli}")

    pipeline_src = open(LAUNCHERS[1]).read()
    audit_fields = ("gated_compressed_rope", "gated_rope_start_layer")
    missing_audit = [field for field in audit_fields if field not in pipeline_src]
    check("R9_pipeline_audits_gated_eval_geometry", not missing_audit,
          "final/smoke JSONs audit the restored gated geometry"
          if not missing_audit else f"missing audit fields: {missing_audit}")

    return summarize(results)


def test_recipe_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
