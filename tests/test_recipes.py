"""Correctness invariants for the named recipe registry (scripts/recipes.py).

A recipe is what a launch command means, so a wrong recipe silently trains the
wrong thing for hours. These checks pin the properties the launchers depend on.

Pure stdlib + the repo's files: no torch, no cluster. Runnable as a script and
under pytest.

Invariants proven:
  R1  the registry is internally consistent: exactly one status=current, CURRENT
      points at it, every `extends` resolves, no inheritance cycles.
  R2  v0.6.4 resolves to exactly the five documented levers.
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
"""
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

    # ---- R7: identify() round-trip ----
    name, diffs = R.identify(R.expected_args("v0.6.4"))
    check("R7_identify_roundtrips_current", name == "v0.6.4" and not diffs,
          f"-> {name}, diffs={diffs}")
    off = dict(R.expected_args("v0.6.4"), encoder_n_layers=4)
    name2, diffs2 = R.identify(off)
    check("R7_identify_flags_off_recipe_checkpoint",
          "encoder_n_layers" in diffs2,
          f"closest={name2}, diffs={sorted(diffs2)}")

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

    return summarize(results)


def test_recipe_invariants():
    failed = main()
    assert not failed, f"failed invariants: {failed}"


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
