#!/usr/bin/env python3
"""Named, versioned training recipes for zip2zip-core.

WHY THIS EXISTS. The canonical recipe is five environment variables and NONE of
them is a code default — each lever was deliberately left off so the frozen
v0.1-v0.6.x baselines stay bit-reproducible. The cost of that choice was that
the recipe lived only in shell history and in docs/finetuning.md: five flags to
retype on every launch, and one forgotten flag silently trains the wrong thing
for hours. A recipe NAME fixes this without touching a single code default, and
keeps historical lever sets selectable by name instead of by archaeology.

    RECIPE=v0.6.4   # -> the five flags below, all at once

USAGE FROM A BASH LAUNCHER:

    RECIPE=${RECIPE:-}
    if [ -n "$RECIPE" ]; then
        _env=$(python scripts/recipes.py "$RECIPE") || exit 1
        eval "$_env"
    fi

Each entry is emitted as `: "${KEY=value}"` followed by `export KEY`, which has
two properties the launchers rely on:

  * an explicitly-set variable always WINS — including an explicit empty value
    or `0`, both of which mean "off" in the launcher convention. So a
    single-variable experiment stays trivial:
        RECIPE=v0.6.4 ENCODER_N_LAYERS=4
  * resolving is IDEMPOTENT, so it is safe for the pipeline to resolve and then
    for the finetune launcher it calls to resolve again.

CLI:
    python scripts/recipes.py v0.6.4           # shell exports (for eval)
    python scripts/recipes.py --list           # every recipe + status
    python scripts/recipes.py --show v0.6.4    # fully resolved, human readable
    python scripts/recipes.py --identify DIR   # which named lever set it matches

ADDING A RECIPE. Add one entry whose `extends` names its parent and whose `env`
holds ONLY the keys that change. That is this project's versioning convention
(vX.Y = parent + exactly one change) expressed in code, so this file doubles as
the experiment ledger: you can read what v0.6.4 is by reading four short blocks.

Stdlib only — the launchers must be able to run this with whatever `python` is
on PATH, before any venv exists. (`--identify` imports torch lazily; no launcher
path touches it.)
"""
import os
import re
import sys

# The recipe this project trains with today. Single source of truth: docs and
# tests both read it from here.
CURRENT = "v0.6.4"

# Every env var a recipe is allowed to set, with the train.py argument it ends up
# as in meta.pt, how to parse the env value, and the value that means "off".
#   flag  -> "1"/anything-but-0-or-empty is True
#   Recipes may ONLY use keys in this table, so a typo like ZEROINIT_ENCODER
#   fails at load time instead of silently doing nothing (the exact class of bug
#   that cost this project six experiments).
ENV_KEYS = {
    # env var                  train.py arg                kind     off
    "DISABLE_DIGIT_IDS":        ("disable_digit_ids",        "flag",  False),
    "UNTIED_HYPER_ENCODER":     ("untied_hyper_encoder",     "flag",  False),
    "SHARE_HYPER_ENCODER_WEIGHTS": ("share_hyper_encoder_weights", "flag", False),
    "HYPER_ENCODER_TYPE":       ("hyper_encoder_type",       "str",   "flat"),
    "BASE_TOKEN_POSITIONS":     ("base_token_positions",     "flag",  False),
    "ZERO_INIT_ENCODER_OUTPUT": ("zero_init_encoder_output", "flag",  False),
    "NO_ENCODER_RESIDUAL":      ("no_encoder_residual",      "flag",  False),
    "ONLINE_CODEBOOK_MASK":     ("online_codebook_mask",     "flag",  False),
    "TOKEN_TYPE_LOSS_WEIGHT":   ("token_type_loss_weight",   "float", 0.0),
    "WARMSTART_STEPS":          ("warmstart_steps",          "int",   0),
    # off = the Phi3.5-mini config default, the only model config this line uses
    "ENCODER_N_LAYERS":         ("encoder_n_layers",         "int",   2),
}

# status: current | candidate | superseded | negative (a measured negative result)
RECIPES = {
    "v0.2": {
        "extends": None,
        "status": "superseded",
        "description": "EOS inside the loss mask; no compression levers yet",
        "data": "$SCRATCH/datasets/phi-1B-sft-8shards-eosfix",
        "env": {},
    },
    "v0.3": {
        "extends": "v0.2",
        "status": "superseded",
        "description": "math docs re-rendered as multi-turn chat (data-only change)",
        "data": "$SCRATCH/datasets/phi-1B-sft-8shards-mathchat",
        "env": {},
        "notes": "differs from v0.2 ONLY by DATA_DIR; it did not pay off",
    },
    "v0.4": {
        "extends": "v0.2",
        "status": "superseded",
        "description": "digit-protected compression: digits never LZW-merge",
        "env": {"DISABLE_DIGIT_IDS": "1"},
        "notes": "biggest single GSM8K jump of the line (+23pt)",
    },
    "v0.5": {
        "extends": "v0.4",
        "status": "superseded",
        "description": "untied hyper-encoder (separate output-role encoder on lm_head)",
        "env": {"UNTIED_HYPER_ENCODER": "1"},
        "notes": "matches the released model's architecture",
    },
    "v0.6": {
        "extends": "v0.5",
        "status": "negative",
        "description": "phased warm-start: decoder-LoRA frozen for the first N steps",
        "env": {"WARMSTART_STEPS": "200"},
        "notes": "GSM8K -6.6pt. Its premise (early transient damages the LoRA) was "
                 "wrong; the transient was the v0.6.4 init bug. Do not revisit.",
    },
    "v0.6.1": {
        "extends": "v0.5",
        "status": "negative",
        "description": "deeper hyper-encoder (2 -> 4 layers)",
        "env": {"ENCODER_N_LAYERS": "4"},
        "notes": "GSM8K -2.6pt; hurt both gap terms. Measured before the v0.6.4 "
                 "init fix, but not being revisited.",
    },
    "v0.6.2": {
        "extends": "v0.5",
        "status": "superseded",
        "description": "RoPE positions follow the uncompressed stream",
        "env": {"BASE_TOKEN_POSITIONS": "1"},
        "notes": "GSM8K +2.7pt; shrank the input term of the gap 6.5 -> 4.3pt",
    },
    "v0.6.3": {
        "extends": "v0.6.2",
        "status": "superseded",
        "description": "auxiliary base-vs-hyper token-type loss",
        "env": {"TOKEN_TYPE_LOSS_WEIGHT": "0.05"},
        "notes": "shrank the weights term 5.2 -> 3.1pt; recovered v0.6.2's MC tax",
    },
    "v0.6.4": {
        "extends": "v0.6.3",
        "status": "current",
        "description": "hyper-encoder starts at exactly zero (fixes a silent init no-op)",
        "env": {"ZERO_INIT_ENCODER_OUTPUT": "1"},
        "notes": "broadest balanced improvement of the line: GSM8K .652 -> .677, "
                 "MC average recovered, best ppl; OBQA/Wino changes within noise.",
    },
    "vx0.6.1": {
        "extends": "v0.6.4",
        "status": "candidate",
        "description": "output role re-encodes lm_head rows using shared hyper-encoder weights",
        "env": {"SHARE_HYPER_ENCODER_WEIGHTS": "1"},
        "notes": "Xinxian exploratory run: keeps v0.6.4's untied input/output roles, "
                 "but removes the second trained hyper_output module. Tests whether "
                 "role-specific hyper-encoders matter",
    },
    "vx0.6.2": {
        "extends": "v0.6.4",
        "status": "candidate",
        "description": "flat hyper-encoder without the residual path",
        "env": {"ZERO_INIT_ENCODER_OUTPUT": "0", "NO_ENCODER_RESIDUAL": "1"},
        "notes": "Xinxian exploratory run: removes the first-token residual from "
                 "the flat hyper-encoder. ZERO_INIT_ENCODER_OUTPUT is forced off "
                 "because train.py intentionally rejects zero-init without the residual.",
    },
    "vx0.6.3": {
        "extends": "v0.6.4",
        "status": "candidate",
        "description": "hierarchical hyper-encoder under the v0.6.4 recipe",
        "env": {"HYPER_ENCODER_TYPE": "hierarchical"},
        "notes": "Xinxian exploratory run: replaces the flat hyper-encoder with "
                 "the left-fold hierarchical composer while keeping every other "
                 "v0.6.4 lever fixed.",
    },
    "vx0.6.4": {
        "extends": "vx0.6.3",
        "status": "candidate",
        "description": "hierarchical hyper-encoder without the residual path",
        "env": {"ZERO_INIT_ENCODER_OUTPUT": "0", "NO_ENCODER_RESIDUAL": "1"},
        "notes": "Xinxian exploratory run: tests whether the first-token residual "
                 "interferes with the recurrent pattern learned by the hierarchical composer.",
    },
    "v0.6.5": {
        "extends": "v0.6.4",
        "status": "negative",
        "description": "decoder-time codebook availability during teacher forcing",
        "env": {"ONLINE_CODEBOOK_MASK": "1"},
        "notes": "MEASURED NEGATIVE (2026-07-26): removing the train/generation "
                 "vocabulary leak made the model WORSE on 6 of 7 metrics — GSM8K "
                 ".677 -> .653, ARC-c .570 -> .550, ARC-e .830 -> .816, HellaSwag "
                 ".723 -> .712 (scored under the legacy mask, like every earlier "
                 "version, so the comparison is like-for-like). The leak was real "
                 "and measurable but apparently acted as a regularizer. v0.6.4 "
                 "remains the standard. Do not revisit without a new argument.",
    },
}

_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._/+:@=-]*$")


class RecipeError(Exception):
    """Bad recipe name or malformed registry — always fatal, never a no-op."""


def _validate_registry():
    """Fail at import time on a typo or a broken parent link."""
    for name, spec in RECIPES.items():
        unknown = set(spec.get("env", {})) - set(ENV_KEYS)
        if unknown:
            raise RecipeError(
                f"recipe {name!r} sets unknown env var(s) {sorted(unknown)}; "
                f"add them to ENV_KEYS or fix the spelling"
            )
        parent = spec.get("extends")
        if parent is not None and parent not in RECIPES:
            raise RecipeError(f"recipe {name!r} extends unknown recipe {parent!r}")
    if CURRENT not in RECIPES:
        raise RecipeError(f"CURRENT={CURRENT!r} is not a recipe")
    if RECIPES[CURRENT].get("status") != "current":
        raise RecipeError(f"CURRENT={CURRENT!r} is not marked status=current")
    currents = [n for n, s in RECIPES.items() if s.get("status") == "current"]
    if currents != [CURRENT]:
        raise RecipeError(f"exactly one recipe may be status=current, found {currents}")


_validate_registry()


def lineage(name):
    """[oldest ancestor, ..., name] — the chain a recipe was built from."""
    if name not in RECIPES:
        raise RecipeError(
            f"unknown recipe {name!r}. Known: {', '.join(sorted(RECIPES))}"
        )
    chain, seen, cur = [], set(), name
    while cur is not None:
        if cur in seen:
            raise RecipeError(f"recipe inheritance cycle at {cur!r}")
        seen.add(cur)
        chain.append(cur)
        cur = RECIPES[cur].get("extends")
    return list(reversed(chain))


def resolve(name):
    """The full env a recipe means, parent values first."""
    env = {}
    for step in lineage(name):
        env.update(RECIPES[step].get("env", {}))
    for key, value in env.items():
        if not _SAFE_VALUE.match(str(value)):
            raise RecipeError(f"unsafe value for {key}: {value!r}")
    return env


def shell_exports(name):
    """Shell for a launcher to eval. Explicit env wins; idempotent."""
    lines = []
    for key, value in sorted(resolve(name).items()):
        # `${K=v}` (no colon) assigns only when K is UNSET, so an explicit empty
        # value or 0 — both "off" by launcher convention — is preserved.
        lines.append(f': "${{{key}={value}}}"')
        lines.append(f"export {key}")
    return "\n".join(lines)


def _as_arg_value(kind, raw):
    """Interpret an env value the way the launcher + train.py will."""
    if kind == "flag":
        return raw not in ("", "0")
    if kind == "float":
        return float(raw)
    if kind == "int":
        return int(raw)
    if kind == "str":
        return str(raw)
    raise RecipeError(f"unknown kind {kind!r}")


def expected_args(name):
    """The train.py args a recipe implies, over EVERY known key (unset = off)."""
    env = resolve(name)
    out = {}
    for key, (arg, kind, off) in ENV_KEYS.items():
        out[arg] = _as_arg_value(kind, env[key]) if key in env else off
    return out


_OFF_BY_ARG = {arg: off for (arg, _kind, off) in ENV_KEYS.values()}


def resolve_data(name):
    """The dataset a recipe implies, inherited from the nearest ancestor that
    declares one (informational: recipes never export DATA_DIR, since the path is
    cluster-specific)."""
    hint = None
    for step in lineage(name):
        if RECIPES[step].get("data"):
            hint = RECIPES[step]["data"]
    return hint


def _effective(arg, train_args):
    """What a checkpoint effectively used for one lever.

    A key MISSING from an old meta.pt means the lever did not exist when that run
    was trained, i.e. it was off — not "unknown". Treating it as unknown would let
    a v0.4 checkpoint match v0.5 (whose only lever, untied, postdates it).
    encoder_n_layers is also recorded as None by runs predating its normalization.
    """
    value = train_args.get(arg, None)
    return _OFF_BY_ARG[arg] if value is None else value


def identify(train_args):
    """Name the recipe a checkpoint was trained with.

    Returns (matches, diffs):
      * matches = every recipe whose levers match EXACTLY. Normally one. It is
        legitimately two when recipes differ only by dataset and no usable
        data_dir was recorded (v0.2 vs v0.3) — reported rather than guessed.
      * diffs   = {} whenever matches is non-empty; otherwise the closest
        recipe's arg -> (recipe_value, checkpoint_value).
    Only the levers in ENV_KEYS are compared: LR, budget and the rest are NOT.
    """
    exact, best, best_diffs = [], None, None
    for name in RECIPES:
        diffs = {
            arg: (want, train_args.get(arg, "<absent>"))
            for arg, want in expected_args(name).items()
            if _effective(arg, train_args) != want
        }
        if not diffs:
            exact.append(name)
        elif best_diffs is None or len(diffs) < len(best_diffs):
            best, best_diffs = name, diffs

    if len(exact) > 1:
        # Tie-break on the recorded dataset, the only thing separating them.
        recorded = os.path.basename(str(train_args.get("data_dir") or "").rstrip("/"))
        if recorded:
            narrowed = [n for n in exact
                        if os.path.basename(str(resolve_data(n) or "")) == recorded]
            if narrowed:
                exact = narrowed
    if exact:
        return exact, {}
    return ([best] if best else []), (best_diffs or {})


def _cmd_list():
    width = max(len(n) for n in RECIPES)
    for name in RECIPES:
        spec = RECIPES[name]
        mark = {
            "current": "*",
            "candidate": "+",
            "negative": "!",
            "superseded": " ",
        }[spec["status"]]
        print(f"{mark} {name:<{width}}  {spec['status']:<10}  {spec['description']}")
    print(
        f"\n* = current standard recipe ({CURRENT})   "
        "+ = unmeasured candidate   ! = measured negative result"
    )


def _cmd_show(name):
    spec = RECIPES[name]
    print(f"recipe:      {name}   [{spec['status']}]")
    print(f"description: {spec['description']}")
    print(f"lineage:     {' -> '.join(lineage(name))}")
    if spec.get("data"):
        print(f"data:        {spec['data']}   (informational — set DATA_DIR yourself)")
    if spec.get("notes"):
        print(f"notes:       {spec['notes']}")
    print("\nresolved environment:")
    env = resolve(name)
    if not env:
        print("  (no levers — launcher defaults)")
    for key, value in sorted(env.items()):
        owner = next(s for s in reversed(lineage(name)) if key in RECIPES[s].get("env", {}))
        print(f"  {key}={value}".ljust(46) + f"# from {owner}")


def _cmd_identify(ckpt_dir):
    import torch  # lazy: no launcher path needs torch

    meta = os.path.join(ckpt_dir, "meta.pt")
    if not os.path.exists(meta):
        raise RecipeError(f"no meta.pt in {ckpt_dir}")
    args = torch.load(meta, map_location="cpu", weights_only=False).get("args", {}) or {}
    matches, diffs = identify(args)
    if matches and not diffs:
        if len(matches) == 1:
            name = matches[0]
            print(f"{name}" + ("  (current)" if name == CURRENT else ""))
        else:
            print(" or ".join(matches)
                  + "  (identical levers; they differ only by dataset, and this"
                    " checkpoint's data_dir does not match either hint)")
        return
    closest = matches[0] if matches else "?"
    print(f"custom — closest is {closest}, differing in:")
    for arg, (want, got) in sorted(diffs.items()):
        print(f"  {arg}: recipe {want!r} vs checkpoint {got!r}")


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    if argv[0] == "--list":
        _cmd_list()
        return 0
    if argv[0] == "--show":
        if len(argv) < 2:
            raise RecipeError("--show needs a recipe name")
        lineage(argv[1])  # validates
        _cmd_show(argv[1])
        return 0
    if argv[0] == "--identify":
        if len(argv) < 2:
            raise RecipeError("--identify needs a checkpoint dir")
        _cmd_identify(argv[1])
        return 0

    name = argv[0]
    out = shell_exports(name)          # raises on an unknown name
    status = RECIPES[name]["status"]
    if status == "negative":
        print(f"[recipe] WARNING: {name} is a MEASURED NEGATIVE RESULT "
              f"({RECIPES[name]['notes']})", file=sys.stderr)
    elif status == "candidate":
        print(f"[recipe] note: {name} is an UNMEASURED CANDIDATE; the current "
              f"standard recipe remains {CURRENT}", file=sys.stderr)
    elif name != CURRENT:
        print(f"[recipe] note: {name} is superseded; the current standard recipe "
              f"is {CURRENT}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except RecipeError as exc:
        print(f"[recipe] FATAL: {exc}", file=sys.stderr)
        sys.exit(2)
