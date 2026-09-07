"""Load eval presets from eval_presets.yaml.

Used as import by eval_harness.py / eval_hf_model.py, and as a script by
validate_phi35_rcp.sh to export shell variables.
"""
import argparse
import os

PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_presets.yaml")


def load_preset(name, preset_file=None):
    import yaml
    with open(preset_file or PRESET_FILE) as f:
        presets = yaml.safe_load(f).get("presets", {})
    if name not in presets:
        raise SystemExit(f"Unknown preset '{name}'. Available: {list(presets.keys())}")
    return presets[name]


def apply_preset(parser, default=None):
    """Extract --preset from argv, load YAML, apply as parser defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=default)
    pre.add_argument("--preset_file", default=None)
    ns, _ = pre.parse_known_args()
    if not ns.preset:
        return None
    preset = load_preset(ns.preset, ns.preset_file)
    defaults = {}
    for key, value in preset.items():
        if key == "description":
            continue
        if key == "tasks" and isinstance(value, list):
            defaults["tasks"] = ",".join(value)
        else:
            defaults[key] = value
    parser.set_defaults(**defaults)
    return ns.preset, preset.get("description", "")


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "default"
    p = load_preset(name)
    tasks = p.get("tasks", [])
    if isinstance(tasks, list):
        tasks = ",".join(tasks)
    print(f"TASKS={tasks}")
    # Empty when the preset has no global few-shot: callers must then omit
    # --num_fewshot so each task keeps its own protocol (postsft).
    nf = p.get("num_fewshot")
    print(f"NUM_FEWSHOT={'' if nf is None else nf}")
    print(f"BATCH_SIZE={p.get('batch_size', 1)}")
    print(f"MAX_LENGTH={p.get('max_length', 4096)}")
    if p.get("apply_chat_template"):
        print("CHAT_TEMPLATE_FLAG=--apply_chat_template")
    if p.get("fewshot_as_multiturn"):
        print("MULTITURN_FLAG=--fewshot_as_multiturn")
