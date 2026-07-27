"""Generate HuggingFace model card (README.md) for zip2zip-core checkpoints."""

import os


def write_model_card(path: str, repo_id: str, step: int, train_args: dict):
    """Write a HuggingFace model card README.md.

    Args:
        path: Output file path for the README.md
        repo_id: HuggingFace repo ID (e.g. epfl-dlab/candidate-xxx)
        step: Training step number
        train_args: Training arguments dict (from meta.pt)
    """
    model_config = train_args.get("model_config", "unknown")
    max_subtokens = train_args.get("max_subtokens", "?")
    max_codebook_size = train_args.get("max_codebook_size", "?")
    init_from = train_args.get("init_from_hf") or train_args.get("resume_from") or "scratch"
    data_dir = train_args.get("data_dir", "?")
    seq_len = train_args.get("seq_len", "?")
    lr = train_args.get("lr", "?")
    max_tokens = train_args.get("max_tokens", "?")
    gated_rope = bool(train_args.get("gated_compressed_rope", False))
    gated_rope_start = train_args.get("gated_rope_start_layer", 0)
    gated_rope_pair = train_args.get("gated_rope_start_pair")
    if gated_rope_pair is None:
        # Pre-suffix gated checkpoints covered every pair and did not record
        # this field.  Keep generated cards truthful for that legacy format.
        gated_rope_pair = 0
    base_replay = train_args.get("base_view_replay_prob", 0.0)

    lines = [
        "---",
        "tags:",
        "  - llaza",
        "  - zip2zip",
        "  - adaptive-tokenization",
        "library_name: zip2zip",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        "Training checkpoint from [zip2zip-core](https://github.com/epfl-dlab/zip2zip-core).",
        "This is a **candidate** model (not production-ready).",
        "",
        "## Training Config",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| model_config | `{model_config}` |",
        f"| init_from | `{init_from}` |",
        f"| max_subtokens | {max_subtokens} |",
        f"| max_codebook_size | {max_codebook_size} |",
        f"| seq_len | {seq_len} |",
        f"| lr | {lr} |",
        f"| max_tokens | {max_tokens} |",
        f"| gated_compressed_rope | {gated_rope} |",
        f"| gated_rope_start_layer | {gated_rope_start} |",
        f"| gated_rope_start_pair | {gated_rope_pair} |",
        f"| base_view_replay_prob | {base_replay} |",
        f"| step | {step} |",
        f"| data | `{os.path.basename(str(data_dir))}` |",
        "",
        "## Usage",
        "",
        "This is a **training checkpoint** (torchtitan format).",
        "",
    ]
    if gated_rope:
        lines.extend(
            [
                "This checkpoint uses gated compressed-coordinate RoPE, which "
                "the external HuggingFace runtime does not implement. Use "
                "`scripts/inference.py` or the in-core evaluation adapter; "
                "export is intentionally refused.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "To use for inference, export to HuggingFace format first:",
                "",
                "```bash",
                "python scripts/zip2zip_hf/export_to_zip2zip.py \\",
                f"    --ckpt_dir <local_path>/step_{step} \\",
                "    --output_dir <export_dir> \\",
                f"    --base_model {init_from} \\",
                f"    --model_config {model_config}",
                "```",
                "",
            ]
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))
