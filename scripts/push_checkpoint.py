"""Export and optionally publish one of the four Zip2Zip++ checkpoints.

The Hugging Face repository layout is:

* ``main``: original zip2zip-core training checkpoint (resume/reproduction)
* ``hf``: self-contained, user-facing zip2zip inference weights

No upload happens unless ``--upload`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "ext" / "torchtitan"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir", "--ckpt_dir", dest="ckpt_dir", type=Path, required=True
    )
    parser.add_argument("--repo-id", "--repo_id", dest="repo_id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload main and hf revisions after export validation.",
    )
    args = parser.parse_args()

    from zip2zip_core.release import export_release, publish_release

    ckpt_dir = args.ckpt_dir.resolve()
    if args.output_dir is None and not args.upload:
        parser.error("pass --output-dir for a local export, or pass --upload")

    temporary_export = None
    if args.output_dir is None:
        temporary_export = tempfile.TemporaryDirectory(
            prefix="zip2zippp_release_"
        )
        export_dir = Path(temporary_export.name)
    else:
        export_dir = args.output_dir.resolve()

    try:
        step, train_args = export_release(
            ckpt_dir,
            export_dir,
            args.repo_id,
            max_shard_size=args.max_shard_size,
        )
        print(f"Validated Zip2Zip++ export: {export_dir}")
        if args.upload:
            publish_release(
                ckpt_dir,
                export_dir,
                args.repo_id,
                step,
                train_args,
            )
    finally:
        if temporary_export is not None:
            temporary_export.cleanup()


if __name__ == "__main__":
    main()
