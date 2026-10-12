"""Create the Hugging Face Space and upload this project to it.

Prerequisites::

    huggingface-cli login --token <WRITE_TOKEN>

Usage::

    python scripts/deploy_space.py <user>/<space-name> [--private]

The Space is created with the Gradio SDK and every file needed at runtime is
uploaded.  Local-only files (``.venv``, ``weights``, caches, ``scripts``) are
skipped; the demo fetches the checkpoints at runtime instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".venv/*", "weights/*", "weights_tmp/*", "weights_gdrive/*",
    "__pycache__/*", "**/__pycache__/*", ".git/*", ".gitattributes",
    "scripts/*", ".vscode/*", "assets/_*", "*.zip",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="e.g. my-user/paramISP")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hardware", default="zero-a10g",
                        help="Space hardware flavor (default: zero-a10g/ZeroGPU, "
                             "required on free accounts; use cpu-basic on Pro).")
    args = parser.parse_args()

    if "/" not in args.repo_id:
        raise SystemExit("repo_id must be of the form <user-or-org>/<space-name>")

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Not logged in — run 'huggingface-cli login' first ({exc})")

    print(f"Creating Space {args.repo_id} (SDK: gradio, hardware: {args.hardware}) ...")
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=args.hardware,
            private=args.private,
            exist_ok=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] create_repo with hardware failed ({exc}); retrying without it")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        try:
            api.request_space_hardware(repo_id=args.repo_id, hardware=args.hardware)
        except Exception as exc2:  # noqa: BLE001
            print(f"[warn] could not set hardware: {exc2}")

    print("Uploading files ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Add ParamISP demo Space",
    )
    print(f"\nDone. Space: https://huggingface.co/spaces/{args.repo_id}")
    print(
        "Note: on a free account, Gradio Spaces must run on ZeroGPU "
        "(the app uses @spaces.GPU when available)."
    )


if __name__ == "__main__":
    main()
