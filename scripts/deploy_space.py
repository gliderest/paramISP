"""Create the Hugging Face Space and upload this project to it.

Prerequisites::

    huggingface-cli login --token <WRITE_TOKEN>

Usage::

    # Free Static Space (serves index.html; no hardware needed)
    python scripts/deploy_space.py <user>/<space-name> --static

    # Gradio Space on ZeroGPU / paid hardware
    python scripts/deploy_space.py <user>/<space-name> [--hardware zero-a10g]

Local-only files (``.venv``, ``weights``, caches, ``scripts``) are skipped; the
demo fetches the checkpoints at runtime instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

IGNORE = [
    ".venv/*", "weights/*", "weights_tmp/*", "weights_gdrive/*",
    "__pycache__/*", "**/__pycache__/*", ".git/*", ".gitattributes",
    "scripts/*", ".vscode/*", "assets/_*", "*.zip",
]


def _static_readme(text: str) -> str:
    """Return README text with the front matter switched to the Static SDK."""
    out = []
    for line in text.splitlines():
        if line.startswith("sdk:") and "sdk_version" not in line:
            out.append("sdk: static")
        elif line.startswith("app_file:") or line.startswith("sdk_version:"):
            continue
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="e.g. my-user/paramISP")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hardware", default="zero-a10g",
                        help="Space hardware flavor (default: zero-a10g/ZeroGPU, "
                             "required on free accounts; use cpu-basic on Pro).")
    parser.add_argument("--static", action="store_true",
                        help="Create a free Static Space instead (serves index.html; "
                             "no hardware). The SDK can be switched to 'gradio' later.")
    args = parser.parse_args()

    if "/" not in args.repo_id:
        raise SystemExit("repo_id must be of the form <user-or-org>/<space-name>")

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Not logged in — run 'huggingface-cli login' first ({exc})")

    if args.static:
        print(f"Creating Static Space {args.repo_id} ...")
        api.create_repo(repo_id=args.repo_id, repo_type="space",
                        space_sdk="static", private=args.private, exist_ok=True)
        print("Uploading files (Static Space serves index.html) ...")
        api.upload_folder(
            repo_id=args.repo_id, repo_type="space", folder_path=str(ROOT),
            ignore_patterns=IGNORE + ["README.md"],
            commit_message="Add ParamISP static demo",
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        api.upload_file(
            repo_id=args.repo_id, repo_type="space",
            path_or_fileobj=_static_readme(readme).encode("utf-8"),
            path_in_repo="README.md",
            commit_message="Set Space SDK to static",
        )
        print(f"\nDone. Space: https://huggingface.co/spaces/{args.repo_id}")
        print("The full Gradio app is in the repo; to switch it on once you have "
              "hardware, set 'sdk: gradio' (plus sdk_version/app_file) in the README.")
        return

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
