"""Upload the local checkpoints to a Hugging Face repo for the Space to fetch.

Usage::

    hf auth login          # or: huggingface-cli login
    python scripts/publish_weights.py <user>/<repo-name>

The files are uploaded as ``inverse/<camera>.ckpt`` and ``forward/<camera>.ckpt``.
Then set the Space variable ``PARAMISP_WEIGHTS_REPO=<user>/<repo-name>`` so the
demo downloads them from the Hub instead of Google Drive.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        print(__doc__)
        raise SystemExit(1)

    repo_id = sys.argv[1]
    if not WEIGHTS.is_dir():
        raise SystemExit(f"no weights directory at {WEIGHTS}")

    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", exist_ok=True)
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(WEIGHTS),
        path_in_repo="",
        allow_patterns=["inverse/*.ckpt", "forward/*.ckpt"],
    )
    print(f"Uploaded checkpoints to https://huggingface.co/{repo_id}")
    print(f"Now set the Space variable PARAMISP_WEIGHTS_REPO={repo_id}")


if __name__ == "__main__":
    main()
