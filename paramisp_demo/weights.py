"""Locate (and optionally download) the ParamISP checkpoints.

Checkpoints are searched, in order, from:

1. the local ``weights/`` directory (``weights/<mode>/<camera>.ckpt``);
2. a Hugging Face Hub repo given by ``PARAMISP_WEIGHTS_REPO`` (e.g.
   ``"my-user/paramisp-weights"``), downloading ``<mode>/<camera>.ckpt``;
3. the Google Drive folder shared by the authors, given by
   ``PARAMISP_GDRIVE_ID`` (defaults to the official folder), downloaded with
   ``gdown``.

``mode`` is ``"inverse"`` (sRGB -> RAW) or ``"forward"`` (RAW -> sRGB).
"""
from __future__ import annotations

import os
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# Folder shared in the official README (dataset example + official weights), and
# the id of the ``weights.zip`` inside it.
DEFAULT_GDRIVE_ID = "1L7ZSikYt5eaCOEINpjjwmPX0CoZLbBG6"
DEFAULT_GDRIVE_ZIP_ID = "1p6I4r6Nz8vewBu_rorhYnJPdVRsNxjex"


def weights_dir() -> Path:
    return Path(os.environ.get("PARAMISP_WEIGHTS_DIR", WEIGHTS_DIR))


def expected_path(mode: str, camera: str) -> Path:
    return weights_dir() / mode / f"{camera}.ckpt"


def is_available(mode: str, camera: str) -> bool:
    return expected_path(mode, camera).exists()


def _download_from_hub(mode: str, camera: str) -> Path | None:
    repo = os.environ.get("PARAMISP_WEIGHTS_REPO")
    if not repo:
        return None
    try:
        from huggingface_hub import hf_hub_download

        try:
            cached = hf_hub_download(repo_id=repo, filename=f"{mode}/{camera}.ckpt")
        except Exception:
            cached = hf_hub_download(repo_id=repo, filename=f"{mode}/{camera}.pth")
        target = expected_path(mode, camera)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(Path(cached).read_bytes())
        return target if target.exists() else None
    except Exception as exc:  # noqa: BLE001
        print(f"[weights] Hub download failed: {exc}")
        return None


def _download_from_gdrive() -> None:
    """Download ``weights.zip`` from the authors' Google Drive and extract it.

    The zip contains ``weights/fine_tuning/{inverse,forward}/<camera>.ckpt``;
    each file is written to ``weights/<mode>/<camera>.ckpt``.
    """
    try:
        import gdown
        import zipfile
    except ImportError:
        print("[weights] gdown/zipfile unavailable; skipping Google Drive download.")
        return

    dest = weights_dir()
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "weights.zip"
    zip_id = os.environ.get("PARAMISP_GDRIVE_ZIP_ID", DEFAULT_GDRIVE_ZIP_ID)

    try:
        if not zip_path.exists():
            print(f"[weights] Downloading weights.zip (id {zip_id}) from Google Drive ...")
            gdown.download(id=zip_id, output=str(zip_path), quiet=False, use_cookies=False)

        if not zip_path.exists():
            print("[weights] Download did not produce weights.zip.")
            return

        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                parts = name.split("/")
                # .../fine_tuning/<mode>/<camera>.ckpt
                if name.endswith(".ckpt") and len(parts) >= 2:
                    mode, filename = parts[-2], parts[-1]
                    if mode not in ("inverse", "forward"):
                        continue
                    target = dest / mode / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as out:
                        out.write(src.read())
                    print(f"[weights] extracted {mode}/{filename}")
    except Exception as exc:  # noqa: BLE001
        print(f"[weights] Google Drive download failed: {exc}")


def ensure_weights(mode: str, camera: str) -> Path:
    """Return a path to the checkpoint, downloading it if necessary."""
    path = expected_path(mode, camera)
    if path.exists():
        return path

    hub_path = _download_from_hub(mode, camera)
    if hub_path is not None and hub_path.exists():
        return hub_path

    if os.environ.get("PARAMISP_SKIP_GDRIVE") != "1":
        _download_from_gdrive()
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Checkpoint '{mode}/{camera}.ckpt' was not found in {weights_dir()}. "
        "Place the official weights there, set PARAMISP_WEIGHTS_REPO to a "
        "Hugging Face repo, or set PARAMISP_GDRIVE_ID to the shared folder."
    )


def list_available() -> dict[str, list[str]]:
    """Return the cameras for which a checkpoint exists, per mode."""
    result = {"inverse": [], "forward": []}
    for mode in result:
        mdir = weights_dir() / mode
        if mdir.is_dir():
            result[mode] = sorted(p.stem for p in mdir.glob("*.ckpt"))
    return result
