"""Built-in demo examples (sRGB + optional ground-truth RAW).

Examples live in ``examples/`` next to a ``manifest.json`` that records, for each
one, the camera, Bayer pattern, white balance, colour matrix and EXIF values —
so the app can run inference with the instance's *real* parameters and compare
against the true RAW.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import PIL.Image

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
MANIFEST = EXAMPLES_DIR / "manifest.json"


def list_examples() -> list[dict]:
    """Return the manifest entries (empty list if no examples are shipped)."""
    if not MANIFEST.exists():
        return []
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8")).get("examples", [])
    except (OSError, ValueError):
        return []


def example_labels() -> list[str]:
    return [e["name"] for e in list_examples()]


def load_example(name: str) -> dict | None:
    """Load one example: sRGB, optional ground-truth RAW, and its parameters."""
    entry = next((e for e in list_examples() if e["name"] == name), None)
    if entry is None:
        return None

    srgb = np.asarray(
        PIL.Image.open(EXAMPLES_DIR / entry["srgb"]).convert("RGB"), dtype=np.float32
    ) / 255.0

    raw = None
    if entry.get("raw"):
        raw = load_raw(EXAMPLES_DIR / entry["raw"])

    return {
        "name": entry["name"],
        "camera": entry["camera"],
        "bayer_pattern": entry.get("bayer_pattern", "RGGB"),
        "white_balance": tuple(entry.get("white_balance", (2.0, 1.0, 1.5))),
        "exif": entry.get("exif") or {},
        "color_matrix": (
            np.asarray(entry["color_matrix"], dtype=np.float32) if entry.get("color_matrix") else None
        ),
        "srgb": srgb,
        "raw": raw,
        "has_gt": raw is not None,
        "note": entry.get("note", ""),
    }


def load_raw(path: str | Path) -> np.ndarray | None:
    """Load a single-channel RAW mosaic as a ``(H, W)`` float array in ``[0, 1]``."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
    elif suffix in (".png", ".jpg", ".jpeg"):
        arr = np.asarray(PIL.Image.open(path))
    else:
        return None

    arr = np.asarray(arr).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    maxv = float(arr.max()) if arr.size else 0.0
    if maxv > 1.5:
        arr = arr / (65535.0 if maxv > 255.0 else 255.0)
    return np.clip(arr, 0.0, 1.0)
