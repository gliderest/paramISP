"""Extract small demo examples (sRGB + ground-truth RAW) from the ParamISP dataset.

The official dataset is a ~2.7 GB ``patchsets.zip`` on Google Drive.  This script
reads a few instances *directly from that archive* (without extracting it) and
writes them to ``examples/``, together with a ``manifest.json`` the Streamlit
demo reads.  Normalisation reuses the vendored upstream code, so the stored RAW
matches what the official pipeline feeds the network.

Usage::

    python scripts/fetch_examples.py                 # download the archive, then extract
    python scripts/fetch_examples.py --zip PATH      # use a local archive
    python scripts/fetch_examples.py --per-camera 2  # more examples per camera
    python scripts/fetch_examples.py --list          # just report the archive contents
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "paramisp_src"))

import PIL.Image  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import utils.camera  # noqa: E402
import utils.convert  # noqa: E402

EXAMPLES_DIR = ROOT / "examples"
PATCHSET_ZIP_ID = "12jiX9fR166CdHHyFogGuh86SKsHvsQ5U"  # patchsets.zip in the authors' Drive folder

# dataset camera_name -> short demo name
CAMERA_SHORT = {
    "SONY/ILCE-7RM3": "A7R3",
    "NIKON CORPORATION/NIKON D7000": "D7000",
    "NIKON CORPORATION/NIKON D90": "D90",
    "NIKON CORPORATION/NIKON D40": "D40",
    "SAMSUNG/SM-G935F": "S7",
}


def find_instances(zf: zipfile.ZipFile) -> list[tuple[str, dict[str, str]]]:
    """Group the archive members into instance folders that hold a full sRGB/RAW pair."""
    by_dir: dict[str, dict[str, str]] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        folder, _, filename = name.rpartition("/")
        if folder:
            by_dir.setdefault(folder, {})[filename] = name

    instances = []
    for folder, files in by_dir.items():
        has_rgb = any(k.startswith("rgb-") and k.endswith(".tif") for k in files)
        has_raw = any(k.startswith("raw-") and k.endswith(".tif") for k in files)
        if "metadata.pt" in files and has_rgb and has_raw:
            instances.append((folder, files))
    return sorted(instances)


def read_metadata(zf: zipfile.ZipFile, files: dict[str, str]) -> dict:
    return torch.load(io.BytesIO(zf.read(files["metadata.pt"])), map_location="cpu", weights_only=False)


def read_exif(zf: zipfile.ZipFile, files: dict[str, str]) -> dict | None:
    if "extra.yml" not in files:
        return None
    import yaml
    extra = yaml.safe_load(zf.read(files["extra.yml"]))
    try:
        return {
            "focal_length": float(Fraction(str(extra["focal_length"]))),
            "f_number": float(Fraction(str(extra["f_number"]))),
            "exposure_time": float(Fraction(str(extra["exposure_time"]))),
            "iso_sensitivity": float(extra["iso_sensitivity"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def build_example(zf: zipfile.ZipFile, folder: str, files: dict[str, str], index: int) -> tuple[dict, dict]:
    """Return ``(manifest_entry, files_to_write)`` for one dataset instance."""
    meta = read_metadata(zf, files)
    bayer = np.asarray(meta["bayer_pattern"])
    black_level = np.asarray(meta["black_level"])
    white_level = meta["white_level"]
    white_balance = np.asarray(meta["white_balance"], dtype=np.float32)
    color_matrix = np.asarray(meta["camera_matrix"], dtype=np.float32)
    camera_name = str(meta["camera_name"])
    short = CAMERA_SHORT.get(camera_name, camera_name.split("/")[-1])

    rgb_name = sorted(k for k in files if k.startswith("rgb-") and k.endswith(".tif"))[0]
    raw_name = sorted(k for k in files if k.startswith("raw-") and k.endswith(".tif"))[0]
    rgb = tifffile.imread(io.BytesIO(zf.read(files[rgb_name])))
    raw = tifffile.imread(io.BytesIO(zf.read(files[raw_name])))

    # normalise exactly like the upstream pipeline
    rgb_f = np.asarray(utils.convert.dequantize(rgb, augmentation=False), dtype=np.float32)
    raw_f = np.asarray(
        utils.convert.dequantize(raw, in_range=(black_level, white_level), bayer_pattern=bayer),
        dtype=np.float32,
    )
    wb_n = np.asarray(utils.camera.normalize_whitebalance(white_balance), dtype=np.float32)

    stem = f"{short}_{index:02d}"
    srgb_rel = f"{stem}_srgb.png"
    raw_rel = f"{stem}_raw.tif"

    entry = {
        "name": f"{short} — dataset instance {index:02d}",
        "camera": short,
        "bayer_pattern": utils.camera.BayerPatternFlag.from_numpy(bayer).name,
        "white_balance": [round(float(v), 6) for v in wb_n],
        "color_matrix": [[round(float(v), 8) for v in row] for row in color_matrix],
        "exif": read_exif(zf, files),
        "srgb": srgb_rel,
        "raw": raw_rel,
        "note": f"From the official ParamISP dataset ({camera_name}).",
    }
    writes = {
        srgb_rel: ("png", np.clip(rgb_f, 0.0, 1.0)),
        raw_rel: ("tif", np.clip(raw_f, 0.0, 1.0)),
    }
    return entry, writes


def write_files(writes: dict) -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for rel, (kind, arr) in writes.items():
        path = EXAMPLES_DIR / rel
        if kind == "png":
            PIL.Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(path)
        else:
            tifffile.imwrite(str(path), (arr * 65535.0 + 0.5).astype(np.uint16))
        print(f"  wrote examples/{rel}")


def ensure_zip(path: Path) -> Path:
    if path.exists():
        return path
    import gdown
    print(f"Downloading patchsets.zip (~2.7 GB) to {path} ...")
    gdown.download(id=PATCHSET_ZIP_ID, output=str(path), quiet=False, use_cookies=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", default=str(ROOT / "patchsets.zip"), help="path to a local patchsets.zip")
    ap.add_argument("--per-camera", type=int, default=1, help="number of examples per camera")
    ap.add_argument("--list", action="store_true", help="only report the archive contents")
    args = ap.parse_args()

    zf_path = ensure_zip(Path(args.zip))
    with zipfile.ZipFile(zf_path) as zf:
        instances = find_instances(zf)
        print(f"{len(instances)} instance folders found in the archive")
        if not instances:
            return

        if args.list:
            counts: dict[str, int] = {}
            for _folder, files in instances:
                name = str(read_metadata(zf, files)["camera_name"])
                counts[name] = counts.get(name, 0) + 1
            for name, n in sorted(counts.items()):
                print(f"  {name}: {n}")
            return

        chosen = []
        per: dict[str, int] = {}
        for folder, files in instances:
            name = str(read_metadata(zf, files)["camera_name"])
            short = CAMERA_SHORT.get(name, name.split("/")[-1])
            if per.get(short, 0) >= args.per_camera:
                continue
            per[short] = per.get(short, 0) + 1
            chosen.append((short, folder, files, per[short] - 1))

        entries = []
        for short, folder, files, idx in chosen:
            print(f"- {short} instance {idx:02d}  ({folder})")
            entry, writes = build_example(zf, folder, files, idx)
            write_files(writes)
            entries.append(entry)

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (EXAMPLES_DIR / "manifest.json").write_text(
        json.dumps({"examples": entries}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote examples/manifest.json with {len(entries)} example(s)")


if __name__ == "__main__":
    main()
