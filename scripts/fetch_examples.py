"""Extract small demo examples (sRGB + ground-truth RAW) from the ParamISP dataset.

The official dataset is a ~2.7 GB ``patchsets.zip`` on Google Drive.  This script
reads a few instances *directly from that archive* (without extracting it) and
writes them to ``examples/``, together with a ``manifest.json`` the Streamlit
demo reads.  Normalisation reuses the vendored upstream code, so the stored RAW
matches what the official pipeline feeds the network.

Examples are picked to be *representative*: each instance is stored as many
512x512 patches, and the first patch (top-left corner) is often blown-out sky or
a dark corner.  Both extremes distort the absolute PSNR/SSIM of the demo, so for
every camera we ship the instance/patch whose RAW brightness is closest to that
camera's median patch brightness (ties broken towards higher contrast).

Usage::

    python scripts/fetch_examples.py                 # download the archive, then extract
    python scripts/fetch_examples.py --zip PATH      # use a local archive
    python scripts/fetch_examples.py --per-camera 2  # more examples per camera
    python scripts/fetch_examples.py --list          # just report the archive contents
    python scripts/fetch_examples.py --patch corner  # old behaviour (top-left patch)
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
    "SM-G935F": "S7",
}


def short_name(camera_name: str) -> str:
    """Map a dataset camera name to the demo's short name (case-insensitive)."""
    return (CAMERA_SHORT.get(camera_name)
            or CAMERA_SHORT.get(camera_name.upper())
            or camera_name.split("/")[-1])


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


def patch_stats(zf: zipfile.ZipFile, files: dict[str, str]) -> list[tuple[str, float, float]]:
    """Return ``(raw_name, mean, std)`` for every sRGB/RAW patch pair of an instance.

    The RAW digital numbers are enough to rank brightness within one camera
    (black/white levels are constant per camera), so nothing is normalised here.
    """
    stats = []
    for raw_name in sorted(k for k in files if k.startswith("raw-") and k.endswith(".tif")):
        if raw_name.replace("raw-", "rgb-", 1) not in files:
            continue
        arr = tifffile.imread(io.BytesIO(zf.read(files[raw_name])))
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        stats.append((raw_name, float(arr.mean()), float(arr.std())))
    return stats


def select_examples(per_cam: dict[str, list], per_camera: int, patch_mode: str):
    """Pick ``per_camera`` representative instances per camera.

    ``per_cam`` maps a short camera name to ``[(folder, files, stats), ...]``.
    With ``patch_mode="median"`` every camera's candidate patches are ranked by
    how close their RAW brightness is to that camera's median patch brightness
    (ties towards higher contrast); with ``"corner"`` the historical "first
    patch" behaviour is kept.
    """
    chosen: list[tuple[str, str, dict, str, int]] = []
    for short in sorted(per_cam):
        instances = per_cam[short]
        means = [m for _f, _files, stats in instances for _rn, m, _s in stats]
        if not means:
            continue
        target = float(np.median(means))
        ranked = []
        for folder, files, stats in instances:
            if patch_mode == "corner":
                raw_name = sorted(stats)[0][0]
            else:
                raw_name = min(stats, key=lambda t: (abs(t[1] - target), -t[2]))[0]
            stats_by_name = {rn: (m, s) for rn, m, s in stats}
            mean, std = stats_by_name[raw_name]
            ranked.append((abs(mean - target), folder, files, raw_name, mean, std))
        ranked.sort(key=lambda t: t[0])
        for index, (_dist, folder, files, raw_name, mean, std) in enumerate(ranked[:per_camera]):
            print(f"- {short} instance {index:02d}  ({folder})")
            print(f"    patch {raw_name}  RAW mean {mean:.1f} std {std:.1f} "
                  f"(camera median {target:.1f})")
            chosen.append((short, folder, files, raw_name, index))
    return chosen


def build_example(zf: zipfile.ZipFile, folder: str, files: dict[str, str], index: int,
                  raw_name: str | None = None) -> tuple[dict, dict]:
    """Return ``(manifest_entry, files_to_write)`` for one dataset instance."""
    meta = read_metadata(zf, files)
    bayer = np.asarray(meta["bayer_pattern"])
    black_level = np.asarray(meta["black_level"])
    white_level = meta["white_level"]
    white_balance = np.asarray(meta["white_balance"], dtype=np.float32)
    color_matrix = np.asarray(meta["camera_matrix"], dtype=np.float32)
    camera_name = str(meta["camera_name"])
    short = short_name(camera_name)

    if raw_name is None:
        raw_name = sorted(k for k in files if k.startswith("raw-") and k.endswith(".tif"))[0]
    rgb_name = raw_name.replace("raw-", "rgb-", 1)
    if rgb_name not in files:
        rgb_name = sorted(k for k in files if k.startswith("rgb-") and k.endswith(".tif"))[0]
    rgb = tifffile.imread(io.BytesIO(zf.read(files[rgb_name])))
    raw = tifffile.imread(io.BytesIO(zf.read(files[raw_name])))
    if raw.ndim == 3 and raw.shape[-1] == 1:  # stored as (H, W, 1)
        raw = raw[..., 0]
    if rgb.ndim == 3 and rgb.shape[-1] == 1:
        rgb = rgb[..., 0]

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
    ap.add_argument("--patch", choices=("median", "corner"), default="median",
                    help="which patch of an instance to ship (default: median brightness)")
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

        # Scan every camera's patches so the shipped example can be the one with
        # a representative (median) exposure.
        per_cam: dict[str, list] = {}
        for folder, files in instances:
            name = str(read_metadata(zf, files)["camera_name"])
            short = short_name(name)
            stats = patch_stats(zf, files)
            if stats:
                per_cam.setdefault(short, []).append((folder, files, stats))

        chosen = select_examples(per_cam, args.per_camera, args.patch)

        entries = []
        for short, folder, files, raw_name, idx in chosen:
            entry, writes = build_example(zf, folder, files, idx, raw_name)
            write_files(writes)
            entries.append(entry)

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (EXAMPLES_DIR / "manifest.json").write_text(
        json.dumps({"examples": entries}, indent=2), encoding="utf-8"
    )
    print(f"\nWrote examples/manifest.json with {len(entries)} example(s)")


if __name__ == "__main__":
    main()
