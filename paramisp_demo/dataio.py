"""Building the network input batch from an image + camera parameters."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parent.parent / "paramisp_src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import layers.bayer  # noqa: E402
import layers.color  # noqa: E402
import utils.camera  # noqa: E402
import data.utils  # noqa: E402

# Short name -> (full camera name used by the encoders, default Bayer pattern).
CAMERA_INFO = {
    "A7R3":  {"name": "SONY/ILCE-7RM3",                "bayer": "RGGB"},
    "D7000": {"name": "NIKON CORPORATION/NIKON D7000", "bayer": "RGGB"},
    "D90":   {"name": "NIKON CORPORATION/NIKON D90",   "bayer": "RGGB"},
    "D40":   {"name": "NIKON CORPORATION/NIKON D40",   "bayer": "RGGB"},
    "S7":    {"name": "SAMSUNG/SM-G935F",              "bayer": "RGGB"},
}
BAYER_CHOICES = ["RGGB", "GRBG", "GBRG", "BGGR"]

# Fallbacks used when the EXIF information is missing.  The upstream pipeline
# encodes "unknown" as 0, but the log/pow terms in the camera-parameter encoders
# blow up to ~1e12 for 0 (which the trained network cannot consume), so the demo
# uses plausible values instead.
DEFAULT_EXIF = {"focal_length": 50.0, "f_number": 8.0, "exposure_time": 0.01, "iso_sensitivity": 200.0}
DEFAULT_WHITE_BALANCE = (2.0, 1.0, 1.5)


def load_srgb(path: str | Path) -> np.ndarray:
    """Load an 8-bit sRGB image as an ``(H, W, 3)`` float array in ``[0, 1]``."""
    with PIL.Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def read_exif_params(path: str | Path) -> dict:
    """Read focal length / f-number / exposure time / ISO from an image's EXIF.

    Returns values of ``0`` for any tag that cannot be found, matching the
    upstream convention for "unknown" camera parameters.
    """
    params = dict(DEFAULT_EXIF)
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return params

    def _ratio(value) -> float:
        try:
            return float(Fraction(str(value)))
        except Exception:
            try:
                return float(value)
            except Exception:
                return 0.0

    if "EXIF FocalLength" in tags:
        params["focal_length"] = _ratio(tags["EXIF FocalLength"])
    if "EXIF FNumber" in tags:
        params["f_number"] = _ratio(tags["EXIF FNumber"])
    if "EXIF ExposureTime" in tags:
        params["exposure_time"] = _ratio(tags["EXIF ExposureTime"])
    for key in ("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity", "MakerNote ISOSetting"):
        if key in tags:
            try:
                params["iso_sensitivity"] = float(str(tags[key]).split(",")[0])
                break
            except Exception:
                pass
    return params


def estimate_white_balance(srgb: np.ndarray) -> tuple[float, float, float]:
    """Simple gray-world white-balance estimate from an sRGB image (green == 1)."""
    means = srgb.reshape(-1, 3).mean(axis=0)
    means = np.clip(means, 1e-6, None)
    wb = means.mean() / means
    wb = wb / wb[1]
    return tuple(float(v) for v in wb)


def load_raw_mosaic(path: str | Path) -> tuple[np.ndarray, str | None]:
    """Load a single-channel RAW mosaic from a RAW/`.npy`/`.tif` file.

    Returns ``(mosaic_float_in_0_1, bayer_hint)`` where ``bayer_hint`` is a Bayer
    pattern string when it can be inferred, else ``None``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".npy", ".npz"):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            arr = arr[arr.files[0]]
        return _normalize_mosaic(np.asarray(arr)), None

    if suffix in (".tif", ".tiff"):
        import tifffile
        arr = np.asarray(tifffile.imread(path))
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4) and arr.shape[0] > 4:
            arr = arr[..., 0]  # take the first channel if it is an interleaved RGB(A) tiff
        while arr.ndim > 2:
            arr = arr[..., 0]
        return _normalize_mosaic(arr), None

    # Fall back to rawpy for real camera RAW files (DNG/ARW/NEF/CR2/...).
    import rawpy
    with rawpy.imread(str(path)) as raw:
        arr = np.asarray(raw.raw_image_visible).astype(np.float32)
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
        hint = None
        try:
            from rawpy import ColorSpace  # noqa: F401
        except Exception:
            pass
    arr = np.clip((arr - black) / max(white - black, 1e-6), 0.0, 1.0)
    return arr.astype(np.float32), hint


def _normalize_mosaic(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(np.float32)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    maxv = float(np.max(arr)) if arr.size else 1.0
    if maxv > 1.5:  # integer data (8/16 bit) -> normalize by its range
        arr = arr / (255.0 if maxv <= 255.0 else 65535.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def bayer_pattern_tensor(bayer_pattern: str, device="cpu") -> torch.Tensor:
    """Return a ``(1, 1, 2, 2)`` long tensor for a Bayer pattern string."""
    arr = utils.camera.BayerPatternFlag.str_to_numpy(bayer_pattern)
    return torch.as_tensor(np.asarray(arr), dtype=torch.long, device=device).view(1, 1, 2, 2)


def build_batch(
    *,
    camera_model: str,
    bayer_pattern: str = "RGGB",
    white_balance=DEFAULT_WHITE_BALANCE,
    focal_length: float = DEFAULT_EXIF["focal_length"],
    f_number: float = DEFAULT_EXIF["f_number"],
    exposure_time: float = DEFAULT_EXIF["exposure_time"],
    iso_sensitivity: float = DEFAULT_EXIF["iso_sensitivity"],
    color_matrix=None,
    rgb: np.ndarray | None = None,
    raw: np.ndarray | None = None,
    device="cpu",
) -> dict:
    """Assemble the batch dictionary consumed by :class:`ParamISPNet`.

    Provide ``rgb`` (``(H, W, 3)`` in ``[0, 1]``) for the inverse direction or
    ``raw`` (``(H, W)`` in ``[0, 1]``) for the forward direction.
    """
    camera_name = CAMERA_INFO[camera_model]["name"]

    if color_matrix is None:
        color_matrix = data.utils.known_color_matrix([camera_name])[0].numpy()

    cm = torch.as_tensor(np.asarray(color_matrix), dtype=torch.float32)
    wb = torch.as_tensor(np.asarray(white_balance, dtype=np.float32), dtype=torch.float32)
    bp = bayer_pattern_tensor(bayer_pattern)

    if rgb is not None:
        rgb_t = torch.as_tensor(np.asarray(rgb), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        raw_t = torch.zeros(1, 1, *rgb_t.shape[-2:])
    else:
        raw_t = torch.as_tensor(np.asarray(raw), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        rgb_t = torch.zeros(1, 3, *raw_t.shape[-2:])

    h, w = rgb_t.shape[-2:]

    batch = {
        "rgb": rgb_t,
        "raw": raw_t,
        "bayer_pattern": bp.view(1, 1, 2, 2),
        "white_balance": wb.view(1, 3),
        "color_matrix": cm.view(1, 3, 3),
        "focal_length": torch.tensor([float(focal_length)]),
        "f_number": torch.tensor([float(f_number)]),
        "exposure_time": torch.tensor([float(exposure_time)]),
        "iso_sensitivity": torch.tensor([float(iso_sensitivity)]),
        "camera_name": [camera_name],
        "inst_id": ["demo-00000"],
        "index": torch.tensor([0]),
        "full_image_size": [(h, w)],
        "window_top_left": [(0, 0)],
        "transform": [(False, False, 0)],
        "quantized_level": torch.tensor([[255.0, 255.0, 255.0]]),
    }
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def visualize_raw(
    mosaic,
    bayer_pattern: str,
    white_balance=DEFAULT_WHITE_BALANCE,
    color_matrix=None,
    camera_model: str = "D7000",
) -> np.ndarray:
    """Render a single-channel Bayer mosaic as an ``(H, W, 3)`` sRGB preview.

    Mirrors ``AlgorithmOnlyModel.visualize_raw``: white balance -> demosaic ->
    color correction -> sRGB transfer.
    """
    if color_matrix is None:
        color_matrix = data.utils.known_color_matrix([CAMERA_INFO[camera_model]["name"]])[0].numpy()

    x = torch.as_tensor(np.asarray(mosaic), dtype=torch.float32)
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[:1]

    bp = bayer_pattern_tensor(bayer_pattern)
    wb = torch.as_tensor(np.asarray(white_balance, dtype=np.float32), dtype=torch.float32).view(1, 3)
    cm = torch.as_tensor(np.asarray(color_matrix), dtype=torch.float32).view(1, 3, 3)

    demosaic = layers.bayer.Demosaic()
    x = layers.bayer.apply_white_balance(x, bp, wb)
    x = demosaic(x, bp)
    x = layers.color.apply_color_matrix(x.clip(0, 1), cm).clip(0, 1)
    x = layers.color.linear_to_srgb(x)

    return x[0].permute(1, 2, 0).clamp(0, 1).numpy()

