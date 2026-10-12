"""High-level inference helpers for the ParamISP demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .net import ParamISPNet
from . import dataio


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_load(path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without ``weights_only``
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "net", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def load_model(ckpt_path: str | Path, inverse: bool, device: torch.device):
    """Instantiate :class:`ParamISPNet` and load a checkpoint into it.

    Returns ``(model, missing_keys, unexpected_keys)``.  The official checkpoints
    are plain PyTorch-Lightning files, so only ``.ckpt["state_dict"]`` is used.
    """
    model = ParamISPNet(inverse=inverse)
    state = _extract_state_dict(_torch_load(ckpt_path))
    state = {
        (k[7:] if k.startswith("model.") else k[4:] if k.startswith("net.") else k): v
        for k, v in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model, list(missing), list(unexpected)


def _resize(x: np.ndarray, nh: int, nw: int, *, upscale: bool) -> np.ndarray:
    import PIL.Image

    arr = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = PIL.Image.fromarray(arr)
    resample = PIL.Image.LANCZOS if not upscale else PIL.Image.BICUBIC
    im = im.resize((nw, nh), resample)
    return np.asarray(im).astype(np.float32) / 255.0


def _fit_scale(h: int, w: int, max_side: int | None, min_side: int | None) -> float:
    scale = 1.0
    if max_side is not None and max(h, w) > max_side:
        scale = max_side / max(h, w)
    if min_side is not None and min(h, w) * scale < min_side:
        scale = min_side / min(h, w)
    return scale


def _geometry(h: int, w: int, multiple: int, max_side: int | None, min_side: int | None):
    """Return ``(scale, nh, nw, top, left, ch, cw)`` for the resize + center-crop."""
    scale = _fit_scale(h, w, max_side, min_side)
    if abs(scale - 1.0) > 1e-6:
        h, w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    ch = (h // multiple) * multiple
    cw = (w // multiple) * multiple
    return scale, h, w, (h - ch) // 2, (w - cw) // 2, ch, cw


def _crop_to_multiple(x: np.ndarray, multiple: int = 4, max_side: int | None = 1024,
                      min_side: int = 128) -> np.ndarray:
    """Resize to ``[min_side, max_side]`` and center-crop to a multiple of ``multiple``.

    The input is a float array in ``[0, 1]``.  ``GlobalNet`` reduces the feature
    map by 64x, so the input must be at least ~96 px per side; large uploads are
    downscaled to keep CPU inference fast.  The final crop guarantees the dims
    are divisible by 4 (the network's two down/up-sampling stages).
    """
    h, w = x.shape[:2]
    scale, nh, nw, top, left, ch, cw = _geometry(h, w, multiple, max_side, min_side)
    if abs(scale - 1.0) > 1e-6:
        x = _resize(x, nh, nw, upscale=scale > 1.0)
    return x[top:top + ch, left:left + cw]


def align_pair(srgb: np.ndarray, gt: np.ndarray, multiple: int = 4,
               max_side: int | None = 1024, min_side: int = 128):
    """Apply *the same* resize + center-crop to an sRGB image and its aligned GT RAW.

    Used before computing metrics so the prediction and the ground truth stay
    pixel-aligned.  Both inputs must share the same spatial size.
    """
    srgb = np.asarray(srgb, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if srgb.shape[:2] != gt.shape[:2]:
        raise ValueError(f"size mismatch: sRGB {srgb.shape[:2]} vs RAW {gt.shape[:2]}")

    h, w = srgb.shape[:2]
    scale, nh, nw, top, left, ch, cw = _geometry(h, w, multiple, max_side, min_side)
    if abs(scale - 1.0) > 1e-6:
        srgb = _resize(srgb, nh, nw, upscale=scale > 1.0)
        gt = _resize(gt, nh, nw, upscale=scale > 1.0)
    return srgb[top:top + ch, left:left + cw], gt[top:top + ch, left:left + cw]


@torch.no_grad()
def predict_srgb_to_raw(model, srgb, device, *, camera_model, bayer_pattern="RGGB",
                        white_balance=dataio.DEFAULT_WHITE_BALANCE, exif=None,
                        color_matrix=None, max_side=1536):
    """Inverse ISP: ``(H, W, 3)`` sRGB in ``[0, 1]`` -> ``(H, W)`` RAW mosaic in ``[0, 1]``."""
    exif = exif or dataio.DEFAULT_EXIF
    srgb = _crop_to_multiple(np.asarray(srgb, dtype=np.float32), 4, max_side)
    batch = dataio.build_batch(
        camera_model=camera_model, bayer_pattern=bayer_pattern, white_balance=white_balance,
        focal_length=exif["focal_length"], f_number=exif["f_number"],
        exposure_time=exif["exposure_time"], iso_sensitivity=exif["iso_sensitivity"],
        color_matrix=color_matrix, rgb=srgb, device=device,
    )
    out = model(batch).clamp(0, 1)
    return out[0, 0].float().cpu().numpy()


@torch.no_grad()
def predict_raw_to_srgb(model, raw, device, *, camera_model, bayer_pattern="RGGB",
                        white_balance=dataio.DEFAULT_WHITE_BALANCE, exif=None,
                        color_matrix=None, max_side=1536):
    """Forward ISP: ``(H, W)`` RAW mosaic in ``[0, 1]`` -> ``(H, W, 3)`` sRGB in ``[0, 1]``."""
    exif = exif or dataio.DEFAULT_EXIF
    raw = _crop_to_multiple(np.asarray(raw, dtype=np.float32), 4, max_side)
    batch = dataio.build_batch(
        camera_model=camera_model, bayer_pattern=bayer_pattern, white_balance=white_balance,
        focal_length=exif["focal_length"], f_number=exif["f_number"],
        exposure_time=exif["exposure_time"], iso_sensitivity=exif["iso_sensitivity"],
        color_matrix=color_matrix, raw=raw, device=device,
    )
    out = model(batch).clamp(0, 1)
    return out[0].float().permute(1, 2, 0).cpu().numpy()


def save_tiff_u16(mosaic: np.ndarray, path: str | Path) -> str:
    """Save a ``[0, 1]`` float mosaic as a 16-bit TIFF."""
    import tifffile
    data = np.clip(np.asarray(mosaic, dtype=np.float32), 0.0, 1.0)
    data = (data * 65535.0 + 0.5).astype(np.uint16)
    tifffile.imwrite(str(path), data)
    return str(path)
