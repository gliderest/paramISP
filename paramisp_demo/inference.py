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
    pixel-aligned.  Both inputs must share the same spatial size — use
    :func:`register_gt` when the RAW is a few sensor pixels larger than the
    image (a very common camera situation).
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


def _luma(x: np.ndarray) -> np.ndarray:
    """Luminance of an ``(H, W[, C])`` float ``[0, 1]`` image."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x
    if x.shape[-1] >= 3:
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return x[..., 0]

def _center_crop(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = a.shape[:2]
    top, left = max(0, (h - th) // 2), max(0, (w - tw) // 2)
    return a[top:top + th, left:left + tw]


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [-1, 1] (brightness-invariant)."""
    u = a - a.mean()
    v = b - b.mean()
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-9:
        return 0.0
    return float(np.sum(u * v) / denom)


def _center_ratio(total: int, sub: int) -> int:
    return max(0, (total - sub) // 2)


def _estimate_offset(big: np.ndarray, small: np.ndarray) -> tuple[int, int, float]:
    """Locate ``small`` inside ``big`` (both full-resolution, same pixel pitch).

    Returns ``(row, col, ncc_score)`` where ``row``/``col`` are the top-left
    pixel of the region of ``big`` that best matches ``small``.  A small central
    template is scanned over every valid offset with a brightness-invariant
    normalised cross-correlation; the scale is reduced only when the offset
    space would be expensive.
    """
    hb, wb = big.shape[:2]
    hs, ws = small.shape[:2]
    if hs > hb or ws > wb:
        raise ValueError(f"image {small.shape[:2]} does not fit inside {big.shape[:2]}")
    if (hb - hs, wb - ws) == (0, 0):
        return 0, 0, 1.0

    bg, sg = _luma(big), _luma(small)

    # Choose a scale so the number of offsets to scan stays small, without
    # losing more than a pixel or two of precision.
    f = 1.0
    while True:
        oy_max = max(0, round(hb * f) - round(hs * f))
        ox_max = max(0, round(wb * f) - round(ws * f))
        if (oy_max + 1) * (ox_max + 1) <= 2000 or f <= 0.0625:
            break
        f *= 0.5

    bh, bw = max(1, round(hb * f)), max(1, round(wb * f))
    sh, sw = max(1, round(hs * f)), max(1, round(ws * f))
    big_s = _resize(bg, bh, bw, upscale=f > 1.0)
    small_s = _resize(sg, sh, sw, upscale=f > 1.0)

    oy_max, ox_max = bh - sh, bw - sw
    if small_s.std() < 0.01 or big_s.std() < 0.01:  # flat content: nothing to register
        return _center_ratio(hb, hs), _center_ratio(wb, ws), 0.0

    # central template of the smaller image
    th = min(384, max(96, round(min(sh, sw) * 0.28)))
    t0y, t0x = (sh - th) // 2, (sw - th) // 2
    templ = small_s[t0y:t0y + th, t0x:t0x + th]

    best = (0, 0)
    best_n = -1.0
    for oy in range(oy_max + 1):
        for ox in range(ox_max + 1):
            n = _ncc(big_s[oy + t0y:oy + t0y + th, ox + t0x:ox + t0x + th], templ)
            if n > best_n:
                best_n, best = n, (oy, ox)

    # Refine around the coarse estimate at twice the resolution (if reduced).
    while f < 1.0:
        f2 = min(2.0 * f, 1.0)
        bh, bw = max(1, round(hb * f2)), max(1, round(wb * f2))
        sh, sw = max(1, round(hs * f2)), max(1, round(ws * f2))
        big_s2 = _resize(bg, bh, bw, upscale=f2 > 1.0)
        small_s2 = _resize(sg, sh, sw, upscale=f2 > 1.0)
        th = min(384, max(96, round(min(sh, sw) * 0.28)))
        t0y, t0x = (sh - th) // 2, (sw - th) // 2
        templ = small_s2[t0y:t0y + th, t0x:t0x + th]
        cy, cx = round(best[0] * f2 / f), round(best[1] * f2 / f)
        oy_max, ox_max = bh - sh, bw - sw
        for oy in range(max(0, cy - 3), min(oy_max, cy + 3) + 1):
            for ox in range(max(0, cx - 3), min(ox_max, cx + 3) + 1):
                n = _ncc(big_s2[oy + t0y:oy + t0y + th, ox + t0x:ox + t0x + th], templ)
                if n > best_n:
                    best_n, best = n, (oy, ox)
        f = f2  # ``best`` is now expressed at scale ``f2``
        if f >= 1.0 or best_n > 0.999:
            break

    return (int(round(best[0] / f)), int(round(best[1] / f)), best_n)


def register_gt(srgb: np.ndarray, gt: np.ndarray, multiple: int = 4,
                max_side: int | None = 1024, min_side: int = 128):
    """Align an sRGB image and a ground-truth RAW mosaic that may differ in size.

    Camera RAW mosaics are usually a handful of sensor rows/columns *larger*
    than the camera JPEG (masked / unused edge pixels), e.g. RAW ``(4020, 6036)``
    vs sRGB ``(4000, 6000)``.  This finds the pixel offset between the two and
    crops the larger one to the smaller one's footprint before the shared
    resize + center-crop.

    Returns ``(aligned_srgb, aligned_gt, warnings)`` so the caller can surface
    a message when the alignment cannot be verified.
    """
    srgb = np.asarray(srgb, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if srgb.shape[:2] == gt.shape[:2]:
        a, b = align_pair(srgb, gt, multiple, max_side, min_side)
        return a, b, []

    hs, ws = srgb.shape[:2]
    hr, wr = gt.shape[:2]
    warnings: list[str] = []

    if hs >= hr and ws >= wr:
        big, small, big_is_srgb = srgb, gt, True
    elif hr >= hs and wr >= ws:
        big, small, big_is_srgb = gt, srgb, False
    else:
        # Crossed dimensions (one image taller, the other wider): compare the
        # common centre region and warn — there is no sane crop for this case.
        th, tw = min(hs, hr), min(ws, wr)
        srgb_c = _center_crop(srgb, th, tw)
        gt_c = _center_crop(gt, th, tw)
        a, b = align_pair(srgb_c, gt_c, multiple, max_side, min_side)
        warnings.append(
            "The sRGB and RAW have different aspect ratios; using their centre overlap."
        )
        return a, b, warnings

    oy, ox, score = _estimate_offset(big, small)
    if score < 0.15:
        warnings.append(
            "Could not reliably align the ground-truth RAW with the image "
            "(weak correlation); using the centre of the RAW."
        )
        oy = _center_ratio(big.shape[0], small.shape[0])
        ox = _center_ratio(big.shape[1], small.shape[1])

    crop = big[oy:oy + small.shape[0], ox:ox + small.shape[1]]
    if big_is_srgb:
        srgb, gt = crop, small
    else:
        srgb, gt = small, crop

    a, b = align_pair(srgb, gt, multiple, max_side, min_side)
    return a, b, warnings


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
