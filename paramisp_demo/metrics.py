"""Image-quality metrics for comparing a predicted RAW with ground truth.

PSNR and SSIM follow the convention of the upstream code
(``AlgorithmOnlyModel.compute_metrics``): a 2-pixel border is cropped before
measuring, and SSIM uses an 11x11 Gaussian window (sigma=1.5) with
``data_range=1.0`` — the ``torchmetrics`` defaults used by the official
implementation.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_BORDER = 2


def _as_4d(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    t = t.float()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[None]
    return t


def crop_border(x: torch.Tensor, border: int = _BORDER) -> torch.Tensor:
    """Drop a ``border``-wide frame (the official metric crops 2 pixels)."""
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def psnr(pred, gt, data_range: float = 1.0, border: int = _BORDER) -> float:
    """Peak signal-to-noise ratio between ``pred`` and ``gt`` (higher is better)."""
    x = crop_border(_as_4d(pred), border)
    y = crop_border(_as_4d(gt), border)
    mse = F.mse_loss(x, y).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(pred, gt, data_range: float = 1.0, border: int = _BORDER,
         kernel_size: int = 11, sigma: float = 1.5) -> float:
    """Structural similarity between ``pred`` and ``gt`` (higher is better)."""
    x = crop_border(_as_4d(pred), border)
    y = crop_border(_as_4d(gt), border)
    pad = kernel_size // 2
    if min(x.shape[-2:]) <= pad:
        return float("nan")

    channels = x.shape[1]
    window = _gaussian_window(kernel_size, sigma).to(x.dtype).to(x.device)
    window = window.expand(channels, 1, kernel_size, kernel_size).contiguous()

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(t, (pad, pad, pad, pad), mode="reflect"), window, groups=channels)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = filt(x), filt(y)
    sigma_x2 = filt(x * x) - mu_x * mu_x
    sigma_y2 = filt(y * y) - mu_y * mu_y
    sigma_xy = filt(x * y) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((numerator / denominator).mean())


def mae(pred, gt, border: int = _BORDER) -> float:
    """Mean absolute error between ``pred`` and ``gt`` (lower is better)."""
    x = crop_border(_as_4d(pred), border)
    y = crop_border(_as_4d(gt), border)
    return float(F.l1_loss(x, y))


def compare(pred, gt, data_range: float = 1.0) -> dict:
    """Return ``{"psnr", "ssim", "mae"}`` for a predicted / ground-truth pair."""
    return {
        "psnr": psnr(pred, gt, data_range),
        "ssim": ssim(pred, gt, data_range),
        "mae": mae(pred, gt),
    }
