"""Validate the sRGB/ground-truth RAW auto-alignment (``inference.register_gt``).

Covers the case reported by a user where the camera RAW (e.g. ``(4020, 6036)``)
is a few sensor rows/columns larger than the sRGB (e.g. ``(4000, 6000)``): the
offset is detected automatically and both images are cropped to the same pixel
size before metrics.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paramisp_demo import inference  # noqa: E402


def _pool2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    a = a[: h // 2 * 2, : w // 2 * 2]
    return a.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _make_pair(raw_h: int, raw_w: int, shift: tuple[int, int]):
    """Return ``(srgb_small, raw_full)`` like a camera JPEG + a larger sensor RAW."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:raw_h, 0:raw_w].astype(np.float32)

    smooth = rng.random((50, 60), dtype=np.float32)
    smooth = inference._resize(smooth, raw_h, raw_w, upscale=True)
    scene = 0.30 * (y / max(raw_h, 1) + x / max(raw_w, 1)) + 0.70 * smooth
    scene = scene / scene.max()

    raw_full = np.clip(scene + 0.05 * rng.random((raw_h, raw_w), dtype=np.float32), 0.02, 0.98)

    # A plausible "sRGB": the RAW scene, smoothed and denoised, in RGB.
    luma = np.kron(_pool2(raw_full), np.ones((2, 2), dtype=np.float32))
    luma = inference._resize(luma, raw_h, raw_w, upscale=True)[:raw_h, :raw_w]
    srgb_full = np.stack([luma] * 3, axis=-1)

    oy, ox = shift
    srgb_small = srgb_full[oy:oy + raw_h - 28, ox:ox + raw_w - 40]
    return srgb_small, raw_full, oy, ox


def test_equal_sizes_noop():
    rng = np.random.default_rng(1)
    srgb = rng.random((80, 120, 3), dtype=np.float32)
    gt = srgb[..., 0].copy()
    a, b, warnings = inference.register_gt(srgb, gt)
    assert a.shape[:2] == b.shape[:2]
    assert a.shape[0] % 4 == 0 and a.shape[1] % 4 == 0
    assert not warnings
    print("OK  equal sizes -> passthrough, no warnings")


def test_raw_larger_than_srgb():
    # RAW (420, 452), sRGB (392, 412): the RAW is 28 rows / 40 cols larger, and
    # the sRGB belongs to raw rows 7..399 / cols 11..423.
    srgb_small, raw_full, oy_t, ox_t = _make_pair(420, 452, (7, 11))
    a, b, warnings = inference.register_gt(srgb_small, raw_full)
    assert a.shape[:2] == b.shape[:2] == (392, 412), (a.shape, b.shape)
    assert not warnings, warnings  # alignment should be trustworthy

    gt_small = b
    region = raw_full[oy_t:oy_t + 392, ox_t:ox_t + 412]
    assert gt_small.shape == region.shape
    diff = float(np.abs(gt_small - region).mean())
    assert diff < 0.02, f"mis-aligned ground truth (MAE {diff:.4f})"
    corr = float(np.corrcoef(gt_small.ravel(), region.ravel())[0, 1])
    assert corr > 0.99, f"mis-aligned ground truth (corr {corr:.4f})"
    print(f"OK  RAW larger than sRGB: aligned shapes {a.shape}, GT MAE {diff:.4f}, "
          f"corr {corr:.4f}")


def test_crossed_dims_warns():
    rng = np.random.default_rng(2)
    srgb = rng.random((100, 160, 3), dtype=np.float32)
    gt = rng.random((120, 140), dtype=np.float32)  # taller but narrower
    a, b, warnings = inference.register_gt(srgb, gt)
    assert a.shape[:2] == b.shape[:2]
    assert a.shape[0] % 4 == 0 and a.shape[1] % 4 == 0
    assert len(warnings) == 1
    print("OK  crossed dimensions -> centre overlap + warning")


def test_srgb_larger_than_raw():
    # Reverse of the camera case: a large rendered image vs a smaller raw crop.
    rng = np.random.default_rng(4)
    H, W = 440, 500
    scene = inference._resize(rng.random((55, 65), dtype=np.float32), H, W, upscale=True)
    scene = (scene - scene.min()) / (scene.max() - scene.min())
    srgb_big = np.stack([scene] * 3, axis=-1).astype(np.float32)
    oy_t, ox_t = 13, 21                     # where the raw lives inside the sRGB
    raw_small = scene[oy_t:oy_t + 392, ox_t:ox_t + 412].astype(np.float32)
    a, b, warnings = inference.register_gt(srgb_big, raw_small)
    assert a.shape[:2] == b.shape[:2]
    assert not warnings, warnings
    expected_s, expected_gt = inference.align_pair(srgb_big[oy_t:oy_t + 392, ox_t:ox_t + 412],
                                                   raw_small)
    assert abs(a - expected_s).max() < 1e-5 and abs(b - expected_gt).max() < 1e-5
    print(f"OK  sRGB larger than RAW: aligned shapes {a.shape[:2]}")


def main():
    test_equal_sizes_noop()
    test_raw_larger_than_srgb()
    test_srgb_larger_than_raw()
    test_crossed_dims_warns()
    print("\nALIGN TEST PASSED")


if __name__ == "__main__":
    main()