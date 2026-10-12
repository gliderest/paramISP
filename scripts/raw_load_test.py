"""Validate ``dataio.load_raw_mosaic`` for mosaics and camera RAW files.

Exercises the three code paths:

* ``.npy`` / ``.tif`` mosaic loading (no optional deps);
* the actionable error raised for camera RAW when ``rawpy`` is missing;
* the camera-RAW path (black-point / white-level normalisation + Bayer hint),
  driven by a stub ``rawpy`` so it runs even without a real ``.nef``/``.dng``.

This mirrors the Streamlit app's forward tab and inverse-tab ground truth, both
of which read uploads through ``dataio.load_raw_mosaic``.
"""
import importlib.abc
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paramisp_demo import dataio  # noqa: E402


class _BlockRawpy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "rawpy":
            raise ModuleNotFoundError("No module named 'rawpy' (blocked)")
        return None


def test_npy_and_tif():
    rng = np.random.default_rng(0)
    mosaic = rng.integers(0, 65536, size=(64, 96), dtype=np.uint16)
    with tempfile.TemporaryDirectory() as d:
        import tifffile
        np.save(Path(d) / "m.npy", mosaic)
        tifffile.imwrite(Path(d) / "m.tif", mosaic)
        a, ha = dataio.load_raw_mosaic(Path(d) / "m.npy")
        b, hb = dataio.load_raw_mosaic(Path(d) / "m.tif")
    assert a.shape == (64, 96) and a.dtype == np.float32, (a.shape, a.dtype)
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0
    assert b.shape == a.shape and ha is None and hb is None
    print("OK  .npy/.tif mosaic loading ->", a.shape)


def test_missing_rawpy():
    blocker = _BlockRawpy()
    sys.meta_path.insert(0, blocker)
    sys.modules.pop("rawpy", None)
    try:
        try:
            dataio.load_raw_mosaic("some_camera.nef")
        except RuntimeError as exc:
            assert "rawpy" in str(exc), exc
            print("OK  missing rawpy -> actionable error:", str(exc)[:68], "...")
            return
        raise AssertionError("expected RuntimeError when rawpy is unavailable")
    finally:
        sys.meta_path = [m for m in sys.meta_path if m is not blocker]
        sys.modules.pop("rawpy", None)


class _StubRaw:
    raw_image_visible = None                       # force the raw_image fallback
    raw_image = np.full((32, 48), 8192, dtype=np.uint16)
    black_level_per_channel = [512, 512, 512, 512]
    white_level = 16383
    raw_pattern = np.array([[0, 1], [3, 2]])       # RGGB, with G2 encoded as 3

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _StubRawpy:
    @staticmethod
    def imread(path):
        return _StubRaw()


def test_camera_raw_path():
    sys.modules["rawpy"] = _StubRawpy
    try:
        arr, hint = dataio.load_raw_mosaic("anything.nef")
    finally:
        sys.modules.pop("rawpy", None)
    assert arr.shape == (32, 48), arr.shape
    assert arr.dtype == np.float32
    assert hint == "RGGB", hint
    expected = (8192 - 512) / (16383 - 512)
    assert abs(float(arr.mean()) - expected) < 1e-4, (float(arr.mean()), expected)
    print("OK  camera-RAW path -> shape", arr.shape, "bayer", hint,
          "value %.4f" % float(arr.mean()))


def test_neutral_preview_for_unknown_camera():
    """A neutral-grey mosaic must not render magenta/green on the upload path.

    ``visualize_raw(..., white_balance=None, color_matrix=None)`` is what the
    app uses for uploaded RAWs (the camera's parameters are unknown): it
    derives a gray-world WB from the mosaic and skips the colour matrix
    (previously another camera's matrix was substituted, which could collapse
    the green channel -> bright magenta previews).
    """
    rng = np.random.default_rng(7)
    mosaic = np.clip(0.2 + 0.02 * rng.random((64, 96)).astype(np.float32), 0.0, 1.0)
    out = dataio.visualize_raw(mosaic, "RGGB", None, None)
    means = [float(out[..., c].mean()) for c in range(3)]
    spread = max(means) - min(means)
    assert out.shape == (64, 96, 3) and not np.isnan(out).any()
    assert spread < 0.1, f"non-neutral preview: {means}"
    # the known-camera path (explicit WB + colour matrix) still runs unchanged
    cm = np.asarray([[3.7, -0.59, -0.11], [-1.59, 7.33, -2.74],
                     [0.05, -1.06, 4.01]], dtype=np.float32)
    out2 = dataio.visualize_raw(mosaic, "RGGB", (2.0, 1.0, 1.5), color_matrix=cm)
    assert out2.shape == out.shape and not np.isnan(out2).any()
    print("OK  neutral GT preview (upload path) + known-matrix path,",
          "spread %.4f" % spread)


def main():
    test_npy_and_tif()
    test_missing_rawpy()
    test_camera_raw_path()
    test_neutral_preview_for_unknown_camera()
    print("\nRAW-LOAD TEST PASSED")


if __name__ == "__main__":
    main()
