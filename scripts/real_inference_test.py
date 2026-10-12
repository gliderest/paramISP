"""Run the demo end to end with the official checkpoints."""
from pathlib import Path

import numpy as np
from PIL import Image

from paramisp_demo import dataio, inference, weights

ROOT = Path(__file__).resolve().parent.parent


def main():
    image_path = ROOT / "assets" / "example_srgb.jpg"
    if not image_path.exists():
        print("example image missing; run scripts/make_example.py first")
        return

    device = inference.resolve_device()
    print("device:", device)

    # ---- inverse ISP: sRGB -> RAW ----
    print("\navailable checkpoints:", weights.list_available())
    ckpt = weights.ensure_weights("inverse", "D7000")
    model, missing, unexpected = inference.load_model(ckpt, inverse=True, device=device)
    print(f"loaded {ckpt.name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    print("  unexpected (expected: ms_ssim_loss.*):", unexpected[:3])

    srgb = dataio.load_srgb(image_path)
    exif = dataio.read_exif_params(image_path)
    print("EXIF:", exif)

    mosaic = inference.predict_srgb_to_raw(
        model, srgb, device, camera_model="D7000", bayer_pattern="RGGB",
        white_balance=(2.0, 1.0, 1.5), exif=exif,
    )
    print("RAW mosaic:", mosaic.shape, mosaic.dtype, "range", float(np.nanmin(mosaic)), float(np.nanmax(mosaic)))
    assert not np.isnan(mosaic).any(), "NaN in prediction"

    preview = dataio.visualize_raw(mosaic, "RGGB", (2.0, 1.0, 1.5), camera_model="D7000")
    Image.fromarray((preview * 255).astype(np.uint8)).save(ROOT / "assets" / "_raw_preview.png")
    inference.save_tiff_u16(mosaic, ROOT / "assets" / "_raw_prediction.tiff")
    print("saved assets/_raw_preview.png and assets/_raw_prediction.tiff")

    # ---- forward ISP: RAW -> sRGB ----
    ckpt_fwd = weights.ensure_weights("forward", "D7000")
    model_fwd, m2, u2 = inference.load_model(ckpt_fwd, inverse=False, device=device)
    print(f"\nloaded {ckpt_fwd.name}: {len(m2)} missing, {len(u2)} unexpected keys")

    srgb_rec = inference.predict_raw_to_srgb(
        model_fwd, mosaic, device, camera_model="D7000", bayer_pattern="RGGB",
        white_balance=(2.0, 1.0, 1.5), exif=exif,
    )
    print("reconstructed sRGB:", srgb_rec.shape, "range", float(srgb_rec.min()), float(srgb_rec.max()))
    assert not np.isnan(srgb_rec).any(), "NaN in forward prediction"
    Image.fromarray((np.clip(srgb_rec, 0, 1) * 255).astype(np.uint8)).save(ROOT / "assets" / "_srgb_reconstruction.png")

    print("\nround-trip |sRGB - fwd(inv(sRGB))| mean =",
          float(np.abs(srgb_rec - srgb[:srgb_rec.shape[0], :srgb_rec.shape[1]]).mean()))
    print("\nREAL INFERENCE TEST PASSED")


if __name__ == "__main__":
    main()
