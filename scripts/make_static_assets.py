"""Generate the preview images embedded in ``index.html`` (official weights).

Produces ``assets/raw_preview.png`` (predicted RAW) and
``assets/srgb_reconstruction.png`` (round-trip sRGB) from the example image.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from paramisp_demo import dataio, inference, weights

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def main():
    device = inference.resolve_device()
    src = ASSETS / "example_srgb.jpg"
    srgb = dataio.load_srgb(src)
    exif = dataio.read_exif_params(src)
    wb = (2.0, 1.0, 1.5)

    inv_model = inference.load_model(weights.ensure_weights("inverse", "D7000"), True, device)[0]
    raw = inference.predict_srgb_to_raw(
        inv_model, srgb, device, camera_model="D7000", bayer_pattern="RGGB",
        white_balance=wb, exif=exif,
    )
    preview = dataio.visualize_raw(raw, "RGGB", wb, camera_model="D7000")
    Image.fromarray((preview * 255).astype(np.uint8)).save(ASSETS / "raw_preview.png")

    fwd_model = inference.load_model(weights.ensure_weights("forward", "D7000"), False, device)[0]
    rec = inference.predict_raw_to_srgb(
        fwd_model, raw, device, camera_model="D7000", bayer_pattern="RGGB",
        white_balance=wb, exif=exif,
    )
    Image.fromarray((np.clip(rec, 0, 1) * 255).astype(np.uint8)).save(ASSETS / "srgb_reconstruction.png")

    print("wrote assets/raw_preview.png and assets/srgb_reconstruction.png")


if __name__ == "__main__":
    main()
