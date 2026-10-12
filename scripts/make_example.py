"""Generate a small example sRGB image used by the demo's ``gr.Examples``."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "assets" / "example_srgb.jpg"


def main():
    w = h = 512
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx / w * 255).astype(np.uint8)
    g = (yy / h * 255).astype(np.uint8)
    b = ((xx + yy) / (w + h) * 255).astype(np.uint8)
    img = Image.fromarray(np.stack([r, g, b], axis=-1))

    draw = ImageDraw.Draw(img)
    draw.ellipse([110, 110, 300, 300], fill=(240, 200, 80))
    draw.rectangle([330, 160, 470, 420], fill=(60, 90, 200))
    draw.ellipse([50, 340, 190, 470], fill=(40, 180, 90))
    draw.line([0, 0, w, h], fill=(255, 255, 255), width=4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, quality=95)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
