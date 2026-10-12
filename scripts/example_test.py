"""Load every built-in example, predict its RAW, and report PSNR / SSIM / MAE.

This mirrors the Streamlit app's "Built-in example" flow (real instance metadata,
aligned ground truth, same metrics).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from paramisp_demo import examples as demo_examples  # noqa: E402
from paramisp_demo import inference, metrics, weights  # noqa: E402


def main():
    device = torch.device("cpu")
    labels = demo_examples.example_labels()
    print(f"{len(labels)} example(s): {labels}\n")

    for name in labels:
        ex = demo_examples.load_example(name)
        model = inference.load_model(weights.ensure_weights("inverse", ex["camera"]), True, device)[0]
        start = time.perf_counter()
        mosaic = inference.predict_srgb_to_raw(
            model, ex["srgb"], device, camera_model=ex["camera"],
            bayer_pattern=ex["bayer_pattern"], white_balance=ex["white_balance"],
            color_matrix=ex["color_matrix"], exif=ex["exif"])
        elapsed = time.perf_counter() - start
        scores = metrics.compare(mosaic, ex["raw"])
        print(f"{name}: PSNR {scores['psnr']:6.2f} dB | SSIM {scores['ssim']:.4f} | "
              f"MAE {scores['mae']:.5f} | {elapsed*1000:.0f} ms | {tuple(mosaic.shape)}")

    print("\nEXAMPLE TEST PASSED")


if __name__ == "__main__":
    main()