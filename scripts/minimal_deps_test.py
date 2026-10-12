"""Verify the demo runs when the optional torchvision / rawpy deps are missing.

Reproduces a minimal deployment environment (e.g. Streamlit Community Cloud,
where a hidden ``torchvision`` import once broke the build) by blocking those
packages, then runs a full sRGB -> RAW prediction.
"""
import importlib.abc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKED = ("torchvision", "rawpy")


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named '{name}' (blocked)")
        return None


def main():
    sys.path.insert(0, str(ROOT / "paramisp_src"))
    sys.path.insert(0, str(ROOT))
    sys.meta_path.insert(0, _Block())

    import numpy as np
    import torch

    from paramisp_demo import dataio, inference, weights

    print("imports OK with", ", ".join(BLOCKED), "blocked")
    device = torch.device("cpu")
    model = inference.load_model(weights.ensure_weights("inverse", "D7000"), True, device)[0]
    raw = inference.predict_srgb_to_raw(
        model, dataio.load_srgb(ROOT / "assets" / "example_srgb.jpg"), device,
        camera_model="D7000", bayer_pattern="RGGB", white_balance=(2.0, 1.0, 1.5),
    )
    assert not np.isnan(raw).any(), "NaN in prediction"
    print("inference OK:", raw.shape)
    print("MINIMAL-DEPS TEST PASSED")


if __name__ == "__main__":
    main()
