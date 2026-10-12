"""Quick smoke test: build the network and run one forward pass per direction."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paramisp_demo import dataio  # noqa: E402
from paramisp_demo.net import ParamISPNet  # noqa: E402


def main():
    torch.manual_seed(0)

    model = ParamISPNet(inverse=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ParamISPNet(inverse=True): {n_params/1e6:.2f}M parameters")

    rgb = np.random.rand(128, 160, 3).astype(np.float32)
    batch = dataio.build_batch(camera_model="D7000", rgb=rgb)
    with torch.no_grad():
        out = model(batch)
    print("inverse output:", tuple(out.shape), "range", float(out.min()), float(out.max()))
    assert out.shape == (1, 1, 128, 160), out.shape

    model_fwd = ParamISPNet(inverse=False).eval()
    raw = np.random.rand(128, 160).astype(np.float32)
    batch_fwd = dataio.build_batch(camera_model="D7000", raw=raw)
    with torch.no_grad():
        out_fwd = model_fwd(batch_fwd)
    print("forward output:", tuple(out_fwd.shape), "range", float(out_fwd.min()), float(out_fwd.max()))
    assert out_fwd.shape == (1, 3, 128, 160), out_fwd.shape

    preview = dataio.visualize_raw(out[0, 0].numpy(), "RGGB", camera_model="D7000")
    print("RAW preview:", preview.shape, preview.dtype, float(preview.min()), float(preview.max()))
    assert preview.shape == (128, 160, 3)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
