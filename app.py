"""ParamISP on Hugging Face Spaces.

Interactive demo for the CVPR 2024 paper **ParamISP: Learned Forward and Inverse
ISPs using Camera Parameters** (https://github.com/woo525/ParamISP).

Two directions are supported:

* **sRGB -> RAW (inverse ISP)**: given an ``.sRGB`` image and its camera
  parameters (ISO, exposure time, focal length, f-number, white balance, color
  matrix) the network predicts the underlying Bayer RAW.
* **RAW -> sRGB (forward ISP)**: given a Bayer RAW mosaic it renders the sRGB
  image a camera ISP would produce.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
import numpy as np

from paramisp_demo import dataio, inference, weights

try:  # ZeroGPU is only meaningful inside a Hugging Face Space.
    import spaces
    _gpu = spaces.GPU(duration=120) if os.environ.get("SPACE_ID") else (lambda fn: fn)
except Exception:  # noqa: BLE001
    def _gpu(fn):
        return fn

CAMERAS = list(dataio.CAMERA_INFO.keys())
_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_model(mode: str, camera: str):
    key = (mode, camera)
    if key not in _MODEL_CACHE:
        path = weights.ensure_weights(mode, camera)
        device = inference.resolve_device()
        model, missing, unexpected = inference.load_model(path, inverse=(mode == "inverse"), device=device)
        print(f"[app] loaded {mode}/{camera} ({len(missing)} missing, {len(unexpected)} unexpected keys)")
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _params_from_inputs(camera, bayer, wr, wg, wb, use_exif, image_path=None,
                        focal=0.0, aperture=0.0, exposure=0.0, iso=0.0):
    exif = dict(dataio.DEFAULT_EXIF)
    if use_exif and image_path:
        exif = dataio.read_exif_params(image_path)
    else:
        exif.update({"focal_length": float(focal), "f_number": float(aperture),
                     "exposure_time": float(exposure), "iso_sensitivity": float(iso)})
    return {
        "camera_model": camera,
        "bayer_pattern": bayer,
        "white_balance": (float(wr), float(wg), float(wb)),
        "exif": exif,
    }


@_gpu
def run_inverse(image_path, camera, bayer, wr, wg, wb, use_exif, focal, aperture, exposure, iso):
    if not image_path:
        raise gr.Error("Please upload an sRGB image.")
    cfg = _params_from_inputs(camera, bayer, wr, wg, wb, use_exif, image_path,
                              focal, aperture, exposure, iso)
    srgb = dataio.load_srgb(image_path)
    model = _get_model("inverse", camera)
    device = inference.resolve_device()
    mosaic = inference.predict_srgb_to_raw(
        model, srgb, device, camera_model=cfg["camera_model"], bayer_pattern=cfg["bayer_pattern"],
        white_balance=cfg["white_balance"], exif=cfg["exif"],
    )
    preview = (dataio.visualize_raw(
        mosaic, cfg["bayer_pattern"], cfg["white_balance"], camera_model=cfg["camera_model"]) * 255
    ).astype(np.uint8)

    out_dir = Path(tempfile.gettempdir()) / "paramisp_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tiff = out_dir / f"{uuid.uuid4().hex[:8]}_raw_prediction.tiff"
    inference.save_tiff_u16(mosaic, out_tiff)

    info = (
        f"Camera: **{cfg['camera_model']}** | Bayer: **{cfg['bayer_pattern']}**\n\n"
        f"Focal length: {cfg['exif']['focal_length']:.2f} mm | "
        f"f/{cfg['exif']['f_number']:.2f} | "
        f"{cfg['exif']['exposure_time']:.5f} s | ISO {cfg['exif']['iso_sensitivity']:.0f}\n\n"
        f"Predicted RAW: {mosaic.shape[0]}x{mosaic.shape[1]} (single channel, [0, 1])."
    )
    return preview, str(out_tiff), info


@_gpu
def run_forward(raw_path, camera, bayer, wr, wg, wb, use_exif, focal, aperture, exposure, iso):
    if not raw_path:
        raise gr.Error("Please upload a RAW file (.dng/.arw/.nef/.cr2/.npy/.tif).")
    cfg = _params_from_inputs(camera, bayer, wr, wg, wb, use_exif, raw_path,
                              focal, aperture, exposure, iso)
    raw, bayer_hint = dataio.load_raw_mosaic(raw_path)
    bayer = bayer_hint or cfg["bayer_pattern"]
    model = _get_model("forward", camera)
    device = inference.resolve_device()
    srgb = inference.predict_raw_to_srgb(
        model, raw, device, camera_model=cfg["camera_model"], bayer_pattern=bayer,
        white_balance=cfg["white_balance"], exif=cfg["exif"],
    )
    out = (np.clip(srgb, 0.0, 1.0) * 255).astype(np.uint8)
    info = (
        f"Camera: **{cfg['camera_model']}** | Bayer: **{bayer}**\n\n"
        f"Rendered sRGB: {out.shape[0]}x{out.shape[1]}."
    )
    return out, info


DESCRIPTION = """
# ParamISP — Learned Forward and Inverse ISPs using Camera Parameters

<sub>Official implementation of the CVPR 2024 paper by Kim, Kim, Lee, Lee, Baek & Cho —
[paper](https://arxiv.org/abs/2312.13313) ·
[code](https://github.com/woo525/ParamISP) ·
[project page](https://woo525.github.io/ParamISP/)</sub>

ParamISP learns the **forward** (RAW → sRGB) and **inverse** (sRGB → RAW) camera
ISP processes, using the camera parameters found in EXIF (ISO, exposure time,
focal length, f-number, white balance, color matrix) to control the network
through a *ParamNet* hyper-network.
"""


def _param_panel(prefix: str):
    """Build the shared camera-parameter controls for one tab."""
    camera = gr.Dropdown(CAMERAS, value="D7000", label="Camera model",
                         info="Which camera's ISP / parameters the model should use.")
    bayer = gr.Dropdown(dataio.BAYER_CHOICES, value="RGGB", label="Bayer pattern")
    use_exif = gr.Checkbox(value=False, label="Read camera parameters from EXIF",
                           info="If off, the manual values below are used (0 = unknown).")
    with gr.Accordion("Manual camera parameters", open=False):
        with gr.Row():
            focal = gr.Number(value=50.0, label="Focal length (mm)")
            aperture = gr.Number(value=8.0, label="F-number")
            exposure = gr.Number(value=0.01, label="Exposure time (s)")
            iso = gr.Number(value=200.0, label="ISO")
    with gr.Accordion("White balance (R, G, B)", open=False):
        with gr.Row():
            wr = gr.Slider(0.2, 4.0, value=2.0, step=0.01, label="R")
            wg = gr.Slider(0.2, 4.0, value=1.0, step=0.01, label="G")
            wb = gr.Slider(0.2, 4.0, value=1.5, step=0.01, label="B")
    return camera, bayer, use_exif, focal, aperture, exposure, iso, wr, wg, wb


def build_demo() -> gr.Blocks:
    available = weights.list_available()
    example_image = str((Path(__file__).resolve().parent / "assets" / "example_srgb.jpg"))
    with gr.Blocks(title="ParamISP", theme=gr.themes.Soft()) as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Tab("sRGB → RAW (inverse ISP)"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="filepath", label="Input sRGB image")
                    camera, bayer, use_exif, focal, aperture, exposure, iso, wr, wg, wb = _param_panel("inv")
                    inv_btn = gr.Button("Predict RAW", variant="primary")
                    gr.Examples(
                        [[example_image, "D7000", "RGGB", False, 50.0, 8.0, 0.01, 200.0, 2.0, 1.0, 1.5]],
                        inputs=[image, camera, bayer, use_exif, focal, aperture, exposure, iso, wr, wg, wb],
                        label="Example",
                    )
                with gr.Column():
                    inv_preview = gr.Image(label="Predicted RAW (white-balanced preview)", type="numpy")
                    inv_file = gr.File(label="Download predicted RAW (16-bit TIFF)")
                    inv_info = gr.Markdown()
            inv_btn.click(
                run_inverse,
                inputs=[image, camera, bayer, wr, wg, wb, use_exif, focal, aperture, exposure, iso],
                outputs=[inv_preview, inv_file, inv_info],
            )

        with gr.Tab("RAW → sRGB (forward ISP)"):
            with gr.Row():
                with gr.Column():
                    raw_file = gr.File(label="Input RAW (.dng/.arw/.nef/.cr2/.npy/.tif)")
                    camera2, bayer2, use_exif2, focal2, aperture2, exposure2, iso2, wr2, wg2, wb2 = _param_panel("fwd")
                    fwd_btn = gr.Button("Render sRGB", variant="primary")
                with gr.Column():
                    fwd_image = gr.Image(label="Rendered sRGB", type="numpy")
                    fwd_info = gr.Markdown()
            fwd_btn.click(
                run_forward,
                inputs=[raw_file, camera2, bayer2, wr2, wg2, wb2, use_exif2, focal2, aperture2, exposure2, iso2],
                outputs=[fwd_image, fwd_info],
            )

        gr.Markdown(
            "Checkpoints currently found: "
            f"`{available or 'none yet — they are downloaded on first use'}`\n\n"
            "Set `PARAMISP_WEIGHTS_REPO` (a Hugging Face repo) or `PARAMISP_GDRIVE_ID` "
            "to point the app at the official weights."
        )
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ParamISP demo")
    ap.add_argument("--share", action="store_true",
                    help="create a temporary public gradio.live link (free)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    _args = ap.parse_args()
    _share = _args.share or os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    build_demo().queue().launch(share=_share, server_port=_args.port, server_name=_args.host)

