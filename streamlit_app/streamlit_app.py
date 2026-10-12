"""ParamISP demo for Streamlit Community Cloud (free, CPU-only).

Deploy at https://share.streamlit.io with **Main file path** =
``streamlit_app/streamlit_app.py``. Dependencies are read from the repository
root ``requirements.txt``.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tifffile

# Make the repo root importable (paramisp_demo / paramisp_src live there).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prefer a repo-local weights/ directory when present; otherwise download into a
# guaranteed-writable temp directory (cloud checkouts may be read-only).
if not (ROOT / "weights").is_dir():
    os.environ.setdefault("PARAMISP_WEIGHTS_DIR", str(Path(tempfile.gettempdir()) / "paramisp_weights"))

from paramisp_demo import dataio, examples as demo_examples, inference, metrics, weights  # noqa: E402

st.set_page_config(page_title="ParamISP", page_icon="📷", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(mode: str, camera: str):
    """Load (and cache) a checkpoint for one direction and camera."""
    ckpt = weights.ensure_weights(mode, camera)
    device = inference.resolve_device(prefer_gpu=False)
    model, _, _ = inference.load_model(ckpt, inverse=(mode == "inverse"), device=device)
    return model, device


def save_upload(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def collect_params(camera, bayer, wb, use_exif, path, focal, aperture, exposure, iso):
    if use_exif:
        exif = dataio.read_exif_params(path)
    else:
        exif = {"focal_length": float(focal), "f_number": float(aperture),
                "exposure_time": float(exposure), "iso_sensitivity": float(iso)}
    return camera, bayer, wb, exif


def param_summary(camera, bayer, exif) -> str:
    return (f"Camera **{camera}** · Bayer **{bayer}** · "
            f"{exif['focal_length']:.1f} mm · f/{exif['f_number']:.1f} · "
            f"{exif['exposure_time']:.4f} s · ISO {exif['iso_sensitivity']:.0f}")


st.title("ParamISP — Learned Forward and Inverse ISPs")
st.caption(
    "CVPR 2024 · [paper](https://arxiv.org/abs/2312.13313) · "
    "[official code](https://github.com/woo525/ParamISP) · "
    "[project page](https://woo525.github.io/ParamISP/) · runs on **CPU**"
)

with st.sidebar:
    st.header("Camera parameters")
    camera = st.selectbox("Camera model", list(dataio.CAMERA_INFO), index=1)
    bayer = st.selectbox("Bayer pattern", dataio.BAYER_CHOICES)
    use_exif = st.checkbox("Read parameters from EXIF", value=False)
    with st.expander("Manual parameters", expanded=not use_exif):
        focal = st.number_input("Focal length (mm)", value=50.0, min_value=0.0)
        aperture = st.number_input("F-number", value=8.0, min_value=0.0)
        exposure = st.number_input("Exposure time (s)", value=0.01, min_value=0.0, format="%.5f")
        iso = st.number_input("ISO", value=200.0, min_value=0.0)
    with st.expander("White balance (R, G, B)"):
        wr = st.slider("R", 0.2, 4.0, 2.0, 0.01)
        wg = st.slider("G", 0.2, 4.0, 1.0, 0.01)
        wbb = st.slider("B", 0.2, 4.0, 1.5, 0.01)
    wb_tuple = (wr, wg, wbb)
    st.caption("Checkpoints download on first use (~95 MB, cached afterwards).")

tab_inv, tab_fwd = st.tabs(["sRGB → RAW (inverse ISP)", "RAW → sRGB (forward ISP)"])

with tab_inv:
    st.markdown(
        "Recover the underlying **Bayer RAW** from an sRGB image. "
        "Optionally supply the original RAW to score the prediction with **PSNR / SSIM**."
    )

    ex_labels = demo_examples.example_labels()
    source_opts = (["Built-in example", "Upload your own"] if ex_labels else ["Upload your own"])
    source = st.radio("Input source", source_opts, horizontal=True, key="inv_source")

    example = None
    up = None
    if source == "Built-in example" and ex_labels:
        choice = st.selectbox("Example", ex_labels, key="inv_example")
        example = demo_examples.load_example(choice)
        if example is not None:
            e1, e2 = st.columns(2)
            e1.image(example["srgb"], caption=f"Example sRGB · {example['camera']} · {example['bayer_pattern']}")
            if example["has_gt"]:
                e2.image(dataio.visualize_raw(example["raw"], example["bayer_pattern"],
                                              example["white_balance"],
                                              color_matrix=example["color_matrix"],
                                              camera_model=example["camera"]),
                         caption="Ground-truth RAW (preview)")
            else:
                e2.info("This example has no ground-truth RAW.")
            if example.get("note"):
                st.caption(example["note"])
    else:
        up = st.file_uploader("Input sRGB image",
                              type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"], key="inv")

    with st.expander("Ground-truth RAW (optional — enables PSNR / SSIM)", expanded=True):
        st.caption("Upload the original RAW for the same image. It may be a few sensor "
                   "pixels larger than the image (e.g. RAW 6036×4020 vs JPEG 6000×4000) — "
                   "the two are aligned and cropped to the same pixel size automatically. "
                   "The built-in examples already ship their ground truth, so no upload is "
                   "needed for those.")
        gt_upload = st.file_uploader("Original RAW (.npy/.tif or camera .dng/.arw/.nef/.cr2)",
                                     type=["npy", "tif", "tiff", "dng", "arw", "nef", "cr2", "raf", "rw2"],
                                     key="inv_gt")

    if st.button("Predict RAW", type="primary", key="inv_go"):
        error = None

        # ---- resolve inputs ----
        if example is not None:
            srgb = example["srgb"]
            cam, bp, wbv = example["camera"], example["bayer_pattern"], example["white_balance"]
            cm = example["color_matrix"]
            exif = example["exif"] or dict(dataio.DEFAULT_EXIF)
            gt, label = example["raw"], example["name"]
        elif up is not None:
            path = save_upload(up)
            cam, bp, wbv, exif = collect_params(camera, bayer, wb_tuple, use_exif, path,
                                                focal, aperture, exposure, iso)
            srgb = dataio.load_srgb(path)
            cm, gt, label = None, None, up.name
        else:
            srgb = cam = bp = wbv = cm = exif = gt = label = None
            error = "Upload an sRGB image, or pick a built-in example."

        if error is None and gt_upload is not None:
            try:
                gt, bayer_hint = dataio.load_raw_mosaic(save_upload(gt_upload))
                if bayer_hint:
                    bp = bayer_hint
            except Exception as exc:  # decode failure / rawpy unavailable
                error = f"Could not read the ground-truth RAW file: {exc}"
                gt = None
            if error is None:
                label = f"{label} (+ ground truth)"

        if error is None and gt is not None:
            srgb, gt, align_warnings = inference.register_gt(srgb, gt)
            for warn_msg in align_warnings:
                st.warning(warn_msg)

        if error is not None:
            st.warning(error)
        else:
            with st.spinner("Loading model and running inference…"):
                model, device = load_model("inverse", cam)
                start = time.perf_counter()
                mosaic = inference.predict_srgb_to_raw(
                    model, srgb, device, camera_model=cam, bayer_pattern=bp,
                    white_balance=wbv, color_matrix=cm, exif=exif)
                elapsed = time.perf_counter() - start
                preview = dataio.visualize_raw(mosaic, bp, wbv, color_matrix=cm, camera_model=cam)

            st.success(f"{param_summary(cam, bp, exif)}  ·  {label}")
            # ---- metrics ----
            if gt is not None:
                scores = metrics.compare(mosaic, gt)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("PSNR", f"{scores['psnr']:.2f} dB")
                m2.metric("SSIM", f"{scores['ssim']:.4f}")
                m3.metric("MAE", f"{scores['mae']:.4f}")
                m4.metric("Inference time", f"{elapsed * 1000:.0f} ms")
            else:
                st.info(f"Inference time: **{elapsed * 1000:.0f} ms** — add the ground-truth "
                        "RAW to compute PSNR / SSIM.")

            # ---- images ----
            if gt is not None:
                c1, c2, c3 = st.columns(3)
                c1.image(srgb, caption="Input sRGB")
                c2.image(preview, caption="Predicted RAW (preview)")
                c3.image(dataio.visualize_raw(gt, bp, wbv, color_matrix=cm, camera_model=cam),
                         caption="Ground-truth RAW (preview)")
                diff = np.abs(mosaic - gt)
                dm = float(np.nanmax(diff)) if diff.size else 1.0
                st.image(np.divide(diff, dm, out=np.zeros_like(diff), where=dm > 0),
                         caption="|predicted − ground truth| (normalised)")
            else:
                c1, c2 = st.columns(2)
                c1.image(srgb, caption="Input sRGB")
                c2.image(preview, caption="Predicted RAW (white-balanced preview)")

            # ---- download ----
            buf = io.BytesIO()
            tifffile.imwrite(buf, (np.clip(mosaic, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16))
            st.download_button("Download predicted RAW (16-bit TIFF)", buf.getvalue(),
                               file_name="paramisp_raw_prediction.tiff", mime="image/tiff")

with tab_fwd:
    st.markdown("Render the **sRGB** image a camera ISP would produce from a RAW mosaic.")
    up2 = st.file_uploader("Input RAW (.npy/.tif or camera .dng/.arw/.nef/.cr2)",
                           type=["npy", "tif", "tiff", "dng", "arw", "nef", "cr2", "raf", "rw2"], key="fwd")
    if st.button("Render sRGB", type="primary", key="fwd_go"):
        if up2 is None:
            st.warning("Upload a RAW file first.")
        else:
            with st.spinner("Loading model and running inference…"):
                path = save_upload(up2)
                cam, bp, wbv, exif = collect_params(
                    camera, bayer, wb_tuple, use_exif, path, focal, aperture, exposure, iso)
                try:
                    raw, hint = dataio.load_raw_mosaic(path)
                except Exception as exc:  # decode failure / rawpy unavailable
                    st.error(f"Could not read the RAW file: {exc}")
                    st.stop()
                bp = hint or bp
                model, device = load_model("forward", cam)
                out = inference.predict_raw_to_srgb(
                    model, raw, device, camera_model=cam, bayer_pattern=bp,
                    white_balance=wbv, exif=exif)
            st.success(param_summary(cam, bp, exif))
            st.image(np.clip(out, 0.0, 1.0), caption="Rendered sRGB")

st.divider()
st.caption(
    "Full source: [github.com/gliderest/paramISP](https://github.com/gliderest/paramISP)"
)
