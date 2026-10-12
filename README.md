# ParamISP — forward / inverse camera ISP demo

Interactive demo for **ParamISP: Learned Forward and Inverse ISPs using Camera
Parameters** (CVPR 2024).

* Paper: <https://arxiv.org/abs/2312.13313>
* Official code: <https://github.com/woo525/ParamISP>
* Project page: <https://woo525.github.io/ParamISP/>

The model is only **~0.7M parameters** and runs on **CPU** — no GPU required.

Both directions of the learned camera ISP are exposed:

| Direction | Input | Output |
|---|---|---|
| **Inverse ISP** (sRGB → RAW) | an sRGB image + camera parameters | the predicted single-channel Bayer RAW |
| **Forward ISP** (RAW → sRGB) | a Bayer RAW file + camera parameters | the rendered sRGB image |

Camera parameters (ISO, exposure time, focal length, f-number, white balance and
the camera→sRGB colour matrix) are read from the uploaded image's EXIF when
available, or set manually.

## Project layout

```
streamlit_app/streamlit_app.py        the web UI (Streamlit)
paramisp_demo/
  net.py                              the ParamISP network (PyTorch port, no Lightning)
  dataio.py                           image/EXIF loading + the network input batch
  inference.py                        checkpoint loading + sRGB<->RAW prediction
  weights.py                          locate / download the official checkpoints
paramisp_src/                         vendored upstream sources (layers/, data/, utils/)
notebooks/paramisp_demo_colab.ipynb   Google Colab runner
scripts/                              tests + helper scripts
requirements.txt                      runtime dependencies
```

`paramisp_demo/net.py` is a verbatim port of the computation in
`paramisp_src/models/paramisp.py` with the PyTorch-Lightning scaffolding removed,
so the official `.ckpt` files load directly. The vendored `layers/` and
`data.utils` modules are used unchanged.

## Running the app

```bash
python -m venv .venv
. .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app/streamlit_app.py
```

## Deploying to Streamlit Community Cloud (free)

Streamlit Community Cloud hosts a persistent public web app for free (no GPU;
apps sleep after 12 h of inactivity and wake on visit).

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → *Deploy a public app from GitHub*.
3. Repository `<user>/paramISP`, Branch `main`,
   **Main file path** `streamlit_app/streamlit_app.py`.
4. *(Optional)* Advanced settings → pick a **Python version** (3.12 is a safe choice).
5. **Deploy.** The first run downloads the checkpoints (~95 MB) once.

Dependencies are read from the repository-root `requirements.txt`.

## Running on Google Colab (free, optional GPU)

Open `notebooks/paramisp_demo_colab.ipynb`. CPU is sufficient; a free T4 GPU is
available via *Runtime ▸ Change runtime type* if you want it.

## Checkpoints

The official weights are not redistributed here. They are located, in order, at:

1. `weights/<mode>/<camera>.ckpt` (e.g. `weights/inverse/D7000.ckpt`);
2. the authors' shared Google Drive archive (`weights.zip`), downloaded on first
   use with `gdown`.

Override the download id with `PARAMISP_GDRIVE_ZIP_ID`, or point
`PARAMISP_WEIGHTS_DIR` at a directory that already holds the checkpoints.

Camera models: `A7R3`, `D7000`, `D90`, `D40`, `S7`. Modes: `inverse`, `forward`.

## Tests

```bash
pip install -r requirements.txt
python scripts/smoke_test.py            # forward pass per direction, no weights
python scripts/real_inference_test.py   # full pipeline with the official weights
python scripts/minimal_deps_test.py     # runs with torchvision / rawpy absent
python scripts/streamlit_smoketest.py   # headless Streamlit AppTest
```

## Citation

```
@inproceedings{kim2024paramisp,
  title={ParamISP: Learned Forward and Inverse ISPs using Camera Parameters},
  author={Kim, Woohyeok and Kim, Geonu and Lee, Junyong and Lee, Seungyong and Baek, Seung-Hwan and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26067--26076},
  year={2024}
}
```
