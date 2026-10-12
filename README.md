---
title: ParamISP
emoji: 📷
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# ParamISP — interactive demo

Hugging Face Space for **ParamISP: Learned Forward and Inverse ISPs using Camera
Parameters** (CVPR 2024).

* Paper: <https://arxiv.org/abs/2312.13313>
* Code: <https://github.com/woo525/ParamISP>
* Project page: <https://woo525.github.io/ParamISP/>

The demo exposes both directions of the learned camera ISP:

| Tab | Direction | Input | Output |
|-----|-----------|-------|--------|
| **sRGB → RAW** | inverse ISP | an sRGB image + camera parameters | the predicted single-channel Bayer RAW |
| **RAW → sRGB** | forward ISP | a Bayer RAW file + camera parameters | the rendered sRGB image |

Camera parameters (ISO, exposure time, focal length, f-number, white balance and
the camera→sRGB color matrix) are read from the uploaded image's EXIF when
available, or set manually.

## How it works

```
app.py                      Gradio interface
paramisp_demo/
  net.py                    the ParamISP network (PyTorch port, no Lightning)
  dataio.py                 image/EXIF loading + building the network input batch
  inference.py              checkpoint loading + sRGB<->RAW prediction
  weights.py                locate / download the official checkpoints
paramisp_src/               vendored upstream sources (layers/, data/, utils/)
```

`paramisp_demo/net.py` is a verbatim port of the computation in
`paramisp_src/models/paramisp.py` with the PyTorch-Lightning scaffolding removed,
so the official `.ckpt` files load directly. The vendored `layers/` and
`data.utils` modules are used unchanged.

## Checkpoints

The official weights are not redistributed here. The app looks for them, in
order, at:

1. `weights/<mode>/<camera>.ckpt` (e.g. `weights/inverse/D7000.ckpt`);
2. a Hugging Face repo set via `PARAMISP_WEIGHTS_REPO`;
3. the authors' shared Google Drive folder (`PARAMISP_GDRIVE_ID`).

Camera models: `A7R3`, `D7000`, `D90`, `D40`, `S7`. Modes: `inverse`, `forward`.

## Running locally

```bash
python -m venv .venv
. .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

`scripts/smoke_test.py` runs a forward pass per direction without weights;
`scripts/real_inference_test.py` runs the full pipeline with the official weights.

## Deploying to Hugging Face Spaces

```bash
huggingface-cli login --token <WRITE_TOKEN>
python scripts/deploy_space.py <user>/<space-name>
```

The Space is created with the Gradio SDK and all runtime files are uploaded.
Checkpoints are **not** committed; they are downloaded on first use from the
authors' shared `weights.zip` (see *Checkpoints* above). To use a Hub repo
instead, upload once with `python scripts/publish_weights.py <user>/<repo>` and
set the Space variable `PARAMISP_WEIGHTS_REPO=<user>/<repo>`.

> On a free Hugging Face account, Gradio Spaces must run on **ZeroGPU**; the app
> automatically uses `spaces.GPU` when the `spaces` package is present.

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
