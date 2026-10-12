"""The ParamISP network, without the PyTorch Lightning scaffolding.

This is a faithful port of the computation performed by the ``ParamISP`` class
in ``paramisp_src/models/paramisp.py`` (inference path only).  Every sub-module
is created with the exact same name and order as in the upstream code, so the
official ``.ckpt`` files load directly into this module.

The only differences from upstream are:
  * it subclasses ``torch.nn.Module`` instead of ``pl.LightningModule``;
  * the training-only branches (noise injection, loss, metrics, logging) and
    the unused ``optics_mask`` (whose result is discarded upstream) are removed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init

# The vendored upstream code imports its packages at the top level
# (``import layers.bayer``, ``import data.utils``, ...), so make it importable.
_SRC_DIR = Path(__file__).resolve().parent.parent / "paramisp_src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import data.utils  # noqa: E402
import layers.bayer  # noqa: E402
import layers.color  # noqa: E402
import layers.misc  # noqa: E402
import layers.nn.v2  # noqa: E402


class _Args:
    """Minimal stand-in for the upstream ``arg.Namespace``."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ParamISPNet(nn.Module):
    """ParamISP network for a single direction.

    Args:
        inverse: ``True``  -> inverse ISP, sRGB  -> RAW (Bayer).
                 ``False`` -> forward ISP, RAW (Bayer) -> sRGB.
    """

    def __init__(self, inverse: bool = True):
        super().__init__()
        self.args = _Args(inverse=inverse)

        # ``AlgorithmOnlyModel.set_attributes``
        self.demosaic = layers.bayer.Demosaic()

        # ``ParamISP.set_attributes`` (order preserved for checkpoint keys)
        self.optics_dropout_probability = 0.2
        self.id_channels = 16  # camera id variant
        self.feat_channels = (3 * (1 + 2 + 4 + 8 + 16) + 3)  # pixel, gradient, soft histogram, over-exposed mask

        self.main_channels = 64
        self.hyper_channels = 64

        self.tonenet = layers.nn.v2.GlobalNet(self.feat_channels, self.main_channels)
        self.localnet = layers.nn.v2.LocalNet(self.feat_channels, self.main_channels, num_block=2, num_scale=2)

        self.embed_id = layers.nn.v2.Linear(self.id_channels, self.hyper_channels)   # placeholder; unused
        self.embed_color = layers.nn.v2.Linear(24, self.hyper_channels)              # white balance, color matrix
        self.embed_focallen = layers.nn.v2.Linear(15, self.hyper_channels)           # focal length
        self.embed_aperture = layers.nn.v2.Linear(15, self.hyper_channels)           # f-number
        self.embed_expstime = layers.nn.v2.Linear(15, self.hyper_channels)           # exposure time
        self.embed_senstvty = layers.nn.v2.Linear(15, self.hyper_channels)           # sensitivity

        self.embednet = nn.Sequential(
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels, self.hyper_channels),
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels, self.hyper_channels),
            nn.ELU(inplace=True), layers.nn.v2.Linear(self.hyper_channels,
                                                      self.hyper_channels, init_weights=lambda x: init.xavier_normal_(x, 0.1)),
        )

        self.hypertonenet = layers.nn.v2.HyperGlobalNet(self.tonenet, self.hyper_channels)
        self.hyperlocalnet = layers.nn.v2.SynHyperLocalNet(self.localnet, self.hyper_channels)

    # ``ParamISP.get_embedding``
    def get_embedding(self, batch: dict) -> torch.Tensor:
        embed: torch.Tensor
        embed = 0 * self.embed_id(data.utils.embed_camera_id(data.utils.get_camera_id_batch(batch), self.id_channels))

        embed = embed + self.embed_color(
            torch.cat([data.utils.embed_white_balance(batch["white_balance"]),
                       data.utils.embed_color_matrix(batch["color_matrix"])], dim=-1))

        embed = embed + self.embed_focallen(data.utils.embed_focal_length(batch["focal_length"]))
        embed = embed + self.embed_aperture(data.utils.embed_aperture(batch["f_number"]))
        embed = embed + self.embed_expstime(data.utils.embed_exposure_time(batch["exposure_time"]))
        embed = embed + self.embed_senstvty(data.utils.embed_sensitivity(batch["iso_sensitivity"]))

        embed = self.embednet(embed)
        return embed

    # ``ParamISP.get_common_features``
    def get_common_features(self, x: torch.Tensor) -> torch.Tensor:
        return layers.nn.v2.over_exposed_mask(x)

    # ``ParamISP.get_input_features``
    def get_input_features(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, layers.nn.v2.content_features(
            x, use_gradient=True, histogram_bins=[4, 8, 16]), z], dim=-3)

    # ``ParamISP.forward`` (evaluation branch)
    def forward(self, batch: dict) -> torch.Tensor:
        embed = self.get_embedding(batch)

        if self.args.inverse:  # sRGB -> RAW
            x: torch.Tensor = batch["rgb"]

            common_features = self.get_common_features(x)

            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)

            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)

            x = layers.color.apply_color_matrix(x, batch["color_matrix"].inverse())
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], 1 / batch["white_balance"], mosaic_flag=False)
            x = layers.bayer.mosaic(x, batch["bayer_pattern"])

        else:  # RAW -> sRGB
            x = batch["raw"]

            x = self.demosaic(x, batch["bayer_pattern"])
            x = layers.bayer.apply_white_balance(x, batch["bayer_pattern"], batch["white_balance"], mosaic_flag=False)
            x = x.clip(0, 1)
            x = layers.color.apply_color_matrix(x, batch["color_matrix"])

            common_features = self.get_common_features(x)

            z = self.get_input_features(x, common_features)
            x = self.hyperlocalnet(x, z, embed)

            z = self.get_input_features(x, common_features)
            x = self.hypertonenet(x, z, embed)

        return x
