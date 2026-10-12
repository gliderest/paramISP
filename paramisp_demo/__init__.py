"""ParamISP demo runtime (Hugging Face Space).

Lightweight, PyTorch-Lightning-free runtime for the CVPR 2024 paper
"ParamISP: Learned Forward and Inverse ISPs using Camera Parameters"
(https://github.com/woo525/ParamISP).

The vendored upstream sources live in ``paramisp_src/`` and are used as-is for
the network layers (``layers``) and the camera-parameter encoders
(``data.utils``).  See ``paramisp_demo/net.py`` for the network definition.
"""

from .net import ParamISPNet  # noqa: F401

__all__ = ["ParamISPNet"]
