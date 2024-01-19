# -*- coding: utf-8 -*-
# File: imgen/gan/realesrgan/scripts/pytorch2onnx.py

"""
Run

```bash
python -m pip install -U onnx onnxscript
```

to install ONNX.
"""

import argparse
import os

import torch
import torch.onnx
from utils.file_ops import get_filename

from imgen import DEVICE
from imgen.gan.realesrgan.archs.srvgg_arch import SRVGGNetCompact
from imgen.gan.realesrgan.basicsr.archs.rrdbnet_arch import RRDBNet


def convert(
    model_path: str,
    output_dir: str = ".",
):
    model_filename = get_filename(model_path)
    if model_filename == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
    elif model_filename in {"realesr-general-x4v3", "realesr-general-wdn-x4v3"}:
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
        )[f"params{'_ema' if model_filename == 'RealESRGAN_x4plus' else ''}"]
    )
    model.eval()

    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64, requires_grad=True),
        os.path.join(output_dir, f"{model_filename}.onnx"),
        opset_version=11,
        export_params=True,
    )


if __name__ == "__main__":
    """Convert pytorch model to onnx models"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="PyTorch model path (.pth).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help='Output directory. Defaults to "."',
    )
    args = parser.parse_args()

    convert(args.model, args.output_dir)
