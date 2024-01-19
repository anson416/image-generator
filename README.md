# Image Generator

This image generation tool uses [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) and Generative Adversarial Network (GAN). Take a look at the demo on [Google Colab](https://colab.research.google.com/drive/1rlifoQyurOjFvd1qgcTACnj4KbpmvQkZ?usp=sharing) (Google Chrome is preferred).

## Installation

_Note_: Use Python 3.10 or above.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (find a version that suits you most).

2. Clone this repository and change directory:

   ```bash
   git clone https://github.com/anson416/image-generator.git
   cd image-generator
   ```

3. Download Real-ESRGAN models ([realesr-general-x4v3.pth
](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth), [realesr-general-wdn-x4v3.pth
](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth), [RealESRGAN_x4plus.pth
](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)) to `./imgen/gan/realesrgan/weights` ([backup](https://drive.google.com/drive/folders/1cmEja-8RX8kxcWnbCpNMfD_4Qx-3ASgQ?usp=share_link)).

4. Install `imgen`:

   ```bash
   python -m pip install -e .
   ```

## Usage

### Stable Diffusion

#### Positive Prompt

Suggested prefix: "(((masterpiece))), (((best quality))), ((ultra-detailed)), ((8k))"

If you want to prevent NSFW content, you can add "(family friendly:0.85)" after the above prefix.

#### Negative Prompt

Suggested prefix: "lowres, worst quality, low quality, standard quality, error, jpeg artifacts, blurry, username, signature, watermark, text"

If you want to prevent NSFW content, you can insert "((nsfw)), ((nude))" before the above prefix.

## Materials

1. [Prompt Templates for Stable Diffusion](https://github.com/Dalabad/stable-diffusion-prompt-templates)
2. [The Code of Quintessence](https://docs.qq.com/doc/DWHl3am5Zb05QbGVs)

## Acknowledgement

- [BasicSR](https://github.com/XPixelGroup/BasicSR): An open-source image and video restoration toolbox based on PyTorch.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): A practical algorithm for general image/video restoration.
