# Image Generator

This image generation tool uses [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) and [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Take a look at the demo on [Google Colab](https://colab.research.google.com/drive/1rlifoQyurOjFvd1qgcTACnj4KbpmvQkZ?usp=sharing).

## Installation

_Note_: Use Python 3.10 or above.

1. Install [PyTorch](https://pytorch.org/get-started/locally/) (find a version that suits you most).

2. Install `imgen`:

   ```bash
   python -m pip install -U git+https://github.com/anson416/image-generator.git
   ```

## Usage

### Stable Diffusion

#### Positive Prompt

Suggested prefix: "(((masterpiece))), (((best quality))), ((ultra-detailed)), ((8k))"

If you want to prevent NSFW content, you can add "(family friendly:0.85)" after the above prefix.

#### Negative Prompt

Suggested prefix: "lowres, worst quality, low quality, standard quality, error, jpeg artifacts, blurry, username, signature, watermark, text"

If you want to prevent NSFW content, you can insert "((nsfw)), ((nude))" before the above prefix.
