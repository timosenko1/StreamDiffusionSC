import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
# taesd_model = "madebyollin/taesd"
# lcm_model = "/home/ubuntu/models/dreamshaper_lcm.safetensors"
lcm_model = "stabilityai/sd-turbo"
# lora_dict = {"/home/ubuntu/models/lcm_kairon.safetensors": 1.0}
# lora_dict = {"/home/ubuntu/models/rembg_kairon.safetensors": 1.0}
# lora_dict = {"/home/ubuntu/models/last-000024.safetensors": 1.0}
lora_dict = None

default_prompt = "kairon, 1boy, white hair, blue eyes, fitted brown quilted jacket, a teal-blue scarf, cybernetic arms, sleek metallic designs, blue accents, comic style, solo"
default_negative_prompt = "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">
    This demo showcases
    <a
    href="https://github.com/cumulo-autumn/StreamDiffusion"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">StreamDiffusion
</a>
Image to Image pipeline using
    <a
    href="https://huggingface.co/stabilityai/sd-turbo"
    target="_blank"
    class="text-blue-500 underline hover:no-underline">SD-Turbo</a
    > with a MJPEG stream server.
</p>
"""


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        # negative_prompt: str = Field(
        #     default_negative_prompt,
        #     title="Negative Prompt",
        #     field="textarea",
        #     id="negative_prompt",
        # )
        width: int = Field(
            768,
            min=2,
            max=15,
            title="Width",
            disabled=True,
            hide=True,
            id="width",
        )
        height: int = Field(
            768,
            min=2,
            max=15,
            title="Height",
            disabled=True,
            hide=True,
            id="height",
        )

    def __init__(
        self, args: Args, device: torch.device, torch_dtype: torch.dtype
    ):
        params = self.InputParams()
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=lcm_model,
            use_tiny_vae=True,
            device=device,
            dtype=torch_dtype,
            t_index_list=[25, 32, 37, 48],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            lora_dict=lora_dict,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="self",
            use_safety_checker=args.safety_checker,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir=args.engine_dir,
        )

        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt=default_negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2,
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        image_tensor = self.stream.preprocess_image(params.image, False, False)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
