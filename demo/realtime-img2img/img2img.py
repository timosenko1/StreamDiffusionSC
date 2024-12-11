import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

import numpy as np
import cv2 as cv
from utils.wrapper import StreamDiffusionWrapper

import torch

from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import math

base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

default_prompt = "good_twenty_guy as a guy with light hair and blue eyes, wearing a fitted brown quilted jacket, a teal-blue scarf, and tan utility pants, paired with sturdy high-cut boots. Both his arms are cybernetic, featuring sleek metallic designs with blue accents"
default_negative_prompt = "black and white, blurry, low resolution, pixelated,  pixel art, low quality, low fidelity"

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion Sonce DEMO</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
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
            512,
            min=2,
            max=15,
            title="Width",
            disabled=True,
            hide=True,
            id="width",
        )
        height: int = Field(
            512,
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
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            lora_dict={"/home/ubuntu/kairon_sdxl.safetensors": 1.0},
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",
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

        # === Preprocessing Configuration ===
        # Set the preprocessing mode here. Options:
        # 'none' - No preprocessing
        # 'noise' - Add Gaussian noise
        # 'grayscale' - Convert to grayscale
        # 'canny' - Apply Canny edge detection
        self.preprocessing_mode = "canny"  # Change this to desired mode

        # Parameters for preprocessing
        self.noise_level = 50  # Adjust the noise level (0-255)
        self.canny_threshold1 = 100  # First threshold for the hysteresis
        self.canny_threshold2 = 200  # Second threshold for the hysteresis

    def add_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to the image."""
        np_image = np.array(image)
        noise = np.random.randint(
            0, self.noise_level, np_image.shape, dtype="uint8"
        )
        noised_image = cv.add(np_image, noise)
        return Image.fromarray(noised_image)

    def convert_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert the image to grayscale."""
        return image.convert("L").convert(
            "RGB"
        )  # Convert back to RGB for consistency

    def apply_canny(self, image: Image.Image) -> Image.Image:
        """Apply Canny edge detection to the image."""
        np_image = np.array(image.convert("L"))  # Convert to grayscale
        edges = cv.Canny(
            np_image, self.canny_threshold1, self.canny_threshold2
        )
        # Convert edges to 3-channel image
        edges_rgb = cv.cvtColor(edges, cv.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply the selected preprocessing to the image."""
        if self.preprocessing_mode == "noise":
            return self.add_noise(image)
        elif self.preprocessing_mode == "grayscale":
            return self.convert_grayscale(image)
        elif self.preprocessing_mode == "canny":
            return self.apply_canny(image)
        else:
            return image  # No preprocessing

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Preprocess the input image based on the selected mode
        preprocessed_image = self.preprocess_image(params.image)

        # Continue with the existing prediction pipeline
        image_tensor = self.stream.preprocess_image(preprocessed_image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)

        return output_image
