# img2img.py

import sys
import os
from typing import Optional
import io
import base64

# Adjust the system path to include the parent directories
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

from utils.wrapper import StreamDiffusionWrapper

import torch
from enum import Enum
from config import Args
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import cv2 as cv  # OpenCV for image processing
import math

# Define model paths
base_model = "stabilityai/sd-turbo"
taesd_model = "madebyollin/taesd"

# Default prompts
default_prompt = "good_twenty_guy as a guy with light hair and blue eyes, wearing a fitted brown quilted jacket, a teal-blue scarf, and tan utility pants, paired with sturdy high-cut boots. Both his arms are cybernetic, featuring sleek metallic designs with blue accents"
default_negative_prompt = "black and white, blurry, low resolution, pixelated, pixel art, low quality, low fidelity"

# Default Parameters (Adjust these locally as needed)
LOCAL_GUIDANCE_SCALE = 1.2
LOCAL_NOISE_AMOUNT = 0.0
LOCAL_GRAYSCALE = False
LOCAL_CONTROLNET_MODE = "none"  # Options: "none", "canny", "depth"
LOCAL_WIDTH = 512
LOCAL_HEIGHT = 512

# Page content for frontend
page_content = """<h1 class="text-3xl font-bold">StreamDiffusion Sonce DEMO</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
"""


class ControlNetMode(str, Enum):
    none = "none"
    canny = "canny"
    depth = "depth"


class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParamsBase(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        guidance_scale: float = Field(
            LOCAL_GUIDANCE_SCALE,
            title="Guidance Scale",
            description="Controls how strongly the image follows the prompt.",
            ge=0.0,
            le=20.0,
            id="guidance_scale",
        )
        noise_amount: float = Field(
            LOCAL_NOISE_AMOUNT,
            title="Noise Amount",
            description="Amount of noise to add to the input image.",
            ge=0.0,
            le=1.0,
            id="noise_amount",
        )
        grayscale: bool = Field(
            LOCAL_GRAYSCALE,
            title="Grayscale",
            description="Convert input image to grayscale.",
            id="grayscale",
        )
        controlnet_mode: ControlNetMode = Field(
            LOCAL_CONTROLNET_MODE,
            title="ControlNet Mode",
            description="Select ControlNet preprocessing mode.",
            id="controlnet_mode",
        )
        width: int = Field(
            LOCAL_WIDTH,
            ge=2,
            le=2048,
            title="Width",
            disabled=True,
            hide=True,
            id="width",
        )
        height: int = Field(
            LOCAL_HEIGHT,
            ge=2,
            le=2048,
            title="Height",
            disabled=True,
            hide=True,
            id="height",
        )

    class InputParams(InputParamsBase):
        image: Image.Image = Field(
            ...,  # This field will be provided during runtime
            title="Input Image",
            description="The input image for img2img processing.",
            id="image",
        )

        class Config:
            arbitrary_types_allowed = (
                True  # Allow arbitrary types like PIL.Image.Image
            )

    def __init__(
        self, args: Args, device: torch.device, torch_dtype: torch.dtype
    ):
        params = self.InputParamsBase()
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=10,
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            lora_dict={"/home/ubuntu/kairon_sdxl.safetensors": 1.0},
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
            guidance_scale=LOCAL_GUIDANCE_SCALE,
        )

    def preprocess_image(
        self, image: Image.Image, params: "Pipeline.InputParams"
    ) -> torch.Tensor:
        """
        Preprocess the input image based on the provided parameters.

        Args:
            image (Image.Image): The input PIL image.
            params (Pipeline.InputParams): The input parameters for preprocessing.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        # Convert to grayscale if required
        if params.grayscale:
            image = image.convert("L").convert("RGB")

        # Add noise if required
        if params.noise_amount > 0.0:
            # Convert PIL image to NumPy array
            img_array = np.array(image).astype(np.float32)

            # Generate noise
            noise = (
                np.random.randn(*img_array.shape) * params.noise_amount * 255
            )

            # Add noise and clip to valid range
            img_noisy = img_array + noise
            img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

            # Convert back to PIL Image
            image = Image.fromarray(img_noisy)

        # Apply ControlNet preprocessing if required
        if params.controlnet_mode == ControlNetMode.canny:
            image = self.apply_canny(image)
        elif params.controlnet_mode == ControlNetMode.depth:
            image = self.apply_depth(image)

        # Preprocess the image using the StreamDiffusionWrapper
        return self.stream.preprocess_image(image)

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        """
        Generate an output image based on the input parameters.

        Args:
            params (Pipeline.InputParams): The input parameters for prediction.

        Returns:
            Image.Image: The generated output image.
        """
        # Preprocess the input image
        image_tensor = self.preprocess_image(params.image, params)

        # Generate the output image
        output_image = self.stream(
            image=image_tensor,
            prompt=params.prompt,
            guidance_scale=params.guidance_scale,
        )

        return output_image

    @staticmethod
    def apply_canny(image: Image.Image) -> Image.Image:
        """
        Apply Canny edge detection to the input image.

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            Image.Image: The image after applying Canny edge detection.
        """
        # Convert PIL Image to grayscale NumPy array
        img_gray = image.convert("L")
        img_array = np.array(img_gray)

        # Apply Canny edge detection
        edges = cv.Canny(img_array, 100, 200)

        # Convert edges back to PIL Image
        edges_image = Image.fromarray(edges).convert("RGB")

        return edges_image

    @staticmethod
    def apply_depth(image: Image.Image) -> Image.Image:
        """
        Apply Depth ControlNet preprocessing to the input image.
        (Currently returns the image unchanged.)

        Args:
            image (Image.Image): The input PIL image.

        Returns:
            Image.Image: The image after applying Depth ControlNet preprocessing.
        """
        # Placeholder for Depth ControlNet processing
        # Currently returns the image unchanged
        return image
