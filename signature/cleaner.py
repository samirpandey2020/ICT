# cleaner.py

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from image_gen_aux import UpscaleWithModel
from io import BytesIO
import requests
import logging

from signature.variables import *

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SignatureCleaner:
    def __init__(self, device: str = "cpu", load_upscaler: bool = True):
        """
        Initialize SignatureCleaner by loading the BiRefNet model and (optionally) the upscaler.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                CLEANER_DETECTION_PATH, trust_remote_code=True, local_files_only=True
            ).to(self.device)
            self.model.eval()
            logger.info("BiRefNet model loaded.")
            self.transform = transforms.Compose(
                [
                    transforms.Resize((512, 512)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        except Exception as e:
            logger.warning(
                f"Failed to load bitrefnet: {e}. Bitrefnet will not be available."
            )

        self.upscaler = None
        if load_upscaler:
            try:
                self.upscaler = UpscaleWithModel.from_pretrained(
                    UPSCALER_DETECTION_PATH
                ).to(self.device)
                logger.info("Upscaler model loaded.")
            except Exception as e:
                logger.warning(
                    f"Failed to load upscaler: {e}. Upscaling will not be available."
                )
                self.upscaler = None

    def _load_image(
        self, image_input: Union[str, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Load an image from path, URL, numpy array, or PIL object.
        """
        try:
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            elif isinstance(image_input, str):
                if image_input.startswith(("http://", "https://")):
                    response = requests.get(image_input)
                    response.raise_for_status()
                    return Image.open(BytesIO(response.content)).convert("RGB")
                else:
                    return Image.open(image_input).convert("RGB")
            else:
                raise ValueError(
                    "Input must be a file path, URL, numpy array, or PIL Image object"
                )
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def _upscale_image(self, image: Image.Image) -> Image.Image:
        """
        Upscale the image using the loaded upscaler model.
        """
        if not self.upscaler:
            logger.warning("Upscaler not loaded. Skipping upscale.")
            return image
        return self.upscaler(image, tiling=True, tile_width=512, tile_height=512)

    def _clean_image_with_model(self, image: Image.Image) -> Image.Image:
        """Remove background using BiRefNet and return cleaned image on a white background."""
        original_size = image.size
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)[-1].sigmoid().cpu()

        mask_pil = transforms.ToPILImage()(prediction[0].squeeze()).resize(
            original_size
        )

        background = Image.new("RGB", image.size, (255, 255, 255))
        image_rgb = image.convert("RGB")

        return Image.composite(image_rgb, background, mask_pil.convert("L"))

    def _normalize_image_layout(
        self, img: np.ndarray, size: Tuple[int, int], padding: int
    ) -> np.ndarray:
        """Normalize image size and position with dynamic canvas sizing.

        Args:
            img: Grayscale numpy array of the image
            size: Target size tuple (width, height)
            padding: Padding to apply around the signature

        Returns:
            Normalized image as numpy array
        """
        max_r_limit, max_c_limit = size

        # Ensure image is 8-bit grayscale for consistency
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)

        # Binarize the image using OTSU's algorithm
        threshold, binarized_image = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Find the coordinates of foreground pixels
        r, c = np.where(binarized_image == 0)

        # If no foreground pixels found, return blank canvas
        if r.size == 0 or c.size == 0:
            logger.warning(
                "No foreground pixels found for normalization. Returning blank canvas."
            )
            return np.ones(size, dtype=np.uint8) * 255

        # Calculate padded bounding box, clamped to image bounds
        r_min = max(r.min() - padding, 0)
        r_max = min(r.max() + padding, img.shape[0] - 1)
        c_min = max(c.min() - padding, 0)
        c_max = min(c.max() + padding, img.shape[1] - 1)

        # Crop the image with padded bounding box
        cropped = img[r_min : r_max + 1, c_min : c_max + 1]

        # Get dimensions of cropped image
        img_r, img_c = cropped.shape

        # Calculate scaling factor to fit within max dimensions
        reserved_padding = padding * 2

        available_width = max_c_limit - reserved_padding
        available_height = max_r_limit - reserved_padding

        # Ensure minimum space for content
        available_width = max(available_width, 10)
        available_height = max(available_height, 10)

        # Calculate scale based on available space
        scale_r = available_height / img_r if img_r > 0 else 0
        scale_c = available_width / img_c if img_c > 0 else 0

        scale = (
            min(scale_r, scale_c)
            if scale_r > 0 and scale_c > 0
            else (scale_r if scale_r > 0 else scale_c)
        )

        if scale == 0:
            logger.warning(
                "Cropped image has zero dimensions after padding. Returning blank canvas."
            )
            return np.ones(size, dtype=np.uint8) * 255

        # Resize cropped image to fit within available space
        new_r = int(img_r * scale)
        new_c = int(img_c * scale)

        if new_r == 0:
            new_r = 1
        if new_c == 0:
            new_c = 1

        cropped = cv2.resize(cropped, (new_c, new_r), interpolation=cv2.INTER_AREA)

        # Update dimensions after resize
        img_r, img_c = cropped.shape

        # Calculate starting positions to center the image
        r_start = (max_r_limit - img_r) // 2
        c_start = (max_c_limit - img_c) // 2

        # Create normalized image with target size, initialized to white
        normalized_image = np.ones(size, dtype=np.uint8) * 255

        # Add cropped and centered image to canvas
        normalized_image[r_start : r_start + img_r, c_start : c_start + img_c] = cropped

        return normalized_image

    def clean_and_normalize(
        self, image_input, normalize_size=(512, 512), upscale=True, padding=40
    ) -> np.ndarray:
        """
        Full pipeline to load, optionally upscale, clean, and normalize a signature image.
        Returns a normalized numpy array (grayscale).
        """
        try:
            pil_image = self._load_image(image_input)

            if upscale:
                pil_image = self._upscale_image(pil_image)

            cleaned_pil = self._clean_image_with_model(pil_image)

            cleaned_gray_np = np.array(cleaned_pil.convert("L"))

            normalized_np = self._normalize_image_layout(
                cleaned_gray_np, normalize_size, padding
            )

            # logger.info("Clean and normalize pipeline complete.")
            return normalized_np

        except Exception as e:
            logger.error(f"Failed in clean_and_normalize pipeline: {e}")
            raise e
