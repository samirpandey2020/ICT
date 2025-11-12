import os
import cv2
import torch
import supervision as sv
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import uuid
import logging
from dataclasses import dataclass, field
from signature.variables import SIGNATURE_DETECTION_PATH, SIGNATURE_DETECTION_THRESHOLD
from signature.verification import clean_signatures

logger = logging.getLogger(__name__)


@dataclass
class DetectedSignature:
    image: np.ndarray
    confidence: float
    bbox: Tuple[int, int, int, int]
    signature_id: str
    unique_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cleaned_image: np.ndarray = None


class SignatureDetector:
    def __init__(self, model_path: str = SIGNATURE_DETECTION_PATH, device: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Determine the best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"YOLO model using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.box_annotator = sv.BoxAnnotator()

    def detect_signatures(
        self,
        image_input,
        confidence_threshold: float = SIGNATURE_DETECTION_THRESHOLD,
        upscale: bool = False,
    ) -> List[DetectedSignature]:
        """Detect signatures in an image using numpy array input only.

        Args:
            image_input: numpy array of the image
            confidence_threshold: minimum confidence for detection
            upscale: whether to upscale small signatures

        Returns:
            List of DetectedSignature objects
        """
        if isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError("image_input must be a numpy array")

        if image is None or image.size == 0:
            raise ValueError("Invalid image array provided")

        results = self.model(image)
        detections = sv.Detections.from_ultralytics(results[0])
        detected_signatures = []

        for i, (xyxy, confidence) in enumerate(
            zip(detections.xyxy, detections.confidence)
        ):
            if confidence < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            padding = 0
            h, w = image.shape[:2]
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

            cropped = image[y1:y2, x1:x2]

            cleaned_image = clean_signatures(cropped, upscale)

            signature_id = f"sig_{i}"

            detected_signatures.append(
                DetectedSignature(
                    image=cropped,
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    signature_id=signature_id,
                    cleaned_image=cleaned_image,
                )
            )
        return detected_signatures

    def detect_and_annotate(self, image: np.ndarray) -> np.ndarray:
        """Annotate an image with detected signature bounding boxes.

        Args:
            image: numpy array of the image

        Returns:
            Annotated image as numpy array
        """
        results = self.model(image)
        detections = sv.Detections.from_ultralytics(results[0])
        annotated_image = self.box_annotator.annotate(
            scene=image.copy(), detections=detections
        )
        return annotated_image

    def process_batch(
        self, images: List[np.ndarray]
    ) -> Dict[int, List[DetectedSignature]]:
        """Process a batch of images and detect signatures in each.

        Args:
            images: List of numpy arrays representing images

        Returns:
            Dictionary mapping image index to list of detected signatures
        """
        results = {}
        for idx, image in enumerate(images):
            try:
                detected_signatures = self.detect_signatures(
                    image_input=image, upscale=False
                )
                results[idx] = detected_signatures
            except Exception as e:
                print(f"Error processing image at index {idx}: {str(e)}")
                results[idx] = []
        return results
