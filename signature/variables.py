import os
import torch
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(__file__).resolve().parents[3]

CHEQUE_PDF_FROM_API = os.path.join(ROOT_DIR, "api_pdf_files")

# models path
PRINTED_DONUT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "cheque-modelv-printed")
HANDWRITTEN_DONUT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "cheque-modelv1")
SIGNATURE_MODEL_PATH = os.path.join(
    ROOT_DIR, "models", "inception_resnetv2_triangular_e5"
)

TR_OCR_PATH = os.path.join(ROOT_DIR, "models", "trocr-large-printed")
NEP_OCR_PATH = os.path.join(ROOT_DIR, "models", "nepali_digits_v3.pth")
LANGUGAE_DETECTION_MODEL = os.path.join(
    ROOT_DIR, "models", "object_detection", "en_ne_classification_model.pth"
)


SIGNATURE_DETECTION_PATH = (
    rf"C:\Users\Sameer\Desktop\Samir\ict\models\signature\kumari.pt"
)

UPSCALER_DETECTION_PATH = rf"C:\Users\Sameer\Desktop\Samir\ict\models\upscaling\4xNomosWebPhoto_RealPLKSR.safetensors"

CLEANER_DETECTION_PATH = "models/cleaning"

SIGNATURE_DETECTION_THRESHOLD = 0.82
SIGNATURE_VERIFICATION_THRESHOLD = 0.785
