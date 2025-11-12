# signature.py (Updated to carry pre-processing data)

import os
import cv2
import logging
import tempfile
import numpy as np
import matplotlib
import concurrent.futures
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

matplotlib.use("Agg")

from signature.variables import SIGNATURE_DETECTION_PATH
from signature.detection import SignatureDetector, DetectedSignature
from signature.signature import (
    preprocess_signature_image,
    compare_boundary_signatures_with_preprocessed,
    compare_texture_signatures,
    normalize_custom,
    hog_intervals,
    hog_outputs,
    scaler_intervals,
    scaler_outputs,
    swift_intervals,
    swift_outputs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- CHANGE 1: ADD FIELDS TO STORE PRE-PROCESSED IMAGES ---
@dataclass
class SignatureMatch:
    signature_id: str
    score: float
    image_path: str
    matched_signature: DetectedSignature
    cheque_signature: DetectedSignature
    comparison_image_path: Optional[str] = None
    cheque_bbox: Optional[Tuple[int, int, int, int]] = None
    reference_bbox: Optional[Tuple[int, int, int, int]] = None
    # These new fields will hold the dictionaries of pre-processed images
    cheque_preprocessed_images: Optional[Dict] = None
    ref_preprocessed_images: Optional[Dict] = None


# --- END OF CHANGE 1 ---


class SignatureVerifier:
    def __init__(self, model_path: str = SIGNATURE_DETECTION_PATH, padding=40):
        self.detector = SignatureDetector(model_path)
        self.padding = padding
        self._performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_comparisons": 0,
            "skipped_comparisons": 0,
        }

    def verify_signatures(
        self,
        cheque_image_path: str,
        reference_image_paths: List[str],
        output_dir: Optional[str] = None,
        generate_comparisons: bool = True,
    ) -> List[SignatureMatch]:
        logger.info(
            f"Starting verification for cheque: {os.path.basename(cheque_image_path)}"
        )
        try:
            cheque_signatures = self.detector.detect_signatures(cheque_image_path)
            if not cheque_signatures:
                logger.error(
                    f"No signatures found in cheque image: {cheque_image_path}"
                )
                return []
            logger.info(f"Found {len(cheque_signatures)} signatures on the cheque.")
        except Exception as e:
            logger.error(f"Error detecting signatures on cheque: {e}")
            return []

        try:
            reference_signatures = self._detect_reference_signatures(
                reference_image_paths, output_dir
            )
            if not reference_signatures:
                logger.error("No signatures found in any reference images.")
                return []
            logger.info(
                f"Found {len(reference_signatures)} total reference signatures."
            )
        except Exception as e:
            logger.error(f"Error detecting reference signatures: {e}")
            return []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            cheque_preprocessed_map = {
                sig.signature_id: pre
                for sig, pre in zip(
                    cheque_signatures,
                    executor.map(self._preprocess_sig, cheque_signatures),
                )
            }
            ref_sigs_list = [s for _, s in reference_signatures]
            ref_preprocessed_map = {
                sig.signature_id: pre
                for sig, pre in zip(
                    ref_sigs_list, executor.map(self._preprocess_sig, ref_sigs_list)
                )
            }

        all_matches = []
        for cheque_sig in cheque_signatures:
            cheque_pre = cheque_preprocessed_map.get(cheque_sig.signature_id)
            if not cheque_pre:
                continue

            for ref_img_path, ref_sig in reference_signatures:
                ref_pre = ref_preprocessed_map.get(ref_sig.signature_id)
                if not ref_pre:
                    continue

                match = self._compare_signatures(
                    cheque_sig,
                    ref_sig,
                    cheque_pre,
                    ref_pre,
                    ref_img_path,
                    output_dir,
                    generate_comparisons,
                )
                if match:
                    all_matches.append(match)
                    # #added
                    # if match.score > 0.75:
                    #     logger.info(f'High confidence found showing only one')
                    #     return [match]
                    # #added ends here

        filtered_matches = [m for m in all_matches if m.score > 0.1]
        filtered_matches.sort(key=lambda m: m.score, reverse=True)

        logger.info(
            f"Verification complete. Found {len(filtered_matches)} potential matches."
        )
        return filtered_matches

    def _preprocess_sig(self, signature: DetectedSignature) -> Optional[Dict]:
        try:
            if signature.cropped_path and os.path.exists(signature.cropped_path):
                return preprocess_signature_image(
                    signature.cropped_path, padding=self.padding
                )
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    cv2.imwrite(tmp.name, signature.image)
                    path = tmp.name
                preprocessed_data = preprocess_signature_image(
                    path, padding=self.padding
                )
                os.unlink(path)
                return preprocessed_data
        except Exception as e:
            logger.error(
                f"Failed to preprocess signature {signature.signature_id}: {e}"
            )
            return None

    def _compare_signatures(
        self,
        cheque_sig: DetectedSignature,
        ref_sig: DetectedSignature,
        cheque_preprocessed: Dict,
        ref_preprocessed: Dict,
        ref_img_path: str,
        output_dir: Optional[str],
        generate_comparison: bool,
    ) -> Optional[SignatureMatch]:
        boundary_results = compare_boundary_signatures_with_preprocessed(
            cheque_preprocessed, ref_preprocessed
        )
        texture_results = compare_texture_signatures(
            cheque_preprocessed, ref_preprocessed
        )

        hog_score = boundary_results.get("hog_similarity", 0.0)
        scalar_score = boundary_results.get("similarities", {}).get("scalar_avg", 0.0)
        swift_score = texture_results.get("swift_similarity", 0.0)
        weight_similarity = boundary_results.get("weight_similarity", 0.0)
        boundary_similarity = boundary_results.get("boundary_similarity", 0.0)

        h_proj = boundary_results.get("similarities", {}).get("h_projection_corr", 0.0)
        v_proj = boundary_results.get("similarities", {}).get("v_projection_corr", 0.0)
        projection_score = np.mean([h_proj, v_proj])

        total_score = (
            normalize_custom(hog_score, hog_intervals, hog_outputs) * 35
            + normalize_custom(scalar_score, scaler_intervals, scaler_outputs) * 35
            + normalize_custom(swift_score, swift_intervals, swift_outputs) * 30
        ) / 100

        adjusted_score = total_score
        if (weight_similarity > 0.65) and not (boundary_similarity > 0.8):
            adjusted_score *= 0.9
        elif not (weight_similarity > 0.65) and (boundary_similarity > 0.8):
            adjusted_score *= 0.8
        elif not (weight_similarity > 0.65) and not (boundary_similarity > 0.8):
            adjusted_score *= 0.7
        if projection_score < 0.5:
            adjusted_score *= 0.9
        if hog_score < 0.43:
            adjusted_score *= 0.9

        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # --- CHANGE 2: POPULATE THE NEW FIELDS WHEN CREATING THE MATCH OBJECT ---
        match = SignatureMatch(
            signature_id=cheque_sig.signature_id,
            score=adjusted_score,
            image_path=ref_img_path,
            matched_signature=ref_sig,
            cheque_signature=cheque_sig,
            cheque_bbox=cheque_sig.bbox,
            reference_bbox=ref_sig.bbox,
            cheque_preprocessed_images=cheque_preprocessed,
            ref_preprocessed_images=ref_preprocessed,
        )
        # --- END OF CHANGE 2 ---

        print("*" * 50)
        print(
            f"Compared Cheque Sig {cheque_sig.signature_id} with Ref Sig {ref_sig.signature_id}: Score={adjusted_score:.4f}"
        )
        print("Boundary Results:", boundary_results.items())
        print("Texture Results:", texture_results.items())
        print("*" * 50)
        if generate_comparison and adjusted_score > 0.1:
            match.comparison_image_path = self._save_comparison_image(
                cheque_sig, ref_sig, adjusted_score, output_dir
            )
        return match

    def _detect_reference_signatures(
        self, image_paths: List[str], output_dir: Optional[str]
    ) -> List[Tuple[str, DetectedSignature]]:
        results = self.detector.process_batch(image_paths, output_dir)
        return [(imgPath, sig) for imgPath, sigs in results.items() for sig in sigs]

    def _save_comparison_image(
        self,
        cheque_sig: DetectedSignature,
        ref_sig: DetectedSignature,
        score: float,
        output_dir: Optional[str],
    ) -> Optional[str]:
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(cv2.cvtColor(cheque_sig.image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Cheque Signature (ID: {cheque_sig.signature_id})")
        axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(ref_sig.image, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Reference Signature\nScore: {score:.2%}")
        axes[1].axis("off")
        plt.tight_layout()
        comp_filename = f"comparison_{cheque_sig.signature_id}_vs_{os.path.basename(ref_sig.cropped_path or 'ref.png')}"
        path = os.path.join(output_dir, comp_filename)
        plt.savefig(path)
        plt.close(fig)
        return path
