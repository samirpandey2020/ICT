# signature.py

import os
import cv2
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from torch.nn import InstanceNorm1d


from signature.variables import *
from signature.detection import (
    SignatureDetector,
    DetectedSignature,
)
from signature.verification import (
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

matplotlib.use("Agg")
logging.basicConfig(level=logging.DEBUG)  # change to INFO for full
logger = logging.getLogger(__name__)


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


class SignatureVerifier:
    def __init__(self, model_path: str = SIGNATURE_DETECTION_PATH, padding=40):
        self.detector = SignatureDetector(model_path)
        self.padding = padding
        self._preprocessing_cache = {}
        self._performance_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_comparisons": 0,
            "skipped_comparisons": 0,
        }

    def verify_signatures(
        self,
        cheque_image_np: np.ndarray,
        reference_images_np: List[np.ndarray],
        output_dir: Optional[str] = None,
        generate_comparisons: bool = True,
    ) -> List[SignatureMatch]:  # < ------ Added ----- >
        """
        Verifies signatures on a cheque against a list of reference signatures.
        """
        import time

        overall_start = time.time()
        logger.info("=" * 80)
        logger.info("🚀 STARTING SIGNATURE VERIFICATION - PERFORMANCE TRACKING")
        logger.info("=" * 80)

        # 1. Detect signatures on the cheque
        step_start = time.time()
        try:
            cheque_signatures = self.detector.detect_signatures(cheque_image_np)
            if not cheque_signatures:
                logger.error(f"No signatures found in cheque image:")
                return []
            step_time = time.time() - step_start
            logger.info(
                f"✅ [STEP 1] Cheque Detection: {step_time:.3f}s - Found {len(cheque_signatures)} signatures"
            )
        except Exception as e:
            logger.error(f"Error detecting signatures on cheque: {e}")
            return []

        # 2. Detect signatures on reference documents
        step_start = time.time()
        try:
            reference_signatures = self._detect_reference_signatures(
                reference_images_np
            )
            if not reference_signatures:
                logger.error("No signatures found in any reference images.")
                return []
            step_time = time.time() - step_start
            logger.info(
                f"✅ [STEP 2] Reference Detection: {step_time:.3f}s - Found {len(reference_signatures)} signatures"
            )
        except Exception as e:
            logger.error(f"Error detecting reference signatures: {e}")
            return []

        # 3. Preprocess all signatures - with timing
        step_start = time.time()
        total_sigs = len(cheque_signatures) + len([s for _, s in reference_signatures])
        logger.info(f"⏳ [STEP 3] Starting preprocessing of {total_sigs} signatures...")

        # Preprocess cheque signatures sequentially
        cheque_preprocessed_map = {}
        for idx, sig in enumerate(cheque_signatures, 1):
            sig_start = time.time()
            cheque_preprocessed_map[sig.unique_id] = self._preprocess_sig(sig)
            sig_time = time.time() - sig_start
            logger.info(
                f"  ├─ Cheque sig {idx}/{len(cheque_signatures)}: {sig_time:.3f}s"
            )

        # Preprocess reference signatures sequentially
        ref_sigs_list = [s for _, s in reference_signatures]
        ref_preprocessed_map = {}
        for idx, sig in enumerate(ref_sigs_list, 1):
            sig_start = time.time()
            ref_preprocessed_map[sig.unique_id] = self._preprocess_sig(sig)
            sig_time = time.time() - sig_start
            logger.info(
                f"  ├─ Reference sig {idx}/{len(ref_sigs_list)}: {sig_time:.3f}s"
            )

        step_time = time.time() - step_start
        cache_stats = f"Cache: {self._performance_stats['cache_hits']} hits, {self._performance_stats['cache_misses']} misses"
        logger.info(
            f"✅ [STEP 3] Total Preprocessing: {step_time:.3f}s - Avg: {step_time/total_sigs:.3f}s/sig - {cache_stats}"
        )

        # 4. Compare each cheque signature against each reference signature
        step_start = time.time()
        total_comparisons = len(cheque_signatures) * len(reference_signatures)
        logger.info(
            f"⏳ [STEP 4] Starting {total_comparisons} signature comparisons..."
        )

        all_matches = []
        high_matches = []
        comparison_count = 0
        for cheque_sig in cheque_signatures:
            cheque_pre = cheque_preprocessed_map.get(cheque_sig.unique_id)
            if not cheque_pre:
                continue

            for ref_img_path, ref_sig in reference_signatures:
                ref_pre = ref_preprocessed_map.get(ref_sig.unique_id)
                if not ref_pre:
                    continue

                comparison_count += 1
                comp_start = time.time()
                match = self._compare_signatures(
                    cheque_sig,
                    ref_sig,
                    cheque_pre,
                    ref_pre,
                    ref_img_path,
                    output_dir,
                    generate_comparisons,
                )
                comp_time = time.time() - comp_start

                if match:
                    all_matches.append(match)
                    logger.info(
                        f"  ├─ Comparison {comparison_count}/{total_comparisons}: {comp_time:.3f}s - Score: {match.score:.4f}"
                    )
                    # added
                    if match.score > SIGNATURE_VERIFICATION_THRESHOLD:
                        logger.info(f"    └─ ✓ High confidence match found!")
                        # return [match]             # UNCOMMENT TO FINISH LOOP AFTER FINDING SIGNATURE_VERIFICATION_CONFIDENCE
                    # added ends here

        # 5. Filter and sort results
        step_time = time.time() - step_start
        logger.info(
            f"✅ [STEP 4] All Comparisons: {step_time:.3f}s - Avg: {step_time/max(total_comparisons, 1):.3f}s per comparison"
        )

        filtered_matches = [
            m for m in all_matches if m.score > SIGNATURE_VERIFICATION_THRESHOLD
        ]  # filter the matches value with score
        filtered_matches.sort(key=lambda m: m.score, reverse=True)

        total_time = time.time() - overall_start
        logger.info("=" * 80)
        logger.info(f"🏁 VERIFICATION COMPLETE - Total Time: {total_time:.3f}s")
        logger.info(
            f"📊 Results: {len(filtered_matches)} matches above threshold (from {len(all_matches)} total)"
        )
        if filtered_matches:
            logger.info(f"🎯 Best Match Score: {filtered_matches[0].score:.4f}")
        logger.info("=" * 80)

        # return [best_match]   #this shows only the best match found so far
        return filtered_matches

    # def _preprocess_sig(self, signature: DetectedSignature) -> Optional[Dict]:
    #     """Helper to preprocess a signature and handle potential errors."""
    #     try:
    #         # The preprocessing function expects a file path.
    #         if signature.cropped_path and os.path.exists(signature.cropped_path):
    #              return preprocess_signature_image(signature.cropped_path, padding=self.padding)
    #         else:
    #             # If no path is available, create a temporary one.
    #             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    #                 cv2.imwrite(tmp.name, signature.image)
    #                 path = tmp.name
    #             preprocessed_data = preprocess_signature_image(path, padding=self.padding)
    #             os.unlink(path)
    #             return preprocessed_data
    #     except Exception as e:
    #         logger.error(f"Failed to preprocess signature {signature.signature_id}: {e}")
    #         return None

    def _preprocess_sig(self, signature: DetectedSignature) -> Optional[Dict]:
        """
        Helper to preprocess a signature using numpy arrays only, with instance-level caching.
        No file I/O operations - works entirely in memory.

        Args:
            signature: DetectedSignature object containing the image as numpy array

        Returns:
            Preprocessed data dictionary or None on failure
        """
        # Use the globally unique ID as the cache key
        cache_key = signature.unique_id

        # Check the instance cache first
        if cache_key in self._preprocessing_cache:
            self._performance_stats["cache_hits"] += 1
            return self._preprocessing_cache[cache_key]

        # If not in cache, process it
        self._performance_stats["cache_misses"] += 1
        try:
            # Work directly with the image numpy array from the signature
            image_array = signature.image

            if image_array is None or image_array.size == 0:
                logger.error(f"Invalid image array for signature {signature.unique_id}")
                return None

            # Call preprocessing with numpy array directly
            padding_value = getattr(self, "padding", 40)
            preprocessed_data = preprocess_signature_image(
                image_array, padding=padding_value
            )

            # Store the result in the cache before returning
            self._preprocessing_cache[cache_key] = preprocessed_data
            return preprocessed_data

        except Exception as e:
            logger.error(f"Failed to preprocess signature {signature.unique_id}: {e}")
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
        """Compares two preprocessed signatures and calculates a final score."""
        import time

        # Boundary comparison
        boundary_start = time.time()
        boundary_results = compare_boundary_signatures_with_preprocessed(
            cheque_preprocessed, ref_preprocessed
        )
        boundary_time = time.time() - boundary_start

        # Texture comparison
        texture_start = time.time()
        texture_results = compare_texture_signatures(
            cheque_preprocessed, ref_preprocessed
        )
        texture_time = time.time() - texture_start

        logger.info(f"    ├─ Boundary features: {boundary_time:.3f}s")
        logger.info(f"    └─ Texture features (SWIFT): {texture_time:.3f}s")

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

        match = SignatureMatch(
            signature_id=cheque_sig.signature_id,
            score=adjusted_score,
            image_path=ref_img_path,
            matched_signature=ref_sig,
            cheque_signature=cheque_sig,
            cheque_bbox=cheque_sig.bbox,
            reference_bbox=ref_sig.bbox,
        )

        return match

    # def _detect_reference_signatures(self, image_paths: List[str], output_dir: Optional[str]) -> List[Tuple[str, DetectedSignature]]:
    #     results = self.detector.process_batch(image_paths, output_dir)
    #     return [(imgPath, sig) for imgPath, sigs in results.items() for sig in sigs]

    def _detect_reference_signatures(
        self, images_np: List[np.ndarray]
    ) -> List[Tuple[int, DetectedSignature]]:  # < ------ Added ----- >
        results = self.detector.process_batch(images_np)  # < ------ Added ----- >
        output = []
        for idx, sigs in enumerate(results.values()):  # < ------ Added ----- >
            for sig in sigs:
                output.append((idx, sig))  # < ------ Added ----- >
        return output  # < ------ Added ----- >

    def clear_cache(self):
        """Clears internal cache of pre processed signatures"""
        initial_size = len(self._preprocessing_cache)
        self._preprocessing_cache.clear()
        logger.info(f"Cache cleared. Removed {initial_size} items")

    # def _save_comparison_image(self, cheque_sig: DetectedSignature, ref_sig: DetectedSignature, score: float, output_dir: Optional[str]) -> Optional[str]:
    #     if not output_dir: return None
    #     os.makedirs(output_dir, exist_ok=True)
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     axes[0].imshow(cv2.cvtColor(cheque_sig.image, cv2.COLOR_BGR2RGB))
    #     axes[0].set_title(f"Cheque Signature (ID: {cheque_sig.signature_id})")
    #     axes[0].axis('off')
    #     axes[1].imshow(cv2.cvtColor(ref_sig.image, cv2.COLOR_BGR2RGB))
    #     axes[1].set_title(f"Reference Signature\nScore: {score:.2%}")
    #     axes[1].axis('off')
    #     plt.tight_layout()
    #     comp_filename = f"comparison_{cheque_sig.signature_id}_vs_{os.path.basename(ref_sig.cropped_path or 'ref.png')}"
    #     path = os.path.join(output_dir, comp_filename)
    #     plt.savefig(path)
    #     plt.close(fig)
    #     return path

    def compare_two_signatures(
        self,
        signature1_np: np.ndarray,
        signature2_np: np.ndarray,
    ) -> Tuple[float, Optional[SignatureMatch]]:
        """
        Directly compares two signature images without running detection.
        Returns confidence score (0-100) and optional match details.

        Args:
            signature1_np: NumPy array of the first signature image (cropped).
            signature2_np: NumPy array of the second signature image (cropped).

        Returns:
            Tuple of (confidence_percentage, SignatureMatch object or None)
            - confidence_percentage: Score from 0-100 representing similarity
            - SignatureMatch: Detailed match object or None if comparison fails
        """
        try:
            logger.info("🔍 Starting direct signature comparison...")

            # Validate inputs
            if signature1_np is None or signature2_np is None:
                logger.error("❌ One or both signature images are None")
                return 0.0, None

            if signature1_np.size == 0 or signature2_np.size == 0:
                logger.error("❌ One or both signature images are empty")
                return 0.0, None

            # Create dummy DetectedSignature objects for compatibility
            sig1 = DetectedSignature(
                signature_id="signature_1",
                confidence=1.0,
                image=signature1_np,
                bbox=None,
                unique_id="sig1_direct_compare",
            )

            sig2 = DetectedSignature(
                signature_id="signature_2",
                confidence=1.0,
                image=signature2_np,
                bbox=None,
                unique_id="sig2_direct_compare",
            )

            # Preprocess both signatures
            logger.info("⏳ Preprocessing signature 1...")
            sig1_pre = self._preprocess_sig(sig1)

            logger.info("⏳ Preprocessing signature 2...")
            sig2_pre = self._preprocess_sig(sig2)

            if sig1_pre is None or sig2_pre is None:
                logger.error("❌ Failed to preprocess one or both signature images.")
                return 0.0, None

            # Perform comparison
            logger.info("🔍 Comparing signatures...")
            match = self._compare_signatures(
                cheque_sig=sig1,
                ref_sig=sig2,
                cheque_preprocessed=sig1_pre,
                ref_preprocessed=sig2_pre,
                ref_img_path="DirectComparison",
                output_dir=None,
                generate_comparison=False,
            )

            if match is None:
                logger.error("❌ Comparison failed to produce results")
                return 0.0, None

            # Convert score to percentage (0-100)
            confidence_percentage = match.score * 100

            logger.info(
                f"✅ Comparison complete - Confidence: {confidence_percentage:.2f}%"
            )
            logger.info(f"   Match Score: {match.score:.4f}")

            return confidence_percentage, match

        except Exception as e:
            logger.error(f"❌ Error comparing two signatures directly: {e}")
            return 0.0, None


if __name__ == "__main__":

    # @st.cache_resource
    def load_verifier() -> SignatureVerifier:
        return SignatureVerifier()

    verifier = load_verifier()
    im1 = rf"models/test_image/sig_1_1.png"
    im2 = rf"models/test_image/sig_1_1.png"

    # im1 = rf"models\test_image\cheque_front_Image.png"
    # im2 = rf"models\test_image\cheque_front_Image.png"
    cheque_np = cv2.imread(im1)
    refrence_images = cv2.imread(
        im2,
    )

    match = verifier.compare_two_signatures(cheque_np, refrence_images)
    confidence, match_details = match
    if match_details:
        print(f"Confidence: {confidence:.2f}%")
        print(f"Match Score: {match_details.score:.4f}")
    else:
        print("Comparison failed.")
