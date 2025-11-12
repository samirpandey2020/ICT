
import os
import traceback
import cv2
import pywt
import math
import logging
import numpy as np
from skimage import morphology
from skimage.feature import hog
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression

# This module now depends on the cleaner for image preprocessing.
from signature.cleaner import SignatureCleaner

logging.basicConfig(level=logging.DEBUG) #change to INFO for full
logger = logging.getLogger(__name__)

ENABLE_PRINT = False #set False to disable prints

def c_print(*args, **kwargs):
    if ENABLE_PRINT:
        print(*args, **kwargs)


# --- CONFIGURATIONS AND UTILITIES (SINGLE SOURCE OF TRUTH) ---

# Initialize the cleaner once to be reused by preprocessing functions.
signature_cleaner = SignatureCleaner()

# Scoring intervals are now defined only here.
hog_intervals = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
hog_outputs = [(0.3, 0.5), (0.5, 0.65), (0.65, 0.9), (0.9, 1.0)]

scaler_intervals = [(0.0, 0.6), (0.6, 0.7), (0.7, 1.0)]
scaler_outputs = [(0.0, 0.2), (0.2, 0.4), (0.4, 1.0)]

swift_intervals = [(0.0, 0.7), (0.7, 0.84), (0.84, 0.9), (0.9, 1.0)]
swift_outputs = [(0.0, 0.1), (0.1, 0.6), (0.6, 0.8), (0.8, 1.0)]

def normalize_custom(score, input_intervals, output_intervals):
    """Custom normalization function for a single score."""
    overall_in_min, overall_in_max = input_intervals[0][0], input_intervals[-1][1]
    score_clamped = max(min(score, overall_in_max), overall_in_min)
    
    for (in_start, in_end), (out_start, out_end) in zip(input_intervals, output_intervals):
        if in_start <= score_clamped <= in_end:
            if (in_end - in_start) == 0: return out_start
            return out_start + (score_clamped - in_start) * (out_end - out_start) / (in_end - in_start)
    
    return output_intervals[-1][1]

# --- PREPROCESSING AND ROTATION ---

def filter_by_angle(lines, max_angle=89):
    filtered = []
    for line_info in lines:
        angle = line_info[3]   # angle is stored at index 3
        if 1<abs(angle) <= max_angle:
            filtered.append(line_info)
    return filtered

def filter_lines_for_dominant_angle(lines_with_angles, min_abs_angle=1.0, max_abs_angle=89.0, max_cluster_spread=15):
    """
    Performs a two-stage filtering process to find the most representative lines for rotation.

    Stage 1: Removes lines that are nearly horizontal or vertical.
    Stage 2: Finds the densest cluster of angles from the remaining lines.

    Args:
        lines_with_angles (list): List of tuples, e.g., (line_coords, length, angle_deg).
        min_abs_angle (float): The minimum absolute angle to keep (removes horizontal lines).
        max_abs_angle (float): The maximum absolute angle to keep (removes vertical lines).
        window_size (int): The number of consecutive angles to consider in a cluster.
        max_cluster_spread (float): The max angle difference allowed in the final cluster.

    Returns:
        list: The double-filtered list of lines representing the dominant slant.
    """
    # --- Stage 1: Hard Cutoff Filter (remove horizontal/vertical lines) ---
    range_filtered = []
    for line_info in lines_with_angles:
        angle = line_info[3]
        if min_abs_angle < abs(angle) <= max_abs_angle:
            range_filtered.append(line_info)
    
    c_print(f"Stage 1 (Range Filter): Kept {len(range_filtered)} of {len(lines_with_angles)} lines.")
    return range_filtered

def perform_rotation(binary_image, rotation_threshold=0.05):
    """
    Performs rotation correction on a binary image using linear regression
    to find the skew angle and then rotates the image to correct it.

    Args:
        binary_image (np.ndarray): The input binary image (expected to be grayscale,
                                   single-channel, with signature pixels > 0).
        rotation_threshold (float): The maximum acceptable absolute slope value.
                                    The loop continues until the slope is below this.

    Returns:
        np.ndarray: The deskewed binary image, resized to 512x512,
                    with a white signature on a black background.
    """
    rotation_threshold=0.05
    try:
        # --- Step 1: Initial Checks and Setup ---
        coords = cv2.findNonZero(binary_image)
        if coords is None:
            # Image is blank, no rotation needed.
            # Ensure output format consistency
            output_size = (512, 512)
            blank_image = np.zeros(output_size, dtype=np.uint8)
            return blank_image # Return a blank black image

        # Get the bounding box of the signature
        x, y, w, h = cv2.boundingRect(coords)
        c_print(f"Input image dimensions: Width={w}, Height={h}")

        # --- Step 2: Prepare Pixel Data for Linear Regression ---
        # Get the (row, column) coordinates of all signature pixels
        # np.where returns (row_indices, col_indices)
        pixels = np.where(binary_image > 0)
        y_coords, x_coords = pixels[0], pixels[1]

        # Scikit-learn expects X to be a 2D array, so we reshape it
        # X will be our independent variable (x-coordinates)
        # y will be our dependent variable (y-coordinates)
        X = x_coords.reshape(-1, 1)
        y = y_coords

        # --- Step 3: Apply Linear Regression to Find Initial Slant ---
        # This provides an initial estimate of the skew.
        model = LinearRegression()
        try:
            model.fit(X, y)
            slope = model.coef_[0]
            c_print(f"📈 Calculated Original Slope (m): {slope:.4f}")
        except ValueError as e:
            logger.warning(f"Linear regression failed: {e}. Not enough data points for regression. Returning original normalized image.")
            # If regression fails (e.g., only one point), return original normalized
            normalized_img = cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
            _, final_normalized = cv2.threshold(normalized_img, 127, 255, cv2.THRESH_BINARY)
            return final_normalized

        # --- Step 4: Iteratively Refine Rotation Angle ---
        # The goal is to rotate the image until the slope of the signature pixels is close to zero.
        # We'll start with the initial angle derived from the first regression.
        total_angle_deg = np.degrees(np.arctan(slope))
        current_rotated_slope = slope
        iterations = 0
        max_iterations = 2 # Safety break to prevent infinite loops

        c_print("\n--- Finding Optimal Rotation Angle ---")
        # Continue as long as the absolute slope is greater than our threshold
        # and we haven't exceeded the maximum number of iterations.
        while abs(current_rotated_slope) > rotation_threshold and iterations < max_iterations:
            (img_h, img_w) = binary_image.shape # Use original image dimensions for rotation center
            center = (img_w // 2, img_h // 2)

            # Get rotation matrix for the current total angle
            M = cv2.getRotationMatrix2D(center, total_angle_deg, 1.0)
            # Perform a temporary rotation to find the new slope
            # borderValue=0 ensures new areas are black
            temp_rotated_binary = cv2.warpAffine(binary_image, M, (img_w, img_h), borderValue=0)

            # Find pixels in the temporarily rotated image
            rotated_pixels = np.where(temp_rotated_binary > 0)
            rotated_y_coords, rotated_x_coords = rotated_pixels[0], rotated_pixels[1]

            # Check if enough pixels remain for regression
            if len(rotated_x_coords) < 2:
                logger.warning("Not enough signature pixels found after rotation. Stopping iteration.")
                break

            # Apply linear regression to the rotated pixels to find the new slope
            rotated_X = rotated_x_coords.reshape(-1, 1)
            rotated_y = rotated_y_coords
            try:
                rotated_model = LinearRegression().fit(rotated_X, rotated_y)
                current_rotated_slope = rotated_model.coef_[0]
            except ValueError as e:
                logger.warning(f"Linear regression failed during iteration {iterations + 1}: {e}. Stopping iteration.")
                break

            c_print(f"Iteration {iterations + 1}: Angle={total_angle_deg:.2f}°, Current Slope={current_rotated_slope:.4f}")

            # If the slope is still too steep, calculate the adjustment angle and add it
            if abs(current_rotated_slope) > rotation_threshold:
                adjustment_angle_rad = np.arctan(current_rotated_slope)
                # Add the adjustment angle in degrees to the total angle
                total_angle_deg += np.degrees(adjustment_angle_rad)

            iterations += 1

        # Report final status of the iterative process
        if abs(current_rotated_slope) <= rotation_threshold:
            c_print("\n✅ Achieved a rotated slope close to 0.")
        else:
            pass
            # logger.warning("\n Could not achieve a slope close to 0 within max iterations. Proceeding with current angle.")

        # --- Step 5: Perform the Final Rotation ---
        c_print(f"\nPerforming final rotation with a total angle of {total_angle_deg:.2f}°")

        # Use the original image dimensions for the final rotation center
        (img_h, img_w) = binary_image.shape
        center = (img_w // 2, img_h // 2)

        # Get the final rotation matrix
        final_M = cv2.getRotationMatrix2D(center, total_angle_deg, 1.0)

        # Perform the final rotation. borderValue=0 ensures the background is black.
        rotated_binary_image = cv2.warpAffine(binary_image, final_M, (img_w, img_h),
                                            flags=cv2.INTER_LINEAR, # INTER_LINEAR is generally good for rotation
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0) # Fill new areas with black


        output_size = (512, 512)
        normalized_rotated = cv2.resize(rotated_binary_image, output_size, interpolation=cv2.INTER_AREA)


        _, final_rotation = cv2.threshold(normalized_rotated, 127, 255, cv2.THRESH_BINARY)


        return final_rotation
    except Exception as e:
        logger.error(f'{traceback.print_exc()}')
def perform_rotation_old(binary_image, rotation_threshold=0.8):
    """Perform double rotation on a binary image to correct its skew."""

    coords = cv2.findNonZero(binary_image)
    if coords is None:
        # Image is blank, no rotation needed.
        return binary_image

    # Get the bounding box of the signature to find its real width and height
    x, y, w, h = cv2.boundingRect(coords)
    c_print(f"The image is of Width x Height ({w},{h})")
    if not (w >= 1.5 * h or h >= 1.5 * w):
        logger.warning("height x width not compatible should be 1.5 more than each other")
        # We must still return an image of the standard size for consistency.
        # Ensure it's binary (threshold again just in case)
        binary = binary_image.copy()
        if binary.ndim == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        
        # Convert to strict binary (white signature / black background)
        _, cropped_binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        # Normalize to fixed size
        output_size=(512, 512)
        normalized = cv2.resize(cropped_binary, output_size, interpolation=cv2.INTER_AREA)

        # Ensure binary after resize
        _, final_normalized = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
        

        return final_normalized

    initial_min_line_length = int(w * 0.9)
    min_line_length = initial_min_line_length
    max_line_gap = int(w * 0.2)
    target_max_lines =15
    loop = 1
    lines = None
    while min_line_length > int(w * 0.2): # Don't let minLineLength get too small
        detected_lines = cv2.HoughLinesP(binary_image, rho=1, theta=np.pi / 180,
                                       threshold=15, minLineLength=min_line_length, maxLineGap=max_line_gap)
        c_print(f"Loop is {loop} ")

        if detected_lines is not None :
            c_print(f" len of detected_lines is {len(detected_lines)}")
        loop = loop + 1
        
        # Check if we found a good number of lines
        if detected_lines is not None :
            if len(detected_lines) > target_max_lines:
                lines = detected_lines
                break 
        min_line_length = int(min_line_length * 0.9) # If we found too many lines or no lines, relax the criteria by 10% and try again
        max_line_gap = int(max_line_gap * 1.1)
    
    if lines is None:
        logger.warning("Could not find a stable set of lines for rotation. Skipping rotation for this image.")
        # We must still return an image of the standard size for consistency.
        normalized_no_rotation = cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
        _, final_no_rotation = cv2.threshold(normalized_no_rotation, 127, 255, cv2.THRESH_BINARY)
        return final_no_rotation
    c_print(f"✅ Detected {len(lines)} line segments on the clean mask.")
    negative_slope_lines = []
    positive_slope_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0: continue
        slope = (y2 - y1) / (x2 - x1)
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle_deg = math.degrees(math.atan(slope))
        if slope < 0:
            negative_slope_lines.append(((x1, y1, x2, y2),slope, length, angle_deg))
        if slope > 0:
            positive_slope_lines.append(((x1, y1, x2, y2),slope, length, angle_deg))
    negative_slope_lines.sort(key=lambda item: item[1], reverse=True)
    positive_slope_lines.sort(key=lambda item: item[1], reverse=True)
    neg = len(negative_slope_lines)
    pos = len(positive_slope_lines)
    difference = abs(neg - pos) / max(neg, pos)
    c_print(f"➡️ Found {neg} segments with a negative slope.")
    c_print(f"➡️ Found {pos} segments with a positive slope.")
    
    top_lines = max([negative_slope_lines, positive_slope_lines], key=len)
    c_print(f"Selected {len(top_lines)} number of lines.")
    top_lines = top_lines[:100] 
    if top_lines:
        #top_lines = filter_by_angle(top_lines)
        top_lines = filter_lines_for_dominant_angle(top_lines)
        slopes = [info[1] for info in top_lines]
        lengths = [info[2] for info in top_lines]
        angle_deg = [info[3] for info in top_lines]
        c_print(f"="*50)
        c_print(f"Angles are : {angle_deg}")
        c_print(f"="*50)

        if not lengths or sum(lengths) == 0:
            logger.warning("Line lengths sum to zero. Skipping rotation.")
            return cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
        weighted_mean_slope = np.average(slopes, weights=lengths)
        c_print(f"\n📊 Weighted mean slope: {weighted_mean_slope:.4f}")

        angle_rad = math.atan(weighted_mean_slope)
        angle_deg = math.degrees(angle_rad)
        sorted_pairs = sorted(zip(slopes, lengths))
        sorted_slopes, sorted_lengths = zip(*sorted_pairs)
        sorted_slopes = np.array(sorted_slopes)
        sorted_lengths = np.array(sorted_lengths)
        cumulative_weights = np.cumsum(sorted_lengths)
        total_weight = np.sum(sorted_lengths)
        median_weight = total_weight / 2
        weighted_median_index = np.where(cumulative_weights >= median_weight)[0][0]
        weighted_median_slope = sorted_slopes[weighted_median_index]
        angle_rad_median = math.atan(weighted_median_slope)
        angle_deg_median = math.degrees(angle_rad_median)
        final_angle_rotation = angle_deg - angle_deg_median

        rotation_angle = angle_deg
        c_print(f"Rotation angle is {rotation_angle}")
        if abs(rotation_angle)  > 70:
            logger.warning("More than 70 degree rotation is not valid so returning no rotation")
            normalized_no_rotation = cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
            _, final_no_rotation = cv2.threshold(normalized_no_rotation, 127, 255, cv2.THRESH_BINARY)
            return final_no_rotation
        

            # --- First Rotation ---
        (h, w) = binary_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

        rotated_mask = cv2.warpAffine(binary_image, M, (w, h),
                                        flags=cv2.INTER_LINEAR, # Keep INTER_LINEAR here for the first pass
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)  # Black border to maintain signature isolation

        # Create white canvas with black signature (as in your original code)
        final_rotated_cv = np.full((h, w), 255, dtype=np.uint8)  # White canvas
        final_rotated_cv[rotated_mask > 128] = 0  # Add black signature


        # --- Second Rotation ---
        # Use the white canvas with black signature from the first rotation for the second.
        # This ensures we're rotating a clean binary image.
        (h_final, w_final) = final_rotated_cv.shape[:2]
        center_final = (w_final // 2, h_final // 2)
        final_angle_rotation = - final_angle_rotation # Corrected angle usage

        M_final = cv2.getRotationMatrix2D(center_final, final_angle_rotation, 1.0)

        abs_cos_final = np.abs(M_final[0, 0])
        abs_sin_final = np.abs(M_final[0, 1])
        nW_final = int(h_final * abs_sin_final + w_final * abs_cos_final)
        nH_final = int(h_final * abs_cos_final + w_final * abs_sin_final)

        M_final[0, 2] += (nW_final / 2) - center_final[0]
        M_final[1, 2] += (nH_final / 2) - center_final[1]

        # Perform the final rotation with a white border (as in your original code)
        final_rotated_cv_2 = cv2.warpAffine(final_rotated_cv, M_final, (nW_final, nH_final),
                                            flags=cv2.INTER_LINEAR, # Keep INTER_LINEAR here
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=255) # White border as in original code

        # Convert to binary and invert to get black background with white signature
        _, final_binary_output = cv2.threshold(final_rotated_cv_2, 127, 255, cv2.THRESH_BINARY)
        # Invert to get black background with white signature (as expected by the pipeline)
        final_binary_output = cv2.bitwise_not(final_binary_output)

        # Normalize back to 512x512 to maintain consistency for similarity scoring
        normalized_rotated = cv2.resize(final_binary_output, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Ensure binary values are maintained after resize
        _, final_normalized = cv2.threshold(normalized_rotated, 127, 255, cv2.THRESH_BINARY)
        
        # Validate that signature content is preserved during normalization
        original_pixels = np.sum(binary_image > 0)
        final_pixels = np.sum(final_normalized > 0)
        preservation_ratio = final_pixels / max(original_pixels, 1)
        
        c_print(f"📊 Signature preservation: {preservation_ratio:.2%} ({original_pixels} → {final_pixels} pixels)")
        
        # If we lost too much signature content, return original normalized to 512x512
        if preservation_ratio < 0.2:  #or difference > 0.5
            if preservation_ratio < 0.2:
                c_print(f"⚠️ Too much signature content lost ({preservation_ratio:.2%} < {rotation_threshold:.2%}), returning original normalized")
            if difference > 0.5:
                    c_print(f"⚠️  Differences in lines is  {difference}")
            
            original_normalized = cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
            _, original_final = cv2.threshold(original_normalized, 127, 255, cv2.THRESH_BINARY)
            return original_final
        
        return final_normalized
    else:
        # Normalize to 512x512 for consistency even when no rotation is performed
        normalized_no_rotation = cv2.resize(binary_image, (512, 512), interpolation=cv2.INTER_AREA)
        _, final_no_rotation = cv2.threshold(normalized_no_rotation, 127, 255, cv2.THRESH_BINARY)
        c_print(f"✅ No rotation applied. Normalized to 512x512 for consistency.")
        return final_no_rotation

def clean_signatures(image, upscale, enable_rotation=True, rotation_threshold=0.8, padding=40):
    """High-level wrapper to clean and rotate a signature image."""
    import time
    try:
        clean_start = time.time()
        normalized_np = signature_cleaner.clean_and_normalize(
            image, upscale=upscale, padding=padding
        )
        clean_time = time.time() - clean_start
        logger.info(f"      ├─ BiRefNet+Upscale: {clean_time:.3f}s")
        
        mask_binary = (normalized_np < 255).astype(np.uint8) * 255
        c_print('check check check ')
        
        if enable_rotation:
            rotation_start = time.time()
            mask_binary = perform_rotation(mask_binary, rotation_threshold)
            rotation_time = time.time() - rotation_start
            logger.info(f"      └─ Rotation: {rotation_time:.3f}s")
            
        return mask_binary
    except Exception as e:
        import traceback
        logger.error(traceback.print_exc())
        logger.error(f"Error in clean_signatures: {e}")
        # Return a blank image on failure to prevent downstream errors
        return np.zeros((512, 512), dtype=np.uint8)

def preprocess_signature_image(image: np.ndarray, padding: int = 40):
    """
    Shared preprocessing pipeline for signature analysis.
    Takes a numpy array image, cleans, normalizes, rotates, and prepares masks.
    
    Args:
        image: numpy array of the signature image
        padding: padding to apply during normalization
        
    Returns:
        Dictionary containing preprocessed image data
    """
    import time
    preprocess_start = time.time()
    
    if image is None or image.size == 0:
        raise ValueError("Invalid image array provided")

    # Use the centralized clean_signatures function
    clean_start = time.time()
    final_mask = clean_signatures(image, upscale=(image.shape[0] < 256 or image.shape[1] < 256), padding=padding)
    clean_time = time.time() - clean_start
    logger.info(f"    ├─ clean_signatures: {clean_time:.3f}s")

    thin_start = time.time()
    thinned_mask = thin_signature(final_mask)
    thin_time = time.time() - thin_start
    logger.info(f"    └─ thin_signature: {thin_time:.3f}s")
    
    total_time = time.time() - preprocess_start
    logger.info(f"    Total preprocess_signature_image: {total_time:.3f}s")
    
    return {
        'normalized_np': final_mask,
        'mask_binary': final_mask,
        'thinned': thinned_mask,
    }


# --- MORPHOLOGICAL AND BOUNDARY OPERATIONS ---

def thin_signature(binary_mask):
    """Thins the binary signature mask to a single-pixel width skeleton."""
    return morphology.skeletonize(binary_mask > 0).astype(np.uint8) * 255

def extract_signature_boundary_points_from_preprocessed(pre_data):
    """Extracts boundary points from preprocessed data."""
    mask = pre_data['mask_binary']
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], np.zeros_like(mask)
    
    largest_contour = max(contours, key=cv2.contourArea)
    boundary_points = largest_contour.reshape(-1, 2).tolist()
    
    # Reconstruct for weight comparison
    reconstructed = np.zeros_like(mask)
    # cv2.drawContours(reconstructed, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return boundary_points, reconstructed

# --- FEATURE EXTRACTION ---
def extract_swift_features(image, wavelets=['db1', 'db2'], levels=[1, 2]):
    """Extract SWIFT features from preprocessed signature image."""
    # Ensure image is properly formatted
    image = image.astype(np.float32)
    
    # Debug: Check input image
    c_print(f"SWIFT input image shape: {image.shape}, max: {np.max(image)}, min: {np.min(image)}")
    
    # Handle empty images
    if np.all(image == 0):
        c_print("Warning: Input image is all zeros for SWIFT extraction")
        features = {'energy': {}, 'entropy': {}, 'mean': {}, 'std': {}, 'correlation': {}}
        for wavelet in wavelets:
            for level in levels:
                for band in ['cA', 'cH', 'cV', 'cD']:
                    key = f'{band}_{wavelet}_l{level}'
                    features['energy'][key] = 0.0
                    features['entropy'][key] = 0.0
                    features['mean'][key] = 0.0
                    features['std'][key] = 0.0
                for corr in ['HV', 'HD', 'VD']:
                    features['correlation'][f'{corr}_{wavelet}_l{level}'] = 0.0
        return features
    
    features = {'energy': {}, 'entropy': {}, 'mean': {}, 'std': {}, 'correlation': {}}
    
    for wavelet in wavelets:
        for level in levels:
            try:
                cA, cH, cV, cD = apply_swt(image, wavelet, level)
                    # Debug: Check SWT coefficients
                c_print(f"SWT {wavelet} level {level} - cA shape: {cA.shape}, max: {np.max(cA)}")
                    # Normalize coefficients to prevent numerical issues
                norm = np.linalg.norm(cA) + 1e-8
                cA, cH, cV, cD = cA / norm, cH / norm, cV / norm, cD / norm
                    # Calculate energy features with log transformation for stability
                features['energy'][f'cA_{wavelet}_l{level}'] = np.log1p(np.sum(cA ** 2))
                features['energy'][f'cH_{wavelet}_l{level}'] = np.log1p(np.sum(cH ** 2))
                features['energy'][f'cV_{wavelet}_l{level}'] = np.log1p(np.sum(cV ** 2))
                features['energy'][f'cD_{wavelet}_l{level}'] = np.log1p(np.sum(cD ** 2))
                    # Calculate entropy features
                features['entropy'][f'cA_{wavelet}_l{level}'] = calculate_entropy(cA)
                features['entropy'][f'cH_{wavelet}_l{level}'] = calculate_entropy(cH)
                features['entropy'][f'cV_{wavelet}_l{level}'] = calculate_entropy(cV)
                features['entropy'][f'cD_{wavelet}_l{level}'] = calculate_entropy(cD)
                    # Calculate statistical features
                features['mean'][f'cA_{wavelet}_l{level}'] = np.mean(cA)
                features['mean'][f'cH_{wavelet}_l{level}'] = np.mean(cH)
                features['mean'][f'cV_{wavelet}_l{level}'] = np.mean(cV)
                features['mean'][f'cD_{wavelet}_l{level}'] = np.mean(cD)
                features['std'][f'cA_{wavelet}_l{level}'] = np.std(cA)
                features['std'][f'cH_{wavelet}_l{level}'] = np.std(cH)
                features['std'][f'cV_{wavelet}_l{level}'] = np.std(cV)
                features['std'][f'cD_{wavelet}_l{level}'] = np.std(cD)
                    # Calculate correlation features with error handling
                try:
                    features['correlation'][f'HV_{wavelet}_l{level}'] = float(np.corrcoef(cH.flatten(), cV.flatten())[0, 1])
                except Exception:
                    features['correlation'][f'HV_{wavelet}_l{level}'] = 0.0
                try:
                    features['correlation'][f'HD_{wavelet}_l{level}'] = float(np.corrcoef(cH.flatten(), cD.flatten())[0, 1])
                except Exception:
                    features['correlation'][f'HD_{wavelet}_l{level}'] = 0.0
                try:
                    features['correlation'][f'VD_{wavelet}_l{level}'] = float(np.corrcoef(cV.flatten(), cD.flatten())[0, 1])
                except Exception:
                    features['correlation'][f'VD_{wavelet}_l{level}'] = 0.0
            except Exception as e:
                    c_print(f"Failed to extract SWIFT features for {wavelet} level {level}: {e}")
                    # Set default values for failed extraction
                    for band in ['cA', 'cH', 'cV', 'cD']:
                        key = f'{band}_{wavelet}_l{level}'
                        features['energy'][key] = 0.0
                        features['entropy'][key] = 0.0
                        features['mean'][key] = 0.0
                        features['std'][key] = 0.0
                    for corr in ['HV', 'HD', 'VD']:
                        features['correlation'][f'{corr}_{wavelet}_l{level}'] = 0.0
                    continue
        # Debug: Check final features
        c_print(f"SWIFT features extracted - energy keys: {list(features['energy'].keys())}")
        c_print(f"SWIFT energy values sample: {list(features['energy'].values())[:3]}")
        
        # Normalize features to [0, 1] range for each feature type with better handling
        for key in features:
            vals = list(features[key].values())
            if len(vals) > 0:
                min_v, max_v = np.min(vals), np.max(vals)
            if max_v > min_v:
                for k in features[key]:
                    features[key][k] = (features[key][k] - min_v) / (max_v - min_v)
            else: # If all values are the same, set to 0.5 (neutral similarity)
                for k in features[key]:
                    features[key][k] = 0.5
        
    return features

def apply_swt(image, wavelet='db1', level=2):
    """Apply Stationary Wavelet Transform (SWT) to the image, padding if necessary."""
    image = image.astype(np.float32)
    if image.size == 0 or np.all(image == 0):
        h, w = image.shape if image.size > 0 else (64, 64)
        return np.zeros((h//2, w//2)), np.zeros((h//2, w//2)), np.zeros((h//2, w//2)), np.zeros((h//2, w//2))
    min_size = 2**level
    if image.shape[0] < min_size or image.shape[1] < min_size:
        target_size = (max(min_size, image.shape[1]), max(min_size, image.shape[0]))
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    required_multiple = 2**level
    h, w = image.shape
    pad_h = (required_multiple - h % required_multiple) % required_multiple
    pad_w = (required_multiple - w % required_multiple) % required_multiple
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
    try:
        coeffs = pywt.swt2(image, wavelet, level=level, start_level=0)
        cA, (cH, cV, cD) = coeffs[level-1]
        return cA, cH, cV, cD
    except Exception as e:
        c_print(f"SWT failed: {e}")
        h, w = image.shape
        return np.zeros((h//2, w//2)), np.zeros((h//2, w//2)), np.zeros((h//2, w//2)), np.zeros((h//2, w//2))

def calculate_entropy(coeffs):
    hist, _ = np.histogram(coeffs.flatten(), bins=256, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    return -np.sum(hist * np.log2(hist))

def extract_hog_features(signature_img):
    """Extracts Histogram of Oriented Gradients (HOG) features."""
    if signature_img.size == 0 or np.all(signature_img == 0):
    # Return a zero vector of fixed size if the image is empty
        return np.zeros(324)  # Standard HOG feature vector size for 64x128 image
        
    # Resize for consistent HOG feature vector size
    resized_img = cv2.resize(signature_img, (64, 128), interpolation=cv2.INTER_AREA)
    
    try:
        hog_features = hog(
            resized_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )
        return hog_features
    except Exception as e:
        c_print(f"HOG feature extraction failed: {e}")
        return np.zeros(324)

def extract_enhanced_signature_features(mask_binary, thinned_mask, boundary_coords):
    """Extract an extensive set of features consistent with core logic."""
    features = {}
    signature = mask_binary
    h, w = signature.shape
    # Aspect and density
    features['aspect_ratio'] = w / h if h > 0 else 0
    total_pixels = h * w
    signature_pixels = np.sum(signature > 0)
    features['density'] = signature_pixels / total_pixels if total_pixels > 0 else 0
    # Boundary-based scalars
    features['num_boundary_points'] = len(boundary_coords)
    features['boundary_density'] = len(boundary_coords) / total_pixels if total_pixels > 0 else 0
    if signature_pixels > 0:
        y_coords, x_coords = np.where(signature > 0)
        features['centroid_x'] = np.mean(x_coords) / w if w > 0 else 0
        features['centroid_y'] = np.mean(y_coords) / h if h > 0 else 0
        features['std_x'] = np.std(x_coords) / w if w > 0 else 0
        features['std_y'] = np.std(y_coords) / h if h > 0 else 0
    else:
        features['centroid_x'] = features['centroid_y'] = 0.5
        features['std_x'] = features['std_y'] = 0
    if len(boundary_coords) > 0:
        boundary_array = np.array(boundary_coords)
        features['boundary_centroid_x'] = np.mean(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_centroid_y'] = np.mean(boundary_array[:, 1]) / h if h > 0 else 0
        features['boundary_std_x'] = np.std(boundary_array[:, 0]) / w if w > 0 else 0
        features['boundary_std_y'] = np.std(boundary_array[:, 1]) / h if h > 0 else 0
        min_x, max_x = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])
        min_y, max_y = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])
        features['boundary_width'] = (max_x - min_x) / w if w > 0 else 0
        features['boundary_height'] = (max_y - min_y) / h if h > 0 else 0
    else:
        features['boundary_centroid_x'] = features['boundary_centroid_y'] = 0.5
        features['boundary_std_x'] = features['boundary_std_y'] = 0
        features['boundary_width'] = features['boundary_height'] = 0
    # Projections
    h_projection = np.sum(signature, axis=0)
    v_projection = np.sum(signature, axis=1)
    h_projection = h_projection / np.max(h_projection) if np.max(h_projection) > 0 else h_projection
    v_projection = v_projection / np.max(v_projection) if np.max(v_projection) > 0 else v_projection
    features['h_projection'] = h_projection
    features['v_projection'] = v_projection
    # Components and contour stats
    num_labels, labels = cv2.connectedComponents(signature)
    features['num_components'] = num_labels - 1
    contours, _ = cv2.findContours(signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        features['contour_area_ratio'] = contour_area / (h * w) if (h * w) > 0 else 0
        perimeter = cv2.arcLength(largest_contour, True)
        features['perimeter'] = perimeter / max(h, w) if max(h, w) > 0 else 0
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        features['convexity'] = contour_area / hull_area if hull_area > 0 else 0
        x, y, w_box, h_box = cv2.boundingRect(largest_contour)
        bbox_area = w_box * h_box
        features['bbox_ratio'] = contour_area / bbox_area if bbox_area > 0 else 0
    else:
        features['contour_area_ratio'] = 0
        features['perimeter'] = 0
        features['convexity'] = 0
        features['bbox_ratio'] = 0
    # SWIFT features
    swift_features = extract_swift_features(mask_binary)
    features['swift_energy'] = swift_features['energy']
    features['swift_entropy'] = swift_features['entropy']
    features['swift_mean'] = swift_features['mean']
    features['swift_std'] = swift_features['std']
    features['swift_correlation'] = swift_features['correlation']
    # HOG features
    features['hog'] = extract_hog_features(mask_binary)
    # Path Signature features
    path_points = extract_ordered_skeleton_path(mask_binary)
    # pathsig = extract_path_signature_features(path_points, level=3)
    # features['pathsig'] = pathsig
    # Hu Moments
    features['hu_moments'] = extract_hu_moments(mask_binary)
    return features

# --- FEATURE COMPARISON ---

def compare_hog_features(hog1, hog2):
    """Compares HOG features using cosine similarity."""
    if len(hog1) == 0 or len(hog2) == 0:
        return 0.0
    
    try:
        # Calculate cosine similarity
        dot_product = np.dot(hog1, hog2)
        norm1 = np.linalg.norm(hog1)
        norm2 = np.linalg.norm(hog2)
        
        # Ensure denominator is not zero
        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            # Ensure the result is between 0 and 1
            return max(0.0, min(1.0, cosine_sim))
        else:
            return 0.0
    except Exception as e:
        c_print(f"HOG cosine similarity calculation failed: {e}")
        return 0.0

def compare_enhanced_features(features1, features2):
    """Compares two dictionaries of enhanced features."""
    # sim = {}
    # # Scalar features
    # scalar_keys = ['density', 'num_components']
    # scalar_sims = [1 - abs(features1[k] - features2[k]) / max(features1[k], features2[k], 1) for k in scalar_keys]
    # sim['scalar_avg'] = np.mean(scalar_sims)
    
    # # HOG
    # sim['hog_similarity'] = compare_hog_features(features1['hog'], features2['hog'])

    # # Projection
    # h1, h2 = features1['h_projection'], features2['h_projection']
    # v1, v2 = features1['v_projection'], features2['v_projection']
    # sim['h_projection_corr'] = pearsonr(h1, h2)[0] if np.std(h1) > 0 and np.std(h2) > 0 else 0
    # sim['v_projection_corr'] = pearsonr(v1, v2)[0] if np.std(v1) > 0 and np.std(v2) > 0 else 0
    
    # # SWIFT
    # energy_sims, entropy_sims = [], []
    # for k in features1['energy']:
    #     e1, e2 = features1['energy'][k], features2['energy'][k]
    #     s1, s2 = features1['entropy'][k], features2['entropy'][k]
    #     energy_sims.append(1 - abs(e1 - e2) / max(e1, e2, 1e-6))
    #     entropy_sims.append(1 - abs(s1 - s2) / max(s1, s2, 1e-6))
    
    # # SWIFT similarity is based only on energy for the final scoring
    # sim['swift_similarity'] = np.mean(energy_sims) if energy_sims else 0
    
    # return {k: max(0.0, v if not np.isnan(v) else 0) for k, v in sim.items()}
     
    similarities = {}

    # 1. Scalar Features Comparison
    scalar_features = [
        'density', 'centroid_x', 'centroid_y',
        'std_x', 'std_y', 'num_components', 'contour_area_ratio',
        'perimeter', 'convexity', 'bbox_ratio', 'boundary_density',
        'boundary_centroid_x', 'boundary_centroid_y', 'boundary_std_x',
        'boundary_std_y', 'boundary_width', 'boundary_height'
    ]

    scalar_diffs = []
    for feature in scalar_features:
        val1 = features1.get(feature, 0)
        val2 = features2.get(feature, 0)

        max_val = max(abs(val1), abs(val2), 1e-6)
        diff = abs(val1 - val2) / max_val
        scalar_diffs.append(diff)
        similarities[f'{feature}_similarity'] = 1 - diff

    similarities['scalar_avg'] = 1 - np.mean(scalar_diffs)

    # 2. SWIFT Feature Comparison
    swift_features_to_compare = {
        'energy': True,   # Apply log transform
        'entropy': False,
        'mean': False,
        'std': False,
        'correlation': False
    }

    swift_similarities = []

    for feature_type, use_log_transform in swift_features_to_compare.items():
        swift1 = features1.get(f'swift_{feature_type}', {})
        swift2 = features2.get(f'swift_{feature_type}', {})

        for subband in swift1:
            if subband not in swift2:
                continue  # skip if subband missing in second set

            val1 = swift1[subband]
            val2 = swift2[subband]

            if use_log_transform:
                val1 = np.log1p(max(0, val1))
                val2 = np.log1p(max(0, val2))

            max_val = max(abs(val1), abs(val2), 1e-6)
            diff = abs(val1 - val2) / max_val
            swift_similarities.append(1 - diff)

    similarities['swift_avg'] = np.mean(swift_similarities) if swift_similarities else 0

    # 3. HOG Feature Comparison using Cosine Similarity
    hog1 = features1.get('hog', [])
    hog2 = features2.get('hog', [])
    similarities['hog_similarity'] = compare_hog_features(hog1, hog2)

    # 4. Projection Correlation (Horizontal & Vertical)
    h_proj1 = np.array(features1.get('h_projection', []))
    h_proj2 = np.array(features2.get('h_projection', []))
    v_proj1 = np.array(features1.get('v_projection', []))
    v_proj2 = np.array(features2.get('v_projection', []))

    min_len_h = min(len(h_proj1), len(h_proj2))
    min_len_v = min(len(v_proj1), len(v_proj2))

    if min_len_h > 5:
        h_proj1_resized = cv2.resize(h_proj1.reshape(-1, 1), (1, min_len_h)).flatten()
        h_proj2_resized = cv2.resize(h_proj2.reshape(-1, 1), (1, min_len_h)).flatten()

        if np.std(h_proj1_resized) > 0 and np.std(h_proj2_resized) > 0:
            h_corr, _ = pearsonr(h_proj1_resized, h_proj2_resized)
            # Ensure correlation is between 0 and 1
            similarities['h_projection_corr'] = max(0.0, min(1.0, h_corr if not np.isnan(h_corr) else 0))
        else:
            similarities['h_projection_corr'] = 0.0
    else:
        similarities['h_projection_corr'] = 0.0

    if min_len_v > 5:
        v_proj1_resized = cv2.resize(v_proj1.reshape(-1, 1), (1, min_len_v)).flatten()
        v_proj2_resized = cv2.resize(v_proj2.reshape(-1, 1), (1, min_len_v)).flatten()

        if np.std(v_proj1_resized) > 0 and np.std(v_proj2_resized) > 0:
            v_corr, _ = pearsonr(v_proj1_resized, v_proj2_resized)
            # Ensure correlation is between 0 and 1
            similarities['v_projection_corr'] = max(0.0, min(1.0, v_corr if not np.isnan(v_corr) else 0))
        else:
            similarities['v_projection_corr'] = 0.0
    else:
        similarities['v_projection_corr'] = 0.0

    # # 4. Path Signature Feature Comparison
    # pathsig1 = features1.get('pathsig', np.zeros(14))
    # pathsig2 = features2.get('pathsig', np.zeros(14))
    # similarities['pathsig_similarity'] = compare_path_signature_features(pathsig1, pathsig2)

    # 5. Hu Moments Feature Comparison (log-transformed, Euclidean distance to similarity)
    hu1 = features1.get('hu_moments', np.zeros(7))
    hu2 = features2.get('hu_moments', np.zeros(7))
    if len(hu1) == 7 and len(hu2) == 7:
        dist = euclidean(hu1, hu2)
        # Use a more robust similarity conversion that ensures 0-1 range
        similarities['hu_similarity'] = np.exp(-dist / 50.0)  # Exponential decay with scale factor
    else:
        similarities['hu_similarity'] = 0.0

    # Validate all similarity scores
    similarities = validate_similarity_scores(similarities)

    return similarities


def validate_similarity_scores(similarities):
    """Validate that all similarity scores are properly normalized to [0, 1] range."""
    c_print("\n🔍 VALIDATING SIMILARITY SCORES:")
    c_print("=" * 50)
    
    for key, value in similarities.items():
        if isinstance(value, (int, float)):
            if value < 0.0 or value > 1.0:
                c_print(f"❌ {key}: {value:.4f} (OUT OF RANGE!)")
                # Clamp the value
                similarities[key] = max(0.0, min(1.0, value))
            else:
                c_print(f"✅ {key}: {value:.4f}")
        else:
            c_print(f"⚠️  {key}: {value} (not a number)")
    
    c_print("=" * 50)
    return similarities


def compare_boundary_points(boundary1, boundary2):
    """Compare boundary point distributions with tolerance for minor variations"""
    # Better input validation
    if not boundary1 or not boundary2 or len(boundary1) == 0 or len(boundary2) == 0:
        return 0.0
    
    try:
        points1 = np.array(boundary1, dtype=np.float64)
        points2 = np.array(boundary2, dtype=np.float64)
        
        if points1.ndim != 2 or points2.ndim != 2 or points1.shape[1] != 2 or points2.shape[1] != 2:
            return 0.0
            
    except (ValueError, TypeError):
        return 0.0
    
    def normalize_points(points):
        if len(points) == 0:
            return points
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)
        normalized = (points - min_vals) / range_vals
        return np.clip(normalized, 0, 1)
    
    norm_points1 = normalize_points(points1)
    norm_points2 = normalize_points(points2)
    
    # Use coarser bins to be more forgiving of small variations
    bins = 5
    
    try:
        hist1, _, _ = np.histogram2d(
            norm_points1[:, 0], norm_points1[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        hist2, _, _ = np.histogram2d(
            norm_points2[:, 0], norm_points2[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        
        from scipy.ndimage import gaussian_filter
        hist1 = gaussian_filter(hist1, sigma=0.8)
        hist2 = gaussian_filter(hist2, sigma=0.8)
        
    except ImportError:
        hist1, _, _ = np.histogram2d(
            norm_points1[:, 0], norm_points1[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
        hist2, _, _ = np.histogram2d(
            norm_points2[:, 0], norm_points2[:, 1], 
            bins=bins, range=[[0, 1], [0, 1]], density=True
        )
    except Exception:
        return 0.0
    
    hist1_sum = np.sum(hist1)
    hist2_sum = np.sum(hist2)
    
    if hist1_sum > 0:
        hist1 = hist1 / hist1_sum
    if hist2_sum > 0:
        hist2 = hist2 / hist2_sum
    
    hist1_flat = hist1.flatten()
    hist2_flat = hist2.flatten()
    
    try:
        # 1. Histogram intersection (most forgiving)
        intersection = np.sum(np.minimum(hist1_flat, hist2_flat))
        
        # 2. Cosine similarity (angle between vectors)
        dot_product = np.dot(hist1_flat, hist2_flat)
        norm1 = np.linalg.norm(hist1_flat)
        norm2 = np.linalg.norm(hist2_flat)
        cosine_sim = dot_product / (norm1 * norm2 + 1e-10)
        cosine_sim = max(0, cosine_sim)
        
        # 3. Correlation (but with less weight)
        correlation_matrix = np.corrcoef(hist1_flat, hist2_flat)
        correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
        correlation = max(0, correlation)
        
        # Weight intersection and cosine similarity more heavily
        # These are more forgiving of small shifts
        combined_similarity = (intersection * 0.5 + cosine_sim * 0.3 + correlation * 0.2)
        
        # Apply a boost for high-quality matches to be less harsh
        if combined_similarity > 0.6:
            combined_similarity = combined_similarity * 0.8 + 0.2 * (combined_similarity ** 0.5)
        
        # Ensure the result is properly bounded between 0 and 1
        return max(0.0, min(1.0, combined_similarity))
        
    except Exception:
        return 0.0

def extract_ordered_skeleton_path(binary_img):
    """Extract an ordered path from the skeletonized binary image for Path Signature."""
    if np.max(binary_img) > 1:
        binary_img = (binary_img > 0).astype(np.uint8)
    skeleton = morphology.skeletonize(binary_img)
    points = np.column_stack(np.where(skeleton))
    if len(points) == 0:
        return np.zeros((2, 2))
    from scipy.ndimage import convolve
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    endpoints = np.argwhere((skeleton == 1) & ((neighbor_count == 11) | (neighbor_count == 12)))
    if len(endpoints) == 0:
        return points
    path = [tuple(endpoints[0])]
    visited = set(path)
    current = tuple(endpoints[0])
    while True:
        neighbors = [(current[0]+i, current[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (i==0 and j==0)]
        next_pts = [pt for pt in neighbors if pt in map(tuple, points) and pt not in visited]
        if not next_pts:
            break
        current = next_pts[0]
        path.append(current)
        visited.add(current)
    return np.array(path)

# def extract_path_signature_features(path_points, level=3):
#     """Extract Path Signature features from an ordered path (Nx2 array)."""
#     if path_points.shape[0] < 2:
#         return np.zeros(iisignature.siglength(2, level))
#     min_vals = np.min(path_points, axis=0)
#     max_vals = np.max(path_points, axis=0)
#     denom = np.where((max_vals - min_vals) == 0, 1, (max_vals - min_vals))
#     norm_path = (path_points - min_vals) / denom
#     return iisignature.sig(norm_path, level)

def compare_path_signature_features(sig1, sig2):
    """Compare two Path Signature feature vectors using cosine similarity."""
    if len(sig1) == 0 or len(sig2) == 0:
        return 0.0
    dot = np.dot(sig1, sig2)
    norm1 = np.linalg.norm(sig1)
    norm2 = np.linalg.norm(sig2)
    cosine_sim = dot / (norm1 * norm2 + 1e-10)
    return max(0.0, min(1.0, cosine_sim))

def extract_hu_moments(signature_img):
    """Extract the 7 Hu Moments, invariant to scale, rotation, and translation."""
    if signature_img.size == 0 or np.all(signature_img == 0):
        return np.zeros(7)
    moments = cv2.moments(signature_img)
    hu_moments = cv2.HuMoments(moments).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
        log_hu_moments[np.isinf(log_hu_moments) | np.isnan(log_hu_moments)] = 0
    return log_hu_moments

def extract_features(processed_image):
    """Extract simple weight features to compute weight similarity (compat with core)."""
    if processed_image.ndim < 2:
        return None
    height, width = processed_image.shape[:2]
    raw_weight = np.sum(processed_image == 255)
    total_pixels = height * width
    normalized_weight = raw_weight / total_pixels if total_pixels > 0 else 0
    return {
        'dims': (height, width),
        'raw_weight': raw_weight,
        'normalized_weight': normalized_weight
    }

def compare_signatures(image1, image2):
    """Compare signature weights similar to core's logic."""
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    if features1 is None or features2 is None:
        return {'similarity': 0.0, 'weight1': 0.0, 'weight2': 0.0}
    if features1['dims'] == features2['dims']:
        weight1 = features1['raw_weight']
        weight2 = features2['raw_weight']
        similarity = min(weight1, weight2) / max(weight1, weight2) if max(weight1, weight2) > 0 else 1.0
    else:
        weight1 = features1['normalized_weight']
        weight2 = features2['normalized_weight']
        similarity = min(weight1, weight2) / max(weight1, weight2) if max(weight1, weight2) > 0 else 1.0
    return {
        'similarity': similarity,
        'weight1': features1['raw_weight'],
        'weight2': features2['raw_weight']
    }

# --- HIGH-LEVEL COMPARISON WORKFLOWS ---

def compare_boundary_signatures_with_preprocessed(pre1, pre2):
    """Compares signatures using boundary points with enriched core-aligned logic.
    Keeps the same return structure as before for compatibility.
    """
    boundary1, recon1 = extract_signature_boundary_points_from_preprocessed(pre1)
    boundary2, recon2 = extract_signature_boundary_points_from_preprocessed(pre2)
    features1 = extract_enhanced_signature_features(pre1['mask_binary'], pre1['thinned'], boundary1)
    features2 = extract_enhanced_signature_features(pre2['mask_binary'], pre2['thinned'], boundary2)
    similarities = compare_enhanced_features(features1, features2)
    # Use weight similarity aligned with core method
    weights_info = compare_signatures(recon1, recon2)
    weight_similarity = weights_info['similarity']
    w1 = np.sum(pre1['mask_binary'] > 0)
    w2 = np.sum(pre2['mask_binary'] > 0)
    
    # The weight similarity calculation itself is correct.
    weight_similarity = min(w1, w2) / max(w1, w2, 1)
    return {
        'similarities': similarities,
        'boundary_similarity': compare_boundary_points(boundary1, boundary2),
        'hog_similarity': similarities['hog_similarity'],
        'weight_similarity': weight_similarity
    }

def compare_texture_signatures(pre1, pre2):
    """Compare texture features (SWIFT + gradient) similar to core."""
    try:
        sig1_swift_mask = pre1['mask_binary']
        sig2_swift_mask = pre2['mask_binary']
        sig1_grad_mask = pre1['mask_binary']
        sig2_grad_mask = pre2['mask_binary']
        features1 = extract_swift_features(sig1_swift_mask)
        features2 = extract_swift_features(sig2_swift_mask)
        swift_vector1 = []
        swift_vector2 = []
        for sub_key in features1['energy']:
            val1 = np.log1p(max(0, features1['energy'][sub_key]))
            val2 = np.log1p(max(0, features2['energy'][sub_key]))
            swift_vector1.append(val1)
            swift_vector2.append(val2)
        for key in ['entropy', 'mean', 'std', 'correlation']:
            for sub_key in features1[key]:
                swift_vector1.append(features1[key][sub_key])
                swift_vector2.append(features2[key][sub_key])
        swift_vector1 = np.array(swift_vector1)
        swift_vector2 = np.array(swift_vector2)
        distance = euclidean(swift_vector1, swift_vector2)
        swift_similarity = np.exp(-distance / 10.0)
        grad1_x = cv2.Sobel(sig1_grad_mask, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(sig1_grad_mask, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(sig2_grad_mask, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(sig2_grad_mask, cv2.CV_64F, 0, 1, ksize=3)
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        if grad1_mag.size > 0 and np.std(grad1_mag) > 0 and np.std(grad2_mag) > 0:
            gradient_similarity_raw, _ = pearsonr(grad1_mag.flatten(), grad2_mag.flatten())
            gradient_similarity = (gradient_similarity_raw + 1) / 2
        else:
            gradient_similarity = 0.0
        weights = {'swift': 0.6, 'gradient': 0.4}
        final_score = (weights['swift'] * swift_similarity + weights['gradient'] * gradient_similarity)
        return {
            'swift_similarity': swift_similarity,
            'gradient_similarity': gradient_similarity,
            'final_score': final_score
        }
    except Exception as e:
        return {
            'swift_similarity': 0.0,
            'gradient_similarity': 0.0,
            'final_score': 0.0,
            'error': str(e)
        }