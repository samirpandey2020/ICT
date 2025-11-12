"""
Simple test to demonstrate compare_two_signatures function
"""

import cv2
import sys

sys.path.insert(0, ".")

from signature.signature import SignatureVerifier

# Initialize verifier
print("🔧 Initializing SignatureVerifier...")
verifier = SignatureVerifier()

# Load two test signature images
print("📂 Loading test images...")
sig1_path = "models/test_image/sig_1_1.png"
sig2_path = "models/test_image/sig_1_1.png"

sig1 = cv2.imread(sig1_path)
sig2 = cv2.imread(sig2_path)

if sig1 is None or sig2 is None:
    print("❌ Failed to load images. Check paths:")
    print(f"   Image 1: {sig1_path}")
    print(f"   Image 2: {sig2_path}")
    sys.exit(1)

print(f"✅ Loaded images successfully")
print(f"   Signature 1 shape: {sig1.shape}")
print(f"   Signature 2 shape: {sig2.shape}")
print()

# Compare the two signatures
print("=" * 60)
print("🔍 COMPARING SIGNATURES")
print("=" * 60)

confidence, match_details = verifier.compare_two_signatures(sig1, sig2)

print("=" * 60)
print("📊 RESULTS")
print("=" * 60)

if match_details:
    print(f"✅ Comparison successful!")
    print(f"   Confidence: {confidence:.2f}%")
    print(f"   Raw Score: {match_details.score:.4f}")
    print(f"   Signature ID: {match_details.signature_id}")

    if confidence >= 80:
        print(f"   Status: ✅ HIGH MATCH - Signatures are very similar")
    elif confidence >= 60:
        print(f"   Status: ⚠️ MEDIUM MATCH - Signatures are somewhat similar")
    else:
        print(f"   Status: ❌ LOW MATCH - Signatures are different")
else:
    print("❌ Comparison failed")

print("=" * 60)
