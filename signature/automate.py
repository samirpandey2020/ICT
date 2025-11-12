import os
import random
import tempfile
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from itertools import zip_longest

from signature.variables import *
from signature.signature_show import SignatureVerifier  # your existing verifier

# --- CONFIGURATION ---
base_dir = "D:/signatue_detection/signature_testing/clean_directory"
output_dir = "D:/signatue_detection/signature_testing/reports"

model_path = SIGNATURE_DETECTION_PATH
# --- SETUP ---

os.makedirs(output_dir, exist_ok=True)
verifier = SignatureVerifier(model_path=model_path)
width, height = A4


# --- HELPER FUNCTION (NO CHANGE) ---
def add_image_to_pdf(
    pdf_canvas, img_path, x, y, max_width=3 * inch, max_height=2 * inch
):
    """Adds a resized image to the PDF and returns the height it occupied."""
    try:
        img = Image.open(img_path)
        img_width, img_height = img.size
        if img_width == 0 or img_height == 0:
            return 0
        ratio = min(max_width / img_width, max_height / img_height)
        new_width = img_width * ratio
        new_height = img_height * ratio
        pdf_canvas.drawImage(
            img_path, x, y - new_height, width=new_width, height=new_height
        )
        return new_height
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {img_path}")
        return 0
    except Exception as e:
        print(f"❌ Could not draw image {img_path}: {e}")
        return 0


# --- MAIN PROCESSING LOOP ---
print("🚀 Starting batch verification process with new systematic layout...")
for root, dirs, files in os.walk(base_dir):
    img_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(img_files) >= 2:
        dir_name = os.path.basename(root)
        output_pdf_path = os.path.join(output_dir, f"report_{dir_name}.pdf")

        c = canvas.Canvas(output_pdf_path, pagesize=A4)
        y_pos = height - inch

        margin = inch
        col1_x = margin
        col2_x = width / 2 + margin / 2

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                selected = random.sample(img_files, 2)
                cheque_file, ref_file = selected
                cheque_path = os.path.join(root, cheque_file)
                ref_path = os.path.join(root, ref_file)

                matches = verifier.verify_signatures(
                    cheque_image_path=cheque_path,
                    reference_image_paths=[ref_path],
                    output_dir=temp_dir,
                )

                # --- SECTION 1: REPORT HEADER AND SOURCE IMAGES (CORRECTED) ---
                c.setFont("Helvetica-Bold", 14)
                c.drawString(margin, y_pos, f"Verification Report: {dir_name}")
                y_pos -= 0.5 * inch

                c.setFont("Helvetica-Bold", 11)
                c.drawString(
                    col2_x, y_pos, "Source Reference Image"
                )  # Draw reference title first

                # =============================================================
                # === FIX: Check for annotated image and use original as fallback
                # =============================================================
                annotated_cheque_path = os.path.join(
                    temp_dir, f"annotated_{os.path.basename(cheque_path)}"
                )

                if os.path.exists(annotated_cheque_path):
                    display_cheque_path = annotated_cheque_path
                    c.drawString(col1_x, y_pos, "Source Cheque Image (Annotated)")
                else:
                    display_cheque_path = cheque_path  # Fallback to the original
                    c.drawString(col1_x, y_pos, "Source Cheque Image (Original)")
                    print(
                        f"⚠️ Annotated image not found for {dir_name}. Using original cheque image."
                    )

                y_pos -= 0.2 * inch

                h1 = add_image_to_pdf(
                    c, display_cheque_path, col1_x, y_pos, max_width=3.5 * inch
                )
                # =============================================================

                h2 = add_image_to_pdf(c, ref_path, col2_x, y_pos, max_width=3.5 * inch)
                y_pos -= max(h1, h2) + 0.3 * inch

                if not matches:
                    c.setFont("Helvetica", 12)
                    c.drawString(margin, y_pos, "⚠️ No matching signatures found.")

                # --- SECTION 2: INDIVIDUAL MATCH DETAILS (NO CHANGE) ---
                for i, match in enumerate(matches):
                    y_pos -= 0.2 * inch
                    if y_pos < height / 2:
                        c.showPage()
                        y_pos = height - inch

                    c.setStrokeColorRGB(0.8, 0.8, 0.8)
                    c.line(margin, y_pos, width - margin, y_pos)
                    y_pos -= 0.4 * inch

                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(
                        margin,
                        y_pos,
                        f"Match #{i+1} - Similarity Score: {match.score:.2%}",
                    )
                    y_pos -= 0.3 * inch

                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(col1_x, y_pos, "Detected Cheque Signature")
                    c.drawString(col2_x, y_pos, "Matched Reference Signature")
                    y_pos -= 0.2 * inch

                    cheque_sig_path = os.path.join(
                        temp_dir, f"match_{i}_cheque_sig.png"
                    )
                    Image.fromarray(match.cheque_signature.image).save(cheque_sig_path)

                    ref_sig_path = os.path.join(temp_dir, f"match_{i}_ref_sig.png")
                    Image.fromarray(match.matched_signature.image).save(ref_sig_path)

                    h1 = add_image_to_pdf(
                        c, cheque_sig_path, col1_x, y_pos, max_height=1.5 * inch
                    )
                    h2 = add_image_to_pdf(
                        c, ref_sig_path, col2_x, y_pos, max_height=1.5 * inch
                    )
                    y_pos -= max(h1, h2) + 0.3 * inch

                    # --- SECTION 3: COLUMNAR PRE-PROCESSING STEPS (NO CHANGE) ---
                    if (
                        match.cheque_preprocessed_images
                        and match.ref_preprocessed_images
                    ):
                        if y_pos < 3 * inch:
                            c.showPage()
                            y_pos = height - inch

                        c.setFont("Helvetica-Bold", 11)
                        c.drawString(margin, y_pos, "Pre-processing Comparison")
                        y_pos -= 0.3 * inch

                        cheque_steps = list(match.cheque_preprocessed_images.items())
                        ref_steps = list(match.ref_preprocessed_images.items())

                        for cheque_item, ref_item in zip_longest(
                            cheque_steps, ref_steps
                        ):
                            if y_pos < 2.5 * inch:
                                c.showPage()
                                y_pos = height - inch

                            h1, h2 = 0, 0

                            if cheque_item:
                                step_name, step_img = cheque_item
                                c.setFont("Helvetica", 9)
                                c.drawString(
                                    col1_x,
                                    y_pos,
                                    f"Cheque Step: {step_name.replace('_', ' ').title()}",
                                )
                                temp_img_path = os.path.join(
                                    temp_dir, f"match_{i}_cheque_{step_name}.png"
                                )
                                Image.fromarray(step_img).save(temp_img_path)
                                h1 = add_image_to_pdf(
                                    c,
                                    temp_img_path,
                                    col1_x,
                                    y_pos - 0.15 * inch,
                                    max_width=3 * inch,
                                    max_height=1.5 * inch,
                                )

                            if ref_item:
                                step_name, step_img = ref_item
                                c.setFont("Helvetica", 9)
                                c.drawString(
                                    col2_x,
                                    y_pos,
                                    f"Reference Step: {step_name.replace('_', ' ').title()}",
                                )
                                temp_img_path = os.path.join(
                                    temp_dir, f"match_{i}_ref_{step_name}.png"
                                )
                                Image.fromarray(step_img).save(temp_img_path)
                                h2 = add_image_to_pdf(
                                    c,
                                    temp_img_path,
                                    col2_x,
                                    y_pos - 0.15 * inch,
                                    max_width=3 * inch,
                                    max_height=1.5 * inch,
                                )

                            y_pos -= max(h1, h2) + 0.3 * inch

        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(margin, y_pos, f"❌ An error occurred: {e}")

        c.save()
        print(f"✅ Report saved for '{dir_name}' at {output_pdf_path}")

print("🎉 Batch processing complete.")
