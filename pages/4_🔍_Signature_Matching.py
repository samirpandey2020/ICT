import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
import time

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_database, add_game_item, ASSETS_FOLDER
import os

st.set_page_config(page_title="Signature Matching 🔍", page_icon="🔍", layout="wide")

# Initialize database
init_database()


@st.cache_resource
def load_verifier():
    """Load signature verifier with caching"""
    try:
        from signature.signature import SignatureVerifier

        return SignatureVerifier()
    except Exception as e:
        st.error(f"❌ Failed to load signature verifier: {e}")
        return None


def signature_matching_page():
    """Signature Matching mode interface"""

    st.title("🔍 Signature Matching ")

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <h3 style="color: white; margin: 0;">📝 How it works:</h3>
            <p style="color: white; font-size: 1.1rem; margin-top: 10px; margin-bottom: 0;">
                Upload two signature images and our AI will analyze them to determine how similar they are.
                Perfect for signature verification and authentication!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "image1" not in st.session_state:
        st.session_state.image1 = None
    if "image2" not in st.session_state:
        st.session_state.image2 = None
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "image1_hash" not in st.session_state:
        st.session_state.image1_hash = None
    if "image2_hash" not in st.session_state:
        st.session_state.image2_hash = None

    # Two columns for image upload
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📸 Signature 1")
        uploaded_file1 = st.file_uploader(
            "Upload first signature image",
            type=["jpg", "jpeg", "png"],
            key="sig1",
            help="Upload the first signature to compare",
        )

        if uploaded_file1:
            # Read and display image
            file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
            image1 = cv2.imdecode(file_bytes1, cv2.IMREAD_COLOR)

            if image1 is not None:
                # Check if image changed - clear results if so
                current_hash = hash(image1.tobytes())
                if st.session_state.image1_hash != current_hash:
                    st.session_state.comparison_result = None
                    st.session_state.image1_hash = current_hash
                    # Clear cache when images change
                    load_verifier.clear()

                # Display with fixed height CSS
                st.markdown(
                    """
                    <style>
                    div[data-testid="stImage"] img {
                        max-height: 350px !important;
                        object-fit: contain !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(
                    cv2.cvtColor(image1, cv2.COLOR_BGR2RGB),
                    caption="✅ Signature 1 Loaded",
                    use_container_width=True,
                )
                st.session_state.image1 = image1
                st.success(
                    f"📊 Image 1 size: {image1.shape[1]}x{image1.shape[0]} pixels"
                )
            else:
                st.error("❌ Failed to load image 1")

    with col2:
        st.markdown("### 📸 Signature 2")
        uploaded_file2 = st.file_uploader(
            "Upload second signature image",
            type=["jpg", "jpeg", "png"],
            key="sig2",
            help="Upload the second signature to compare",
        )

        if uploaded_file2:
            # Read and display image
            file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)
            image2 = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)

            if image2 is not None:
                # Check if image changed - clear results if so
                current_hash = hash(image2.tobytes())
                if st.session_state.image2_hash != current_hash:
                    st.session_state.comparison_result = None
                    st.session_state.image2_hash = current_hash
                    # Clear cache when images change
                    load_verifier.clear()

                # Display with fixed height CSS
                st.markdown(
                    """
                    <style>
                    div[data-testid="stImage"] img {
                        max-height: 350px !important;
                        object-fit: contain !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(
                    cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
                    caption="✅ Signature 2 Loaded",
                    use_container_width=True,
                )
                st.session_state.image2 = image2
                st.success(
                    f"📊 Image 2 size: {image2.shape[1]}x{image2.shape[0]} pixels"
                )
            else:
                st.error("❌ Failed to load image 2")

    st.markdown("---")

    # Compare button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("🔍 Compare Signatures", use_container_width=True):
            if st.session_state.image1 is None or st.session_state.image2 is None:
                st.warning("⚠️ Please upload both signatures first!")
            else:
                # Show processing message
                with st.spinner("🧠 Analyzing signatures... Please wait..."):
                    time.sleep(0.5)  # Brief delay for UX

                    # Load verifier
                    verifier = load_verifier()

                    if verifier is None:
                        st.error("❌ Failed to initialize signature verifier")
                    else:
                        try:
                            # Compare signatures
                            confidence, match_details = verifier.compare_two_signatures(
                                st.session_state.image1, st.session_state.image2
                            )

                            # Store result
                            st.session_state.comparison_result = {
                                "confidence": confidence,
                                "match": match_details,
                            }

                            # Trigger balloons and rerun to show results
                            st.balloons()
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Comparison failed: {str(e)}")
                            st.session_state.comparison_result = None

    # Display results
    if st.session_state.comparison_result is not None:
        result = st.session_state.comparison_result
        confidence = result["confidence"]
        match_details = result["match"]

        st.markdown("---")

        # Determine match status
        if confidence >= 85:
            status = "HIGH MATCH"
            status_emoji = "✅"
            status_color = "#00ff87"
            status_bg = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            message = (
                "The signatures are highly similar! Very likely from the same person."
            )
        elif confidence >= 80:
            status = "MEDIUM MATCH"
            status_emoji = "⚠️"
            status_color = "#FFA500"
            status_bg = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            message = (
                "The signatures show some similarities but have notable differences."
            )
        else:
            status = "LOW MATCH"
            status_emoji = "❌"
            status_color = "#FF6B6B"
            status_bg = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            message = "The signatures are different. Likely from different people."

        # Results card
        st.markdown(
            f"""
            <div style="background: {status_bg}; padding: 40px; border-radius: 25px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.3); animation: popIn 0.6s ease-out;">
                <div style="background: {status_color}; color: white; padding: 15px 35px; border-radius: 50px; display: inline-block; margin-bottom: 25px; font-size: 1.6rem; font-weight: bold; box-shadow: 0 5px 20px rgba(0,0,0,0.3); animation: pulse 1.5s infinite;">
                    {status_emoji} {status} {status_emoji}
                </div>
                <div style="margin: 30px 0;">
                    <div style="font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-bottom: 15px;">
                        🎯 SIMILARITY CONFIDENCE
                    </div>
                    <div style="font-size: 5rem; color: white; font-weight: bold; text-shadow: 3px 3px 6px rgba(0,0,0,0.3);">
                        {confidence:.1f}%
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.3); height: 50px; border-radius: 25px; overflow: hidden; margin: 25px 0;">
                    <div style="background: {status_color}; height: 100%; width: {confidence}%; border-radius: 25px; animation: fillBar 1.5s ease-out; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; font-size: 1.2rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {confidence:.1f}%
                    </div>
                </div>
                <p style="color: white; font-size: 1.3rem; margin-top: 20px; margin-bottom: 0; line-height: 1.6;">
                    {message}
                </p>
            </div>
            <style>
            @keyframes popIn {{
                0% {{ transform: scale(0.3) rotate(-5deg); opacity: 0; }}
                50% {{ transform: scale(1.05) rotate(2deg); }}
                100% {{ transform: scale(1) rotate(0deg); opacity: 1; }}
            }}
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.1); }}
            }}
            @keyframes fillBar {{
                0% {{ width: 0%; }}
                100% {{ width: {confidence}%; }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Additional details in expandable section
        with st.expander("📊 View Detailed Analysis"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 🔢 Technical Details")
                st.write(f"**Raw Score:** {match_details.score:.4f}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write(f"**Signature ID:** {match_details.signature_id}")

            with col_b:
                st.markdown("### 🎯 Match Criteria")
                if confidence >= 80:
                    st.success("✅ Passes high-confidence threshold")
                elif confidence >= 60:
                    st.warning("⚠️ Moderate confidence level")
                else:
                    st.error("❌ Below confidence threshold")

        # Add to Game Database Section
        st.markdown("---")
        st.markdown("### 💾 Save to Game Database")

        col_save1, col_save2, col_save3 = st.columns([1, 2, 1])
        with col_save2:
            st.info("🎮 Add this signature pair to your game for players to guess!")

            with st.form("save_to_db_form"):
                description = st.text_input(
                    "📝 Description (optional):",
                    placeholder="e.g., Similar signatures from different people",
                )

                # Use AI confidence as the actual score
                st.write(
                    f"🎯 **AI Score (will be used as actual score):** {int(confidence)}%"
                )

                submit_button = st.form_submit_button(
                    "💾 ✨ Save Pair to Database ✨", use_container_width=True
                )

                if submit_button:
                    try:
                        # Create unique filenames
                        import datetime

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Save images to assets folder
                        ASSETS_FOLDER.mkdir(exist_ok=True)

                        img1_filename = f"sig_{timestamp}_1.png"
                        img2_filename = f"sig_{timestamp}_2.png"

                        img1_path = ASSETS_FOLDER / img1_filename
                        img2_path = ASSETS_FOLDER / img2_filename

                        # Save the images
                        cv2.imwrite(str(img1_path), st.session_state.image1)
                        cv2.imwrite(str(img2_path), st.session_state.image2)

                        # Add to database
                        add_game_item(
                            category="✍️ Signature Sleuth",
                            item1_path=str(img1_path),
                            item2_path=str(img2_path),
                            actual_score=int(confidence),
                            description=(
                                description
                                if description
                                else f"AI Confidence: {confidence:.1f}%"
                            ),
                        )

                        st.success("🎉 Successfully saved to game database!")
                        st.balloons()

                    except Exception as e:
                        st.error(f"❌ Failed to save: {str(e)}")


# Run the page
signature_matching_page()
