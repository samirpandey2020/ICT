import streamlit as st
import os
import logging
from pathlib import Path
from database import get_random_game_item, save_game_result, get_all_categories

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def display_item(item_path, label):
    """Display an item based on its file type in fixed container"""
    if not item_path or not os.path.exists(item_path):
        st.warning(f"{label}: File not found")
        return

    file_ext = Path(item_path).suffix.lower()

    if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
        # Display image with fixed height
        st.image(item_path, caption=label, use_container_width=True)
    elif file_ext in [".mp3", ".wav", ".ogg"]:
        st.audio(item_path)
        st.caption(label)
    elif file_ext in [".mp4", ".avi", ".mov"]:
        st.video(item_path)
        st.caption(label)
    elif file_ext in [".txt"]:
        with open(item_path, "r", encoding="utf-8") as f:
            st.text_area(label, f.read(), height=200)
    else:
        st.info(f"{label}: {item_path}")


def player_page():
    """Player mode interface"""
    logger.info("🎮 Player page loaded")

    # Initialize session state
    if "player_name" not in st.session_state:
        st.session_state.player_name = ""
        logger.info("📝 Initialized player_name in session state")
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
        logger.info("📝 Initialized selected_category in session state")
    if "current_item" not in st.session_state:
        st.session_state.current_item = None
        logger.info("📝 Initialized current_item in session state")
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
        logger.info("📝 Initialized game_started in session state")
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
        logger.info("📝 Initialized feedback in session state")
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True
        logger.info("📝 Initialized show_welcome in session state")

    # Step 1: Welcome Screen and Player Name
    if st.session_state.show_welcome and not st.session_state.player_name:
        logger.info("👋 Showing welcome screen")
        st.markdown(
            """
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 4rem; margin-bottom: 20px;">🎉 Welcome! 🎉</h1>
                <h2 style="color: #d0d0d0; font-size: 2.5rem;">Guess The Similarity Game</h2>
                <p style="font-size: 1.3rem; color: #b0b0b0; margin-top: 20px;">
                    ✨ Test your perception skills! ✨<br>
                    Compare two items and guess how similar they are! and win exciting prizes!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            player_name_input = st.text_input(
                "👤 Enter Your Name to Begin:",
                placeholder="Your awesome name...",
                key="name_input",
            )

            if st.button("✨ Let's Play! ✨", use_container_width=True):
                if player_name_input:
                    logger.info(f"✅ Player entered name: {player_name_input}")
                    st.session_state.player_name = player_name_input
                    st.session_state.show_welcome = False
                    st.rerun()
                else:
                    logger.warning(
                        "⚠️ Player attempted to proceed without entering name"
                    )
                    st.warning("⚠️ Please enter your name first!")
        return

    # Step 2: Category Selection
    if st.session_state.player_name and not st.session_state.selected_category:
        logger.info(f"🎮 Player '{st.session_state.player_name}' selecting category")
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px;">
                <h1 style="color: #f0f0f0;">🎉 Hello, {st.session_state.player_name}! 🎉</h1>
                <p style="font-size: 1.5rem; color: #b0b0b0; margin-top: 20px;">
                    Choose your game category below:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Get available categories from database
            categories = get_all_categories()
            logger.info(f"📊 Retrieved {len(categories)} categories from database")

            if not categories:
                logger.warning("⚠️ No game categories available in database")
                st.error(
                    "😞 No games available yet. Please ask admin to add game pairs!"
                )

                if st.button("⬅️ Back"):
                    st.session_state.player_name = ""
                    st.session_state.show_welcome = True
                    st.rerun()
                return

            selected = st.selectbox(
                "🎮 Select Game Category:", categories, key="category_select"
            )

            if st.button("🎮 Start Game! 🎮", use_container_width=True):
                logger.info(f"✅ Player selected category: {selected}")
                st.session_state.selected_category = selected
                st.rerun()

            if st.button("⬅️ Change Name", use_container_width=True):
                logger.info("🔄 Player wants to change name")
                st.session_state.player_name = ""
                st.session_state.show_welcome = True
                st.rerun()
        return

    # Step 3: Game Play
    if st.session_state.player_name and st.session_state.selected_category:
        logger.info(
            f"🎮 Game active - Player: {st.session_state.player_name}, Category: {st.session_state.selected_category}"
        )

        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px;">
                <h1 style="color: #f0f0f0;">🎉 Hello, {st.session_state.player_name}! 🎉</h1>
                <p style="font-size: 1.5rem; color: #b0b0b0; margin-top: 20px;">
                    let's play  {st.session_state.selected_category}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Start new round button
        if not st.session_state.game_started or st.session_state.feedback:
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🎲 ✨ Start New Round ✨ 🎲", use_container_width=True):
                    logger.info(
                        f"🎲 Starting new round for category: {st.session_state.selected_category}"
                    )
                    item = get_random_game_item(
                        st.session_state.selected_category, st.session_state.player_name
                    )
                    if item:
                        logger.info(f"✅ Retrieved game item ID: {item[0]}")
                        st.session_state.current_item = item
                        st.session_state.game_started = True
                        st.session_state.feedback = None
                        st.rerun()
                    else:
                        # No more unseen items available
                        logger.info(
                            f"🏆 Player completed all items in category: {st.session_state.selected_category}"
                        )
                        st.warning(
                            "🎉 Congratulations! You've completed all available pairs in this category! 🎉"
                        )
                        st.info(
                            "🔄 Check back later for new pairs or try a different category!"
                        )
                        st.session_state.game_started = False

        # Display current game
        if (
            st.session_state.current_item
            and st.session_state.game_started
            and not st.session_state.feedback
        ):
            item = st.session_state.current_item
            item_id, category, item1_path, item2_path, actual_score, description = item
            logger.info(
                f"📊 Displaying game item - ID: {item_id}, Score: {actual_score}"
            )

            if description:
                st.info(f"📝 {description}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✨ Item 1 ✨")
                st.markdown(
                    """
                    <style>
                    div[data-testid="stImage"] {
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 400px;
                        max-height: 400px;
                    }
                    div[data-testid="stImage"] img {
                        max-height: 400px !important;
                        object-fit: contain !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                display_item(item1_path, "Item 1")

            with col2:
                st.markdown("### ✨ Item 2 ✨")
                display_item(item2_path, "Item 2")

            st.markdown("---")

            col_input1, col_input2, col_input3 = st.columns([1, 2, 1])
            with col_input2:
                guessed_score = st.number_input(
                    "🎯 Your Similarity Guess (0-100):",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    key="score_input",
                )

                if st.button("✨ Submit Guess ✨", use_container_width=True):
                    logger.info(
                        f"🎯 Player guessed: {guessed_score}, Actual: {actual_score}"
                    )
                    # Show calculating message
                    with st.spinner("🔍 Analyzing similarities... 🧠"):
                        import time

                        time.sleep(1)  # Simulate processing

                    # Calculate if guess is correct
                    tolerance = 10
                    is_correct = abs(guessed_score - actual_score) <= tolerance
                    logger.info(
                        f"✅ Result: {'CORRECT' if is_correct else 'INCORRECT'} (tolerance: {tolerance})"
                    )

                    # Save result with game_item_id
                    save_game_result(
                        st.session_state.player_name,
                        st.session_state.selected_category,
                        guessed_score,
                        actual_score,
                        is_correct,
                        item_id,
                    )
                    logger.info(
                        f"💾 Saved game result to database for player: {st.session_state.player_name}"
                    )

                    # Set feedback
                    difference = abs(guessed_score - actual_score)
                    if is_correct:
                        st.session_state.feedback = "correct"
                    else:
                        st.session_state.feedback = "incorrect"

                    st.session_state.feedback_data = {
                        "guessed": guessed_score,
                        "actual": actual_score,
                        "difference": difference,
                    }
                    logger.info(
                        f"📢 Showing feedback: {st.session_state.feedback}, difference: {difference}"
                    )
                    st.rerun()

        # Display feedback
        if st.session_state.feedback:
            logger.info(f"📣 Displaying feedback UI: {st.session_state.feedback}")
            if st.session_state.feedback == "correct":
                # Auto-scroll to feedback with advanced animation
                difference = st.session_state.feedback_data["difference"]

                # Calculate score percentage
                accuracy = 100 - difference

                # Determine performance level
                if difference == 0:
                    performance = "PERFECT MATCH!"
                    performance_emoji = "🎯🌟"
                    performance_color = "#FFD700"
                elif difference <= 3:
                    performance = "OUTSTANDING!"
                    performance_emoji = "🎆🎉"
                    performance_color = "#FF6347"
                elif difference <= 7:
                    performance = "EXCELLENT!"
                    performance_emoji = "🌟✨"
                    performance_color = "#FFA500"
                else:
                    performance = "GREAT JOB!"
                    performance_emoji = "👏🎈"
                    performance_color = "#32CD32"

                st.markdown(
                    f"""
                    <div id="feedback-popup" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; text-align: center; margin: 20px auto; max-width: 500px; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6); animation: popIn 0.5s ease-out;">
                        <div style="font-size: 4rem; margin-bottom: 15px; animation: bounce 0.8s ease-in-out;">
                            {performance_emoji}
                        </div>
                        <h2 style="color: white; font-size: 2rem; margin: 10px 0; font-weight: bold;">
                            {performance}
                        </h2>
                        <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                            <div style="display: flex; justify-content: space-around; align-items: center;">
                                <div>
                                    <div style="font-size: 2.5rem; color: white; font-weight: bold;">{st.session_state.feedback_data['guessed']}</div>
                                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Your Guess</div>
                                </div>
                                <div style="font-size: 2rem; color: #FFD700;">✓</div>
                                <div>
                                    <div style="font-size: 2.5rem; color: white; font-weight: bold;">{st.session_state.feedback_data['actual']}</div>
                                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Actual Score</div>
                                </div>
                            </div>
                        </div>
                        <div style="background: rgba(255,255,255,0.25); height: 8px; border-radius: 10px; overflow: hidden; margin: 15px 0;">
                            <div style="background: linear-gradient(90deg, #00ff87 0%, #60efff 100%); height: 100%; width: {accuracy}%; border-radius: 10px; animation: fillBar 1s ease-out;"></div>
                        </div>
                        <p style="color: #FFD700; font-size: 1.1rem; margin: 15px 0 0 0; font-weight: 600;">
                            {accuracy}% Accurate! ✨
                        </p>
                    </div>
                    <style>
                    @keyframes popIn {{
                        0% {{ transform: scale(0.8); opacity: 0; }}
                        100% {{ transform: scale(1); opacity: 1; }}
                    }}
                    @keyframes bounce {{
                        0%, 100% {{ transform: translateY(0); }}
                        50% {{ transform: translateY(-10px); }}
                    }}
                    @keyframes fillBar {{
                        0% {{ width: 0%; }}
                        100% {{ width: {accuracy}%; }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                # Show balloons for success
                st.balloons()

                # Play success sound - Congratulations You Won!
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.myinstants.com/media/sounds/congratulations-you-won.mp3" type="audio/mpeg">
                    </audio>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Incorrect answer with enhanced feedback
                difference = st.session_state.feedback_data["difference"]

                # Determine how close they were
                if difference <= 15:
                    closeness = "You were SO close!"
                    closeness_emoji = "😅"
                elif difference <= 25:
                    closeness = "Not bad! Almost there!"
                    closeness_emoji = "😊"
                else:
                    closeness = "Keep practicing!"
                    closeness_emoji = "💪"

                st.markdown(
                    f"""
                    <div id="feedback-popup" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 20px; text-align: center; margin: 20px auto; max-width: 500px; box-shadow: 0 10px 40px rgba(245, 87, 108, 0.6); animation: shake 0.5s ease-out;">
                        <div style="font-size: 4rem; margin-bottom: 15px;">
                            {closeness_emoji}
                        </div>
                        <h2 style="color: white; font-size: 2rem; margin: 10px 0; font-weight: bold;">
                            {closeness}
                        </h2>
                        <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 15px; margin: 20px 0;">
                            <div style="display: flex; justify-content: space-around; align-items: center;">
                                <div>
                                    <div style="font-size: 2.5rem; color: white; font-weight: bold;">{st.session_state.feedback_data['guessed']}</div>
                                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Your Guess</div>
                                </div>
                                <div style="font-size: 2rem; color: #FFE4E1;">✗</div>
                                <div>
                                    <div style="font-size: 2.5rem; color: white; font-weight: bold;">{st.session_state.feedback_data['actual']}</div>
                                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">Actual Score</div>
                                </div>
                            </div>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 12px; margin: 15px 0; border-left: 4px solid #FFD700;">
                            <div style="font-size: 1rem; color: #FFD700; margin-bottom: 5px; font-weight: 600;">
                                💡 Tip
                            </div>
                            <div style="font-size: 0.95rem; color: white;">
                                Look for small details and patterns!
                            </div>
                        </div>
                        <p style="color: #FFE4E1; font-size: 1.1rem; margin: 15px 0 0 0; font-weight: 600;">
                            Keep trying! You'll get it next time! 💪
                        </p>
                    </div>
                    <style>
                    @keyframes shake {{
                        0%, 100% {{ transform: translateX(0); }}
                        25% {{ transform: translateX(-8px); }}
                        75% {{ transform: translateX(8px); }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Play incorrect sound
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://assets.mixkit.co/active_storage/sfx/2003/2003-preview.mp3" type="audio/mpeg">
                    </audio>
                """,
                    unsafe_allow_html=True,
                )
