import streamlit as st
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Signature Detective ✍️", page_icon="✍️", layout="wide")

# Maximum plays allowed per user
MAX_PLAYS_PER_USER = 1


def get_player_play_count(player_name):
    """Get the number of times a player has played the signature game"""
    import sqlite3

    conn = sqlite3.connect("game_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM leaderboard 
        WHERE player_name = ? AND category = '✍️ Signature Sleuth'
        """,
        (player_name,),
    )
    count = cursor.fetchone()[0]
    conn.close()
    logger.info(f"📊 Player '{player_name}' has played {count} times")
    return count


def get_random_game_item(category=None, player_name=None):
    """Fetch a random game item from database that the player hasn't seen yet"""
    import sqlite3

    logger.info(f"🎲 Fetching game item - Category: {category}, Player: {player_name}")
    conn = sqlite3.connect("game_data.db")
    cursor = conn.cursor()

    if player_name:
        # Get items the player has already seen
        if category and category != "All":
            cursor.execute(
                """
                SELECT * FROM game_items 
                WHERE category = ? 
                AND id NOT IN (
                    SELECT game_item_id FROM leaderboard 
                    WHERE player_name = ? AND category = ? AND game_item_id IS NOT NULL
                )
                ORDER BY RANDOM() LIMIT 1
                """,
                (category, player_name, category),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM game_items 
                WHERE id NOT IN (
                    SELECT game_item_id FROM leaderboard 
                    WHERE player_name = ? AND game_item_id IS NOT NULL
                )
                ORDER BY RANDOM() LIMIT 1
                """,
                (player_name,),
            )
    else:
        # No player tracking, just get random item
        if category and category != "All":
            cursor.execute(
                "SELECT * FROM game_items WHERE category = ? ORDER BY RANDOM() LIMIT 1",
                (category,),
            )
        else:
            cursor.execute("SELECT * FROM game_items ORDER BY RANDOM() LIMIT 1")

    item = cursor.fetchone()
    conn.close()
    if item:
        logger.info(f"✅ Found game item ID: {item[0]}")
    else:
        logger.warning("⚠️ No game items available for player")
    return item


def save_game_result(
    player_name, category, guessed_score, actual_score, is_correct, game_item_id
):
    """Save game result to leaderboard"""
    import sqlite3

    logger.info(
        f"💾 Saving result - Player: {player_name}, Correct: {is_correct}, Item ID: {game_item_id}"
    )
    conn = sqlite3.connect("game_data.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO leaderboard (player_name, category, guessed_score, actual_score, is_correct, game_item_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
        (player_name, category, guessed_score, actual_score, is_correct, game_item_id),
    )
    conn.commit()
    conn.close()
    logger.info("✅ Game result saved successfully")


def signature_guessing_game():
    """Signature Guessing Game interface"""
    logger.info("✍️ Signature Guessing Game loaded")

    # Initialize session state
    if "sig_player_name" not in st.session_state:
        st.session_state.sig_player_name = ""
        logger.info("📝 Initialized sig_player_name in session state")
    if "sig_current_item" not in st.session_state:
        st.session_state.sig_current_item = None
        logger.info("📝 Initialized sig_current_item in session state")
    if "sig_game_started" not in st.session_state:
        st.session_state.sig_game_started = False
        logger.info("📝 Initialized sig_game_started in session state")
    if "sig_feedback" not in st.session_state:
        st.session_state.sig_feedback = None
        logger.info("📝 Initialized sig_feedback in session state")
    if "sig_show_welcome" not in st.session_state:
        st.session_state.sig_show_welcome = True
        logger.info("📝 Initialized sig_show_welcome in session state")

    # Step 1: Welcome Screen and Player Name
    if st.session_state.sig_show_welcome and not st.session_state.sig_player_name:
        logger.info("👋 Showing welcome screen")
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; position: relative;">
                <div style="
                    background: linear-gradient(135deg, rgba(40, 40, 40, 0.3) 0%, rgba(30, 30, 30, 0.3) 100%);
                    border: 2px solid rgba(120, 120, 120, 0.4);
                    border-radius: 30px;
                    padding: 50px;
                    box-shadow: 0 25px 70px rgba(0,0,0,0.6),
                                inset 0 0 50px rgba(120, 120, 120, 0.2);
                    backdrop-filter: blur(15px);
                    animation: floatIn 1s ease-out;
                ">
                    <div style="font-size: 6rem; margin-bottom: 20px;
                                animation: glow 2s ease-in-out infinite alternate;
                                filter: drop-shadow(0 0 40px rgba(120, 120, 120, 1));">
                        ✍️
                    </div>
                    <h1 style="
                        font-size: 4.5rem;
                        margin-bottom: 20px;
                        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 50%, #c0c0c0 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        font-weight: 900;
                        letter-spacing: 3px;
                        text-shadow: 0 0 60px rgba(255, 255, 255, 0.7);
                    ">SIGNATURE DETECTIVE</h1>
                    <div style="
                        height: 4px;
                        width: 200px;
                        background: linear-gradient(90deg, #ffffff, #e0e0e0, #c0c0c0);
                        margin: 30px auto;
                        border-radius: 2px;
                        box-shadow: 0 0 30px rgba(255, 255, 255, 1);
                    "></div>
                    <h2 style="color: #ffffff; font-size: 2.2rem; font-weight: 600; margin: 20px 0;">
                         Master the Art of Signature Analysis
                    </h2>
                    <p style="font-size: 1.4rem; color: #f0f0f0; margin-top: 25px; line-height: 1.8;">
                         <strong>Challenge Your Perception</strong> 🔍<br>
                        Compare signatures • Guess similarity • Beat the AI<br>
                        <span style="color: #ffffff; font-weight: bold;">Are you ready for the ultimate test?</span>
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Name input directly below the main welcome content but within the same visual area
        st.markdown(
            """
            <div style="text-align: center; margin-top: -100px; margin-bottom: 40px; position: relative; z-index: 10;">
                <div style="
                    background: rgba(60, 60, 60, 0.4);
                    border-radius: 20px;
                    padding: 30px;
                    max-width: 600px;
                    margin: 0 auto;
                    border: 1px solid rgba(120, 120, 120, 0.5);
                    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
                ">
                    <h3 style="color: #ffffff; margin-top: 0; font-size: 1.8rem; font-weight: 600;">Enter Your Name to Begin</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Center the input and button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Custom CSS for bigger input and button
            st.markdown(
                """
                <style>
                .big-input-container {
                    background: linear-gradient(135deg, rgba(60, 60, 60, 0.4) 0%, rgba(40, 40, 40, 0.4) 100%);
                    border-radius: 40px;
                    padding: 60px;
                    margin: 50px auto;
                    max-width: 1000px;
                    border: 5px solid rgba(180, 180, 180, 1.0);
                    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.9),
                                inset 0 0 60px rgba(255, 255, 255, 0.2);
                }
                .big-input > div > div > input {
                    background: rgba(20, 20, 20, 0.95) !important;
                    color: #ffffff !important;
                    border: 6px solid rgba(200, 200, 200, 1.0) !important;
                    border-radius: 35px !important;
                    padding: 70px 80px !important;
                    font-size: 3.5rem !important;
                    text-align: center !important;
                    box-shadow: 0 30px 70px rgba(0, 0, 0, 1.0),
                                inset 0 0 60px rgba(255, 255, 255, 0.5) !important;
                    transition: all 0.5s ease !important;
                    letter-spacing: 3px !important;
                    font-weight: 300 !important;
                }
                .big-input > div > div > input:focus {
                    border-color: #ffffff !important;
                    box-shadow: 0 0 90px rgba(255, 255, 255, 1.0),
                                inset 0 0 70px rgba(255, 255, 255, 0.7) !important;
                    transform: scale(1.07) !important;
                }
                .big-input > div > div > input::placeholder {
                    color: rgba(220, 220, 220, 0.9) !important;
                    font-size: 3.5rem !important;
                    font-weight: 300 !important;
                }
                .big-button > button {
                    background: linear-gradient(135deg, #404040 0%, #202020 100%) !important;
                    color: white !important;
                    border-radius: 35px !important;
                    padding: 65px 90px !important;
                    font-size: 3.2rem !important;
                    font-weight: 800 !important;
                    border: 6px solid rgba(180, 180, 180, 1.0) !important;
                    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.9),
                                inset 0 0 50px rgba(255, 255, 255, 0.35) !important;
                    transition: all 0.5s ease !important;
                    width: 100% !important;
                    letter-spacing: 4px !important;
                    margin-top: 50px !important;
                    text-transform: uppercase !important;
                }
                .big-button > button:hover {
                    transform: translateY(-20px) !important;
                    box-shadow: 0 35px 90px rgba(0, 0, 0, 1.0),
                                inset 0 0 60px rgba(255, 255, 255, 0.6) !important;
                    background: linear-gradient(135deg, #505050 0%, #303030 100%) !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Bigger text input with container
            player_name_input = st.text_input(
                "Player Name",  # Non-empty label to avoid deprecation warning
                placeholder="Enter your name here...",
                key="sig_name_input",
                label_visibility="collapsed",  # Hide the label visually
                help="Please enter your name to start playing the signature guessing game",
            )

            # Apply the big button styling
            st.markdown('<div class="big-button">', unsafe_allow_html=True)
            if st.button("✨ Start Playing! ✨", use_container_width=True):
                if player_name_input and player_name_input.strip():
                    logger.info(f"✅ Player entered name: {player_name_input}")
                    st.session_state.sig_player_name = player_name_input.strip()
                    st.session_state.sig_show_welcome = False
                    st.rerun()
                else:
                    logger.warning(
                        "⚠️ Player attempted to proceed without entering name"
                    )
                    st.warning("⚠️ Please enter your name first!")
            st.markdown("</div>", unsafe_allow_html=True)
        return

    # Step 2: Game Play
    if st.session_state.sig_player_name:
        logger.info(f"🎮 Game active - Player: {st.session_state.sig_player_name}")

        # Check if player has reached the play limit
        play_count = get_player_play_count(st.session_state.sig_player_name)
        plays_remaining = MAX_PLAYS_PER_USER - play_count

        if play_count >= MAX_PLAYS_PER_USER:
            logger.info(
                f"🚫 Player '{st.session_state.sig_player_name}' has reached the play limit"
            )
            st.markdown(
                f"""
                <div style="text-align: center; padding: 50px;">
                    <h1 style="font-size: 3rem; margin-bottom: 20px; color: #ffffff;">🎯 Game Limit Reached 🎯</h1>
                    <p style="font-size: 1.5rem; color: #f0f0f0; margin-top: 20px;">
                        Thanks for playing, {st.session_state.sig_player_name}! 🎉<br>
                        You've completed the maximum of {MAX_PLAYS_PER_USER} signature challenges.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, rgba(60, 60, 60, 0.4) 0%, rgba(40, 40, 40, 0.4) 100%);
                    padding: 30px;
                    border-radius: 15px;
                    margin: 20px auto;
                    max-width: 600px;
                    text-align: center;
                    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
                    border: 1px solid rgba(120, 120, 120, 0.5);
                ">
                    <h3 style="color: #ffffff; margin: 0; font-weight: 600;">🏆 Challenge Complete!</h3>
                    <p style="color: #f0f0f0; font-size: 1.2rem; margin-top: 15px; margin-bottom: 0;">
                        You've used all your attempts for this signature challenge.<br>
                        Check the leaderboard to see how you ranked! 📊
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⬅️ Exit Game", use_container_width=True):
                    st.session_state.sig_player_name = ""
                    st.session_state.sig_show_welcome = True
                    st.session_state.sig_current_item = None
                    st.session_state.sig_game_started = False
                    st.session_state.sig_feedback = None
                    st.rerun()
            return

        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px;">
                <div style="
                    background: linear-gradient(135deg, rgba(60, 60, 60, 0.3) 0%, rgba(40, 40, 40, 0.3) 100%);
                    border: 2px solid rgba(120, 120, 120, 0.5);
                    border-radius: 25px;
                    padding: 30px;
                    box-shadow: 0 15px 50px rgba(0,0,0,0.6),
                                inset 0 0 40px rgba(120, 120, 120, 0.2);
                    backdrop-filter: blur(15px);
                ">
                    <h1 style="
                        font-size: 3rem;
                        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
                        -webkit-background-clip: text;
                        background-clip: text;
                        font-weight: 900;
                        margin-bottom: 10px;
                        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
                    ">✍️ Welcome, {st.session_state.sig_player_name}! ✍️</h1>
                    <p style="font-size: 1.4rem; color: #f0f0f0; margin-top: 15px; font-weight: 600;">
                        🎯 Signature Similarity Challenge
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Display remaining plays
        if plays_remaining > 0:
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 25px;">
                    <span style="
                        background: linear-gradient(135deg, rgba(60, 60, 60, 0.4) 0%, rgba(40, 40, 40, 0.4) 100%);
                        color: #ffffff;
                        padding: 15px 35px;
                        border-radius: 25px;
                        font-weight: 800;
                        font-size: 1.4rem;
                        border: 2px solid rgba(120, 120, 120, 0.6);
                        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5),
                                    inset 0 0 25px rgba(255, 255, 255, 0.1);
                        display: inline-block;
                        letter-spacing: 1px;
                        animation: pulse 2s ease-in-out infinite;
                    ">
                        🎮 ROUNDS LEFT: {plays_remaining}/{MAX_PLAYS_PER_USER}
                    </span>
                </div>
                <style>
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4); }}
                    50% {{ transform: scale(1.05); box-shadow: 0 8px 35px rgba(102, 126, 234, 0.7); }}
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

        # Instructions
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(60, 60, 60, 0.3) 0%, rgba(40, 40, 40, 0.3) 100%);
                padding: 30px;
                border-radius: 20px;
                margin-bottom: 30px;
                border: 2px solid rgba(120, 120, 120, 0.5);
                box-shadow: 0 15px 50px rgba(0,0,0,0.6),
                            inset 0 0 40px rgba(120, 120, 120, 0.2);
                backdrop-filter: blur(15px);
            ">
                <h3 style="
                    color: #ffffff;
                    margin: 0 0 20px 0;
                    font-size: 1.8rem;
                    font-weight: 700;
                    text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
                ">📝 Mission Briefing:</h3>
                <div style="color: #f0f0f0; font-size: 1.2rem; line-height: 2; text-align: left; margin-left: 20px;">
                    <div style="margin: 10px 0;">1️⃣ <strong style="color: #ffffff;">Analyze</strong> both signatures carefully</div>
                    <div style="margin: 10px 0;">2️⃣ <strong style="color: #ffffff;">Estimate</strong> similarity (0-100%)</div>
                    <div style="margin: 10px 0;">3️⃣ <strong style="color: #ffffff;">Match</strong> the AI's precision!</div>
                    <div style="margin: 15px 0 0 0; padding: 15px; background: rgba(60, 60, 60, 0.3); border-radius: 10px; border-left: 4px solid #ffffff;">
                        <strong style="color: #ffffff; font-size: 1.3rem;">🎯 Win Condition:</strong> <span style="color: #ffffff;">Within ±2 points!</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Start new round button
        if not st.session_state.sig_game_started or st.session_state.sig_feedback:
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("🎲 ✨ Start New Round ✨ 🎲", use_container_width=True):
                    logger.info("🎲 Starting new signature round")
                    # Get only signature items
                    item = get_random_game_item(
                        "✍️ Signature Sleuth", st.session_state.sig_player_name
                    )
                    if item:
                        logger.info(f"✅ Retrieved game item ID: {item[0]}")
                        st.session_state.sig_current_item = item
                        st.session_state.sig_game_started = True
                        st.session_state.sig_feedback = None
                        st.rerun()
                    else:
                        logger.info("🏆 Player completed all signature pairs")
                        st.warning(
                            "🎉 Congratulations! You've completed all available signature pairs! 🎉"
                        )
                        st.info(
                            "🔄 Check back later for new signatures or ask admin to add more!"
                        )
                        st.session_state.sig_game_started = False

                # Reset/Change Name button
                if st.button("⬅️ Change Name", use_container_width=True):
                    logger.info("🔄 Player wants to change name")
                    st.session_state.sig_player_name = ""
                    st.session_state.sig_show_welcome = True
                    st.session_state.sig_current_item = None
                    st.session_state.sig_game_started = False
                    st.session_state.sig_feedback = None
                    st.rerun()

        # Display current game
        if (
            st.session_state.sig_current_item
            and st.session_state.sig_game_started
            and not st.session_state.sig_feedback
        ):
            item = st.session_state.sig_current_item
            item_id, category, item1_path, item2_path, actual_score, description = item
            logger.info(
                f"📊 Displaying signature pair - ID: {item_id}, AI Score: {actual_score}"
            )

            if description:
                st.info(f"📝 {description}")

            st.markdown("---")

            # Display signatures side by side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                    <div style="text-align: center; margin-bottom: 15px;">
                        <h3 style="
                            color: #ffffff;
                            font-size: 1.6rem;
                            font-weight: 800;
                            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
                            letter-spacing: 2px;
                        ">📸 SIGNATURE A</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(item1_path, use_container_width=True)

            with col2:
                st.markdown(
                    """
                    <div style="text-align: center; margin-bottom: 15px;">
                        <h3 style="
                            color: #ffffff;
                            font-size: 1.6rem;
                            font-weight: 800;
                            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
                            letter-spacing: 2px;
                        ">📸 SIGNATURE B</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(item2_path, use_container_width=True)

            st.markdown("---")

            # Guess input
            col_input1, col_input2, col_input3 = st.columns([1, 2, 1])
            with col_input2:
                st.markdown(
                    """
                    <div style="
                        text-align: center;
                        margin-bottom: 20px;
                        background: linear-gradient(135deg, rgba(60, 60, 60, 0.3) 0%, rgba(40, 40, 40, 0.3) 100%);
                        padding: 20px;
                        border-radius: 15px;
                        border: 2px solid rgba(120, 120, 120, 0.5);
                        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    ">
                        <p style="
                            font-size: 1.5rem;
                            color: #ffffff;
                            font-weight: 800;
                            margin: 0;
                            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
                            letter-spacing: 1px;
                        ">
                            🤔 WHAT'S YOUR VERDICT?
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                guessed_score = st.number_input(
                    "🎯 Your Similarity Guess (0-100%):",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=1,
                    key="sig_score_input",
                    help="0 = Completely different, 100 = Identical",
                )

                if st.button("✨ Submit Guess ✨", use_container_width=True):
                    logger.info(
                        f"🎯 Player guessed: {guessed_score}, AI Score: {actual_score}"
                    )

                    # Show calculating message
                    with st.spinner("🔍 Analyzing signatures... 🧠"):
                        import time

                        time.sleep(3)  # Simulate processing

                    # Calculate if guess is correct
                    tolerance = 2
                    is_correct = abs(guessed_score - actual_score) <= tolerance
                    logger.info(
                        f"✅ Result: {'CORRECT' if is_correct else 'INCORRECT'} (tolerance: {tolerance})"
                    )

                    # Save result with game_item_id
                    save_game_result(
                        st.session_state.sig_player_name,
                        "✍️ Signature Sleuth",
                        guessed_score,
                        actual_score,
                        is_correct,
                        item_id,
                    )
                    logger.info(
                        f"💾 Saved game result to database for player: {st.session_state.sig_player_name}"
                    )

                    # Set feedback
                    difference = abs(guessed_score - actual_score)
                    if is_correct:
                        st.session_state.sig_feedback = "correct"
                    else:
                        st.session_state.sig_feedback = "incorrect"

                    st.session_state.sig_feedback_data = {
                        "guessed": guessed_score,
                        "actual": actual_score,
                        "difference": difference,
                    }
                    logger.info(
                        f"📢 Showing feedback: {st.session_state.sig_feedback}, difference: {difference}"
                    )
                    st.rerun()

        # Display feedback
        if st.session_state.sig_feedback:
            logger.info(f"📣 Displaying feedback UI: {st.session_state.sig_feedback}")
            if st.session_state.sig_feedback == "correct":
                difference = st.session_state.sig_feedback_data["difference"]
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
                    <div id="feedback-popup" style="
                        background: linear-gradient(135deg, rgba(60, 60, 60, 0.4) 0%, rgba(40, 40, 40, 0.4) 100%);
                        border: 3px solid rgba(120, 120, 120, 0.7);
                        padding: 40px;
                        border-radius: 30px;
                        text-align: center;
                        margin: 30px auto;
                        max-width: 550px;
                        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.6),
                                    inset 0 0 50px rgba(120, 120, 120, 0.2);
                        backdrop-filter: blur(20px);
                        animation: successPop 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                    ">
                        <div style="
                            font-size: 5rem;
                            margin-bottom: 20px;
                            filter: drop-shadow(0 0 40px rgba(102, 234, 138, 1));
                            animation: spin 0.8s ease-in-out;
                        ">
                            {performance_emoji}
                        </div>
                        <h2 style="
                            background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                            font-size: 2.8rem;
                            margin: 15px 0;
                            font-weight: 900;
                            letter-spacing: 3px;
                            text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
                        ">
                            {performance}
                        </h2>
                        <div style="
                            background: rgba(120, 120, 120, 0.2);
                            padding: 30px;
                            border-radius: 20px;
                            margin: 25px 0;
                            border: 2px solid rgba(120, 120, 120, 0.5);
                        ">
                            <div style="text-align: center;">
                                <div style="font-size: 4rem; color: #ffffff; margin-bottom: 15px; filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.8));">🎯</div>
                                <div style="font-size: 2.3rem; color: #ffffff; font-weight: 800; margin-bottom: 10px; text-shadow: 0 0 20px rgba(255, 255, 255, 0.6);">WINNER!</div>
                                <div style="font-size: 1.3rem; color: #f0f0f0; font-weight: 600;">Your guess hit the target zone!</div>
                            </div>
                        </div>
                        <p style="color: #ffffff; font-size: 1.4rem; margin: 20px 0 0 0; font-weight: 700; text-shadow: 0 0 15px rgba(255, 255, 255, 0.5);">
                            ⚡ CHALLENGE CONQUERED ⚡
                        </p>
                    </div>
                    <style>
                    @keyframes successPop {{
                        0% {{ transform: scale(0.3) rotate(-10deg); opacity: 0; }}
                        60% {{ transform: scale(1.1) rotate(5deg); }}
                        100% {{ transform: scale(1) rotate(0deg); opacity: 1; }}
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg) scale(0.5); }}
                        100% {{ transform: rotate(360deg) scale(1); }}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                # Show balloons for success
                st.balloons()

                # Play success sound
                st.markdown(
                    """
                    <audio autoplay>
                        <source src="https://www.myinstants.com/media/sounds/congratulations-you-won.mp3" type="audio/mpeg">
                    </audio>
                """,
                    unsafe_allow_html=True,
                )
            else:
                # Incorrect answer
                difference = st.session_state.sig_feedback_data["difference"]

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
                    <div id="feedback-popup" style="
                        background: linear-gradient(135deg, rgba(60, 60, 60, 0.4) 0%, rgba(40, 40, 40, 0.4) 100%);
                        border: 3px solid rgba(120, 120, 120, 0.7);
                        padding: 40px;
                        border-radius: 30px;
                        text-align: center;
                        margin: 30px auto;
                        max-width: 550px;
                        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.6),
                                    inset 0 0 50px rgba(120, 120, 120, 0.2);
                        backdrop-filter: blur(20px);
                        animation: failShake 0.6s cubic-bezier(0.36, 0.07, 0.19, 0.97);
                    ">
                        <div style="
                            font-size: 5rem;
                            margin-bottom: 20px;
                            filter: drop-shadow(0 0 40px rgba(245, 87, 108, 1));
                        ">
                            {closeness_emoji}
                        </div>
                        <h2 style="
                            background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            background-clip: text;
                            font-size: 2.8rem;
                            margin: 15px 0;
                            font-weight: 900;
                            letter-spacing: 3px;
                            text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
                        ">
                            {closeness}
                        </h2>
                        <div style="
                            background: rgba(120, 120, 120, 0.2);
                            padding: 30px;
                            border-radius: 20px;
                            margin: 25px 0;
                            border: 2px solid rgba(120, 120, 120, 0.5);
                        ">
                            <div style="text-align: center;">
                                <div style="font-size: 4rem; color: #ffffff; margin-bottom: 15px; filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.8));">❌</div>
                                <div style="font-size: 2.3rem; color: #ffffff; font-weight: 800; margin-bottom: 10px; text-shadow: 0 0 20px rgba(255, 255, 255, 0.6);">MISS!</div>
                                <div style="font-size: 1.3rem; color: #f0f0f0; font-weight: 600;">Outside the target zone</div>
                            </div>
                        </div>
                        <div style="
                            background: rgba(220, 220, 220, 0.2);
                            padding: 20px;
                            border-radius: 15px;
                            margin: 20px 0;
                            border-left: 4px solid #ffffff;
                        ">
                            <div style="font-size: 1.1rem; color: #ffffff; margin-bottom: 8px; font-weight: 700;">
                                💡 PRO TIP
                            </div>
                            <div style="font-size: 1.05rem; color: #f0f0f0; font-weight: 500;">
                                Study the curves, pressure, spacing & flow patterns!
                            </div>
                        </div>
                        <p style="color: #ffffff; font-size: 1.4rem; margin: 20px 0 0 0; font-weight: 700; text-shadow: 0 0 15px rgba(255, 255, 255, 0.5);">
                            🔥 NEXT ROUND AWAITS! 🔥
                        </p>
                    </div>
                    <style>
                    @keyframes failShake {{
                        0%, 100% {{ transform: translateX(0) rotate(0deg); }}
                        10%, 30%, 50%, 70%, 90% {{ transform: translateX(-10px) rotate(-2deg); }}
                        20%, 40%, 60%, 80% {{ transform: translateX(10px) rotate(2deg); }}
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


# Run the game
signature_guessing_game()
