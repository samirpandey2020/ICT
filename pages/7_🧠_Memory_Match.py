import streamlit as st
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Memory Match 🧠", page_icon="🧠", layout="wide")


def memory_match():
    """Simple memory matching game"""

    st.markdown(
        """
        <div style="text-align: center; padding: 30px;">
            <h1 style="font-size: 3rem;">🧠 Memory Match Game</h1>
            <p style="font-size: 1.2rem; color: #666;">Find all the matching pairs!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize game
    if "memory_cards" not in st.session_state:
        emojis = ["🎮", "🎯", "🎲", "🎨", "🎭", "🎪", "🎸", "🎺"]
        cards = emojis + emojis  # Create pairs
        random.shuffle(cards)
        st.session_state.memory_cards = cards
        st.session_state.revealed = [False] * 16
        st.session_state.selected = []
        st.session_state.matched = []
        st.session_state.moves = 0

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            f"### 🎯 Moves: {st.session_state.moves} | Pairs Found: {len(st.session_state.matched)//2}/8"
        )

        # Create 4x4 grid
        for row in range(4):
            cols = st.columns(4)
            for col in range(4):
                idx = row * 4 + col

                with cols[col]:
                    if (
                        idx in st.session_state.matched
                        or idx in st.session_state.selected
                    ):
                        # Show card
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center; font-size: 2.5rem; cursor: pointer;">
                                {st.session_state.memory_cards[idx]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Hidden card
                        if st.button("❓", key=f"card_{idx}", use_container_width=True):
                            if (
                                len(st.session_state.selected) < 2
                                and idx not in st.session_state.selected
                            ):
                                st.session_state.selected.append(idx)

                                if len(st.session_state.selected) == 2:
                                    st.session_state.moves += 1
                                    idx1, idx2 = st.session_state.selected

                                    if (
                                        st.session_state.memory_cards[idx1]
                                        == st.session_state.memory_cards[idx2]
                                    ):
                                        # Match found!
                                        st.session_state.matched.extend([idx1, idx2])
                                        st.session_state.selected = []

                                st.rerun()

        # Clear selection after showing briefly
        if len(st.session_state.selected) == 2:
            import time

            time.sleep(1)
            st.session_state.selected = []
            st.rerun()

        # Check if won
        if len(st.session_state.matched) == 16:
            st.balloons()
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; text-align: center; margin: 20px 0; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);">
                    <h2 style="color: white; font-size: 2.5rem;">🎉 YOU WON! 🎉</h2>
                    <p style="color: #FFD700; font-size: 1.5rem;">Completed in {st.session_state.moves} moves!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <audio autoplay>
                    <source src="https://www.myinstants.com/media/sounds/congratulations-you-won.mp3" type="audio/mpeg">
                </audio>
                """,
                unsafe_allow_html=True,
            )

        # Reset button
        if st.button("🔄 New Game", use_container_width=True):
            del st.session_state.memory_cards
            st.rerun()


memory_match()
