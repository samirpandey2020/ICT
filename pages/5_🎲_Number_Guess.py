import streamlit as st
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Number Guess 🎲", page_icon="🎲", layout="wide")


def number_guess_game():
    """Simple number guessing game"""

    st.markdown(
        """
        <div style="text-align: center; padding: 30px;">
            <h1 style="font-size: 3rem;">🎲 Number Guessing Game</h1>
            <p style="font-size: 1.2rem; color: #666;">Can you guess the secret number?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize game state
    if "secret_number" not in st.session_state:
        st.session_state.secret_number = random.randint(1, 100)
        st.session_state.attempts = 0
        st.session_state.won = False
        st.session_state.hint = "Make your first guess!"

    # Game info
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"### 🎯 Attempts: {st.session_state.attempts}")
        st.info(f"💡 Hint: {st.session_state.hint}")

        if not st.session_state.won:
            guess = st.number_input(
                "🔢 Enter your guess (1-100):",
                min_value=1,
                max_value=100,
                value=50,
                step=1,
                key="guess_input",
            )

            if st.button("🎲 Submit Guess", use_container_width=True):
                st.session_state.attempts += 1

                if guess == st.session_state.secret_number:
                    st.session_state.won = True
                    st.balloons()
                    st.markdown(
                        """
                        <audio autoplay>
                            <source src="https://www.myinstants.com/media/sounds/congratulations-you-won.mp3" type="audio/mpeg">
                        </audio>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.rerun()
                elif guess < st.session_state.secret_number:
                    st.session_state.hint = f"📈 Too low! Try higher than {guess}"
                    st.rerun()
                else:
                    st.session_state.hint = f"📉 Too high! Try lower than {guess}"
                    st.rerun()

        if st.session_state.won:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; text-align: center; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);">
                    <h2 style="color: white; font-size: 2.5rem; margin: 10px 0;">🎉 YOU WON! 🎉</h2>
                    <p style="color: white; font-size: 1.5rem;">The secret number was {st.session_state.secret_number}</p>
                    <p style="color: #FFD700; font-size: 1.2rem;">You guessed it in {st.session_state.attempts} attempts!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("🔄 Play Again", use_container_width=True):
                st.session_state.secret_number = random.randint(1, 100)
                st.session_state.attempts = 0
                st.session_state.won = False
                st.session_state.hint = "Make your first guess!"
                st.rerun()


number_guess_game()
