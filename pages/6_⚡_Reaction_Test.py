import streamlit as st
import random
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Reaction Test ⚡", page_icon="⚡", layout="wide")


def reaction_test():
    """Simple reaction time game"""

    st.markdown(
        """
        <div style="text-align: center; padding: 30px;">
            <h1 style="font-size: 3rem;">⚡ Reaction Time Test</h1>
            <p style="font-size: 1.2rem; color: #666;">How fast are your reflexes?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize game state
    if "reaction_game_started" not in st.session_state:
        st.session_state.reaction_game_started = False
        st.session_state.reaction_times = []
        st.session_state.current_color = "red"

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if not st.session_state.reaction_game_started:
            st.markdown("### 🎯 How to Play:")
            st.info(
                "1. Click 'Start Test'\n2. Wait for the box to turn GREEN\n3. Click as fast as you can!"
            )

            if st.button("🚀 Start Test", use_container_width=True):
                st.session_state.reaction_game_started = True
                st.session_state.wait_time = random.uniform(2, 5)
                st.session_state.start_time = time.time()
                st.session_state.current_color = "red"
                st.rerun()
        else:
            # Check if it's time to turn green
            elapsed = time.time() - st.session_state.start_time

            if (
                elapsed >= st.session_state.wait_time
                and st.session_state.current_color == "red"
            ):
                st.session_state.current_color = "green"
                st.session_state.green_time = time.time()

            # Display colored box
            color = st.session_state.current_color
            message = "🔴 WAIT..." if color == "red" else "🟢 CLICK NOW!"

            st.markdown(
                f"""
                <div style="background: {'#ff6b6b' if color == 'red' else '#51cf66'}; padding: 80px; border-radius: 20px; text-align: center; margin: 30px 0; box-shadow: 0 10px 40px rgba(0,0,0,0.2);">
                    <h1 style="color: white; font-size: 3rem; margin: 0;">{message}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("⚡ CLICK!", use_container_width=True, key="click_button"):
                if st.session_state.current_color == "green":
                    reaction_time = (time.time() - st.session_state.green_time) * 1000
                    st.session_state.reaction_times.append(reaction_time)
                    st.session_state.reaction_game_started = False
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Too early! Wait for GREEN!")
                    st.session_state.reaction_game_started = False
                    st.rerun()

        # Show results
        if st.session_state.reaction_times:
            st.markdown("### 📊 Your Results:")
            avg_time = sum(st.session_state.reaction_times) / len(
                st.session_state.reaction_times
            )
            best_time = min(st.session_state.reaction_times)

            st.success(f"⚡ Last: {st.session_state.reaction_times[-1]:.0f}ms")
            st.info(f"🏆 Best: {best_time:.0f}ms")
            st.info(f"📊 Average: {avg_time:.0f}ms")

            if st.button("🔄 Play Again", use_container_width=True):
                st.session_state.reaction_game_started = False
                st.rerun()


reaction_test()

# Auto-refresh for waiting period
if (
    st.session_state.get("reaction_game_started")
    and st.session_state.get("current_color") == "red"
):
    time.sleep(0.1)
    st.rerun()
