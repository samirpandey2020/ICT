import streamlit as st
from datetime import datetime
from database import get_leaderboard


def leaderboard_page():
    """Dedicated leaderboard page for external display"""

    st.title("🏆 Top Players Leaderboard")

    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)

    # Champion Podium Display
    leaderboard_data = get_leaderboard(100)  # Get all for podium

    if leaderboard_data and len(leaderboard_data) > 0:
        # Display champion (first place)
        champion = leaderboard_data[0]
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
                padding: 40px;
                border-radius: 25px;
                text-align: center;
                box-shadow: 0 15px 35px rgba(255,215,0,0.4);
                margin: 30px auto;
                max-width: 600px;
                border: 4px solid #FFD700;
                animation: pulse 2s ease-in-out infinite;
            ">
                <h1 style="font-size: 3.5rem; margin: 0; color: white;">
                    🌟 CHAMPION 🌟
                </h1>
                <h2 style="font-size: 2.5rem; margin: 20px 0; color: #8B4513; font-weight: bold;">
                    👑 {champion[0]} 👑
                </h2>
                <div style="display: flex; justify-content: space-around; margin-top: 30px;">
                    <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 2rem; color: white; font-weight: bold;">{champion[2]}</div>
                        <div style="font-size: 1rem; color: rgba(255,255,255,0.9);">Correct Guesses</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 2rem; color: white; font-weight: bold;">{champion[1]}</div>
                        <div style="font-size: 1rem; color: rgba(255,255,255,0.9);">Total Games</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 15px; border-radius: 10px;">
                        <div style="font-size: 2rem; color: white; font-weight: bold;">{champion[3]}</div>
                        <div style="font-size: 1rem; color: rgba(255,255,255,0.9);">Avg Difference</div>
                    </div>
                </div>
            </div>
            <style>
            @keyframes pulse {{
                0%, 100% {{ transform: scale(1); }}
                50% {{ transform: scale(1.02); }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Display rest of the leaderboard
        if len(leaderboard_data) > 1:
            st.markdown("### 🎖️ Top Players")
            for idx, player in enumerate(leaderboard_data[1:11], start=2):
                # Medal assignment
                if idx == 2:
                    medal = "🥈"
                elif idx == 3:
                    medal = "🥉"
                else:
                    medal = "⭐"

                # Color scheme based on position
                if idx == 2:
                    gradient = "linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%)"
                    border_color = "#C0C0C0"
                elif idx == 3:
                    gradient = "linear-gradient(135deg, #CD7F32 0%, #B87333 100%)"
                    border_color = "#CD7F32"
                else:
                    gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                    border_color = "#667eea"

                st.markdown(
                    f"""
                    <div style="
                        background: {gradient};
                        padding: 20px;
                        border-radius: 15px;
                        margin: 10px 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border-left: 5px solid {border_color};
                    ">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center; gap: 15px;">
                                <span style="font-size: 2rem;">{medal}</span>
                                <div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: white;">
                                        #{idx} {player[0]}
                                    </div>
                                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">
                                        {player[1]} games played
                                    </div>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.8rem; font-weight: bold; color: white;">
                                    🎯 {player[2]}
                                </div>
                                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8);">
                                    correct guesses
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 20px;
                margin: 30px 0;
            ">
                <h2 style="font-size: 2.5rem; color: white; margin: 0;">
                    🎮 Be the First Champion! 🎮
                </h2>
                <p style="font-size: 1.3rem; color: rgba(255,255,255,0.9); margin-top: 20px;">
                    ✨ No one has played yet. Start playing and claim your crown! ✨
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Auto-refresh implementation
    if auto_refresh:
        import time

        time.sleep(30)
        st.rerun()
