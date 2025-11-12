import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Leaderboard 🏆", page_icon="🏆", layout="wide")

# Lazy load modules
from leaderboard import leaderboard_page

# Display leaderboard page
leaderboard_page()
