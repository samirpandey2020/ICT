import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Admin Mode 🔧", page_icon="🔧", layout="wide")

# Lazy load modules only when page is accessed
from admin import admin_page

# Display admin page
admin_page()
