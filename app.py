import streamlit as st


def main():
    """Main application entry point - Player Mode"""
    st.set_page_config(page_title="Player Mode 🎮", page_icon="✨", layout="wide")

    # Sophisticated Neutral Theme
    st.markdown(
        """
        <style>
    
        /* Elegant dark neutral background */
        .stApp {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 50%, #1e1e1e 100%);
            color: #e0e0e0;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Sidebar content */
        [data-testid="stSidebar"] * {
            color: #f0f0f0 !important;
        }
        
        /* Sidebar links */
        [data-testid="stSidebar"] a {
            color: #f0f0f0 !important;
            text-decoration: none;
            padding: 10px 15px;
            border-radius: 10px;
            display: block;
            transition: all 0.3s ease;
        }
        
        [data-testid="stSidebar"] a:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }
        
        /* Subtle neutral buttons with gentle glow */
        .stButton > button {
            background: linear-gradient(135deg, #3a3a3a 0%, #2a2a2a 100%);
            color: #f0f0f0;
            border-radius: 25px;
            padding: 16px 35px;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3),
                        inset 0 0 15px rgba(255, 255, 255, 0.05);
            transition: all 0.3s ease;
            font-size: 1.1rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4),
                        inset 0 0 20px rgba(255, 255, 255, 0.1);
            background: linear-gradient(135deg, #4a4a4a 0%, #3a3a3a 100%);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background: rgba(50, 50, 50, 0.3);
            color: #f0f0f0 !important;
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 14px;
            font-size: 1.05rem;
            box-shadow: 0 3px 15px rgba(0, 0, 0, 0.2);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #606060;
            box-shadow: 0 0 20px rgba(90, 90, 90, 0.4);
            color: #f0f0f0 !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: rgba(220, 220, 220, 0.6) !important;
        }
        
        /* Label styling */
        label {
            color: #d0d0d0 !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Lazy load modules AFTER page config and styling
    from database import init_database
    from player import player_page

    # Initialize database
    init_database()

    # Display player page
    player_page()


if __name__ == "__main__":
    main()
