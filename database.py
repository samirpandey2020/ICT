import sqlite3
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize database and assets folder
DB_PATH = "game_data.db"
ASSETS_FOLDER = Path("assets")
ASSETS_FOLDER.mkdir(exist_ok=True)
logger.info(f"📁 Assets folder ready: {ASSETS_FOLDER}")

# Game categories list
GAME_CATEGORIES = [
    "✍️ Signature Sleuth",
    "🎨 Art Twins",
    "🔊 Sound-Alike",
    "📸 Photo Match",
    "✏️ Handwriting Detective",
    "🎭 Logo Lookalike",
    "🎵 Beat Buddies",
    "🖼️ Picture Perfect",
]


def init_database():
    """Initialize SQLite database with required tables (cached)"""
    import streamlit as st

    # Use Streamlit's session state to ensure DB is initialized only once per session
    if "db_initialized" not in st.session_state:
        logger.info("💾 Initializing database...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create game_items table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS game_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                item1_path TEXT NOT NULL,
                item2_path TEXT NOT NULL,
                actual_score INTEGER NOT NULL,
                description TEXT
            )
        """
        )

        # Create leaderboard table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                category TEXT NOT NULL,
                guessed_score INTEGER NOT NULL,
                actual_score INTEGER NOT NULL,
                is_correct BOOLEAN NOT NULL,
                game_item_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Add game_item_id column if it doesn't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE leaderboard ADD COLUMN game_item_id INTEGER")
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists
            pass

        conn.commit()
        conn.close()

        # Mark as initialized
        st.session_state.db_initialized = True
        logger.info("✅ Database initialized successfully")
    else:
        logger.info("✅ Database already initialized (cached)")


def get_random_game_item(category=None, player_name=None):
    """Fetch a random game item from database that the player hasn't seen yet"""
    logger.info(f"🎲 Fetching game item - Category: {category}, Player: {player_name}")
    conn = sqlite3.connect(DB_PATH)
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


def get_all_categories():
    """Get all unique categories from database"""
    logger.info("📊 Fetching all categories from database")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM game_items")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    logger.info(f"✅ Found {len(categories)} categories: {categories}")
    return categories


def save_game_result(
    player_name, category, guessed_score, actual_score, is_correct, game_item_id
):
    """Save game result to leaderboard"""
    logger.info(
        f"💾 Saving result - Player: {player_name}, Correct: {is_correct}, Item ID: {game_item_id}"
    )
    conn = sqlite3.connect(DB_PATH)
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


def get_leaderboard(limit=10):
    """Get top players from leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            player_name,
            COUNT(*) as total_games,
            SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_guesses,
            ROUND(AVG(ABS(guessed_score - actual_score)), 2) as avg_difference,
            MAX(timestamp) as last_played
        FROM leaderboard
        GROUP BY player_name
        ORDER BY correct_guesses DESC, avg_difference ASC
        LIMIT ?
    """,
        (limit,),
    )

    leaderboard_data = cursor.fetchall()
    conn.close()
    return leaderboard_data


def add_game_item(category, item1_path, item2_path, actual_score, description):
    """Add a new game item to database"""
    logger.info(
        f"➕ Adding new game item - Category: {category}, Score: {actual_score}"
    )
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO game_items (category, item1_path, item2_path, actual_score, description)
        VALUES (?, ?, ?, ?, ?)
    """,
        (category, item1_path, item2_path, actual_score, description),
    )
    conn.commit()
    conn.close()
    logger.info("✅ Game item added successfully")


def get_all_game_items():
    """Get all game items"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM game_items")
    items = cursor.fetchall()
    conn.close()
    return items


def delete_game_item(item_id):
    """Delete a game item"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM game_items WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()


def update_game_item(item_id, category, actual_score, description):
    """Update an existing game item"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE game_items 
        SET category = ?, actual_score = ?, description = ?
        WHERE id = ?
    """,
        (category, actual_score, description, item_id),
    )
    conn.commit()
    conn.close()


def reset_leaderboard():
    """Reset the leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM leaderboard")
    conn.commit()
    conn.close()
