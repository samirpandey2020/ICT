import streamlit as st
from datetime import datetime
from pathlib import Path
from database import (
    GAME_CATEGORIES,
    ASSETS_FOLDER,
    add_game_item,
    get_all_game_items,
    update_game_item,
    delete_game_item,
    reset_leaderboard,
)


def admin_page():
    """Admin mode interface"""

    st.title(body="🔧 Admin Mode")

    tab1, tab2, tab3 = st.tabs(
        ["➕ Add New Pair", "📝 Manage Pairs", "🗑️ Reset Leaderboard"]
    )

    # Tab 1: Add New Item
    with tab1:
        st.subheader("✨ Add New Game Pair")

        # Game category dropdown
        selected_category = st.selectbox(
            "🎮 Select Game Type",
            GAME_CATEGORIES,
            index=0,
            help="Choose the type of comparison game",
        )

        description = st.text_area(
            "Description (optional)",
            placeholder="e.g., John Doe's signature from 2020 vs 2024",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✨ Item 1**")
            item1_type = st.radio(
                "Item 1 Type", ["Upload File", "Text Input"], key="item1_type"
            )

            if item1_type == "Upload File":
                item1_file = st.file_uploader(
                    "Upload Item 1",
                    type=["jpg", "jpeg", "png", "gif"],
                    key="item1_file",
                )
                item1_path = None
                if item1_file:
                    # Show image preview immediately
                    st.image(
                        item1_file, caption="Preview - Item 1", use_container_width=True
                    )
                    item1_path = (
                        ASSETS_FOLDER
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item1_file.name}"
                    )
                    with open(item1_path, "wb") as f:
                        f.write(item1_file.read())
            else:
                item1_text = st.text_area("Enter Text for Item 1", key="item1_text")
                item1_path = None
                if item1_text:
                    # Show text preview
                    st.info(
                        f"Preview: {item1_text[:100]}..."
                        if len(item1_text) > 100
                        else f"Preview: {item1_text}"
                    )
                    item1_path = (
                        ASSETS_FOLDER
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_item1.txt"
                    )
                    with open(item1_path, "w", encoding="utf-8") as f:
                        f.write(item1_text)

        with col2:
            st.markdown("**✨ Item 2**")
            item2_type = st.radio(
                "Item 2 Type", ["Upload File", "Text Input"], key="item2_type"
            )

            if item2_type == "Upload File":
                item2_file = st.file_uploader(
                    "Upload Item 2",
                    type=["jpg", "jpeg", "png", "gif"],
                    key="item2_file",
                )
                item2_path = None
                if item2_file:
                    # Show image preview immediately
                    st.image(
                        item2_file, caption="Preview - Item 2", use_container_width=True
                    )
                    item2_path = (
                        ASSETS_FOLDER
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{item2_file.name}"
                    )
                    with open(item2_path, "wb") as f:
                        f.write(item2_file.read())
            else:
                item2_text = st.text_area("Enter Text for Item 2", key="item2_text")
                item2_path = None
                if item2_text:
                    # Show text preview
                    st.info(
                        f"Preview: {item2_text[:100]}..."
                        if len(item2_text) > 100
                        else f"Preview: {item2_text}"
                    )
                    item2_path = (
                        ASSETS_FOLDER
                        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_item2.txt"
                    )
                    with open(item2_path, "w", encoding="utf-8") as f:
                        f.write(item2_text)

        st.markdown("---")
        st.markdown("### 🎯 Set Similarity Score")
        actual_score = st.number_input(
            "✨ Actual Similarity Score (0-100)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            help="Enter the actual similarity score between the two items",
        )

        if st.button("✨ Add Game Pair ✨", use_container_width=True):
            if item1_path and item2_path:
                add_game_item(
                    selected_category,
                    str(item1_path),
                    str(item2_path),
                    actual_score,
                    description,
                )
                st.success(f"🎉 Game pair added to '{selected_category}' successfully!")
            else:
                st.error("❌ Please upload/enter both items.")

    # Tab 2: Manage Items
    with tab2:
        st.subheader("Manage Game Pairs")

        items = get_all_game_items()
        if items:
            for item in items:
                item_id, category, item1_path, item2_path, actual_score, description = (
                    item
                )

                with st.expander(
                    f"ID: {item_id} | {category} | Score: {actual_score} | {description or 'No description'}"
                ):
                    st.write(f"**Category:** {category}")
                    st.write(f"**Description:** {description or 'N/A'}")
                    st.write(f"**Item 1:** {item1_path}")
                    st.write(f"**Item 2:** {item2_path}")

                    col1, col2 = st.columns(2)

                    with col1:
                        new_score = st.number_input(
                            "Actual Score", 0, 100, actual_score, key=f"score_{item_id}"
                        )
                    with col2:
                        new_desc = st.text_input(
                            "Description",
                            value=description or "",
                            key=f"desc_{item_id}",
                        )

                    col_update, col_delete = st.columns(2)
                    with col_update:
                        if st.button("Update", key=f"update_{item_id}"):
                            update_game_item(item_id, category, new_score, new_desc)
                            st.success("Game pair updated!")
                            st.rerun()
                    with col_delete:
                        if st.button("Delete", key=f"delete_{item_id}"):
                            delete_game_item(item_id)
                            st.success("Game pair deleted!")
                            st.rerun()
        else:
            st.info("No game pairs found.")

    # Tab 3: Reset Leaderboard
    with tab3:
        st.subheader("Reset Leaderboard")
        st.warning("⚠️ This will permanently delete all leaderboard entries!")

        if st.button("Reset Leaderboard", type="primary"):
            reset_leaderboard()
            st.success("Leaderboard has been reset!")
