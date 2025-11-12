# Signature Guessing Game

An interactive web application built with Streamlit for playing signature similarity guessing games.

## Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux operating system

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ict
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

**Windows:**
```bash
venv\Scripts\python.exe -m streamlit run app.py
```

**macOS/Linux:**
```bash
streamlit run app.py
```

The application will open in your default web browser. If it doesn't, navigate to `http://localhost:8501` in your browser.

## Application Features

- **Player Mode**: Play signature similarity guessing games
- **Admin Mode**: Add, manage, and delete game items
- **Leaderboard**: View top players and their scores
- **Multiple Game Categories**: Signature Sleuth, Art Twins, Sound-Alike, and more

## How to Play

1. Enter your name in the player mode
2. Select a game category
3. Compare two items (signatures, images, etc.)
4. Guess the similarity score (0-100%)
5. See how close your guess is to the actual score

## Admin Features

- Add new game pairs with upload functionality
- Manage existing game items
- Reset leaderboard data

## Project Structure

```
ict/
├── app.py              # Main application entry point
├── database.py         # Database operations and initialization
├── player.py           # Player mode functionality
├── admin.py            # Admin mode functionality
├── leaderboard.py      # Leaderboard display
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore file
├── game_data.db        # SQLite database (created automatically)
├── assets/             # Uploaded files and images
└── pages/              # Streamlit multi-page files
    ├── 1_🔧_Admin_Mode.py
    ├── 2_🏆_Leaderboard.py
    ├── 3_🎮_Player_Mode.py
    └── ...             # Other game pages
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure you've activated the virtual environment and installed dependencies
2. **Port already in use**: Streamlit will automatically try different ports, or you can specify one:
   ```bash
   streamlit run app.py --server.port 8502
   ```
3. **Permission errors on Windows**: Run the command prompt as administrator if needed

### Windows-Specific Notes

- Always use `venv\Scripts\python.exe` to run Python scripts within the virtual environment
- If you encounter encoding issues, ensure your command prompt is set to UTF-8:
  ```bash
  chcp 65001
  ```

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is proprietary and confidential. All rights reserved.