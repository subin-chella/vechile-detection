# To process the full video, set PROCESS_FIRST_N_SECONDS to None or -1
PROCESS_FIRST_N_SECONDS = 15

# Database configuration
# If True, use an in-memory SQLite database with shared cache (fast, ephemeral).
# If False, use a file-backed SQLite database at DB_FILE_PATH (persistent across runs).
USE_IN_MEMORY_DB = False

# Path for persistent SQLite database when USE_IN_MEMORY_DB = False
# Defaults to outputs/app.db within the project directory
import os as _os
DB_FILE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs", "app.db")
