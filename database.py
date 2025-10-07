import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

try:
    # Optional config to switch between in-memory and file DB
    from config import USE_IN_MEMORY_DB, DB_FILE_PATH
except Exception:
    # Sensible defaults if config is missing
    USE_IN_MEMORY_DB = True
    DB_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "app.db")


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        original_filename TEXT,
        created_at TEXT NOT NULL,
        output_video_path TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        frame_nmr INTEGER NOT NULL,
        car_id INTEGER NOT NULL,
        car_bbox TEXT NOT NULL,
        license_plate_bbox TEXT,
        license_plate_bbox_score REAL,
        license_number TEXT,
        license_number_score REAL,
        FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interpolated_detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,
        frame_nmr INTEGER NOT NULL,
        car_id INTEGER NOT NULL,
        car_bbox TEXT NOT NULL,
        license_plate_bbox TEXT,
        license_plate_bbox_score REAL,
        license_number TEXT,
        license_number_score REAL,
        is_imputed INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_detections_video_frame
        ON detections(video_id, frame_nmr)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_interp_video_frame
        ON interpolated_detections(video_id, frame_nmr)
    """
]


def create_connection() -> sqlite3.Connection:
    """Return a SQLite connection (in-memory shared-cache or file-backed based on config)."""
    if USE_IN_MEMORY_DB:
        conn = sqlite3.connect(
            "file:vehicle_detection?mode=memory&cache=shared",
            uri=True,
            check_same_thread=False,
        )
    else:
        db_dir = os.path.dirname(DB_FILE_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Enforce foreign key constraints for ON DELETE CASCADE, etc.
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception:
        pass
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create database tables if they do not exist."""
    with conn:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)


@contextmanager
def transaction(conn: sqlite3.Connection):
    """Context manager wrapping operations in a transaction."""
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()


def register_video(
    conn: sqlite3.Connection,
    video_id: str,
    original_filename: str,
    output_video_path: Optional[str] = None
) -> None:
    """Insert or update metadata for a processed video."""
    with conn:
        conn.execute(
            """
            INSERT INTO videos(video_id, original_filename, created_at, output_video_path)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                original_filename = excluded.original_filename,
                output_video_path = excluded.output_video_path
            """,
            (video_id, original_filename, datetime.utcnow().isoformat(), output_video_path)
        )


def update_video_output_path(conn: sqlite3.Connection, video_id: str, output_video_path: str) -> None:
    with conn:
        conn.execute(
            "UPDATE videos SET output_video_path = ? WHERE video_id = ?",
            (output_video_path, video_id)
        )


def clear_video_data(conn: sqlite3.Connection, video_id: str) -> None:
    """Remove detections and interpolations for a specific video."""
    with conn:
        conn.execute("DELETE FROM detections WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM interpolated_detections WHERE video_id = ?", (video_id,))


def _serialize_bbox(bbox: Iterable[float]) -> str:
    return json.dumps([float(coord) for coord in bbox])


def _deserialize_bbox(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def store_detections(conn: sqlite3.Connection, video_id: str, results: Dict[int, Dict[int, Dict]]) -> None:
    """Persist raw detection results keyed by frame and car."""
    if not results:
        return

    rows: List[Tuple] = []
    for frame_nmr, cars in results.items():
        for car_id, payload in cars.items():
            car_bbox = payload.get("car", {}).get("bbox")
            lp_payload = payload.get("license_plate", {})
            lp_bbox = lp_payload.get("bbox")
            rows.append(
                (
                    video_id,
                    int(frame_nmr),
                    int(car_id),
                    _serialize_bbox(car_bbox) if car_bbox else None,
                    _serialize_bbox(lp_bbox) if lp_bbox else None,
                    float(lp_payload.get("bbox_score", 0) or 0),
                    lp_payload.get("text"),
                    float(lp_payload.get("text_score", 0) or 0)
                )
            )

    with conn:
        conn.executemany(
            """
            INSERT INTO detections(
                video_id,
                frame_nmr,
                car_id,
                car_bbox,
                license_plate_bbox,
                license_plate_bbox_score,
                license_number,
                license_number_score
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )


def fetch_detections(conn: sqlite3.Connection, video_id: str) -> List[sqlite3.Row]:
    cursor = conn.execute(
        """
        SELECT frame_nmr,
               car_id,
               car_bbox,
               license_plate_bbox,
               license_plate_bbox_score,
               license_number,
               license_number_score
        FROM detections
        WHERE video_id = ?
        ORDER BY frame_nmr ASC, car_id ASC
        """,
        (video_id,)
    )
    return list(cursor.fetchall())


def store_interpolated_detections(
    conn: sqlite3.Connection,
    video_id: str,
    data: Iterable[Dict[str, object]]
) -> None:
    rows = []
    for row in data:
        rows.append(
            (
                video_id,
                int(row["frame_nmr"]),
                int(row["car_id"]),
                _serialize_bbox(row["car_bbox"]),
                _serialize_bbox(row.get("license_plate_bbox")) if row.get("license_plate_bbox") else None,
                float(row.get("license_plate_bbox_score", 0) or 0),
                row.get("license_number"),
                float(row.get("license_number_score", 0) or 0),
                int(row.get("is_imputed", 0))
            )
        )

    with conn:
        conn.executemany(
            """
            INSERT INTO interpolated_detections(
                video_id,
                frame_nmr,
                car_id,
                car_bbox,
                license_plate_bbox,
                license_plate_bbox_score,
                license_number,
                license_number_score,
                is_imputed
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows
        )


def fetch_interpolated_detections(conn: sqlite3.Connection, video_id: str) -> List[Dict[str, object]]:
    cursor = conn.execute(
        """
        SELECT frame_nmr,
               car_id,
               car_bbox,
               license_plate_bbox,
               license_plate_bbox_score,
               license_number,
               license_number_score,
               is_imputed
        FROM interpolated_detections
        WHERE video_id = ?
        ORDER BY frame_nmr ASC, car_id ASC
        """,
        (video_id,)
    )

    rows = []
    for record in cursor.fetchall():
        rows.append(
            {
                "frame_nmr": record["frame_nmr"],
                "car_id": record["car_id"],
                "car_bbox": _deserialize_bbox(record["car_bbox"]) or [],
                "license_plate_bbox": _deserialize_bbox(record["license_plate_bbox"]) or [],
                "license_plate_bbox_score": record["license_plate_bbox_score"],
                "license_number": record["license_number"],
                "license_number_score": record["license_number_score"],
                "is_imputed": record["is_imputed"],
            }
        )
    return rows


def delete_video(conn: sqlite3.Connection, video_id: str) -> None:
    with conn:
        conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))


def get_video_output_path(conn: sqlite3.Connection, video_id: str) -> Optional[str]:
    cursor = conn.execute(
        "SELECT output_video_path FROM videos WHERE video_id = ?",
        (video_id,)
    )
    row = cursor.fetchone()
    return row["output_video_path"] if row and row["output_video_path"] else None