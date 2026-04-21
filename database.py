import sqlite3
import os
import time
from datetime import datetime
from pathlib import Path
import json

DB_PATH = "ppe_monitoring.db"
SCREENSHOTS_DIR = "screenshots"


def init_db():
    Path(SCREENSHOTS_DIR).mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            class_name  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            severity    TEXT NOT NULL,
            screenshot  TEXT,
            zone        TEXT DEFAULT 'Zone A'
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            compliance_score REAL NOT NULL,
            total_detections INTEGER NOT NULL,
            total_violations INTEGER NOT NULL,
            fps              REAL
        )
    """)

    conn.commit()
    conn.close()


def log_frame(compliance_score: float, total_detections: int,
              total_violations: int, fps: float = 0.0):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO frames (timestamp, compliance_score, total_detections,
                            total_violations, fps)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), compliance_score,
          total_detections, total_violations, fps))
    conn.commit()
    conn.close()


def log_violation(class_name: str, confidence: float,
                  severity: str, screenshot_path: str = None,
                  zone: str = "Zone A"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO violations (timestamp, class_name, confidence,
                                severity, screenshot, zone)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), class_name, confidence,
          severity, screenshot_path, zone))
    conn.commit()
    conn.close()


def get_violations(limit: int = 100, hours: int = 24):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    since = datetime.fromtimestamp(time.time() - hours * 3600).isoformat()
    rows = conn.execute("""
        SELECT * FROM violations
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (since, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_compliance_history(hours: int = 24):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    since = datetime.fromtimestamp(time.time() - hours * 3600).isoformat()
    rows = conn.execute("""
        SELECT timestamp, compliance_score, total_violations
        FROM frames
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, (since,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats(hours: int = 24):
    conn = sqlite3.connect(DB_PATH)
    since = datetime.fromtimestamp(time.time() - hours * 3600).isoformat()

    total_violations = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE timestamp >= ?", (since,)
    ).fetchone()[0]

    avg_compliance = conn.execute(
        "SELECT AVG(compliance_score) FROM frames WHERE timestamp >= ?", (since,)
    ).fetchone()[0] or 100.0

    violations_by_type = conn.execute("""
        SELECT class_name, COUNT(*) as count
        FROM violations WHERE timestamp >= ?
        GROUP BY class_name ORDER BY count DESC
    """, (since,)).fetchall()

    conn.close()
    return {
        "total_violations": total_violations,
        "avg_compliance":   round(avg_compliance, 1),
        "by_type":          {r[0]: r[1] for r in violations_by_type},
    }


def clear_old_data(days: int = 7):
    conn = sqlite3.connect(DB_PATH)
    since = datetime.fromtimestamp(time.time() - days * 86400).isoformat()
    conn.execute("DELETE FROM violations WHERE timestamp < ?", (since,))
    conn.execute("DELETE FROM frames WHERE timestamp < ?", (since,))
    conn.commit()
    conn.close()
