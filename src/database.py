
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "artifacts" / "security_events.db"


def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS dlp_audit_logs (
                scan_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                analyst_username TEXT NOT NULL,
                filename TEXT NOT NULL,
                destination TEXT NOT NULL,
                destination_authorized INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                finding_count INTEGER NOT NULL,
                findings_json TEXT NOT NULL,
                review_required INTEGER NOT NULL,
                mode TEXT NOT NULL DEFAULT 'detection_only',
                automated_blocking INTEGER NOT NULL DEFAULT 0
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                employee_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                department TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
                # Safely add email column to existing employee databases.
        employee_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(employees)"
            ).fetchall()
        }

        if "email" not in employee_columns:
            connection.execute(
                "ALTER TABLE employees ADD COLUMN email TEXT"
            )

        # Store email security analysis results.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_security_events (
                event_id TEXT PRIMARY KEY,
                employee_id TEXT NOT NULL,
                gmail_message_id TEXT NOT NULL,
                thread_id TEXT,
                sender TEXT,
                subject TEXT,
                received_at TEXT,
                ai_analysis_json TEXT NOT NULL,
                rule_analysis_json TEXT NOT NULL,
                final_risk_level TEXT NOT NULL
                    CHECK (
                        final_risk_level IN (
                            'low', 'medium', 'high', 'critical'
                        )
                    ),
                created_at TEXT NOT NULL,

                UNIQUE(employee_id, gmail_message_id)
            )
        """)

        # Track security notification attempts.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS email_notification_logs (
                notification_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                employee_id TEXT NOT NULL,
                recipient_email TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'pending', 'accepted', 'failed'
                        )
                    ),
                gmail_message_id TEXT,
                attempted_at TEXT NOT NULL,
                error_message TEXT
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS employee_face_enrollments (
                employee_id TEXT PRIMARY KEY,
                consent_at TEXT NOT NULL,
                enrolled_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                model_version TEXT NOT NULL DEFAULT 'lbph-v1',
                FOREIGN KEY (employee_id)
                    REFERENCES employees(employee_id)
                    ON DELETE CASCADE
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                risk_level TEXT NOT NULL
                    CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'under_review', 'resolved')),
                prediction TEXT,
                risk_score REAL,
                model_threshold REAL,
                automated_blocking INTEGER NOT NULL DEFAULT 0
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS usb_authorizations (
                authorization_id TEXT PRIMARY KEY,
                employee_id TEXT NOT NULL,
                device_instance_id TEXT NOT NULL,
                device_name TEXT NOT NULL,
                authorized_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('authorized', 'revoked', 'expired')),
                verification_method TEXT NOT NULL
                    DEFAULT 'face',
                FOREIGN KEY (employee_id)
                    REFERENCES employees(employee_id)
            )
        """)



if __name__ == "__main__":
    initialize_database()
    print(f"Database initialized at: {DATABASE_PATH}")
