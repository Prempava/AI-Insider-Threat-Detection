
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.database import get_connection, initialize_database


class SecurityEvent(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    event_type: str = Field(min_length=1, max_length=50)
    description: str = Field(min_length=1, max_length=500)
    risk_level: str = Field(
        default="medium",
        pattern="^(low|medium|high|critical)$",
    )


def record_event(
    event: SecurityEvent,
    analysis: dict | None = None,
) -> dict:
    """Save an event and its optional LSTM analysis in SQLite."""
    analysis = analysis or {}

    record = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": event.user_id,
        "event_type": event.event_type,
        "description": event.description,
        "risk_level": event.risk_level,
        "status": "open",
        "prediction": analysis.get("prediction"),
        "risk_score": analysis.get("risk_score"),
        "model_threshold": analysis.get("model_threshold"),
        "automated_blocking": False,
    }

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO security_events (
                event_id, timestamp, user_id, event_type,
                description, risk_level, status, prediction,
                risk_score, model_threshold, automated_blocking
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["event_id"],
                record["timestamp"],
                record["user_id"],
                record["event_type"],
                record["description"],
                record["risk_level"],
                record["status"],
                record["prediction"],
                record["risk_score"],
                record["model_threshold"],
                0,
            ),
        )

    return record


def list_events() -> list[dict]:
    """Return recorded events, newest first."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM security_events ORDER BY timestamp DESC"
        ).fetchall()

    events = []
    for row in rows:
        event = dict(row)
        event["automated_blocking"] = bool(event["automated_blocking"])
        events.append(event)

    return events


def get_event(event_id: str) -> dict:
    """Retrieve one event by ID."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM security_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")

    event = dict(row)
    event["automated_blocking"] = bool(event["automated_blocking"])
    return event


def update_event_status(event_id: str, status: str) -> dict:
    """Update an event's review status."""
    allowed_statuses = {"open", "under_review", "resolved"}

    if status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {sorted(allowed_statuses)}",
        )

    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE security_events SET status = ? WHERE event_id = ?",
            (status, event_id),
        )

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Event not found")

    return get_event(event_id)


initialize_database()
