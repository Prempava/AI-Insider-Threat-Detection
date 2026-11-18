
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from src.database import get_connection


def record_dlp_scan(
    analyst_username: str,
    filename: str,
    destination: str,
    scan_result: dict,
    transfer_policy: dict,
) -> dict:
    scan_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    findings = scan_result.get("findings", [])

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO dlp_audit_logs (
                scan_id,
                timestamp,
                analyst_username,
                filename,
                destination,
                destination_authorized,
                risk_level,
                finding_count,
                findings_json,
                review_required,
                mode,
                automated_blocking
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                timestamp,
                analyst_username,
                filename,
                destination,
                int(transfer_policy["destination_authorized"]),
                transfer_policy["risk_level"],
                scan_result["finding_count"],
                json.dumps(findings),
                int(transfer_policy["review_required"]),
                transfer_policy["mode"],
                int(transfer_policy["automated_blocking"]),
            ),
        )

    return {
        "scan_id": scan_id,
        "timestamp": timestamp,
    }


def list_dlp_scans(limit: int = 50) -> list[dict]:
    limit = max(1, min(limit, 100))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM dlp_audit_logs
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["destination_authorized"] = bool(
            item["destination_authorized"]
        )
        item["review_required"] = bool(item["review_required"])
        item["automated_blocking"] = bool(item["automated_blocking"])
        item["findings"] = json.loads(item.pop("findings_json"))
        results.append(item)

    return results
