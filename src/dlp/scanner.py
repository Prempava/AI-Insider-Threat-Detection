
import re
from typing import Any


PATTERNS = {
    "email_address": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "phone_number": re.compile(
        r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"
    ),
    "possible_secret": re.compile(
        r"\b(?:api[_-]?key|password|secret[_-]?key|access[_-]?token)"
        r"\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
}


def scan_text(text: str) -> dict[str, Any]:
    """Detect configured sensitive-data patterns without returning values."""
    findings = []

    for category, pattern in PATTERNS.items():
        count = sum(1 for _ in pattern.finditer(text))

        if count:
            findings.append({
                "type": category,
                "count": count,
            })

    total = sum(item["count"] for item in findings)
    types_found = {item["type"] for item in findings}

    if "possible_secret" in types_found:
        risk_level = "high"
        recommendation = "Review urgently; a possible secret was detected."
    elif total:
        risk_level = "medium"
        recommendation = "Review the file before transferring it."
    else:
        risk_level = "low"
        recommendation = "No configured sensitive patterns detected."

    return {
        "contains_sensitive_data": total > 0,
        "finding_count": total,
        "risk_level": risk_level,
        "findings": findings,
        "recommendation": recommendation,
    }


def assess_transfer_policy(
    scan_result: dict[str, Any],
    destination: str,
    authorized_destinations: set[str],
) -> dict[str, Any]:
    """Assess a proposed destination against an explicit allowlist."""
    normalized_destination = destination.strip().lower()
    allowed = {
        item.strip().lower()
        for item in authorized_destinations
    }

    destination_authorized = (
        bool(normalized_destination)
        and normalized_destination in allowed
    )

    risk_level = scan_result["risk_level"]
    reasons = []

    if not destination_authorized:
        risk_level = "high"
        reasons.append("Destination is not on the authorized allowlist.")

    if scan_result["risk_level"] == "high":
        reasons.append("A possible secret was detected.")
    elif scan_result["risk_level"] == "medium":
        reasons.append("Sensitive-data patterns require review.")

    review_required = (
        risk_level in {"medium", "high"}
    )

    return {
        "risk_level": risk_level,
        "destination_authorized": destination_authorized,
        "review_required": review_required,
        "policy_reasons": reasons,
        "mode": "detection_only",
        "automated_blocking": False,
    }
