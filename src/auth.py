
import os
from datetime import datetime, timedelta, timezone

import jwt
from pwdlib import PasswordHash

from src.database import get_connection


ANALYST_USERNAME = os.getenv("ANALYST_USERNAME", "")
ANALYST_PASSWORD_HASH = os.getenv("ANALYST_PASSWORD_HASH", "")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Hash a password for secure storage."""
    return password_hasher.hash(password)


def authenticate_analyst(username: str, password: str) -> bool:
    """Verify analyst credentials."""
    if not ANALYST_USERNAME or not ANALYST_PASSWORD_HASH:
        raise RuntimeError("Analyst authentication is not configured.")

    if username != ANALYST_USERNAME:
        return False

    try:
        return password_hasher.verify(password, ANALYST_PASSWORD_HASH)
    except (ValueError, TypeError):
        return False


def create_access_token(username: str) -> str:
    """Create a short-lived JWT for an analyst."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "role": "analyst",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


def create_employee_access_token(employee_id: str) -> str:
    """Create a short-lived JWT for an employee."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": employee_id,
        "role": "employee",
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=ALGORITHM)


def verify_access_token(token: str) -> dict:
    """Validate a JWT belonging to an analyst."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])

    if payload.get("role") != "analyst" or not payload.get("sub"):
        raise jwt.InvalidTokenError("Invalid analyst token.")

    return payload


def verify_employee_access_token(token: str) -> dict:
    """Validate a JWT belonging to an employee."""
    if not JWT_SECRET_KEY:
        raise RuntimeError("JWT_SECRET_KEY is not configured.")

    payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])

    if payload.get("role") != "employee" or not payload.get("sub"):
        raise jwt.InvalidTokenError("Invalid employee token.")

    return payload


def authenticate_employee(employee_id: str, password: str) -> bool:
    """Verify an active employee's password against the database."""
    with get_connection() as connection:
        employee = connection.execute(
            """
            SELECT password_hash
            FROM employees
            WHERE employee_id = ? AND is_active = 1
            """,
            (employee_id,),
        ).fetchone()

    if employee is None:
        return False

    try:
        return password_hasher.verify(password, employee["password_hash"])
    except (ValueError, TypeError):
        return False
