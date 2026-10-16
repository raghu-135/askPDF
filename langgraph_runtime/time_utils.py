"""Shared UTC datetime parsing and formatting helpers."""

from datetime import datetime, timezone
from typing import Any, Optional

from dateutil.parser import isoparse


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def parse_datetime_utc(value: Any) -> Optional[datetime]:
    """Parse a datetime-like value and normalize it to timezone-aware UTC."""
    if value in (None, ""):
        return None

    try:
        if isinstance(value, datetime):
            dt = value
        else:
            dt = isoparse(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def iso_utc_z(value: Any = None) -> str:
    """Format a datetime-like value as an ISO-8601 UTC timestamp ending in Z."""
    dt = parse_datetime_utc(utc_now() if value is None else value)
    if dt is None:
        raise ValueError(f"Cannot parse datetime value: {value!r}")
    return dt.isoformat().replace("+00:00", "Z")


def maybe_iso_utc_z(value: Any) -> Any:
    """Return a normalized ISO UTC string when parseable, otherwise keep value."""
    if value in (None, ""):
        return value
    dt = parse_datetime_utc(value)
    if dt is None:
        return value
    return iso_utc_z(dt)
