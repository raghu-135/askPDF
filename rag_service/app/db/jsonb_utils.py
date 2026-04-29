"""
jsonb_utils.py - Utilities for safe JSONB field mutations with SQLAlchemy.

Critical for PostgreSQL JSONB change tracking. Without these helpers,
mutations to nested JSONB fields may not be detected or persisted.
"""

from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm.attributes import flag_modified


def set_jsonb_field(obj: Any, field_name: str, key: str, value: Any) -> None:
    """
    Safely set a key in a JSONB field with proper change tracking.
    
    Usage:
        file = File(file_hash="abc", file_name="test.pdf")
        set_jsonb_field(file, "file_status", "parsing", {"status": "running"})
    """
    current = getattr(obj, field_name) or {}
    # Create new dict to ensure SQLAlchemy detects change
    new_value = dict(current)
    new_value[key] = value
    new_value["updated_at"] = datetime.utcnow().isoformat()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)


def merge_jsonb_field(obj: Any, field_name: str, updates: Dict[str, Any]) -> None:
    """
    Merge updates into a JSONB field with proper change tracking.
    
    Usage:
        thread = Thread(id="t1", name="Test")
        merge_jsonb_field(thread, "settings", {"max_iterations": 10})
    """
    current = getattr(obj, field_name) or {}
    new_value = {**current, **updates}
    new_value["updated_at"] = datetime.utcnow().isoformat()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)


def replace_jsonb_field(obj: Any, field_name: str, new_value: Dict[str, Any]) -> None:
    """
    Completely replace a JSONB field value.
    
    Usage:
        replace_jsonb_field(file, "file_status", {"parsing": {"status": "completed"}})
    """
    new_value = dict(new_value)  # Ensure it's a new dict object
    new_value["updated_at"] = datetime.utcnow().isoformat()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)
