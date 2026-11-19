"""
jsonb_utils.py - Utilities for safe JSONB field mutations with SQLAlchemy.

Critical for PostgreSQL JSONB change tracking. Without these helpers,
mutations to nested JSONB fields may not be detected or persisted.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict

from sqlalchemy.orm.attributes import flag_modified
from app.time_utils import iso_utc_z


_INVALID_JSON_U = re.compile(r"\\u(?![0-9a-fA-F]{4})")


def _scrub_unicode(value: Any) -> Any:
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    if isinstance(value, dict):
        return {key: _scrub_unicode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_unicode(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub_unicode(item) for item in value]
    return value


def postgres_jsonb_safe(value: Any) -> Any:
    """Round-trip through JSON with ASCII escapes so asyncpg accepts the payload."""
    scrubbed = _scrub_unicode(value)
    text = json.dumps(scrubbed, ensure_ascii=True, default=str)
    text = _INVALID_JSON_U.sub(r"\\\\u", text)
    return json.loads(text)


def set_jsonb_field(
    obj: Any,
    field_name: str,
    key: str,
    value: Any,
    *,
    touch_updated_at: bool = False,
) -> None:
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
    if touch_updated_at:
        new_value["updated_at"] = iso_utc_z()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)


def merge_jsonb_field(
    obj: Any,
    field_name: str,
    updates: Dict[str, Any],
    *,
    touch_updated_at: bool = False,
) -> None:
    """
    Merge updates into a JSONB field with proper change tracking.
    
    Usage:
        project = Project(id="p1", name="Test", embedding_model="BAAI/bge-m3")
        thread = Thread(
            id="t1",
            project_id=project.id,
            name="Test",
            embedding_model=project.embedding_model,
        )
        merge_jsonb_field(thread, "settings", {"replans": 10})
    """
    current = getattr(obj, field_name) or {}
    new_value = {**current, **updates}
    if touch_updated_at:
        new_value["updated_at"] = iso_utc_z()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)


def replace_jsonb_field(
    obj: Any,
    field_name: str,
    new_value: Any,
    *,
    touch_updated_at: bool = False,
) -> None:
    """
    Completely replace a JSONB field value.
    
    Usage:
        replace_jsonb_field(file, "file_status", {"parsing": {"status": "completed"}})
    """
    new_value = deepcopy(new_value)
    if touch_updated_at:
        if not isinstance(new_value, dict):
            raise TypeError("touch_updated_at requires an object-valued JSONB field")
        new_value["updated_at"] = iso_utc_z()
    setattr(obj, field_name, new_value)
    flag_modified(obj, field_name)
