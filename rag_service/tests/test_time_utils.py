from datetime import datetime, timezone

from app.time_utils import iso_utc_z, maybe_iso_utc_z, parse_datetime_utc, utc_now


def test_utc_now_returns_timezone_aware_utc_datetime():
    now = utc_now()

    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 0


def test_parse_datetime_utc_normalizes_supported_inputs():
    assert parse_datetime_utc("2026-06-25T19:15:00Z") == datetime(
        2026, 6, 25, 19, 15, tzinfo=timezone.utc
    )
    assert parse_datetime_utc("2026-06-25T14:15:00-05:00") == datetime(
        2026, 6, 25, 19, 15, tzinfo=timezone.utc
    )
    assert parse_datetime_utc("2026-06-25T19:15:00") == datetime(
        2026, 6, 25, 19, 15, tzinfo=timezone.utc
    )
    assert parse_datetime_utc(datetime(2026, 6, 25, 19, 15)) == datetime(
        2026, 6, 25, 19, 15, tzinfo=timezone.utc
    )


def test_parse_datetime_utc_returns_none_for_empty_or_invalid_values():
    assert parse_datetime_utc(None) is None
    assert parse_datetime_utc("") is None
    assert parse_datetime_utc("not a date") is None


def test_iso_utc_z_formats_none_and_datetime_like_values():
    assert iso_utc_z().endswith("Z")
    assert iso_utc_z("2026-06-25T14:15:00-05:00") == "2026-06-25T19:15:00Z"
    assert iso_utc_z(datetime(2026, 6, 25, 19, 15)) == "2026-06-25T19:15:00Z"


def test_maybe_iso_utc_z_preserves_invalid_values():
    assert maybe_iso_utc_z("2026-06-25T14:15:00-05:00") == "2026-06-25T19:15:00Z"
    assert maybe_iso_utc_z("not a date") == "not a date"
    assert maybe_iso_utc_z(None) is None
