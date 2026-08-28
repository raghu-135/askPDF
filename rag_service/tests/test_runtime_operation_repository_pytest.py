from datetime import timedelta
from types import SimpleNamespace

import pytest

from app.services import runtime_operation_repository as repository
from app.time_utils import utc_now


class _Statement:
    def values(self, **kwargs):
        self.values_data = kwargs
        return self

    def on_conflict_do_nothing(self, **kwargs):
        return self

    def where(self, *args):
        return self

    def with_for_update(self):
        return self


class _Result:
    def __init__(self, *, rowcount=0, record=None):
        self.rowcount = rowcount
        self.record = record

    def scalar_one_or_none(self):
        return self.record


class _Session:
    def __init__(self, results):
        self.results = iter(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def begin(self):
        return self

    async def execute(self, _statement):
        return next(self.results)


class _SessionFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


def _patch_claim_dependencies(monkeypatch, results):
    session = _Session(results)
    monkeypatch.setattr(repository, "async_session_maker", _SessionFactory(session))
    monkeypatch.setattr(repository, "insert", lambda _model: _Statement())
    monkeypatch.setattr(repository, "select", lambda _model: _Statement())


def test_runtime_operation_lease_defaults_to_five_minutes(monkeypatch):
    monkeypatch.delenv("RUNTIME_OPERATION_LEASE_SECONDS", raising=False)
    assert repository.runtime_operation_lease_seconds() == 300


def test_runtime_operation_lease_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("RUNTIME_OPERATION_LEASE_SECONDS", "0")
    with pytest.raises(RuntimeError, match="greater than zero"):
        repository.runtime_operation_lease_seconds()


@pytest.mark.asyncio
async def test_fresh_runtime_operation_claim_is_returned(monkeypatch):
    claimed_at = utc_now()
    record = SimpleNamespace(status="in_progress", request_fingerprint="fingerprint")
    _patch_claim_dependencies(monkeypatch, [_Result(rowcount=1), _Result(record=record)])

    result = await repository.claim_runtime_operation(
        run_id="run-1",
        operation="run.cancel",
        idempotency_key="key-1",
        request_fingerprint="fingerprint",
    )

    assert result is record


@pytest.mark.asyncio
async def test_expired_runtime_operation_claim_is_reclaimed(monkeypatch):
    existing = SimpleNamespace(
        status="in_progress",
        request_fingerprint="fingerprint",
        claim_expires_at=utc_now() - timedelta(seconds=1),
        error_json={"retryable": True},
        result_json={"old": True},
        completed_at=utc_now(),
    )
    _patch_claim_dependencies(monkeypatch, [_Result(rowcount=0), _Result(record=existing)])

    result = await repository.claim_runtime_operation(
        run_id="run-1",
        operation="run.cancel",
        idempotency_key="key-1",
        request_fingerprint="fingerprint",
    )

    assert result is existing
    assert result.status == "in_progress"
    assert result.error_json is None
    assert result.result_json == {}
    assert result.completed_at is None
    assert result.claim_expires_at > result.claimed_at


@pytest.mark.asyncio
async def test_active_runtime_operation_claim_is_not_reclaimed(monkeypatch):
    existing = SimpleNamespace(
        status="in_progress",
        request_fingerprint="fingerprint",
        claim_expires_at=utc_now() + timedelta(minutes=1),
    )
    _patch_claim_dependencies(monkeypatch, [_Result(rowcount=0), _Result(record=existing)])

    with pytest.raises(repository.RuntimeOperationConflict) as error:
        await repository.claim_runtime_operation(
            run_id="run-1",
            operation="run.cancel",
            idempotency_key="key-1",
            request_fingerprint="fingerprint",
        )

    assert error.value.code == "runtime_operation_in_progress"
