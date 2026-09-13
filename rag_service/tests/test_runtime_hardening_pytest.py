from __future__ import annotations

import pytest

from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


def test_langgraph_connector_requires_authentication(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_RUNTIME_TOKEN", raising=False)
    with pytest.raises(AgentRuntimeError, match="TOKEN"):
        HttpLangGraphRuntimeAdapter("http://runtime")
