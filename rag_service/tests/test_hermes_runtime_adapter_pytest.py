import pytest

from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, ContinuationBinding
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.runtime.errors import RuntimeError


@pytest.mark.asyncio
async def test_hermes_adapter_has_independent_identity():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    assert adapter.framework == "hermes"
    assert adapter.builder_id == "hermes_agent"


@pytest.mark.asyncio
async def test_hermes_resume_is_explicitly_unsupported():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    with pytest.raises(RuntimeError, match="resume"):
        await adapter.resume(request, interrupt={}, context=None)


def test_hermes_continuation_binding_is_opaque():
    binding = ContinuationBinding("hermes_session", {"session_id": "session-1"})
    assert binding.to_dict()["payload"]["session_id"] == "session-1"
