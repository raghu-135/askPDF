import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_runtime.hermes_pinned_patch.approval_bridge import wrap_mcp_handler, decode_approval_request
from runtime_protocol.tool_approval import ApprovalMode, ToolApprovalPolicy, tool_approval_request


def test_hermes_wrapper_uses_native_approval_and_retries_same_invocation():
    request = tool_approval_request("arbitrary_tool", {"query": "example"}, policy=ToolApprovalPolicy(ApprovalMode.ASK), caller="agent", response_operation="run.resume")
    request["proposed_tool"]["invocation_id"] = "invocation"
    handler = Mock(side_effect=[json.dumps({"structuredContent": {"artifacts": {"approval_request": request}}}), '{"result":"executed"}'])
    approve = Mock(return_value={"approved": True})
    wrapped = wrap_mcp_handler(handler, approve, tool_name="arbitrary_tool")
    assert wrapped({"query": "example"}) == '{"result":"executed"}'
    first_id = handler.call_args_list[0].args[0]["_askpdf_invocation_id"]
    assert handler.call_args_list[1].args[0]["_askpdf_invocation_id"] == first_id
    replay = Mock(return_value='{"structuredContent":{}}')
    wrap_mcp_handler(replay, approve, tool_name="arbitrary_tool")({"query": "example"})
    assert replay.call_args.args[0]["_askpdf_invocation_id"] == first_id
    event = {"pattern_key": "plugin_rule:" + approve.call_args.kwargs["rule_key"]}
    assert decode_approval_request(event) == request


def test_approval_bridge_does_not_import_runtime_protocol():
    import hermes_runtime.hermes_pinned_patch.approval_bridge as bridge
    assert "runtime_protocol" not in Path(bridge.__file__).read_text()


def test_hermes_native_denial_never_retries_tool():
    request = {"proposed_tool": {"name": "tool"}}
    handler = Mock(return_value=json.dumps({"structuredContent": {"artifacts": {"approval_request": request}}}))
    result = wrap_mcp_handler(handler, Mock(return_value={"approved": False}))({})
    assert "skipped" in result
    handler.assert_called_once()


@pytest.mark.asyncio
async def test_native_stream_reconnect_replays_lost_frames_and_keeps_tools_until_terminal(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from hermes_runtime.hermes_pinned_patch import event_stream

    queue = asyncio.Queue()
    for event in [{"event": "tool.started"}, {"event": "approval.request"}, {"event": "tool.completed"}, None]:
        queue.put_nowait(event)
    adapter = SimpleNamespace(
        _check_auth=lambda _: None, _run_streams={"run-1": queue}, _run_stream_subscribers=set(),
    )
    request = SimpleNamespace(match_info={"run_id": "run-1"})
    retired = AsyncMock()
    responses = []

    class Response:
        def __init__(self, **kwargs):
            self.writes = []
            self.disconnect = not responses
            responses.append(self)

        async def prepare(self, request):
            pass

        async def write(self, frame):
            if self.disconnect and len(self.writes) == 2:
                raise ConnectionResetError("human boundary closed the socket")
            self.writes.append(frame)

    web = SimpleNamespace(StreamResponse=Response)
    encode = lambda event: json.dumps(event).encode()
    await event_stream.serve_run_events(adapter, request, web=web, encode_frame=encode, on_terminal=retired)
    retired.assert_not_awaited()
    assert "run-1" in adapter._run_streams
    assert not adapter._run_stream_subscribers
    await event_stream.serve_run_events(adapter, request, web=web, encode_frame=encode, on_terminal=retired)
    retired.assert_awaited_once()
    replayed = [json.loads(frame) for frame in responses[1].writes if not frame.startswith(b":")]
    assert [event["event"] for event in replayed] == ["tool.started", "approval.request", "tool.completed"]
    assert [event["event_id"] for event in replayed] == ["run-1:0", "run-1:1", "run-1:2"]
    assert responses[0].writes == responses[1].writes[:2]


def test_live_approval_survives_stream_ttl_and_terminal_replay_is_reclaimed():
    from types import SimpleNamespace
    from hermes_runtime.hermes_pinned_patch.event_stream import sweep_run_events

    adapter = SimpleNamespace(
        _active_run_tasks={"live": SimpleNamespace(done=lambda: False), "done": SimpleNamespace(done=lambda: True)},
        _run_streams_created={"live": 0, "done": 0},
        _run_streams={"live": object(), "done": object()},
        _askpdf_event_replays={"live": object(), "done": object()},
    )
    def native_sweep(adapter, now):
        for run_id, timestamp in list(adapter._run_streams_created.items()):
            if now - timestamp > 300:
                adapter._run_streams.pop(run_id)
                adapter._run_streams_created.pop(run_id)
    sweep_run_events(adapter, native_sweep, now=600)
    assert set(adapter._run_streams) == {"live"}
    assert set(adapter._askpdf_event_replays) == {"live"}
