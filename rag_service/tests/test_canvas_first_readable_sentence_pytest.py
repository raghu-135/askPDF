from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.services.canvas_compiler import _first_sentence_for_chunk


@pytest.mark.asyncio
async def test_first_sentence_for_chunk_skips_non_readable_header():
    file_hash = "a" * 32
    row = SimpleNamespace(sentence_ids=["0", "5"])
    parsed = {
        "sentences": [
            {"id": 0, "readable": False, "text": "arXiv:1706.03762"},
            {"id": 5, "readable": True, "text": "Abstract body sentence."},
        ],
    }

    class FakeResult:
        def scalar_one_or_none(self):
            return row

    class BeginCtx:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *_args):
            return None

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def begin(self):
            return BeginCtx()

        async def execute(self, *_args, **_kwargs):
            return FakeResult()

    class SessionMaker:
        def __call__(self):
            return FakeSession()

    with (
        patch("app.services.canvas_compiler.async_session_maker", SessionMaker()),
        patch(
            "app.services.canvas_compiler.get_file_parsed_sentences",
            new=AsyncMock(return_value=parsed),
        ),
    ):
        sentence_id = await _first_sentence_for_chunk(file_hash, "chunk-locator")

    assert sentence_id == 5
