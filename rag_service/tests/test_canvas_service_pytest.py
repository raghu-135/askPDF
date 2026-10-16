import pytest

from app.models.canvas import CanvasCreateRequest, parse_canvas_spec, resolve_quote_blocks
from app.services.canvas_service import CanvasService, CanvasValidationError


def _quote_spec(file_hash: str = "doc1", sentence_id: int = 1):
    return parse_canvas_spec({
        "schema_version": 1,
        "title": "Quoted canvas",
        "sections": [
            {
                "title": "Passage",
                "blocks": [
                    {
                        "type": "quote",
                        "file_hash": file_hash,
                        "sentence_id": sentence_id,
                        "label": "Key line",
                    },
                ],
            }
        ],
    })


def test_resolve_quote_blocks_fills_canonical_text():
    spec = _quote_spec()
    resolved = resolve_quote_blocks(
        spec,
        sentences_by_file={
            "doc1": {
                "sentences": [
                    {"id": 0, "text": "Ignored.", "page": 1},
                    {"id": 1, "text": "Canonical sentence.", "page": 2},
                ]
            }
        },
    )
    quote = resolved.sections[0].blocks[0]
    assert quote.type == "quote"
    assert quote.text == "Canonical sentence."
    assert quote.page == 2


@pytest.mark.asyncio
async def test_canvas_service_create_quote_failure_surfaces_validation_error(monkeypatch):
    spec = _quote_spec(sentence_id=5)

    async def fake_sentences(file_hash: str):
        return {"sentences": [{"id": 1, "text": "Other."}]}

    monkeypatch.setattr("app.services.canvas_service.get_file_parsed_sentences", fake_sentences)

    class FakeRepo:
        async def get(self, thread_id: str):
            return object()

    service = CanvasService(canvases=object(), threads=FakeRepo())
    with pytest.raises(CanvasValidationError):
        await service.create("thread-1", CanvasCreateRequest(spec=spec))
