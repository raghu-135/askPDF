from fastapi import APIRouter, HTTPException, Query

from app.services.canvas_service import (
    CanvasNotFoundError,
    CanvasService,
    CanvasValidationError,
    validate_canvas_payload,
)

router = APIRouter(tags=["canvases"])
_service = CanvasService()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CanvasNotFoundError):
        return HTTPException(status_code=404, detail={"code": "not_found", "message": str(exc)})
    if isinstance(exc, CanvasValidationError):
        return HTTPException(status_code=400, detail={"code": "invalid_canvas", "message": str(exc)})
    raise exc


@router.post("/threads/{thread_id}/canvases")
async def create_thread_canvas(thread_id: str, payload: dict):
    try:
        request = validate_canvas_payload(payload)
        return await _service.create(thread_id, request)
    except (CanvasNotFoundError, CanvasValidationError) as exc:
        raise _http_error(exc) from exc


@router.get("/threads/{thread_id}/canvases")
async def list_thread_canvases(
    thread_id: str,
    current_only: bool = Query(default=True),
):
    try:
        canvases = await _service.list_for_thread(thread_id, current_only=current_only)
        return {"thread_id": thread_id, "canvases": canvases}
    except CanvasNotFoundError as exc:
        raise _http_error(exc) from exc


@router.get("/threads/{thread_id}/canvases/{canvas_id}")
async def get_thread_canvas(thread_id: str, canvas_id: str):
    try:
        return await _service.get(thread_id, canvas_id)
    except (CanvasNotFoundError, CanvasValidationError) as exc:
        raise _http_error(exc) from exc
