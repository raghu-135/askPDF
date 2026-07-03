from fastapi import APIRouter

from app.agent.tool_registry import list_tool_contract_metadata


router = APIRouter(tags=["tools"])


@router.get("/tools/contracts")
async def list_tool_contracts():
    """Return backend tool contract metadata for validation and debug UIs."""

    return {"tools": list_tool_contract_metadata()}
