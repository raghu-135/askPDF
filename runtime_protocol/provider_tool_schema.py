"""JSON Schema shaping for provider function-calling (Gemini-compatible)."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping


_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "$schema",
    "$id",
    "$comment",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "$vocabulary",
    "default",
    "examples",
    "discriminator",
})


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _resolve_pointer(document: Mapping[str, Any], pointer: str) -> Any:
    if pointer == "#":
        return document
    if not pointer.startswith("#/"):
        raise ValueError(f"unsupported JSON pointer: {pointer}")
    node: Any = document
    for raw in pointer[2:].split("/"):
        key = _decode_pointer_token(raw)
        if isinstance(node, Mapping):
            node = node[key]
        else:
            raise KeyError(pointer)
    return node


def _contains_ref_or_defs(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "$ref" in value or "$defs" in value:
            return True
        return any(_contains_ref_or_defs(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref_or_defs(item) for item in value)
    return False


def provider_compatible_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Inline local ``$ref`` targets and drop schema keywords Gemini rejects."""

    root = copy.deepcopy(dict(schema))
    defs = dict(root.pop("$defs", {}) or {})
    document: dict[str, Any] = {"$defs": defs, **root}

    def transform(node: Any, *, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [transform(item, stack=stack) for item in node]
        if not isinstance(node, Mapping):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str):
            if ref in stack:
                return {"type": "object"}
            try:
                resolved = _resolve_pointer(document, ref)
            except (KeyError, ValueError, TypeError):
                return {"type": "object"}
            inlined = transform(resolved, stack=stack + (ref,))
            extras = {
                key: transform(value, stack=stack)
                for key, value in node.items()
                if key != "$ref" and key not in _UNSUPPORTED_SCHEMA_KEYS
            }
            if isinstance(inlined, Mapping) and extras:
                merged = dict(inlined)
                for key, value in extras.items():
                    if key not in merged:
                        merged[key] = value
                return merged
            return inlined

        return {
            key: transform(value, stack=stack)
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS and key != "$defs"
        }

    normalized = transform(root, stack=())
    if _contains_ref_or_defs(normalized):
        raise ValueError("provider tool schema still contains $ref or $defs after normalization")
    return normalized


def provider_compatible_function_tools(tools: list[Any] | None) -> list[Any] | None:
    """Normalize OpenAI-style function tools before provider submission."""

    if not tools:
        return tools
    sanitized: list[Any] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            sanitized.append(tool)
            continue
        entry = dict(tool)
        function = entry.get("function")
        if isinstance(function, Mapping):
            function = dict(function)
            parameters = function.get("parameters")
            if isinstance(parameters, Mapping):
                function["parameters"] = provider_compatible_json_schema(parameters)
            entry["function"] = function
        sanitized.append(entry)
    return sanitized
