"""Load embedding model catalog (JSON) for chunking, prefetch, and API display."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "embedding_models.json"
)


def _strip_provider_variant(model: str) -> str:
    name = str(model or "").strip()
    if name.endswith(":free"):
        return name[: -len(":free")]
    return name


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def catalog_path() -> Path:
    override = os.environ.get("EMBEDDING_MODELS_CONFIG", "").strip()
    if override:
        return Path(override)
    return DEFAULT_CATALOG_PATH


@dataclass(frozen=True)
class GenericSettings:
    encoding: str
    effective_input_limit: int
    pack_margin_ratio: float

    def pack_limit(self) -> int:
        margin = self.pack_margin_ratio
        if margin <= 0 or margin > 1:
            margin = 1.0
        return max(1, int(self.effective_input_limit * margin))


@dataclass(frozen=True)
class EmbeddingDisplayConfig:
    tier: str
    tokenizer: str | None
    tokenizer_revision: str | None
    effective_input_limit: int
    pack_margin_ratio: float | None
    encoding: str | None
    prefix: str
    suffix: str
    query_prefix: str | None
    query_suffix: str | None
    hf_prefetched: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
            "effective_input_limit": self.effective_input_limit,
            "pack_margin_ratio": self.pack_margin_ratio,
            "encoding": self.encoding,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "query_prefix": self.query_prefix,
            "query_suffix": self.query_suffix,
            "hf_prefetched": self.hf_prefetched,
        }


@dataclass(frozen=True)
class ResolvedEmbeddingSettings:
    """Settings consumed by resolve_embedding_tokenizer (no HF load)."""

    tier: str
    provider_model: str
    tokenizer: str | None
    revision: str | None
    effective_input_limit: int
    pack_margin_ratio: float | None
    encoding: str | None
    prefix: str
    suffix: str
    query_prefix: str | None
    query_suffix: str | None
    prefetch: bool


def _parse_generic_block(raw: Mapping[str, Any]) -> GenericSettings:
    encoding = str(raw.get("encoding") or "cl100k_base").strip() or "cl100k_base"
    limit = int(raw.get("effective_input_limit") or 0)
    margin = float(raw.get("pack_margin_ratio") or 0.85)
    limit = _env_int("EMBEDDING_GENERIC_INPUT_LIMIT", limit or 8192)
    margin = _env_float("EMBEDDING_GENERIC_PACK_MARGIN", margin)
    return GenericSettings(encoding=encoding, effective_input_limit=limit, pack_margin_ratio=margin)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Embedding catalog {path} must be a JSON object")
    return data


@lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> tuple[GenericSettings, dict[str, dict[str, Any]]]:
    resolved = Path(path) if path else catalog_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"Embedding catalog not found: {resolved}")
    data = _load_json(resolved)
    generic = _parse_generic_block(data.get("generic") if isinstance(data.get("generic"), Mapping) else {})
    models_raw = data.get("models")
    models: dict[str, dict[str, Any]] = {}
    if isinstance(models_raw, Mapping):
        for key, fields in models_raw.items():
            name = str(key).strip()
            if not name or not isinstance(fields, Mapping):
                continue
            models[name] = dict(fields)
    return generic, models


def reload_catalog() -> None:
    load_catalog.cache_clear()


def lookup_provider_model(provider_id: str) -> dict[str, Any] | None:
    name = str(provider_id or "").strip()
    if not name:
        return None
    _, models = load_catalog()
    if name in models:
        return dict(models[name])
    variant = _strip_provider_variant(name)
    if variant in models:
        return dict(models[variant])
    return None


def has_exact_hf_config(provider_id: str) -> bool:
    entry = lookup_provider_model(provider_id)
    if not entry:
        return False
    tokenizer = str(entry.get("tokenizer") or "").strip()
    if not tokenizer:
        return False
    if entry.get("prefetch") is False and entry.get("tier") == "generic":
        return False
    return True


def resolve_settings_for_provider(provider_id: str) -> ResolvedEmbeddingSettings:
    """Resolve tier and fields for a provider model id (not LOCAL_EMBEDDING_MODEL)."""
    name = str(provider_id or "").strip()
    generic, _ = load_catalog()
    entry = lookup_provider_model(name)

    if entry:
        tier = str(entry.get("tier") or "").strip().lower()
        tokenizer = str(entry.get("tokenizer") or "").strip() or None
        if not tier:
            tier = "exact" if tokenizer else "generic"
        if tier == "generic" or not tokenizer:
            limit = int(entry.get("effective_input_limit") or generic.effective_input_limit)
            margin = entry.get("pack_margin_ratio")
            margin_f = float(margin) if margin is not None else generic.pack_margin_ratio
            encoding = str(entry.get("encoding") or generic.encoding).strip() or generic.encoding
            return ResolvedEmbeddingSettings(
                tier="generic",
                provider_model=name,
                tokenizer=None,
                revision=None,
                effective_input_limit=limit,
                pack_margin_ratio=margin_f,
                encoding=encoding,
                prefix=str(entry.get("prefix") or ""),
                suffix=str(entry.get("suffix") or ""),
                query_prefix=(
                    str(entry["query_prefix"]) if entry.get("query_prefix") is not None else None
                ),
                query_suffix=(
                    str(entry["query_suffix"]) if entry.get("query_suffix") is not None else None
                ),
                prefetch=False,
            )
        limit = int(entry.get("effective_input_limit") or 0)
        if limit <= 0:
            limit = generic.effective_input_limit
        return ResolvedEmbeddingSettings(
            tier="exact",
            provider_model=name,
            tokenizer=tokenizer,
            revision=str(entry.get("revision") or "").strip() or None,
            effective_input_limit=limit,
            pack_margin_ratio=None,
            encoding=None,
            prefix=str(entry.get("prefix") or ""),
            suffix=str(entry.get("suffix") or ""),
            query_prefix=(str(entry["query_prefix"]) if "query_prefix" in entry else None),
            query_suffix=(str(entry["query_suffix"]) if "query_suffix" in entry else None),
            prefetch=entry.get("prefetch") is not False,
        )

    return ResolvedEmbeddingSettings(
        tier="generic",
        provider_model=name,
        tokenizer=None,
        revision=None,
        effective_input_limit=generic.effective_input_limit,
        pack_margin_ratio=generic.pack_margin_ratio,
        encoding=generic.encoding,
        prefix="",
        suffix="",
        query_prefix=None,
        query_suffix=None,
        prefetch=False,
    )


def unique_hf_tokenizer_identities() -> list[tuple[str, str | None]]:
    """HF tokenizer ids to prefetch from the catalog."""
    _, models = load_catalog()
    seen: dict[str, str | None] = {}
    for fields in models.values():
        if fields.get("prefetch") is False:
            continue
        identity = str(fields.get("tokenizer") or "").strip()
        if not identity or identity in seen:
            continue
        revision = str(fields.get("revision") or "").strip() or None
        seen[identity] = revision
    return sorted(seen.items(), key=lambda item: item[0].casefold())


def resolve_display_config(provider_id: str, *, local_model: str | None = None) -> EmbeddingDisplayConfig:
    name = str(provider_id or "").strip()
    local = str(local_model or os.environ.get("LOCAL_EMBEDDING_MODEL", "")).strip()
    if name == local:
        limit = _env_int("LOCAL_EMBEDDING_INPUT_LIMIT", 512)
        tok = os.environ.get("LOCAL_EMBEDDING_TOKENIZER", name).strip() or name
        return EmbeddingDisplayConfig(
            tier="local",
            tokenizer=tok,
            tokenizer_revision=os.environ.get("LOCAL_EMBEDDING_TOKENIZER_REVISION") or None,
            effective_input_limit=limit,
            pack_margin_ratio=None,
            encoding=None,
            prefix=os.environ.get("LOCAL_EMBEDDING_FORMAT_PREFIX", ""),
            suffix=os.environ.get("LOCAL_EMBEDDING_FORMAT_SUFFIX", ""),
            query_prefix=None,
            query_suffix=None,
            hf_prefetched=True,
        )

    resolved = resolve_settings_for_provider(name)
    hf_prefetched = resolved.tier == "exact" and resolved.prefetch
    prefetch_set = {item[0].casefold() for item in unique_hf_tokenizer_identities()}
    if resolved.tokenizer:
        hf_prefetched = resolved.tokenizer.casefold() in prefetch_set
    pack_limit = resolved.effective_input_limit
    if resolved.tier == "generic" and resolved.pack_margin_ratio:
        pack_limit = max(1, int(resolved.effective_input_limit * resolved.pack_margin_ratio))

    return EmbeddingDisplayConfig(
        tier=resolved.tier,
        tokenizer=resolved.tokenizer,
        tokenizer_revision=resolved.revision,
        effective_input_limit=pack_limit if resolved.tier == "generic" else resolved.effective_input_limit,
        pack_margin_ratio=resolved.pack_margin_ratio,
        encoding=resolved.encoding,
        prefix=resolved.prefix,
        suffix=resolved.suffix,
        query_prefix=resolved.query_prefix,
        query_suffix=resolved.query_suffix,
        hf_prefetched=hf_prefetched,
    )


def embedding_configs_for_ids(provider_ids: list[str], *, local_ids: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    local_model = os.environ.get("LOCAL_EMBEDDING_MODEL", "").strip()
    for model_id in local_ids:
        out[str(model_id)] = resolve_display_config(str(model_id), local_model=local_model).to_dict()
    for model_id in provider_ids:
        if model_id in out:
            continue
        out[str(model_id)] = resolve_display_config(str(model_id), local_model=local_model).to_dict()
    return out


__all__ = [
    "EmbeddingDisplayConfig",
    "GenericSettings",
    "ResolvedEmbeddingSettings",
    "catalog_path",
    "embedding_configs_for_ids",
    "has_exact_hf_config",
    "load_catalog",
    "lookup_provider_model",
    "reload_catalog",
    "resolve_display_config",
    "resolve_settings_for_provider",
    "unique_hf_tokenizer_identities",
]