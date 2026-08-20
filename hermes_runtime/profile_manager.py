"""Render immutable Hermes config and isolated, short-lived run profiles."""

from __future__ import annotations

import hashlib
import base64
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass


MIN_CONTEXT_LENGTH = 2048
PINNED_MIN_CONTEXT_LENGTH = 64_000
PROFILE_PREFIX = "askpdf-run-"
TOKEN_HEADER = "X-AskPDF-Execution-Context"
PINNED_HERMES_UID = 10_000
PINNED_HERMES_GID = 10_000
TOMBSTONE_FILE = ".askpdf-retired.json"


@dataclass(frozen=True)
class RunProfile:
    name: str
    directory: Path
    token_digest: str
    token_expires_at: int | None
    policy_profile: str
    expected_tools: tuple[str, ...]
    config_fingerprint: str
    activation_fingerprint: str
    mcp_server_name: str
    lifecycle_state: str = "active"

    def continuation_metadata(self) -> dict[str, object]:
        return {
            "runtime_profile": self.name,
            "profile_digest": self.config_fingerprint,
            "profile_activation_digest": self.activation_fingerprint,
            "token_digest": self.token_digest,
            "token_expires_at": self.token_expires_at,
            "profile_state": self.lifecycle_state,
            "mcp_server_name": self.mcp_server_name,
        }


def _token_expiry(token: str) -> int | None:
    """Read the signed token's non-secret expiry for lifecycle bookkeeping."""
    try:
        encoded = token.split(".", 1)[0]
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        return int(payload["exp"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def configured_context_length() -> int:
    raw = os.getenv("HERMES_MODEL_CONTEXT_LENGTH", "").strip()
    if not raw or raw.lower() in {"true", "false", "yes", "no", "on", "off"}:
        raise RuntimeError("HERMES_MODEL_CONTEXT_LENGTH must be an integer")
    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise RuntimeError("HERMES_MODEL_CONTEXT_LENGTH must be an integer") from exc
    if value < MIN_CONTEXT_LENGTH:
        raise RuntimeError(f"HERMES_MODEL_CONTEXT_LENGTH must be at least {MIN_CONTEXT_LENGTH}")
    return value


def render_bootstrap_config() -> None:
    """Materialize checked-in templates with a concrete numeric context length."""

    context_length = configured_context_length()
    template_root = Path(os.getenv("HERMES_CONFIG_TEMPLATE_ROOT", "/app/hermes_runtime"))
    data_root = Path(os.getenv("HERMES_DATA_ROOT", "/opt/data"))
    targets = {
        template_root / "config.yaml": data_root / "config.yaml",
        template_root / "profiles/askpdf-deep-offline/config.yaml": data_root / "profiles/askpdf-deep-offline/config.yaml",
        template_root / "profiles/askpdf-deep-external/config.yaml": data_root / "profiles/askpdf-deep-external/config.yaml",
    }
    for source, target in targets.items():
        provider = configured_provider()
        validate_provider_context(provider, context_length)
        rendered = (
            source.read_text()
            .replace("__HERMES_MODEL_CONTEXT_LENGTH__", str(context_length))
            .replace("__HERMES_MODEL_PROVIDER__", provider)
        )
        if "${HERMES_MODEL_CONTEXT_LENGTH}" in rendered:
            raise RuntimeError(f"unrendered Hermes context length in {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered)


def configured_provider() -> str:
    provider = os.getenv("HERMES_MODEL_PROVIDER", "custom").strip().lower()
    if not provider or any(character.isspace() for character in provider):
        raise RuntimeError("HERMES_MODEL_PROVIDER must be a provider identifier")
    return provider


def validate_provider_context(provider: str, context_length: int) -> None:
    if provider != "lmstudio" and context_length < PINNED_MIN_CONTEXT_LENGTH:
        raise RuntimeError(
            f"Hermes provider {provider!r} requires HERMES_MODEL_CONTEXT_LENGTH "
            f"of at least {PINNED_MIN_CONTEXT_LENGTH}"
        )


class RunProfileManager:
    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or os.getenv("HERMES_PROFILE_ROOT", "/opt/data/profiles"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: set[str] = set()

    def create(
        self,
        *,
        run_id: str,
        policy_profile: str,
        context_token: str,
        allowed_tools: list[str],
        selected_model: str,
        selected_provider: str,
    ) -> RunProfile:
        if not context_token:
            raise RuntimeError("Hermes MCP execution context is required")
        api_server_key = os.getenv("HERMES_API_TOKEN", "").strip()
        if not api_server_key:
            raise RuntimeError("HERMES_API_TOKEN is required for run profiles")
        selected_model = selected_model.strip()
        selected_provider = selected_provider.strip()
        if not selected_model or not selected_provider:
            raise RuntimeError("Hermes run profiles require a selected model and provider")
        selected_provider = selected_provider.lower()
        context_length = configured_context_length()
        validate_provider_context(selected_provider, context_length)
        provider_url = os.getenv("LLM_API_URL", "").strip()
        if not provider_url:
            raise RuntimeError("LLM_API_URL is required for run profiles")
        provider_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        secret_values = (
            api_server_key, context_token, provider_url, provider_api_key,
            selected_model, selected_provider,
        )
        if any(character in value for value in secret_values for character in "\r\n"):
            raise RuntimeError("Hermes run-profile secrets must be single-line values")
        if policy_profile not in {"askpdf-deep-offline", "askpdf-deep-external"}:
            raise RuntimeError("Unsupported Hermes managed profile")
        suffix = hashlib.sha256(run_id.encode()).hexdigest()[:20]
        profile_name = PROFILE_PREFIX + suffix
        destination = self.root / profile_name
        if destination.exists():
            shutil.rmtree(destination)
        temporary = Path(tempfile.mkdtemp(prefix=f".{profile_name}-", dir=self.root))
        endpoint = "external" if policy_profile.endswith("external") else "offline"
        # Pinned Hermes keeps MCP connections in a process-global registry keyed
        # only by server name. A stable `askpdf` key would reuse another run's
        # connection and its headers, so every run requires its own namespace.
        mcp_server_name = "askpdf_" + suffix
        tools = json.dumps(sorted(set(allowed_tools)))
        activation_fingerprint = hashlib.sha256(
            f"{profile_name}\0{mcp_server_name}\0{tools}\0{context_length}".encode()
        ).hexdigest()
        config = (
            "# Generated run profile. Secret-bearing; retire immediately after the run.\n"
            "_config_version: 37\n"
            "model:\n"
            f"  default: {json.dumps(selected_model)}\n"
            f"  provider: {json.dumps(selected_provider)}\n"
            f"  base_url: {json.dumps(provider_url)}\n"
            f"  context_length: {context_length}\n"
            "askpdf_runtime_profile:\n"
            f"  name: {json.dumps(profile_name)}\n"
            f"  config_fingerprint: {json.dumps(activation_fingerprint)}\n"
            "platforms:\n"
            "  api_server:\n"
            "    enabled: false\n"
            "auxiliary:\n"
            "  title_generation:\n"
            "    enabled: false\n"
            "mcp_servers:\n"
            f"  {mcp_server_name}:\n"
            f"    url: http://rag-service:8000/internal/hermes-mcp/{endpoint}/\n"
            "    enabled: true\n"
            "    headers:\n"
            f"      {TOKEN_HEADER}: {json.dumps(context_token)}\n"
            "    tools:\n"
            f"      include: {tools}\n"
        )
        config_fingerprint = hashlib.sha256(config.encode()).hexdigest()
        (temporary / "config.yaml").write_text(config)
        env_file = temporary / ".env"
        profile_environment = [
            f"API_SERVER_KEY={api_server_key}",
        ]
        if provider_api_key:
            profile_environment.append(f"OPENAI_API_KEY={provider_api_key}")
        env_file.write_text("\n".join(profile_environment) + "\n")
        profile_uid = int(os.getenv("HERMES_PROFILE_UID", str(PINNED_HERMES_UID)))
        profile_gid = int(os.getenv("HERMES_PROFILE_GID", str(PINNED_HERMES_GID)))
        config_file = temporary / "config.yaml"
        for path, mode in ((temporary, 0o750), (config_file, 0o600), (env_file, 0o600)):
            os.chown(path, profile_uid, profile_gid)
            os.chmod(path, mode)
        temporary.rename(destination)
        self._active.add(profile_name)
        return RunProfile(
            name=profile_name,
            directory=destination,
            token_digest=hashlib.sha256(context_token.encode()).hexdigest(),
            token_expires_at=_token_expiry(context_token),
            policy_profile=policy_profile,
            expected_tools=tuple(sorted(set(allowed_tools))),
            config_fingerprint=config_fingerprint,
            activation_fingerprint=activation_fingerprint,
            mcp_server_name=mcp_server_name,
        )

    def retire(self, profile: RunProfile | str | None) -> None:
        """Remove secrets immediately but retain the directory for live loggers."""
        profile_name = profile.name if isinstance(profile, RunProfile) else profile
        if not profile_name or not re.fullmatch(r"askpdf-run-[0-9a-f]{20}", profile_name):
            return
        destination = self.root / profile_name
        if destination.parent == self.root and destination.exists():
            token_digest = profile.token_digest if isinstance(profile, RunProfile) else None
            config_fingerprint = profile.config_fingerprint if isinstance(profile, RunProfile) else None
            env_file = destination / ".env"
            if env_file.exists():
                env_file.write_text("")
                env_file.unlink()
            # Preserve a valid, secret-free config. The pinned gateway keeps
            # profile-specific log handlers alive after the run and its
            # reconciler removes directories that no longer look like profiles.
            config_file = destination / "config.yaml"
            if config_file.exists():
                config_file.write_text(
                    "# Retired askPDF runtime profile; contains no secrets.\n"
                    "_config_version: 37\n"
                    "platforms: {}\n"
                    "auxiliary:\n"
                    "  title_generation:\n"
                    "    enabled: false\n"
                    "mcp_servers: {}\n"
                )
            (destination / "logs").mkdir(exist_ok=True)
            tombstone = {
                "retired_at": int(time.time()),
                "profile_digest": config_fingerprint,
                "token_digest": token_digest,
            }
            (destination / TOMBSTONE_FILE).write_text(json.dumps(tombstone, sort_keys=True))
        self._active.discard(profile_name)

    def remove(self, profile_name: str | None) -> None:
        """Permanently remove an inactive generated profile."""
        if not profile_name or not re.fullmatch(r"askpdf-run-[0-9a-f]{20}", profile_name):
            return
        if profile_name in self._active:
            return
        destination = self.root / profile_name
        if destination.parent == self.root and destination.exists():
            shutil.rmtree(destination)

    def is_reusable(self, profile_name: str | None) -> bool:
        if not profile_name or profile_name not in self._active:
            return False
        destination = self.root / profile_name
        return (destination / "config.yaml").is_file() and (destination / ".env").is_file()

    def verify(self, profile: RunProfile) -> bool:
        config_path = profile.directory / "config.yaml"
        try:
            return (
                profile.name in self._active
                and profile.directory.parent == self.root
                and hashlib.sha256(config_path.read_bytes()).hexdigest() == profile.config_fingerprint
                and (profile.directory / ".env").is_file()
            )
        except OSError:
            return False

    def sweep_stale(self, *, max_age_seconds: int = 86_400) -> int:
        cutoff = time.time() - max(60, max_age_seconds)
        removed = 0
        for candidate in self.root.glob(PROFILE_PREFIX + "*"):
            if candidate.name not in self._active and candidate.is_dir() and candidate.stat().st_mtime < cutoff:
                self.remove(candidate.name)
                removed += 1
        return removed


if __name__ == "__main__":
    render_bootstrap_config()
