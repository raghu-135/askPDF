"""Compatibility re-exports for shared runtime protocol errors."""

from runtime_protocol.errors import ProtocolDecodeError, ProtocolVersionError, RuntimeError

__all__ = ["ProtocolDecodeError", "ProtocolVersionError", "RuntimeError"]
