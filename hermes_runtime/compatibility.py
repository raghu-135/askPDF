"""Compatibility import retained inside the isolated runtime package."""

try:  # rag-service test/runtime source tree
    from app.runtime.hermes_compatibility import *  # type: ignore # noqa: F403
except ModuleNotFoundError:  # isolated Hermes runtime image copy
    from hermes_compatibility import *  # type: ignore # noqa: F403
