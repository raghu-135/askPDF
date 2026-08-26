"""Product-owned runtime capability descriptors."""

from __future__ import annotations

from dataclasses import replace

from app.runtime.contracts import (
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeCapabilities,
    conditional,
    native,
)


PUBLIC_OPERATION_IDS = frozenset({
    RuntimeOperationId.RUN_GET,
    RuntimeOperationId.RUN_LIST,
    RuntimeOperationId.RUN_EVENTS,
    RuntimeOperationId.RUN_CANCEL,
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_SEND_FOLLOWUP,
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.RUN_UPDATE_STATE,
    RuntimeOperationId.TASK_START,
    RuntimeOperationId.TASK_PAUSE,
    RuntimeOperationId.TASK_RESUME,
    RuntimeOperationId.TASK_CANCEL,
    RuntimeOperationId.TASK_RETRY,
    RuntimeOperationId.ARTIFACT_LIST,
})


def product_operation_descriptors() -> dict[RuntimeOperationId, RuntimeOperationDescriptor]:
    """Return descriptors for operations implemented by askPDF itself."""

    return {
        RuntimeOperationId.RUN_EVENTS: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics="persisted_product_event_journal",
        ),
        RuntimeOperationId.RUN_GET: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics="product_run_inspection",
        ),
        RuntimeOperationId.RUN_LIST: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics="product_run_listing",
        ),
        RuntimeOperationId.ARTIFACT_LIST: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics="product_task_artifact_listing",
        ),
        RuntimeOperationId.TASK_START: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics="product_task_start",
        ),
        RuntimeOperationId.TASK_PAUSE: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics="product_task_pause",
        ),
        RuntimeOperationId.TASK_RESUME: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics="product_task_resume",
        ),
        RuntimeOperationId.TASK_CANCEL: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics="product_task_cancel",
        ),
        RuntimeOperationId.TASK_RETRY: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics="product_task_retry",
        ),
    }


def project_public_capabilities(capabilities: RuntimeCapabilities) -> RuntimeCapabilities:
    """Return only operations addressable through the askPDF product API."""

    return replace(
        capabilities,
        operations={
            operation: descriptor
            for operation, descriptor in capabilities.operations.items()
            if operation in PUBLIC_OPERATION_IDS
        },
    )
