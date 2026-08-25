"""Product-owned runtime capability descriptors."""

from __future__ import annotations

from app.runtime.contracts import (
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    conditional,
    native,
)


def product_operation_descriptors() -> dict[RuntimeOperationId, RuntimeOperationDescriptor]:
    """Return descriptors for operations implemented by askPDF itself."""

    return {
        RuntimeOperationId.RUN_EVENTS: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics="persisted_product_event_journal",
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
