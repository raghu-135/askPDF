"""Product-owned runtime capability descriptors."""

from __future__ import annotations

from app.runtime.contracts import (
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
)


def product_operation_descriptors() -> dict[str, RuntimeOperationDescriptor]:
    """Return descriptors for operations implemented by askPDF itself."""

    return {
        RuntimeOperationId.RUN_EVENTS.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.NATIVE,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="persisted_product_event_journal",
        ),
        RuntimeOperationId.TASK_START.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="product_task_start",
        ),
        RuntimeOperationId.TASK_PAUSE.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="product_task_pause",
        ),
        RuntimeOperationId.TASK_RESUME.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="product_task_resume",
        ),
        RuntimeOperationId.TASK_CANCEL.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="product_task_cancel",
        ),
        RuntimeOperationId.TASK_RETRY.value: RuntimeOperationDescriptor(
            RuntimeSupportLevel.CONDITIONAL,
            RuntimeOperationOwner.PRODUCT,
            True,
            semantics="product_task_retry",
        ),
    }
