"""Product-owned runtime capability descriptors."""

from __future__ import annotations

from dataclasses import replace

from app.runtime.contracts import (
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeCapabilities,
    RuntimeCapabilitySemantics,
    RuntimeConfirmationMode,
    RuntimeTerminalState,
    conditional,
    native,
)


PUBLIC_OPERATION_IDS = frozenset({
    RuntimeOperationId.RUN_GET,
    RuntimeOperationId.RUN_LIST,
    RuntimeOperationId.RUN_EVENTS,
    RuntimeOperationId.RUN_INSPECT_STATE,
    RuntimeOperationId.RUN_CANCEL,
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_APPROVAL_RESPOND,
    RuntimeOperationId.RUN_SEND_FOLLOWUP,
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.TASK_START,
    RuntimeOperationId.TASK_PAUSE,
    RuntimeOperationId.TASK_RESUME,
    RuntimeOperationId.TASK_CANCEL,
    RuntimeOperationId.TASK_RETRY,
    RuntimeOperationId.TASK_RESULT_REVIEW_RESPOND,
    RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND,
    RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT,
    RuntimeOperationId.ARTIFACT_LIST,
})


def product_operation_descriptors() -> dict[RuntimeOperationId, RuntimeOperationDescriptor]:
    """Return descriptors for operations implemented by askPDF itself."""

    return {
        RuntimeOperationId.RUN_EVENTS: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics=RuntimeCapabilitySemantics.PERSISTED_PRODUCT_EVENT_JOURNAL,
        ),
        RuntimeOperationId.RUN_GET: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics=RuntimeCapabilitySemantics.PRODUCT_RUN_INSPECTION,
        ),
        RuntimeOperationId.RUN_LIST: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics=RuntimeCapabilitySemantics.PRODUCT_RUN_LISTING,
        ),
        RuntimeOperationId.ARTIFACT_LIST: native(
            owner=RuntimeOperationOwner.PRODUCT,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_ARTIFACT_LISTING,
        ),
        RuntimeOperationId.TASK_START: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_START,
        ),
        RuntimeOperationId.TASK_PAUSE: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_PAUSE,
            confirmation=RuntimeConfirmationMode.BOUNDED,
            terminal_states=(RuntimeTerminalState.INTERRUPTED,),
            preserves_run_id=True,
            preserves_session_id=True,
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.TASK_RESUME: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_RESUME,
            confirmation=RuntimeConfirmationMode.BOUNDED,
            preserves_run_id=True,
            preserves_session_id=True,
            requires_runtime_binding=True,
        ),
        RuntimeOperationId.TASK_CANCEL: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_CANCEL,
        ),
        RuntimeOperationId.TASK_RETRY: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_RETRY,
        ),
        RuntimeOperationId.TASK_RESULT_REVIEW_RESPOND: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_RESULT_REVIEW,
        ),
        RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_BUDGET_REVIEW,
        ),
        RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT: conditional(
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
            semantics=RuntimeCapabilitySemantics.PRODUCT_TASK_COURSE_CORRECTION,
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
