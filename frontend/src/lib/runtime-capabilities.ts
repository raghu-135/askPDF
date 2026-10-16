import type {
  AgentRuntimeCapabilityResponse,
  RuntimeCapabilityDisabledReason,
  RuntimeOperationDescriptor,
} from './api';

export type RuntimeControlOperation =
  | 'run.cancel'
  | 'run.resume'
  | 'run.approval.respond'
  | 'run.send_followup'
  | 'run.interrupt_with_input'
  | 'run.steer_live'
  | 'task.start'
  | 'task.pause'
  | 'task.resume'
  | 'task.cancel'
  | 'task.retry'
  | 'task.result_review.respond'
  | 'task.budget_review.respond'
  | 'task.course_correction.submit';

export type TaskControlAction = 'start' | 'pause' | 'resume' | 'cancel' | 'retry';

export const TASK_CONTROL_CATALOG: ReadonlyArray<{
  action: TaskControlAction;
  operation: Extract<RuntimeControlOperation, `task.${string}`>;
  label: string;
}> = [
  { action: 'start', operation: 'task.start', label: 'start' },
  { action: 'pause', operation: 'task.pause', label: 'pause' },
  { action: 'resume', operation: 'task.resume', label: 'resume' },
  { action: 'cancel', operation: 'task.cancel', label: 'cancel' },
  { action: 'retry', operation: 'task.retry', label: 'retry' },
];

export type RuntimeInterruptResponseOperation = 'run.resume' | 'run.approval.respond' | 'task.budget_review.respond';

export function runtimeInterruptResponseOperation(
  interrupt: { response_operation?: unknown } | null | undefined,
): RuntimeInterruptResponseOperation | undefined {
  const operation = interrupt?.response_operation;
  return operation === 'run.resume' || operation === 'run.approval.respond' || operation === 'task.budget_review.respond' ? operation : undefined;
}

export type RuntimeOperationAvailability = {
  descriptor?: RuntimeOperationDescriptor;
  visible: boolean;
  enabled: boolean;
  disabledReason?: RuntimeCapabilityDisabledReason;
};

export const RUNTIME_CAPABILITY_UNAVAILABLE: RuntimeCapabilityDisabledReason = 'runtime_capability_unavailable';

export function runtimeOperationAvailability(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  operation: RuntimeControlOperation,
): RuntimeOperationAvailability {
  const descriptor = response?.runtime_available === true ? response.capabilities?.operations?.[operation] : undefined;
  if (!descriptor || !['native', 'emulated', 'conditional'].includes(descriptor.support)
    || !['product', 'runtime'].includes(descriptor.owner) || typeof descriptor.enabled !== 'boolean') {
    return { visible: false, enabled: false };
  }
  return {
    descriptor,
    visible: true,
    enabled: descriptor.enabled,
    disabledReason: descriptor.enabled ? undefined : descriptor.disabled_reason || RUNTIME_CAPABILITY_UNAVAILABLE,
  };
}

export function isRuntimeOperationEnabled(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  operation: RuntimeControlOperation,
): boolean {
  return runtimeOperationAvailability(response, operation).enabled;
}

export function runtimeOperationDisabledReason(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  operation: RuntimeControlOperation,
): RuntimeCapabilityDisabledReason | undefined {
  return runtimeOperationAvailability(response, operation).disabledReason;
}

export function runtimeCapabilityResponseMatchesRun(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  runId: string,
): boolean {
  return response?.resource === 'run' && response.run_id === runId
    && typeof response.runtime_available === 'boolean'
    && (!response.runtime_available || (
      response.capabilities !== null && typeof response.capabilities === 'object'
      && response.capabilities.operations !== null && typeof response.capabilities.operations === 'object'
      && !Array.isArray(response.capabilities.operations)
    ));
}
