import type {
  AgentRuntimeCapabilityResponse,
  RuntimeOperationDescriptor,
} from './api';

export type RuntimeControlOperation =
  | 'run.cancel'
  | 'run.resume'
  | 'run.approval.respond'
  | 'run.send_followup'
  | 'run.interrupt_with_input'
  | 'run.steer_live'
  | 'run.update_state'
  | 'task.start'
  | 'task.pause'
  | 'task.resume'
  | 'task.cancel'
  | 'task.retry';

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

export type RuntimeInterruptResponseOperation = 'run.resume' | 'run.approval.respond';

export function runtimeInterruptResponseOperation(
  interrupt: { response_operation?: unknown } | null | undefined,
): RuntimeInterruptResponseOperation | undefined {
  const operation = interrupt?.response_operation;
  return operation === 'run.resume' || operation === 'run.approval.respond' ? operation : undefined;
}

export type RuntimeOperationAvailability = {
  descriptor?: RuntimeOperationDescriptor;
  visible: boolean;
  enabled: boolean;
  disabledReason?: string;
};

export function runtimeOperationAvailability(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  operation: RuntimeControlOperation,
): RuntimeOperationAvailability {
  const descriptor = response?.available ? response.capabilities?.operations[operation] : undefined;
  if (!descriptor || descriptor.support === 'unsupported') {
    return { visible: false, enabled: false };
  }
  return {
    descriptor,
    visible: true,
    enabled: descriptor.enabled,
    disabledReason: descriptor.enabled ? undefined : descriptor.disabled_reason || 'runtime_capability_unavailable',
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
): string | undefined {
  return runtimeOperationAvailability(response, operation).disabledReason;
}

export function isCurrentRuntimeCapabilityRequest(requestId: number, currentRequestId: number): boolean {
  return requestId === currentRequestId;
}

export function runtimeCapabilityResponseMatchesRun(
  response: AgentRuntimeCapabilityResponse | null | undefined,
  runId: string,
): boolean {
  return response?.resource === 'run' && response.run_id === runId;
}
