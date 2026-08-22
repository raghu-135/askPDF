import type {
  AgentRuntimeCapabilityResponse,
  RuntimeOperationDescriptor,
} from './api';

export type RuntimeControlOperation =
  | 'run.cancel'
  | 'run.pause'
  | 'run.resume'
  | 'run.retry'
  | 'run.approval.respond'
  | 'interrupt.respond'
  | 'run.send_followup'
  | 'run.interrupt_with_input'
  | 'run.steer_live'
  | 'run.update_state';

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
