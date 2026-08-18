import type {
  MemoryChangeReceipt,
  MemoryCuratorOperation,
  MemoryScopeType,
} from './api';

export const memoryScopeLabel = (scope?: MemoryScopeType | string | null, threadLabel = 'Thread') => {
  if (scope === 'thread') return threadLabel;
  if (scope === 'project') return 'Project';
  if (scope === 'user') return 'Global';
  return 'Stored';
};

export const memoryWorkspaceTitle = (scope?: MemoryScopeType | string | null) => {
  if (scope === 'thread') return 'Thread memory';
  if (scope === 'project') return 'Project memory';
  return 'Global memory';
};

export const formatMemoryTimestamp = (value?: string | null) => {
  if (!value) return 'Unknown';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

export const memoryIndexStatusColor = (status: string): 'default' | 'success' | 'warning' | 'error' | 'info' => {
  if (status === 'indexed') return 'success';
  if (status === 'failed') return 'error';
  if (status === 'indexing') return 'info';
  if (status === 'pending') return 'warning';
  return 'default';
};

export const memoryRecallReasonLabel = (reason?: string | null) => {
  if (reason === 'memory_disabled') return 'Memory recall off';
  if (reason === 'project_opt_out') return 'Project recall off';
  if (reason === 'thread_opt_out') return 'Thread recall off';
  if (reason === 'not_requested') return 'Recall off';
  return 'Recall off';
};

export const memoryOperationLabel = (operation: MemoryCuratorOperation) => {
  const scope = memoryScopeLabel(operation.scope_type);
  return `${operation.action.toUpperCase()} ${scope || ''}`.trim();
};

export const memoryReceiptMessage = (receipt: MemoryChangeReceipt) => {
  const result = receipt.result_memory_id ? ` (${receipt.result_memory_id})` : '';
  if (receipt.action === 'move') {
    return `Moved the memory from ${memoryScopeLabel(receipt.source_scope?.scope_type)} to ${memoryScopeLabel(receipt.destination_scope?.scope_type)}${result}.`;
  }
  if (receipt.action === 'delete') return `Deleted the ${memoryScopeLabel(receipt.source_scope?.scope_type)} memory.`;
  if (receipt.action === 'set_overrides') return `Updated the memory's override relationships${result}.`;
  if (receipt.action === 'update') return `Updated the ${memoryScopeLabel(receipt.destination_scope?.scope_type)} memory${result}.`;
  return `Created the ${memoryScopeLabel(receipt.destination_scope?.scope_type)} memory${result}.`;
};
