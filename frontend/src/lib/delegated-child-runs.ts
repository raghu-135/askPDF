import type { AgentRunDetails } from './api';

export type DelegatedChildRunRef = {
  run_id: string;
  workflow_id?: string | null;
  order?: number;
};

export function delegatedChildRunsFromTechnical(
  technical: Record<string, unknown> | null | undefined,
): DelegatedChildRunRef[] {
  if (!technical || typeof technical !== 'object') return [];
  const raw = technical.delegated_child_runs;
  if (Array.isArray(raw)) {
    const rows: DelegatedChildRunRef[] = [];
    raw.forEach((item, index) => {
      if (!item || typeof item !== 'object') return;
      const row = item as Record<string, unknown>;
      const runId = String(row.run_id || '').trim();
      if (!runId) return;
      const workflowId = String(row.workflow_id || '').trim();
      rows.push({
        run_id: runId,
        workflow_id: workflowId || undefined,
        order: typeof row.order === 'number' ? row.order : index + 1,
      });
    });
    return rows;
  }
  const legacyRunId = String(technical.child_run_id || technical.delegate_child_run_id || '').trim();
  if (!legacyRunId) return [];
  const legacyWorkflowId = String(technical.child_workflow_id || technical.delegate_child_workflow_id || '').trim();
  return [{ run_id: legacyRunId, workflow_id: legacyWorkflowId || undefined, order: 1 }];
}

function technicalDetailsFromRun(runDetails: AgentRunDetails): Record<string, unknown> | undefined {
  const debug = runDetails.debug;
  if (debug?.technical_details && typeof debug.technical_details === 'object') {
    return debug.technical_details as Record<string, unknown>;
  }
  const trace = debug?.trace as Record<string, unknown> | undefined;
  if (trace?.technical_details && typeof trace.technical_details === 'object') {
    return trace.technical_details as Record<string, unknown>;
  }
  const legacy = (runDetails as { debug_trace?: { technical_details?: Record<string, unknown> } }).debug_trace;
  if (legacy?.technical_details && typeof legacy.technical_details === 'object') {
    return legacy.technical_details;
  }
  return undefined;
}

export function delegatedChildRunsFromRunDetails(runDetails: AgentRunDetails | null | undefined): DelegatedChildRunRef[] {
  if (!runDetails) return [];
  return delegatedChildRunsFromTechnical(technicalDetailsFromRun(runDetails));
}

export function delegatedChildRunLinkLabel(workflowId?: string | null): string {
  const id = String(workflowId || '').trim();
  return id ? `Delegated: ${id}` : 'Delegated child';
}