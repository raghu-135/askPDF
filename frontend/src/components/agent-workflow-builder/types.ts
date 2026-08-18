export type BuilderSelection =
  | { kind: 'node'; nodeId: string }
  | { kind: 'edge'; edgeIndex: number }
  | null;

export interface BuilderValidationIssue {
  id: string;
  severity: 'error' | 'warning';
  message: string;
  selection: BuilderSelection;
  code?: string;
  fix?: { kind: string; [key: string]: any } | null;
}
