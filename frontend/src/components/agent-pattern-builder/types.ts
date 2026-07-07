export type BuilderSelection =
  | { kind: 'node'; nodeId: string }
  | { kind: 'edge'; edgeIndex: number }
  | null;

