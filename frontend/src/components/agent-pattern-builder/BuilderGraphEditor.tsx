import React, { useCallback, useMemo, useState } from 'react';
import AddIcon from '@mui/icons-material/Add';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import {
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import dynamic from 'next/dynamic';
import type { AgentPatternCatalogResponse } from '../../lib/api';
import type { AgentGraphSelection } from '../agent-graph/agent-graph-types';
import type { AgentPatternBuilderState, BuilderEdgeState } from '../../lib/agent-pattern-builder';
import { assembleAgentPatternSpec, canConnectNodes } from '../../lib/agent-pattern-builder';
import type { BuilderSelection, BuilderValidationIssue } from './types';

const AgentGraphCanvas = dynamic(() => import('../agent-graph/AgentGraphCanvas'), { ssr: false });

const edgeLabel = (edge: BuilderEdgeState) => {
  if (edge.conditional) {
    const routes = Object.entries(edge.routes || {})
      .map(([route, target]) => `${route}:${target}`)
      .join(', ');
    return `${edge.from} routes ${routes || '(empty)'}`;
  }
  return `${edge.from} -> ${edge.to}`;
};

const edgeMatches = (candidate: BuilderEdgeState, raw: Record<string, any>) => (
  candidate.from === raw.from
  && candidate.to === raw.to
  && Boolean(candidate.conditional) === Boolean(raw.conditional)
  && JSON.stringify(candidate.routes || {}) === JSON.stringify(raw.routes || {})
);

export default function BuilderGraphEditor({
  catalog,
  state,
  selection,
  validationIssues,
  disabled,
  onSelectionChange,
  onAddEdge,
}: {
  catalog: AgentPatternCatalogResponse;
  state: AgentPatternBuilderState;
  selection: BuilderSelection;
  validationIssues: BuilderValidationIssue[];
  disabled?: boolean;
  onSelectionChange: (selection: BuilderSelection) => void;
  onAddEdge: (edge: BuilderEdgeState) => void;
}) {
  const [source, setSource] = useState('START');
  const [target, setTarget] = useState(state.nodes[0]?.id || 'END');
  const spec = useMemo(() => assembleAgentPatternSpec(state), [state]);
  const compatibility = canConnectNodes(catalog, state, source, target);
  const issueCountForSelection = useCallback((targetSelection: BuilderSelection) => (
    validationIssues.filter((issue) => {
      if (!issue.selection || !targetSelection || issue.selection.kind !== targetSelection.kind) return false;
      if (issue.selection.kind === 'node' && targetSelection.kind === 'node') {
        return issue.selection.nodeId === targetSelection.nodeId;
      }
      if (issue.selection.kind === 'edge' && targetSelection.kind === 'edge') {
        return issue.selection.edgeIndex === targetSelection.edgeIndex;
      }
      return false;
    }).length
  ), [validationIssues]);

  const handleGraphSelection = useCallback((graphSelection: AgentGraphSelection) => {
    if (!graphSelection) {
      onSelectionChange(null);
      return;
    }
    if (graphSelection.kind === 'node') {
      onSelectionChange({ kind: 'node', nodeId: graphSelection.node.id });
      return;
    }
    const raw = graphSelection.edge.raw || {};
    const edgeIndex = (spec.config.graph?.edges || []).findIndex((edge) => edgeMatches(edge as BuilderEdgeState, raw));
    if (edgeIndex >= 0) {
      onSelectionChange({ kind: 'edge', edgeIndex });
    }
  }, [onSelectionChange, spec.config.graph?.edges]);

  const handleAddEdge = () => {
    if (!compatibility.ok) return;
    onAddEdge({ from: source, to: target });
  };

  return (
    <Box sx={{ display: 'grid', gridTemplateRows: 'minmax(0, 1fr) auto', gap: 1, minHeight: 0 }}>
      <Box sx={{ minHeight: 0 }}>
        <AgentGraphCanvas
          resolvedSpec={spec}
          nodeCatalog={catalog.node_catalog}
          mode="builder"
          showInspector={false}
          onSelectionChange={handleGraphSelection}
        />
      </Box>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', lg: 'minmax(0, 1fr) minmax(280px, 360px)' },
          gap: 1,
          minHeight: 0,
        }}
      >
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1, minWidth: 0 }}>
          <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.75, fontWeight: 700 }}>
            <AccountTreeIcon fontSize="small" /> Graph Elements
          </Typography>
          <Divider sx={{ my: 1 }} />
          <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', rowGap: 0.75 }}>
            {state.nodes.map((node) => (
              <Box key={node.id} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.35 }}>
                <Chip
                  clickable
                  color={selection?.kind === 'node' && selection.nodeId === node.id ? 'primary' : 'default'}
                  variant={selection?.kind === 'node' && selection.nodeId === node.id ? 'filled' : 'outlined'}
                  label={`${node.id} · ${node.type}`}
                  onClick={() => onSelectionChange({ kind: 'node', nodeId: node.id })}
                />
                {issueCountForSelection({ kind: 'node', nodeId: node.id }) > 0 ? (
                  <Chip size="small" color="error" label={issueCountForSelection({ kind: 'node', nodeId: node.id })} />
                ) : null}
              </Box>
            ))}
          </Stack>
          <Stack direction="row" spacing={0.75} sx={{ flexWrap: 'wrap', rowGap: 0.75, mt: 1 }}>
            {state.edges.map((edge, index) => (
              <Box key={`${edge.from}-${edge.to || 'routes'}-${index}`} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.35 }}>
                <Chip
                  clickable
                  color={selection?.kind === 'edge' && selection.edgeIndex === index ? 'primary' : 'default'}
                  variant={selection?.kind === 'edge' && selection.edgeIndex === index ? 'filled' : 'outlined'}
                  label={edgeLabel(edge)}
                  onClick={() => onSelectionChange({ kind: 'edge', edgeIndex: index })}
                />
                {issueCountForSelection({ kind: 'edge', edgeIndex: index }) > 0 ? (
                  <Chip size="small" color="error" label={issueCountForSelection({ kind: 'edge', edgeIndex: index })} />
                ) : null}
              </Box>
            ))}
          </Stack>
        </Box>
        <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            Add Sequential Edge
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 1, mt: 1, alignItems: 'center' }}>
            <FormControl size="small" disabled={disabled}>
              <InputLabel id="builder-edge-source-label">Source</InputLabel>
              <Select
                labelId="builder-edge-source-label"
                label="Source"
                value={source}
                onChange={(event: SelectChangeEvent) => setSource(event.target.value)}
              >
                <MenuItem value="START">START</MenuItem>
                {state.nodes.map((node) => (
                  <MenuItem key={node.id} value={node.id}>{node.id}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" disabled={disabled}>
              <InputLabel id="builder-edge-target-label">Target</InputLabel>
              <Select
                labelId="builder-edge-target-label"
                label="Target"
                value={target}
                onChange={(event: SelectChangeEvent) => setTarget(event.target.value)}
              >
                {state.nodes.map((node) => (
                  <MenuItem key={node.id} value={node.id}>{node.id}</MenuItem>
                ))}
                <MenuItem value="END">END</MenuItem>
              </Select>
            </FormControl>
            <Button
              size="small"
              variant="contained"
              startIcon={<AddIcon />}
              disabled={disabled || !compatibility.ok}
              onClick={handleAddEdge}
              sx={{ borderRadius: 1, whiteSpace: 'nowrap' }}
            >
              Add
            </Button>
          </Box>
          {disabled ? (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
              Authoring is disabled; graph edits are read-only.
            </Typography>
          ) : !compatibility.ok ? (
            <Typography variant="caption" color="error" sx={{ display: 'block', mt: 0.75 }}>
              {compatibility.reason}
            </Typography>
          ) : null}
        </Box>
      </Box>
    </Box>
  );
}
