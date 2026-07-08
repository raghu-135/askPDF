import React, { useMemo, useState } from 'react';
import DeleteIcon from '@mui/icons-material/Delete';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import PersonAddAltIcon from '@mui/icons-material/PersonAddAlt';
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
  TextField,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import type { AgentWorkflowCatalogResponse } from '../../lib/api';
import type { AgentWorkflowBuilderState, BuilderEdgeState, BuilderNodeState } from '../../lib/agent-workflow-builder';
import {
  canConnectNodes,
  getAllowedRouteFunctionsForNode,
  getAllowedToolContractsForNode,
  getRouteLabelsForFunction,
} from '../../lib/agent-workflow-builder';
import type { BuilderSelection } from './types';

const asArrayValue = (value: unknown): string[] => (
  Array.isArray(value) ? value.map(String) : String(value || '').split(',').filter(Boolean)
);

const nodeLabel = (node?: BuilderNodeState) => (
  node ? `${node.id} · ${node.type}` : ''
);

const targetOptionsForSource = (
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  sourceId: string,
) => [
  ...state.nodes.map((node) => node.id),
  'END',
].filter((targetId) => targetId !== sourceId && canConnectNodes(catalog, state, sourceId, targetId).ok);

const sourceOptionsForTarget = (
  catalog: AgentWorkflowCatalogResponse,
  state: AgentWorkflowBuilderState,
  targetId: string,
) => [
  'START',
  ...state.nodes.map((node) => node.id),
].filter((sourceId) => sourceId !== targetId && canConnectNodes(catalog, state, sourceId, targetId).ok);

function NodeInspector({
  catalog,
  state,
  node,
  disabled,
  onUpdateNode,
  onRemoveNode,
  onAddHitlGate,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  node: BuilderNodeState;
  disabled?: boolean;
  onUpdateNode: (nodeId: string, patch: Partial<BuilderNodeState>) => void;
  onRemoveNode: (nodeId: string) => void;
  onAddHitlGate: (targetNodeId: string) => void;
}) {
  const entry = catalog.node_catalog[node.type];
  const allowedTools = getAllowedToolContractsForNode(catalog, node.type);
  const selectedToolIds = node.tool_contract_ids || [];
  const capabilities = entry?.capabilities || [];
  const isHitl = node.type === 'hitl_gate';
  const selectedActions = node.hitl?.allowed_actions || ['approve', 'reject', 'continue_without'];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Node Inspector
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {nodeLabel(node)}
        </Typography>
      </Box>
      <Divider />
      <TextField
        size="small"
        label="Node ID"
        value={node.id}
        slotProps={{ input: { readOnly: true } }}
      />
      <TextField
        size="small"
        label="Label"
        value={node.label || ''}
        disabled={disabled}
        onChange={(event) => onUpdateNode(node.id, { label: event.target.value })}
      />
      <TextField
        size="small"
        label="Description"
        value={node.description || ''}
        multiline
        minRows={2}
        disabled={disabled}
        onChange={(event) => onUpdateNode(node.id, { description: event.target.value })}
      />
      <Box>
        <Typography variant="caption" sx={{ display: 'block', fontWeight: 700, mb: 0.5 }}>
          Catalog Facts
        </Typography>
        <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', rowGap: 0.5 }}>
          <Chip size="small" variant="outlined" label={entry?.display_name || node.type} />
          {entry?.category ? <Chip size="small" variant="outlined" label={entry.category} /> : null}
          {typeof entry?.max_instances === 'number' ? <Chip size="small" variant="outlined" label={`max ${entry.max_instances}`} /> : null}
          {capabilities.map((capability) => (
            <Chip key={capability} size="small" variant="outlined" label={capability} />
          ))}
        </Stack>
        {entry?.context_policy ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
            Context: {entry.context_policy.mode} / {entry.context_policy.input_budget} / {entry.context_policy.output_budget}
          </Typography>
        ) : null}
        {entry?.observability ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
            Observability: {entry.observability.span_kind} / {entry.observability.event_prefix}
          </Typography>
        ) : null}
      </Box>
      <FormControl size="small" disabled={disabled || allowedTools.length === 0}>
        <InputLabel id={`node-tools-${node.id}`}>Tool contracts</InputLabel>
        <Select<string[]>
          labelId={`node-tools-${node.id}`}
          multiple
          label="Tool contracts"
          value={selectedToolIds}
          onChange={(event) => {
            const next = asArrayValue(event.target.value);
            onUpdateNode(node.id, { tool_contract_ids: next.length ? next : undefined });
          }}
          renderValue={(selected) => (selected as string[]).join(', ')}
        >
          {allowedTools.map((contract) => (
            <MenuItem key={contract.id} value={contract.id}>
              {contract.display_name || contract.id}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      {isHitl ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>
            HITL Gate
          </Typography>
          <TextField
            size="small"
            label="Title"
            value={node.hitl?.title || ''}
            disabled={disabled}
            onChange={(event) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), title: event.target.value } })}
          />
          <TextField
            size="small"
            label="Prompt"
            value={node.hitl?.prompt || node.hitl?.body || ''}
            multiline
            minRows={2}
            disabled={disabled}
            onChange={(event) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), prompt: event.target.value, body: event.target.value } })}
          />
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
            <FormControl size="small" disabled={disabled}>
              <InputLabel id={`hitl-mode-${node.id}`}>Mode</InputLabel>
              <Select
                labelId={`hitl-mode-${node.id}`}
                label="Mode"
                value={node.hitl?.mode || 'approval'}
                onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), mode: event.target.value } })}
              >
                <MenuItem value="approval">Approval</MenuItem>
                <MenuItem value="review">Review</MenuItem>
                <MenuItem value="choice">Choice</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" disabled={disabled}>
              <InputLabel id={`hitl-phase-${node.id}`}>Phase</InputLabel>
              <Select
                labelId={`hitl-phase-${node.id}`}
                label="Phase"
                value={node.hitl?.phase || 'before'}
                onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), phase: event.target.value } })}
              >
                <MenuItem value="before">Before</MenuItem>
                <MenuItem value="after">After</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`hitl-actions-${node.id}`}>Allowed actions</InputLabel>
            <Select<string[]>
              labelId={`hitl-actions-${node.id}`}
              multiple
              label="Allowed actions"
              value={selectedActions}
              onChange={(event) => {
                const next = asArrayValue(event.target.value);
                onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), allowed_actions: next } });
              }}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              {['approve', 'approve_selected', 'reject', 'edit', 'continue_without'].map((action) => (
                <MenuItem key={action} value={action}>{action}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`hitl-default-${node.id}`}>Default action</InputLabel>
            <Select
              labelId={`hitl-default-${node.id}`}
              label="Default action"
              value={node.hitl?.default_action || 'continue_without'}
              onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), default_action: event.target.value } })}
            >
              {selectedActions.map((action) => (
                <MenuItem key={action} value={action}>{action}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      ) : (
        <Button
          size="small"
          variant="outlined"
          startIcon={<PersonAddAltIcon />}
          onClick={() => onAddHitlGate(node.id)}
          disabled={disabled || !catalog.node_catalog.hitl_gate}
          sx={{ borderRadius: 1 }}
        >
          Add HITL Gate
        </Button>
      )}
      <Button
        size="small"
        color="error"
        variant="outlined"
        startIcon={<DeleteIcon />}
        onClick={() => onRemoveNode(node.id)}
        disabled={disabled}
        sx={{ borderRadius: 1 }}
      >
        Remove Node
      </Button>
    </Box>
  );
}

function EdgeInspector({
  catalog,
  state,
  edge,
  edgeIndex,
  disabled,
  onUpdateEdge,
  onRemoveEdge,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  edge: BuilderEdgeState;
  edgeIndex: number;
  disabled?: boolean;
  onUpdateEdge: (edgeIndex: number, patch: Partial<BuilderEdgeState>) => void;
  onRemoveEdge: (edgeIndex: number) => void;
}) {
  const sourceNode = state.nodes.find((node) => node.id === edge.from);
  const sourceType = sourceNode?.type;
  const routeFns = sourceType ? getAllowedRouteFunctionsForNode(catalog, sourceType) : [];
  const currentRouteFn = edge.route_fn || routeFns[0] || '';
  const routeLabels = currentRouteFn ? getRouteLabelsForFunction(catalog, currentRouteFn) : [];
  const currentRoutes = edge.routes || {};
  const sourceOptions = sourceOptionsForTarget(catalog, state, edge.to || 'END');
  const targetOptions = targetOptionsForSource(catalog, state, edge.from);
  const compatibility = edge.conditional
    ? { ok: true }
    : canConnectNodes(catalog, state, edge.from, edge.to || '');
  const [routeToAdd, setRouteToAdd] = useState('');
  const [routeTarget, setRouteTarget] = useState(targetOptions[0] || 'END');

  const visibleRouteLabels = useMemo(() => {
    const existing = Object.keys(currentRoutes);
    if (routeLabels === null) {
      const hitlActions = sourceNode?.hitl?.allowed_actions || [];
      return Array.from(new Set([...existing, ...hitlActions]));
    }
    return existing;
  }, [currentRoutes, routeLabels, sourceNode?.hitl?.allowed_actions]);

  const missingRouteLabels = useMemo(() => {
    const existing = new Set(Object.keys(currentRoutes));
    const possible = routeLabels === null
      ? sourceNode?.hitl?.allowed_actions || []
      : routeLabels || [];
    return possible.filter((label) => !existing.has(label));
  }, [currentRoutes, routeLabels, sourceNode?.hitl?.allowed_actions]);

  const updateRouteTarget = (route: string, target: string) => {
    onUpdateEdge(edgeIndex, {
      routes: {
        ...currentRoutes,
        [route]: target,
      },
    });
  };

  const removeRoute = (route: string) => {
    const next = { ...currentRoutes };
    delete next[route];
    onUpdateEdge(edgeIndex, { routes: next });
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Edge Inspector
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {edge.conditional ? 'Conditional route edge' : 'Sequential edge'}
        </Typography>
      </Box>
      <Divider />
      {edge.conditional ? (
        <>
          <TextField size="small" label="Source" value={edge.from} slotProps={{ input: { readOnly: true } }} />
          <FormControl size="small" disabled={disabled || routeFns.length === 0}>
            <InputLabel id={`edge-route-fn-${edgeIndex}`}>Route function</InputLabel>
            <Select
              labelId={`edge-route-fn-${edgeIndex}`}
              label="Route function"
              value={currentRouteFn}
              onChange={(event: SelectChangeEvent) => {
                const nextRouteFn = event.target.value;
                const labels = getRouteLabelsForFunction(catalog, nextRouteFn);
                const nextRoutes = labels === null
                  ? currentRoutes
                  : Object.fromEntries(Object.entries(currentRoutes).filter(([route]) => labels.includes(route)));
                onUpdateEdge(edgeIndex, { route_fn: nextRouteFn, routes: nextRoutes });
              }}
            >
              {routeFns.map((routeFn) => (
                <MenuItem key={routeFn} value={routeFn}>{routeFn}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Typography variant="caption" sx={{ fontWeight: 700 }}>
              Route Targets
            </Typography>
            {visibleRouteLabels.map((route) => (
              <Box key={route} sx={{ display: 'grid', gridTemplateColumns: 'minmax(80px, 120px) 1fr auto', gap: 1, alignItems: 'center' }}>
                <Typography variant="caption" sx={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {route}
                </Typography>
                <FormControl size="small" disabled={disabled}>
                  <InputLabel id={`route-target-${edgeIndex}-${route}`}>Target</InputLabel>
                  <Select
                    labelId={`route-target-${edgeIndex}-${route}`}
                    label="Target"
                    value={currentRoutes[route] || targetOptions[0] || 'END'}
                    onChange={(event: SelectChangeEvent) => updateRouteTarget(route, event.target.value)}
                  >
                    {targetOptions.map((target) => (
                      <MenuItem key={target} value={target}>{target}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button size="small" color="error" disabled={disabled} onClick={() => removeRoute(route)}>
                  Remove
                </Button>
              </Box>
            ))}
            {missingRouteLabels.length > 0 ? (
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 1, alignItems: 'center' }}>
                <FormControl size="small" disabled={disabled}>
                  <InputLabel id={`add-route-label-${edgeIndex}`}>Route</InputLabel>
                  <Select
                    labelId={`add-route-label-${edgeIndex}`}
                    label="Route"
                    value={routeToAdd}
                    onChange={(event: SelectChangeEvent) => setRouteToAdd(event.target.value)}
                  >
                    {missingRouteLabels.map((route) => (
                      <MenuItem key={route} value={route}>{route}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" disabled={disabled}>
                  <InputLabel id={`add-route-target-${edgeIndex}`}>Target</InputLabel>
                  <Select
                    labelId={`add-route-target-${edgeIndex}`}
                    label="Target"
                    value={routeTarget}
                    onChange={(event: SelectChangeEvent) => setRouteTarget(event.target.value)}
                  >
                    {targetOptions.map((target) => (
                      <MenuItem key={target} value={target}>{target}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  size="small"
                  variant="outlined"
                  disabled={disabled || !routeToAdd}
                  onClick={() => {
                    updateRouteTarget(routeToAdd, routeTarget);
                    setRouteToAdd('');
                  }}
                  sx={{ borderRadius: 1 }}
                >
                  Add
                </Button>
              </Box>
            ) : null}
          </Box>
        </>
      ) : (
        <>
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`edge-source-${edgeIndex}`}>Source</InputLabel>
            <Select
              labelId={`edge-source-${edgeIndex}`}
              label="Source"
              value={edge.from}
              onChange={(event: SelectChangeEvent) => {
                const nextSource = event.target.value;
                const targets = targetOptionsForSource(catalog, state, nextSource);
                onUpdateEdge(edgeIndex, { from: nextSource, to: targets.includes(edge.to || '') ? edge.to : targets[0] });
              }}
            >
              {sourceOptions.map((source) => (
                <MenuItem key={source} value={source}>{source}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`edge-target-${edgeIndex}`}>Target</InputLabel>
            <Select
              labelId={`edge-target-${edgeIndex}`}
              label="Target"
              value={edge.to || ''}
              onChange={(event: SelectChangeEvent) => onUpdateEdge(edgeIndex, { to: event.target.value })}
            >
              {targetOptions.map((target) => (
                <MenuItem key={target} value={target}>{target}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Chip
            size="small"
            color={compatibility.ok ? 'success' : 'error'}
            label={compatibility.ok ? 'Compatible' : compatibility.reason}
            sx={{ alignSelf: 'flex-start' }}
          />
        </>
      )}
      <Button
        size="small"
        color="error"
        variant="outlined"
        startIcon={<DeleteIcon />}
        onClick={() => onRemoveEdge(edgeIndex)}
        disabled={disabled}
        sx={{ borderRadius: 1 }}
      >
        Remove Edge
      </Button>
    </Box>
  );
}

export default function BuilderInspector({
  catalog,
  state,
  selection,
  disabled,
  onUpdateNode,
  onUpdateEdge,
  onRemoveNode,
  onRemoveEdge,
  onAddHitlGate,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  selection: BuilderSelection;
  disabled?: boolean;
  onUpdateNode: (nodeId: string, patch: Partial<BuilderNodeState>) => void;
  onUpdateEdge: (edgeIndex: number, patch: Partial<BuilderEdgeState>) => void;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (edgeIndex: number) => void;
  onAddHitlGate: (targetNodeId: string) => void;
}) {
  const selectedNode = selection?.kind === 'node'
    ? state.nodes.find((node) => node.id === selection.nodeId)
    : undefined;
  const selectedEdge = selection?.kind === 'edge' ? state.edges[selection.edgeIndex] : undefined;

  if (selectedNode) {
    return (
      <NodeInspector
        catalog={catalog}
        state={state}
        node={selectedNode}
        disabled={disabled}
        onUpdateNode={onUpdateNode}
        onRemoveNode={onRemoveNode}
        onAddHitlGate={onAddHitlGate}
      />
    );
  }

  if (selectedEdge && selection?.kind === 'edge') {
    return (
      <EdgeInspector
        catalog={catalog}
        state={state}
        edge={selectedEdge}
        edgeIndex={selection.edgeIndex}
        disabled={disabled}
        onUpdateEdge={onUpdateEdge}
        onRemoveEdge={onRemoveEdge}
      />
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0 }}>
      <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.75, fontWeight: 700 }}>
        <FactCheckIcon fontSize="small" /> Inspector
      </Typography>
      <Typography variant="body2" color="text.secondary">
        Select a node or edge from the canvas or graph element list.
      </Typography>
    </Box>
  );
}
