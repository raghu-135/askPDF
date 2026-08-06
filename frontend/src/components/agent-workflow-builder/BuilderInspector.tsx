import React, { useMemo, useState } from 'react';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import PersonAddAltIcon from '@mui/icons-material/PersonAddAlt';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import {
  Box,
  Chip,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import type { AgentWorkflowCatalogResponse } from '../../lib/api';
import type { AgentWorkflowBuilderState, BuilderEdgeState, BuilderNodeState } from '../../lib/agent-workflow-builder';
import { AgentRunResumeAction, BuiltinAgentNodeType, HitlMode, HitlPhase } from '../../lib/enums';
import {
  canConnectNodes,
  getImmediateSuccessorIds,
  getAllowedRouteFunctionsForNode,
  getAllowedToolContractsForNode,
  getRouteLabelsForFunction,
} from '../../lib/agent-workflow-builder';
import type { BuilderSelection } from './types';
import { JsonPreview } from '../agent-graph/AgentGraphInspectorPrimitives';

const asArrayValue = (value: unknown): string[] => (
  Array.isArray(value) ? value.map(String) : String(value || '').split(',').filter(Boolean)
);

const nodeLabel = (node?: BuilderNodeState) => (
  node ? `${node.id} · ${node.type}` : ''
);

const sectionLabelSx = {
  fontWeight: 700,
  color: 'text.secondary',
  textTransform: 'uppercase',
  letterSpacing: 0.3,
  fontSize: '0.68rem',
} as const;

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
  onUpdateHitlBypass,
  onRemoveNode,
  onAddHitlGate,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  node: BuilderNodeState;
  disabled?: boolean;
  onUpdateNode: (nodeId: string, patch: Partial<BuilderNodeState>) => void;
  onUpdateHitlBypass: (gateNodeId: string, targetId?: string) => void;
  onRemoveNode: (nodeId: string) => void;
  onAddHitlGate: (targetNodeId: string) => void;
}) {
  const entry = catalog.node_catalog[node.type];
  const allowedTools = getAllowedToolContractsForNode(catalog, node.type);
  const selectedToolIds = node.tool_contract_ids || [];
  const capabilities = entry?.capabilities || [];
  const isHitl = node.type === BuiltinAgentNodeType.HitlGate;
  const selectedActions = node.hitl?.allowed_actions || [AgentRunResumeAction.Approve, AgentRunResumeAction.Reject];
  const hitlRouteEdge = isHitl
    ? state.edges.find((edge) => edge.from === node.id && edge.conditional)
    : undefined;
  const hitlRoutes = hitlRouteEdge?.routes || node.hitl?.routes || {};
  const approvedTarget = hitlRoutes[AgentRunResumeAction.Approve];
  const bypassTargets = approvedTarget ? getImmediateSuccessorIds(state, approvedTarget) : [];
  const continueWithoutTarget = hitlRoutes[AgentRunResumeAction.ContinueWithout] || '';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto auto', gap: 0.5, alignItems: 'start' }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="subtitle2" noWrap sx={{ fontWeight: 700 }}>
            Node Inspector
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
            {nodeLabel(node)}
          </Typography>
        </Box>
        {!isHitl ? (
          <Tooltip title="Add HITL Gate">
            <span>
              <IconButton
                size="small"
                color="primary"
                onClick={() => onAddHitlGate(node.id)}
                disabled={disabled || !catalog.node_catalog[BuiltinAgentNodeType.HitlGate]}
                aria-label="Add HITL Gate"
              >
                <PersonAddAltIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        ) : null}
        <Tooltip title="Remove node">
          <span>
            <IconButton
              size="small"
              color="error"
              onClick={() => onRemoveNode(node.id)}
              disabled={disabled}
              aria-label="Remove node"
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      {entry?.ui?.summary ? (
        <Box sx={{ px: 1, py: 0.75, borderRadius: 1, bgcolor: 'action.hover' }}>
          <Typography variant="body2" sx={{ fontWeight: 700 }}>{entry.display_name || node.type}</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>{entry.ui.summary}</Typography>
          {entry.ui.use_when ? <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>{entry.ui.use_when}</Typography> : null}
        </Box>
      ) : null}
      <TextField
        size="small"
        label="Purpose"
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
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.65 }}>
        <Typography variant="caption" sx={sectionLabelSx}>Catalog details</Typography>
        <TextField
          fullWidth
          size="small"
          label="Node ID"
          value={node.id}
          slotProps={{ input: { readOnly: true } }}
        />
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
        <JsonPreview value={node} maxHeight={220} />
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
                value={node.hitl?.mode || HitlMode.Approval}
                onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), mode: event.target.value } })}
              >
                <MenuItem value={HitlMode.Approval}>Approval</MenuItem>
                <MenuItem value={HitlMode.Review}>Review</MenuItem>
                <MenuItem value={HitlMode.Choice}>Choice</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" disabled={disabled}>
              <InputLabel id={`hitl-phase-${node.id}`}>Phase</InputLabel>
              <Select
                labelId={`hitl-phase-${node.id}`}
                label="Phase"
                value={node.hitl?.phase || HitlPhase.Before}
                onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), phase: event.target.value } })}
              >
                <MenuItem value={HitlPhase.Before}>Before</MenuItem>
                <MenuItem value={HitlPhase.After}>After</MenuItem>
              </Select>
            </FormControl>
          </Box>
          <FormControl size="small" disabled={disabled || !approvedTarget || bypassTargets.length === 0}>
            <InputLabel id={`hitl-bypass-${node.id}`}>Continue without target</InputLabel>
            <Select
              labelId={`hitl-bypass-${node.id}`}
              label="Continue without target"
              value={continueWithoutTarget}
              onChange={(event: SelectChangeEvent) => (
                onUpdateHitlBypass(node.id, event.target.value || undefined)
              )}
            >
              <MenuItem value=""><em>Disabled</em></MenuItem>
              {bypassTargets.map((targetId) => (
                <MenuItem key={targetId} value={targetId}>
                  {targetId === 'END'
                    ? 'End'
                    : state.nodes.find((candidate) => candidate.id === targetId)?.label
                      || catalog.node_catalog[state.nodes.find((candidate) => candidate.id === targetId)?.type || '']?.display_name
                      || targetId}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {bypassTargets.length > 1 && !continueWithoutTarget ? (
            <Typography variant="caption" color="text.secondary">
              Continue without is disabled until a bypass target is selected.
            </Typography>
          ) : null}
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`hitl-actions-${node.id}`}>Allowed actions</InputLabel>
            <Select<string[]>
              labelId={`hitl-actions-${node.id}`}
              multiple
              label="Allowed actions"
              value={selectedActions}
              onChange={(event) => {
                const next = asArrayValue(event.target.value)
                  .filter((action) => action !== AgentRunResumeAction.ContinueWithout);
                if (continueWithoutTarget) next.push(AgentRunResumeAction.ContinueWithout);
                onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), allowed_actions: next } });
              }}
              renderValue={(selected) => (selected as string[]).join(', ')}
            >
              {Object.values(AgentRunResumeAction)
                .filter((action) => action !== AgentRunResumeAction.ContinueWithout)
                .map((action) => (
                <MenuItem key={action} value={action}>{action}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" disabled={disabled}>
            <InputLabel id={`hitl-default-${node.id}`}>Default action</InputLabel>
            <Select
              labelId={`hitl-default-${node.id}`}
              label="Default action"
              value={node.hitl?.default_action || AgentRunResumeAction.Approve}
              onChange={(event: SelectChangeEvent) => onUpdateNode(node.id, { hitl: { ...(node.hitl || {}), default_action: event.target.value } })}
            >
              {selectedActions.map((action) => (
                <MenuItem key={action} value={action}>{action}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      ) : null}
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
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 0.5, alignItems: 'start' }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography variant="subtitle2" noWrap sx={{ fontWeight: 700 }}>
            Edge Inspector
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap sx={{ display: 'block' }}>
            {edge.conditional ? 'Conditional route edge' : 'Sequential edge'}
          </Typography>
        </Box>
        <Tooltip title="Remove edge">
          <span>
            <IconButton
              size="small"
              color="error"
              onClick={() => onRemoveEdge(edgeIndex)}
              disabled={disabled}
              aria-label="Remove edge"
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
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
                <Tooltip title="Remove route">
                  <span>
                    <IconButton size="small" color="error" disabled={disabled} onClick={() => removeRoute(route)} aria-label={`Remove ${route} route`}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
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
                <Tooltip title="Add route target">
                  <span>
                    <IconButton
                      size="small"
                      color="primary"
                      disabled={disabled || !routeToAdd}
                      onClick={() => {
                        updateRouteTarget(routeToAdd, routeTarget);
                        setRouteToAdd('');
                      }}
                      aria-label="Add route target"
                    >
                      <AddIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
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
    </Box>
  );
}

function ParallelPolicyEditor({
  catalog,
  policy,
  disabled,
  onChange,
}: {
  catalog: AgentWorkflowCatalogResponse;
  policy: Record<string, boolean | number>;
  disabled?: boolean;
  onChange: (patch: Record<string, boolean | number>) => void;
}) {
  const contract = catalog.defaults.parallel_policy;
  if (!contract) return null;
  const numericFields = Object.entries(contract.fields).filter(([, field]) => field.type === 'integer');
  const partialField = contract.fields.continue_on_partial_failure;
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75, pt: 1, mt: 1, borderTop: 1, borderColor: 'divider' }}>
      <Typography variant="caption" sx={sectionLabelSx}>Parallel policy</Typography>
      {numericFields.map(([key, field]) => (
        <Box key={key} sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 0.5 }}>
          <TextField
            size="small"
            type="number"
            label={field.label}
            value={policy[key] ?? field.default}
            disabled={disabled}
            slotProps={{ htmlInput: { min: field.minimum, max: field.maximum, step: field.step || 1 } }}
            helperText={field.unit ? `${field.minimum}–${field.maximum} ${field.unit}` : `${field.minimum}–${field.maximum}`}
            onChange={(event) => {
              const parsed = Number(event.target.value);
              const value = Math.max(Number(field.minimum), Math.min(Number(field.maximum), Number.isFinite(parsed) ? parsed : Number(field.default)));
              onChange({ [key]: value });
            }}
          />
          <Tooltip title={`Reset ${field.label.toLocaleLowerCase()}`}>
            <span><IconButton size="small" disabled={disabled} onClick={() => onChange({ [key]: Number(field.default) })}><RestartAltIcon fontSize="small" /></IconButton></span>
          </Tooltip>
        </Box>
      ))}
      {partialField ? (
        <FormControlLabel
          control={<Switch size="small" checked={Boolean(policy.continue_on_partial_failure)} disabled={disabled} onChange={(_, checked) => onChange({ continue_on_partial_failure: checked })} />}
          label={<Typography variant="caption">{partialField.label}</Typography>}
        />
      ) : null}
    </Box>
  );
}

export default function BuilderInspector({
  catalog,
  state,
  selection,
  disabled,
  onUpdateNode,
  onUpdateHitlBypass,
  onUpdateEdge,
  onRemoveNode,
  onRemoveEdge,
  onAddHitlGate,
  onUpdateSettings,
  onUpdateParallelPolicy,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  selection: BuilderSelection;
  disabled?: boolean;
  onUpdateNode: (nodeId: string, patch: Partial<BuilderNodeState>) => void;
  onUpdateHitlBypass: (gateNodeId: string, targetId?: string) => void;
  onUpdateEdge: (edgeIndex: number, patch: Partial<BuilderEdgeState>) => void;
  onRemoveNode: (nodeId: string) => void;
  onRemoveEdge: (edgeIndex: number) => void;
  onAddHitlGate: (targetNodeId: string) => void;
  onUpdateSettings: (patch: Record<string, any>) => void;
  onUpdateParallelPolicy: (patch: Record<string, boolean | number>) => void;
}) {
  const selectedNode = selection?.kind === 'node'
    ? state.nodes.find((node) => node.id === selection.nodeId)
    : undefined;
  const selectedEdge = selection?.kind === 'edge' ? state.edges[selection.edgeIndex] : undefined;

  let inspector: React.ReactNode;
  if (selectedNode) {
    inspector = (
      <NodeInspector
        catalog={catalog}
        state={state}
        node={selectedNode}
        disabled={disabled}
        onUpdateNode={onUpdateNode}
        onUpdateHitlBypass={onUpdateHitlBypass}
        onRemoveNode={onRemoveNode}
        onAddHitlGate={onAddHitlGate}
      />
    );
  } else if (selectedEdge && selection?.kind === 'edge') {
    inspector = (
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
  } else {
    inspector = <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.85, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.75, fontWeight: 700 }}>
          <FactCheckIcon fontSize="small" /> No selection
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Select a graph item to edit it.
        </Typography>
      </Box>
      <TextField
        size="small"
        type="number"
        label="Max replan attempts"
        value={state.extraConfig?.replans || 1}
        disabled={disabled}
        slotProps={{
          htmlInput: { min: 1, max: 5 },
          input: {
            endAdornment: (
              <Tooltip title="Bounds retry loops for workflows that support replanning.">
                <InfoOutlinedIcon fontSize="small" color="action" />
              </Tooltip>
            ),
          },
        }}
        onChange={(event) => onUpdateSettings({ replans: Math.max(1, Math.min(5, Number(event.target.value) || 1)) })}
      />
    </Box>;
  }
  const hasParallelRegion = state.nodes.some((node) => node.type === BuiltinAgentNodeType.ParallelDispatch || node.type === BuiltinAgentNodeType.Aggregator);
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      {inspector}
      {hasParallelRegion && state.parallel_policy ? (
        <ParallelPolicyEditor catalog={catalog} policy={state.parallel_policy} disabled={disabled} onChange={onUpdateParallelPolicy} />
      ) : null}
    </Box>
  );
}
