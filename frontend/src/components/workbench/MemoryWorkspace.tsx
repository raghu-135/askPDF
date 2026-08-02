import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  IconButton,
  InputLabel,
  List,
  ListItem,
  MenuItem,
  Select,
  Stack,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import MemoryIcon from '@mui/icons-material/Memory';
import RefreshIcon from '@mui/icons-material/Refresh';
import ReplayIcon from '@mui/icons-material/Replay';
import {
  deleteMemory,
  listEffectiveMemories,
  listMemories,
  listProjects,
  retryMemoryIndex,
  type MemoryRecord,
  type MemoryScopeType,
  type Project,
  type Thread,
} from '../../lib/api';
import {
  memoryConsentStatus,
  memoryScopesForContext,
  filterMemoryRecords,
  isMemoryResultTruncated,
  resolveMemoryScopeTarget,
} from '../../lib/memory-workspace';
import { createCuratorIntent, type MemoryCuratorIntent } from '../../lib/memory-curator';

const MEMORY_LIMIT = 500;

const scopeLabel = (scope: MemoryScopeType) => {
  if (scope === 'thread') return 'This thread';
  if (scope === 'project') return 'Project';
  return 'Global';
};

const formatTimestamp = (value?: string | null) => {
  if (!value) return 'Unknown';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const statusColor = (status: string): 'default' | 'success' | 'warning' | 'error' | 'info' => {
  if (status === 'indexed') return 'success';
  if (status === 'failed') return 'error';
  if (status === 'indexing') return 'info';
  if (status === 'pending') return 'warning';
  return 'default';
};

function MemoryDetails({ memory }: { memory: MemoryRecord }) {
  const sourceRefs = memory.source_refs_json || memory.source_refs || {};
  return (
    <Box component="details" sx={{ mt: 0.75, minWidth: 0 }}>
      <Typography component="summary" variant="caption" color="text.secondary" sx={{ cursor: 'pointer' }}>
        Details
      </Typography>
      <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 0.75 }}>
        <Typography variant="caption">Created: {formatTimestamp(memory.created_at)}</Typography>
        <Typography variant="caption">Updated: {formatTimestamp(memory.updated_at || memory.created_at)}</Typography>
      </Box>
      {Boolean(memory.overrides?.length) && (
        <Box sx={{ mt: 0.75 }}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>Overrides</Typography>
          {memory.overrides.map((item) => <Typography key={item.id} variant="caption" display="block">{scopeLabel(item.scope_type)}: {item.content}</Typography>)}
        </Box>
      )}
      {Boolean(memory.overridden_by?.length) && (
        <Box sx={{ mt: 0.75 }}>
          <Typography variant="caption" sx={{ fontWeight: 700 }}>Overridden by</Typography>
          {memory.overridden_by.map((item) => <Typography key={item.id} variant="caption" display="block">{scopeLabel(item.scope_type)}: {item.content}</Typography>)}
        </Box>
      )}
      {Object.keys(sourceRefs).length > 0 && (
        <Box component="pre" sx={{ m: 0, mt: 0.75, p: 1, maxHeight: 150, overflow: 'auto', bgcolor: 'action.hover', fontSize: '0.72rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {JSON.stringify({ source_refs: sourceRefs }, null, 2)}
        </Box>
      )}
    </Box>
  );
}

export default function MemoryWorkspace({
  activeThread,
  activeProject: projectContext = null,
  projectInventoryVersion = 0,
  curatorRefreshVersion = 0,
  onOpenCurator,
}: {
  activeThread: Thread | null;
  activeProject?: Project | null;
  projectInventoryVersion?: number;
  curatorRefreshVersion?: number;
  onOpenCurator?: (intent: MemoryCuratorIntent) => void;
}) {
  const [scopeType, setScopeType] = useState<MemoryScopeType>(activeThread ? 'thread' : projectContext ? 'project' : 'user');
  const [viewMode, setViewMode] = useState<'effective' | 'stored'>('effective');
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [actionId, setActionId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<MemoryRecord | null>(null);
  const [unavailableCount, setUnavailableCount] = useState(0);
  const [effectiveTruncated, setEffectiveTruncated] = useState(false);

  const availableScopes = useMemo(
    () => activeThread
      ? memoryScopesForContext(true)
      : projectContext ? (['project', 'user'] as MemoryScopeType[]) : memoryScopesForContext(false),
    [activeThread, projectContext],
  );

  useEffect(() => {
    setScopeType(activeThread ? 'thread' : projectContext ? 'project' : 'user');
    setViewMode('effective');
    setQuery('');
  }, [activeThread?.id, projectContext?.id]);

  useEffect(() => {
    let cancelled = false;
    listProjects()
      .then(({ projects: rows }) => {
        if (cancelled) return;
        setProjects(rows);
        setSelectedProjectId((current) => (
          activeThread?.project_id || projectContext?.id || current || rows[0]?.id || ''
        ));
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : 'Unable to load projects.');
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeThread?.id,
    activeThread?.project_id,
    projectContext?.id,
    projectInventoryVersion,
  ]);

  const target = useMemo(
    () => resolveMemoryScopeTarget({
      scopeType,
      thread: activeThread,
      selectedProjectId: projectContext?.id || selectedProjectId,
    }),
    [activeThread, projectContext, scopeType, selectedProjectId],
  );
  const activeProject = projectContext || projects.find((project) => (
    project.id === (activeThread?.project_id || selectedProjectId)
  )) || null;
  const curatorProject = (
    activeThread || projectContext || scopeType === 'project'
  ) ? activeProject : null;
  const consent = memoryConsentStatus({
    scopeType,
    thread: activeThread,
    project: activeProject,
  });

  useEffect(() => {
    let cancelled = false;
    if (viewMode === 'stored' && !target) {
      setMemories([]);
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setLoading(true);
    setLoadError(null);
    const request = viewMode === 'effective'
      ? listEffectiveMemories({
          threadId: activeThread?.id,
          projectId: activeThread ? null : (projectContext?.id || (scopeType === 'project' ? selectedProjectId : null)),
          limit: MEMORY_LIMIT,
        })
      : listMemories(target!.scopeType, target!.scopeId, MEMORY_LIMIT);
    request
      .then((response) => {
        if (cancelled) return;
        setMemories(response.memories);
        const effectiveMeta = response as { unavailable_memory_count?: number; truncated?: boolean };
        setUnavailableCount(effectiveMeta.unavailable_memory_count ?? 0);
        setEffectiveTruncated(effectiveMeta.truncated ?? false);
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : 'Unable to load memory.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeThread?.id, curatorRefreshVersion, projectContext?.id, refreshVersion, scopeType, selectedProjectId, target, viewMode]);

  const filteredMemories = useMemo(
    () => filterMemoryRecords(memories, query),
    [memories, query],
  );

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      setActionId(deleteTarget.id);
      setLoadError(null);
      await deleteMemory(deleteTarget.id);
      setMemories((current) => current.filter((memory) => memory.id !== deleteTarget.id));
      setDeleteTarget(null);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Unable to delete memory.');
    } finally {
      setActionId(null);
    }
  };

  const handleRetry = async (memory: MemoryRecord) => {
    try {
      setActionId(memory.id);
      setLoadError(null);
      const updated = await retryMemoryIndex(memory.id);
      setMemories((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Unable to retry memory indexing.');
      setRefreshVersion((value) => value + 1);
    } finally {
      setActionId(null);
    }
  };

  const rowsAtLimit = viewMode === 'effective'
    ? effectiveTruncated
    : isMemoryResultTruncated(memories.length, MEMORY_LIMIT);

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto auto minmax(0, 1fr)', bgcolor: 'background.default' }}>
      <Box sx={{ px: 2, pt: 1.5, borderBottom: 1, borderColor: 'divider' }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Memory</Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={viewMode}
              onChange={(_, value) => value && setViewMode(value)}
              aria-label="Memory view"
            >
              <ToggleButton value="effective" sx={{ textTransform: 'none', px: 1.25 }}>Effective</ToggleButton>
              <ToggleButton value="stored" sx={{ textTransform: 'none', px: 1.25 }}>Stored</ToggleButton>
            </ToggleButtonGroup>
          </Stack>
          <Stack direction="row" spacing={0.5}>
            <Tooltip title="Refresh">
              <IconButton size="small" onClick={() => setRefreshVersion((value) => value + 1)} disabled={loading}>
                <RefreshIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Add memory">
              <span>
                <IconButton
                  size="small"
                  color="primary"
                  onClick={() => target && onOpenCurator?.(createCuratorIntent({
                    scopeType: target.scopeType,
                    scopeId: target.scopeId,
                    thread: activeThread,
                    project: curatorProject,
                  }))}
                  disabled={!target || !onOpenCurator}
                >
                  <AddIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        </Stack>
      </Box>

      <Box sx={{ px: 2, py: 1.25, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', borderBottom: 1, borderColor: 'divider' }}>
        {(viewMode === 'stored' || (!activeThread && !projectContext)) && (
          <ToggleButtonGroup
            exclusive
            size="small"
            value={scopeType}
            onChange={(_, value) => value && setScopeType(value)}
            aria-label="Memory scope"
          >
            {(viewMode === 'effective' ? availableScopes.filter((scope) => scope !== 'thread') : availableScopes).map((scope) => (
              <ToggleButton key={scope} value={scope} sx={{ textTransform: 'none', minWidth: 86 }}>
                {scopeLabel(scope)}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        )}
        {!activeThread && !projectContext && scopeType === 'project' && (
          <FormControl size="small" sx={{ minWidth: 190, maxWidth: 300 }}>
            <InputLabel>Project</InputLabel>
            <Select value={selectedProjectId} label="Project" onChange={(event) => setSelectedProjectId(String(event.target.value))}>
              {projects.map((project) => <MenuItem key={project.id} value={project.id}>{project.name}</MenuItem>)}
            </Select>
          </FormControl>
        )}
        {viewMode === 'effective' ? (
          <Chip size="small" variant="outlined" color="success" label="Agent-effective view" />
        ) : (
          <Chip
            size="small"
            variant="outlined"
            color={consent.enabled === false ? 'warning' : consent.enabled === true ? 'success' : 'default'}
            label={consent.label}
          />
        )}
        <TextField
          size="small"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter memory"
          sx={{ ml: { sm: 'auto' }, width: { xs: '100%', sm: 220 } }}
        />
      </Box>

      <Box sx={{ minHeight: 0, overflow: 'auto' }}>
        {loadError && <Alert severity="error" onClose={() => setLoadError(null)} sx={{ m: 1.5 }}>{loadError}</Alert>}
        {viewMode === 'effective' && unavailableCount > 0 && (
          <Alert
            severity="warning"
            sx={{ mx: 1.5, mt: 1.5 }}
            action={<Button size="small" onClick={() => setViewMode('stored')}>View stored</Button>}
          >
            {unavailableCount} stored {unavailableCount === 1 ? 'memory is' : 'memories are'} unavailable until indexing succeeds.
          </Alert>
        )}
        {rowsAtLimit && <Alert severity="info" sx={{ mx: 1.5, mt: 1.5 }}>Showing the first {MEMORY_LIMIT} records.</Alert>}
        {loading ? (
          <Box sx={{ height: '100%', minHeight: 180, display: 'grid', placeItems: 'center' }}><CircularProgress size={28} /></Box>
        ) : (
          filteredMemories.length ? (
            <List disablePadding>
              {filteredMemories.map((memory) => {
                const busy = actionId === memory.id;
                return (
                  <ListItem key={memory.id} divider alignItems="flex-start" sx={{ px: 2, py: 1.5, gap: 1.5 }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
                        <Chip size="small" label={scopeLabel(memory.scope_type)} variant="outlined" />
                        <Chip size="small" label={memory.index_status} color={statusColor(memory.index_status)} />
                        {Boolean(memory.overrides?.length) && <Chip size="small" label={`Overrides ${memory.overrides.length}`} variant="outlined" color="info" />}
                        {Boolean(memory.overridden_by?.length) && <Chip size="small" label={`Overridden by ${memory.overridden_by.length}`} variant="outlined" color="warning" />}
                        <Typography variant="caption" color="text.secondary">{memory.embedding_model}</Typography>
                      </Stack>
                      <Typography variant="body2" sx={{ mt: 0.75, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                        {memory.content}
                      </Typography>
                      {memory.index_error && <Alert severity="error" sx={{ mt: 0.75, py: 0 }}>{memory.index_error}</Alert>}
                      <MemoryDetails memory={memory} />
                    </Box>
                    <Stack direction="row" spacing={0.25}>
                      {viewMode === 'stored' && (
                        <Tooltip title="Edit memory">
                          <span>
                            <IconButton
                              size="small"
                              onClick={() => onOpenCurator?.(createCuratorIntent({
                                scopeType: memory.scope_type,
                                scopeId: memory.scope_id,
                                thread: activeThread,
                                project: (
                                  activeThread || projectContext || memory.scope_type === 'project'
                                ) ? activeProject : null,
                                memory,
                              }))}
                              disabled={busy || !onOpenCurator}
                            >
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </span>
                        </Tooltip>
                      )}
                      {viewMode === 'stored' && ['failed', 'pending'].includes(memory.index_status) && (
                        <Tooltip title="Retry indexing">
                          <span><IconButton size="small" onClick={() => void handleRetry(memory)} disabled={busy}><ReplayIcon fontSize="small" /></IconButton></span>
                        </Tooltip>
                      )}
                      {viewMode === 'stored' && (
                        <Tooltip title="Delete memory">
                          <span><IconButton size="small" color="error" onClick={() => setDeleteTarget(memory)} disabled={busy}><DeleteIcon fontSize="small" /></IconButton></span>
                        </Tooltip>
                      )}
                    </Stack>
                  </ListItem>
                );
              })}
            </List>
          ) : (
            <Box sx={{ height: '100%', minHeight: 220, display: 'grid', placeItems: 'center', color: 'text.secondary', p: 3 }}>
              <Stack alignItems="center" spacing={0.5}><MemoryIcon sx={{ fontSize: 40, opacity: 0.45 }} /><Typography>No {viewMode === 'effective' ? 'effective' : 'stored'} memories</Typography></Stack>
            </Box>
          )
        )}
      </Box>

      <Dialog open={Boolean(deleteTarget)} onClose={() => !actionId && setDeleteTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Delete Memory</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
            {deleteTarget?.content}
          </Typography>
          {Boolean(deleteTarget?.overrides?.length) && (
            <Alert severity="warning" sx={{ mt: 1.5 }}>
              Deleting this memory will restore {deleteTarget?.overrides.length} broader {deleteTarget?.overrides.length === 1 ? 'memory' : 'memories'} to the effective view.
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteTarget(null)} disabled={Boolean(actionId)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={() => void handleDelete()} disabled={Boolean(actionId)}>
            {actionId ? <CircularProgress size={20} /> : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
