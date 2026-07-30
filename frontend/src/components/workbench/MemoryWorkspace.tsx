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
import MemoryIcon from '@mui/icons-material/Memory';
import RefreshIcon from '@mui/icons-material/Refresh';
import ReplayIcon from '@mui/icons-material/Replay';
import {
  createMemory,
  deleteMemory,
  listMemories,
  listProjects,
  retryMemoryIndex,
  type MemoryRecord,
  type MemoryScopeType,
  type MemoryType,
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
  const forkOrigin = memory.fork_origin_json || memory.fork_origin;
  return (
    <Box component="details" sx={{ mt: 0.75, minWidth: 0 }}>
      <Typography component="summary" variant="caption" color="text.secondary" sx={{ cursor: 'pointer' }}>
        Details
      </Typography>
      <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 0.75 }}>
        <Typography variant="caption">Confidence: {Number(memory.confidence ?? 0).toFixed(2)}</Typography>
        <Typography variant="caption">Created by: {memory.created_by || 'Unknown'}</Typography>
        <Typography variant="caption">Updated: {formatTimestamp(memory.updated_at || memory.created_at)}</Typography>
        <Typography variant="caption">Expires: {memory.expires_at ? formatTimestamp(memory.expires_at) : 'Never'}</Typography>
      </Box>
      {Object.keys(sourceRefs).length > 0 && (
        <Box component="pre" sx={{ m: 0, mt: 0.75, p: 1, maxHeight: 150, overflow: 'auto', bgcolor: 'action.hover', fontSize: '0.72rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {JSON.stringify({ source_refs: sourceRefs }, null, 2)}
        </Box>
      )}
      {forkOrigin && (
        <Box component="pre" sx={{ m: 0, mt: 0.75, p: 1, maxHeight: 150, overflow: 'auto', bgcolor: 'action.hover', fontSize: '0.72rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
          {JSON.stringify({ fork_origin: forkOrigin }, null, 2)}
        </Box>
      )}
    </Box>
  );
}

export default function MemoryWorkspace({
  activeThread,
  activeProject: projectContext = null,
  projectInventoryVersion = 0,
}: {
  activeThread: Thread | null;
  activeProject?: Project | null;
  projectInventoryVersion?: number;
}) {
  const [scopeType, setScopeType] = useState<MemoryScopeType>(activeThread ? 'thread' : projectContext ? 'project' : 'user');
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [actionId, setActionId] = useState<string | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [createContent, setCreateContent] = useState('');
  const [createType, setCreateType] = useState<MemoryType>('semantic');
  const [createError, setCreateError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<MemoryRecord | null>(null);

  const availableScopes = useMemo(
    () => activeThread
      ? memoryScopesForContext(true)
      : projectContext ? (['project', 'user'] as MemoryScopeType[]) : memoryScopesForContext(false),
    [activeThread, projectContext],
  );

  useEffect(() => {
    setScopeType(activeThread ? 'thread' : projectContext ? 'project' : 'user');
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
  const consent = memoryConsentStatus({
    scopeType,
    thread: activeThread,
    project: activeProject,
  });

  useEffect(() => {
    let cancelled = false;
    if (!target) {
      setMemories([]);
      setLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setLoading(true);
    setLoadError(null);
    listMemories(target.scopeType, target.scopeId, MEMORY_LIMIT)
      .then((response) => {
        if (cancelled) return;
        setMemories(response.memories);
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
  }, [refreshVersion, target]);

  const filteredMemories = useMemo(
    () => filterMemoryRecords(memories, query),
    [memories, query],
  );

  const handleCreate = async () => {
    const content = createContent.trim();
    if (!target || !content) return;
    try {
      setActionId('create');
      setCreateError(null);
      await createMemory({
        scope_type: target.scopeType,
        scope_id: target.scopeId,
        memory_type: createType,
        content,
        confidence: 1,
        visibility: target.scopeType === 'project' ? 'project' : 'private',
        created_by: 'ui',
      });
      setCreateContent('');
      setCreateType('semantic');
      setCreateOpen(false);
      setRefreshVersion((value) => value + 1);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : 'Unable to create memory.');
    } finally {
      setActionId(null);
    }
  };

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

  const rowsAtLimit = isMemoryResultTruncated(
    memories.length,
    MEMORY_LIMIT,
  );

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto auto minmax(0, 1fr)', bgcolor: 'background.default' }}>
      <Box sx={{ px: 2, pt: 1.5, borderBottom: 1, borderColor: 'divider' }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Active Memory</Typography>
          <Stack direction="row" spacing={0.5}>
            <Tooltip title="Refresh">
              <IconButton size="small" onClick={() => setRefreshVersion((value) => value + 1)} disabled={loading}>
                <RefreshIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Add memory">
              <span>
                <IconButton size="small" color="primary" onClick={() => setCreateOpen(true)} disabled={!target}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        </Stack>
      </Box>

      <Box sx={{ px: 2, py: 1.25, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', borderBottom: 1, borderColor: 'divider' }}>
        <ToggleButtonGroup
          exclusive
          size="small"
          value={scopeType}
          onChange={(_, value) => value && setScopeType(value)}
          aria-label="Memory scope"
        >
          {availableScopes.map((scope) => (
            <ToggleButton key={scope} value={scope} sx={{ textTransform: 'none', minWidth: 86 }}>
              {scopeLabel(scope)}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        {!activeThread && !projectContext && scopeType === 'project' && (
          <FormControl size="small" sx={{ minWidth: 190, maxWidth: 300 }}>
            <InputLabel>Project</InputLabel>
            <Select value={selectedProjectId} label="Project" onChange={(event) => setSelectedProjectId(String(event.target.value))}>
              {projects.map((project) => <MenuItem key={project.id} value={project.id}>{project.name}</MenuItem>)}
            </Select>
          </FormControl>
        )}
        <Chip
          size="small"
          variant="outlined"
          color={consent.enabled === false ? 'warning' : consent.enabled === true ? 'success' : 'default'}
          label={consent.label}
        />
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
                        <Chip size="small" label={memory.memory_type} variant="outlined" />
                        <Chip size="small" label={scopeLabel(memory.scope_type)} variant="outlined" />
                        <Chip size="small" label={memory.index_status} color={statusColor(memory.index_status)} />
                        <Typography variant="caption" color="text.secondary">{memory.embedding_model}</Typography>
                      </Stack>
                      <Typography variant="body2" sx={{ mt: 0.75, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                        {memory.summary || memory.content}
                      </Typography>
                      {memory.summary && memory.summary !== memory.content && (
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                          {memory.content}
                        </Typography>
                      )}
                      {memory.index_error && <Alert severity="error" sx={{ mt: 0.75, py: 0 }}>{memory.index_error}</Alert>}
                      <MemoryDetails memory={memory} />
                    </Box>
                    <Stack direction="row" spacing={0.25}>
                      {['failed', 'pending'].includes(memory.index_status) && (
                        <Tooltip title="Retry indexing">
                          <span><IconButton size="small" onClick={() => void handleRetry(memory)} disabled={busy}><ReplayIcon fontSize="small" /></IconButton></span>
                        </Tooltip>
                      )}
                      <Tooltip title="Delete memory">
                        <span><IconButton size="small" color="error" onClick={() => setDeleteTarget(memory)} disabled={busy}><DeleteIcon fontSize="small" /></IconButton></span>
                      </Tooltip>
                    </Stack>
                  </ListItem>
                );
              })}
            </List>
          ) : (
            <Box sx={{ height: '100%', minHeight: 220, display: 'grid', placeItems: 'center', color: 'text.secondary', p: 3 }}>
              <Stack alignItems="center" spacing={0.5}><MemoryIcon sx={{ fontSize: 40, opacity: 0.45 }} /><Typography>No active memories</Typography></Stack>
            </Box>
          )
        )}
      </Box>

      <Dialog open={createOpen} onClose={() => actionId !== 'create' && setCreateOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add {scopeLabel(scopeType)} Memory</DialogTitle>
        <DialogContent sx={{ display: 'grid', gap: 1.5, pt: '8px !important' }}>
          {createError && <Alert severity="error">{createError}</Alert>}
          <FormControl fullWidth size="small">
            <InputLabel>Memory type</InputLabel>
            <Select value={createType} label="Memory type" onChange={(event) => setCreateType(event.target.value as MemoryType)}>
              <MenuItem value="semantic">Semantic</MenuItem>
              <MenuItem value="episodic">Episodic</MenuItem>
              <MenuItem value="procedural">Procedural</MenuItem>
            </Select>
          </FormControl>
          <TextField
            autoFocus
            fullWidth
            multiline
            minRows={5}
            maxRows={12}
            label="Memory"
            value={createContent}
            onChange={(event) => setCreateContent(event.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateOpen(false)} disabled={actionId === 'create'}>Cancel</Button>
          <Button variant="contained" onClick={() => void handleCreate()} disabled={!createContent.trim() || actionId === 'create'}>
            {actionId === 'create' ? <CircularProgress size={20} /> : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={Boolean(deleteTarget)} onClose={() => !actionId && setDeleteTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Delete Memory</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
            {deleteTarget?.summary || deleteTarget?.content}
          </Typography>
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
