import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  ButtonBase,
  Chip,
  CircularProgress,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  List,
  ListItem,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import PsychologyIcon from '@mui/icons-material/Psychology';
import RefreshIcon from '@mui/icons-material/Refresh';
import ReplayIcon from '@mui/icons-material/Replay';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import {
  deleteMemory,
  getMemoryReviewStatus,
  listEffectiveMemories,
  retryMemoryIndex,
  type MemoryOverrideRef,
  type MemoryScopeType,
  type MemoryWorkspaceRecord,
  type MemoryWorkspaceSection,
  type MemoryReviewStatus,
  type Project,
  type Thread,
} from '../../lib/api';
import {
  filterMemoryWorkspaceSections,
  memorySectionKey,
} from '../../lib/memory-workspace';
import {
  formatMemoryTimestamp,
  memoryIndexStatusColor,
  memoryRecallReasonLabel,
  memoryScopeLabel,
} from '../../lib/memory-ui';
import { createCuratorIntent, memoryReviewCuratorIntent, type MemoryCuratorIntent } from '../../lib/memory-curator';
import { JsonPreview } from '../agent-graph/AgentGraphInspectorPrimitives';

const MEMORY_LIMIT = 500;

function RelationshipTooltip({
  label,
  relationships,
  applied,
  color,
}: {
  label: string;
  relationships: MemoryOverrideRef[];
  applied: MemoryOverrideRef[];
  color: 'info' | 'warning';
}) {
  if (!relationships.length) return null;
  const appliedIds = new Set(applied.map((item) => item.id));
  return (
    <Tooltip
      arrow
      placement="top"
      title={(
        <Stack spacing={1} sx={{ maxWidth: 420, maxHeight: 260, overflow: 'auto', p: 0.5 }}>
          {relationships.map((item) => (
            <Box key={item.id}>
              <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.25 }}>
                <Chip size="small" label={memoryScopeLabel(item.scope_type, 'This thread')} sx={{ height: 20 }} />
                <Typography variant="caption" sx={{ fontWeight: 700 }}>
                  {appliedIds.has(item.id) ? 'Applied here' : 'Inactive here'}
                </Typography>
              </Stack>
              <Typography variant="caption" sx={{ display: 'block', overflowWrap: 'anywhere' }}>
                {item.content.length > 180 ? `${item.content.slice(0, 180)}...` : item.content}
              </Typography>
            </Box>
          ))}
        </Stack>
      )}
    >
      <Chip size="small" label={`${label} ${relationships.length}`} variant="outlined" color={color} sx={{ cursor: 'pointer' }} />
    </Tooltip>
  );
}

function MemoryDetails({
  memory,
  busy,
  onRetry,
}: {
  memory: MemoryWorkspaceRecord;
  busy: boolean;
  onRetry: (embeddingModel?: string) => void;
}) {
  const sourceRefs = memory.source_refs_json || memory.source_refs || {};
  const webSources = Array.isArray(sourceRefs.web_sources) ? sourceRefs.web_sources : [];
  return (
    <Box component="details" sx={{ mt: 0.75, minWidth: 0 }}>
      <Typography component="summary" variant="caption" color="text.secondary" sx={{ cursor: 'pointer' }}>
        Details
      </Typography>
      <Box sx={{ mt: 0.75, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 0.75 }}>
        <Typography variant="caption">Created: {formatMemoryTimestamp(memory.created_at)}</Typography>
        <Typography variant="caption">Updated: {formatMemoryTimestamp(memory.updated_at || memory.created_at)}</Typography>
        <Typography variant="caption">Embedding: {memory.embedding_model}</Typography>
      </Box>
      {Boolean(memory.representations?.length) && (
        <Stack spacing={0.5} sx={{ mt: 0.75 }}>
          <Typography variant="caption" color="text.secondary">Vector representations</Typography>
          {memory.representations?.map((representation) => (
            <Stack key={representation.embedding_model} direction="row" spacing={0.75} alignItems="center" useFlexGap flexWrap="wrap">
              <Chip size="small" variant="outlined" label={representation.primary ? 'Primary' : 'Secondary'} />
              <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>{representation.embedding_model}</Typography>
              <Chip size="small" color={memoryIndexStatusColor(representation.index_status)} label={representation.index_status} />
              {['failed', 'pending'].includes(representation.index_status) && (
                <Tooltip title={`Retry ${representation.embedding_model}`}>
                  <span>
                    <IconButton size="small" disabled={busy} onClick={() => onRetry(representation.primary ? undefined : representation.embedding_model)}>
                      <ReplayIcon fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              )}
              {representation.index_error && <Typography variant="caption" color="error.main">{representation.index_error}</Typography>}
            </Stack>
          ))}
        </Stack>
      )}
      {webSources.length > 0 && (
        <Box sx={{ mt: 0.75 }}>
          <Typography variant="caption" color="text.secondary">Web provenance</Typography>
          {webSources.map((source: Record<string, unknown>, index: number) => (
            <Typography
              key={`${String(source.url || source.title || '')}-${index}`}
              variant="caption"
              component={source.url ? 'a' : 'span'}
              href={source.url ? String(source.url) : undefined}
              target={source.url ? '_blank' : undefined}
              rel={source.url ? 'noopener noreferrer' : undefined}
              sx={{ display: 'block', color: source.url ? 'primary.main' : 'text.primary', overflowWrap: 'anywhere' }}
            >
              {String(source.title || source.url || 'Internet source')}
            </Typography>
          ))}
        </Box>
      )}
      {Object.keys(sourceRefs).length > 0 && (
        <JsonPreview value={{ source_refs: sourceRefs }} maxHeight={180} />
      )}
    </Box>
  );
}

export default function MemoryWorkspace({
  activeThread,
  activeProject: projectContext = null,
  curatorRefreshVersion = 0,
  onOpenCurator,
}: {
  activeThread: Thread | null;
  activeProject?: Project | null;
  projectInventoryVersion?: number;
  curatorRefreshVersion?: number;
  onOpenCurator?: (intent: MemoryCuratorIntent) => void;
}) {
  const [sections, setSections] = useState<MemoryWorkspaceSection[]>([]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const expansionContext = useRef('');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [actionId, setActionId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<MemoryWorkspaceRecord | null>(null);
  const [reviewStatus, setReviewStatus] = useState<MemoryReviewStatus | null>(null);

  useEffect(() => {
    setQuery('');
  }, [activeThread?.id, projectContext?.id]);

  const requestedProjectId = activeThread
    ? null
    : projectContext?.id || null;
  const contextKey = activeThread
    ? `thread:${activeThread.id}`
    : requestedProjectId ? `project:${requestedProjectId}` : 'global:default';

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    listEffectiveMemories({
      threadId: activeThread?.id,
      projectId: requestedProjectId,
      limit: MEMORY_LIMIT,
    })
      .then((response) => {
        if (cancelled) return;
        setSections(response.workspace_sections || []);
        if (expansionContext.current !== contextKey) {
          setExpanded(Object.fromEntries((response.workspace_sections || []).map((section) => [memorySectionKey(section), true])));
          expansionContext.current = contextKey;
        }
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : 'Unable to load memory.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [activeThread?.id, contextKey, curatorRefreshVersion, refreshVersion, requestedProjectId]);

  useEffect(() => {
    let cancelled = false;
    getMemoryReviewStatus({ threadId: activeThread?.id, projectId: projectContext?.id })
      .then((status) => { if (!cancelled) setReviewStatus(status); })
      .catch(() => { if (!cancelled) setReviewStatus(null); });
    return () => { cancelled = true; };
  }, [activeThread?.id, curatorRefreshVersion, projectContext?.id, refreshVersion]);

  const filteredSections = useMemo(
    () => filterMemoryWorkspaceSections(sections, query),
    [query, sections],
  );

  const openCurator = (scopeType: MemoryScopeType, scopeId: string, memory?: MemoryWorkspaceRecord) => {
    onOpenCurator?.(createCuratorIntent({
      scopeType,
      scopeId,
      thread: activeThread,
      project: projectContext,
      memory,
    }));
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      setActionId(deleteTarget.id);
      setLoadError(null);
      await deleteMemory(deleteTarget.id);
      setDeleteTarget(null);
      setRefreshVersion((value) => value + 1);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Unable to delete memory.');
    } finally {
      setActionId(null);
    }
  };

  const handleRetry = async (memory: MemoryWorkspaceRecord, embeddingModel?: string) => {
    try {
      setActionId(memory.id);
      setLoadError(null);
      await retryMemoryIndex(memory.id, embeddingModel);
      setRefreshVersion((value) => value + 1);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : 'Unable to retry memory indexing.');
      setRefreshVersion((value) => value + 1);
    } finally {
      setActionId(null);
    }
  };

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)', bgcolor: 'background.default' }}>
      <Box sx={{ px: 2, py: 1.25, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>Memory &amp; Settings</Typography>
        <Tooltip title="Review related memories for duplicates, conflicts, and stale overrides">
          <span>
            <Button
              size="small"
              startIcon={<FactCheckIcon />}
              color={reviewStatus?.status === 'review_suggested' ? 'warning' : 'primary'}
              onClick={() => onOpenCurator?.(memoryReviewCuratorIntent({ thread: activeThread, project: projectContext }))}
              disabled={!onOpenCurator}
            >
              Review memories
            </Button>
          </span>
        </Tooltip>
        {reviewStatus?.status === 'review_suggested' && <Chip size="small" color="warning" label="Review suggested" />}
        {reviewStatus?.status === 'never_reviewed' && <Chip size="small" variant="outlined" label="Not reviewed" />}
        <TextField
          size="small"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter memory"
          sx={{ ml: { sm: 'auto' }, width: { xs: '100%', sm: 220 } }}
        />
        <Tooltip title="Refresh">
          <IconButton size="small" onClick={() => setRefreshVersion((value) => value + 1)} disabled={loading}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      <Box sx={{ minHeight: 0, overflow: 'auto' }}>
        {loadError && <Alert severity="error" onClose={() => setLoadError(null)} sx={{ m: 1.5 }}>{loadError}</Alert>}
        {loading && sections.length === 0 ? (
          <Box sx={{ height: '100%', minHeight: 180, display: 'grid', placeItems: 'center' }}><CircularProgress size={28} /></Box>
        ) : filteredSections.length ? (
          filteredSections.map((section) => {
            const key = memorySectionKey(section);
            const isExpanded = expanded[key] !== false;
            const totalCount = sections.find((item) => memorySectionKey(item) === key)?.memories.length ?? section.memories.length;
            return (
              <Box component="section" key={key} sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Box sx={{ minHeight: 52, px: 1.5, display: 'flex', alignItems: 'center', gap: 1, bgcolor: 'action.hover' }}>
                  <ButtonBase
                    onClick={() => setExpanded((current) => ({ ...current, [key]: !isExpanded }))}
                    aria-expanded={isExpanded}
                    sx={{ flex: 1, minWidth: 0, justifyContent: 'flex-start', gap: 0.75, py: 1, textAlign: 'left' }}
                  >
                    {isExpanded ? <ExpandMoreIcon fontSize="small" /> : <ChevronRightIcon fontSize="small" />}
                    <Chip size="small" label={memoryScopeLabel(section.scope_type, 'This thread')} variant="outlined" />
                    <Typography variant="caption" color="text.secondary">{totalCount}</Typography>
                    <Chip
                      size="small"
                      variant="outlined"
                      color={section.recall_enabled ? 'success' : 'warning'}
                      label={section.recall_enabled ? 'Recall enabled' : memoryRecallReasonLabel(section.recall_skip_reason)}
                    />
                    {section.truncated && <Chip size="small" color="info" label={`First ${MEMORY_LIMIT}`} />}
                  </ButtonBase>
                  <Tooltip title={`Add ${memoryScopeLabel(section.scope_type, 'this thread').toLowerCase()} memory`}>
                    <span>
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => openCurator(section.scope_type, section.scope_id)}
                        disabled={!onOpenCurator}
                      >
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
                <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                  {section.memories.length ? (
                    <List disablePadding>
                      {section.memories.map((memory) => {
                        const busy = actionId === memory.id;
                        const overridden = memory.resolution_status === 'overridden';
                        const recallDisabled = memory.resolution_status === 'recall_disabled';
                        return (
                          <ListItem key={memory.id} divider alignItems="flex-start" sx={{ px: 2, py: 1.5, gap: 1.5, opacity: recallDisabled ? 0.68 : 1 }}>
                            <Box sx={{ flex: 1, minWidth: 0 }}>
                              <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
                                {overridden && <Chip size="small" color="warning" variant="outlined" label="Not used here" />}
                                {recallDisabled && <Chip size="small" color="warning" variant="outlined" label="Recall off" />}
                                {memory.resolution_status === 'unavailable' && <Chip size="small" label={memory.index_status} color={memoryIndexStatusColor(memory.index_status)} />}
                                <RelationshipTooltip label="Overrides" relationships={memory.overrides || []} applied={memory.applied_overrides || []} color="info" />
                                <RelationshipTooltip label="Overridden by" relationships={memory.overridden_by || []} applied={memory.applied_overridden_by || []} color="warning" />
                              </Stack>
                              <Typography
                                variant="body2"
                                sx={{
                                  mt: 0.75,
                                  whiteSpace: 'pre-wrap',
                                  overflowWrap: 'anywhere',
                                  textDecoration: overridden ? 'line-through' : 'none',
                                  color: overridden ? 'text.secondary' : 'text.primary',
                                }}
                              >
                                {memory.content}
                              </Typography>
                              {memory.index_error && <Alert severity="error" sx={{ mt: 0.75, py: 0 }}>{memory.index_error}</Alert>}
                              <MemoryDetails memory={memory} busy={busy} onRetry={(model) => void handleRetry(memory, model)} />
                            </Box>
                            <Stack direction="row" spacing={0.25}>
                              <Tooltip title="Edit memory">
                                <span><IconButton size="small" onClick={() => openCurator(memory.scope_type, memory.scope_id, memory)} disabled={busy || !onOpenCurator}><EditIcon fontSize="small" /></IconButton></span>
                              </Tooltip>
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
                    <Box sx={{ minHeight: 92, display: 'grid', placeItems: 'center', color: 'text.secondary', p: 2 }}>
                      <Stack alignItems="center" spacing={0.5}>
                        <PsychologyIcon sx={{ fontSize: 28, opacity: 0.4 }} />
                        <Typography variant="body2">{query.trim() ? 'No matching memories' : `No ${memoryScopeLabel(section.scope_type, 'this thread').toLowerCase()} memories`}</Typography>
                      </Stack>
                    </Box>
                  )}
                </Collapse>
              </Box>
            );
          })
        ) : (
          <Box sx={{ height: '100%', minHeight: 220, display: 'grid', placeItems: 'center', color: 'text.secondary', p: 3 }}>
            <Stack alignItems="center" spacing={0.5}><PsychologyIcon sx={{ fontSize: 40, opacity: 0.45 }} /><Typography>No memory scopes available</Typography></Stack>
          </Box>
        )}
      </Box>

      <Dialog open={Boolean(deleteTarget)} onClose={() => !actionId && setDeleteTarget(null)} maxWidth="xs" fullWidth>
        <DialogTitle>Delete Memory</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{deleteTarget?.content}</Typography>
          {Boolean(deleteTarget?.overrides?.length) && (
            <Alert severity="warning" sx={{ mt: 1.5 }}>
              Deleting this memory may restore {deleteTarget?.overrides.length} broader {deleteTarget?.overrides.length === 1 ? 'memory' : 'memories'} here.
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
