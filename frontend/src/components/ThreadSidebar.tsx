import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useTheme } from '@mui/material/styles';

declare const process: {
  env: Record<string, string | undefined>;
};
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Typography,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Chip,
  Paper,
  Collapse,
  CircularProgress,
  Checkbox,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import SpeakerNotesIcon from '@mui/icons-material/SpeakerNotes';
import SpeakerNotesOffIcon from '@mui/icons-material/SpeakerNotesOff';
import DescriptionIcon from '@mui/icons-material/Description';
import LockIcon from '@mui/icons-material/Lock';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import ClearIcon from '@mui/icons-material/Clear';
import CallSplitIcon from '@mui/icons-material/CallSplit';

import {
  Thread,
  createThread,
  listThreads,
  bulkDeleteThreads,
  forkThread,
  updateThread,
} from '../lib/api';
import { fetchAvailableEmbedModels, checkEmbedModelReady } from '../lib/models-api';
import { formatDate } from '../lib/date-utils';


interface ThreadSidebarProps {
  activeThreadId: string | null;
  onThreadSelect: (thread: Thread | null) => void;
  onThreadForked?: (thread: Thread) => void;
  onEmbedModelChange?: (model: string) => void;
  darkMode?: boolean;
}

const ThreadSidebar: React.FC<ThreadSidebarProps> = ({
  activeThreadId,
  onThreadSelect,
  onThreadForked,
  onEmbedModelChange,
  darkMode = false,
}) => {
  const [threads, setThreads] = useState<Thread[]>([]);
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newThreadName, setNewThreadName] = useState(() => {
    const now = new Date();
    return `Thread ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;
  });
  const [newThreadEmbedModel, setNewThreadEmbedModel] = useState('');
  const [availableEmbedModels, setAvailableEmbedModels] = useState<{
    local_embedding_models: string[];
    embedding_models: string[];
    not_embedding_models: string[];
  }>({ local_embedding_models: [], embedding_models: [], not_embedding_models: [] });
  const [creating, setCreating] = useState(false);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [expanded, setExpanded] = useState(true);
  const [isEmbedModelValid, setIsEmbedModelValid] = useState<boolean | null>(null);
  const [isCheckingEmbedModel, setIsCheckingEmbedModel] = useState(false);
  const [selectedThreadIds, setSelectedThreadIds] = useState<Set<string>>(new Set());
  const [lastSelectedThreadId, setLastSelectedThreadId] = useState<string | null>(null);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [forkingThreadId, setForkingThreadId] = useState<string | null>(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [focusedThreadId, setFocusedThreadId] = useState<string | null>(null);
  const threadRowRefs = useRef<Record<string, HTMLLIElement | null>>({});

  const selectedCount = selectedThreadIds.size;
  const allThreadsSelected = threads.length > 0 && selectedCount === threads.length;
  const someThreadsSelected = selectedCount > 0 && !allThreadsSelected;
  const threadsById = useMemo(
    () => new Map(threads.map(thread => [thread.id, thread])),
    [threads]
  );


  // Helper function to get icon and color for model type
  const getModelIcon = (modelName: string) => {
    if (availableEmbedModels.embedding_models.includes(modelName)) {
      return <CheckCircleIcon fontSize="inherit" color="primary" />;
    } else if (availableEmbedModels.local_embedding_models.includes(modelName)) {
      return <CheckCircleIcon fontSize="inherit" sx={{ color: 'orange' }} />;
    } else if (availableEmbedModels.not_embedding_models.includes(modelName)) {
      return <ErrorIcon fontSize="inherit" color="error" />;
    }
    return null;
  };

  // Load threads and embedding models on mount
  useEffect(() => {
    loadThreads();
    fetchAvailableEmbedModels().then((models) => {
      setAvailableEmbedModels(models);
      const allModels = [...models.embedding_models, ...models.local_embedding_models, ...models.not_embedding_models];
      const defaultModel = allModels[0] || '';
      if (!newThreadEmbedModel && defaultModel) {
        setNewThreadEmbedModel(defaultModel);
      }
    });
  }, []);

  const loadThreads = async () => {
    try {
      setLoading(true);
      const response = await listThreads();
      setThreads(response.threads);
    } catch (error) {
      console.error('Failed to load threads:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateThread = async () => {
    if (!newThreadName.trim() || !newThreadEmbedModel) return;

    try {
      // Check if the embedding model is valid before proceeding
      const isReady = await checkEmbedModelReady(newThreadEmbedModel);
      if (!isReady) {
        setIsEmbedModelValid(false);
        return;
      }

      setIsEmbedModelValid(true);
      setCreating(true);
      const thread = await createThread(newThreadName.trim(), newThreadEmbedModel);
      setThreads(prev => [thread, ...prev]);
      onThreadSelect(thread);
      if (onEmbedModelChange) {
        onEmbedModelChange(newThreadEmbedModel);
      }
      setCreateDialogOpen(false);
      setNewThreadName('');
      setNewThreadEmbedModel('');
    } catch (error) {
      console.error('Failed to create thread:', error);
    } finally {
      setCreating(false);
    }
  };

  const toggleThreadSelection = (
    threadId: string,
    checked: boolean,
    isShiftClick: boolean
  ) => {
    setSelectedThreadIds(prev => {
      const next = new Set(prev);
      const currentIndex = threads.findIndex(thread => thread.id === threadId);
      const lastIndex = lastSelectedThreadId
        ? threads.findIndex(thread => thread.id === lastSelectedThreadId)
        : -1;

      if (isShiftClick && currentIndex !== -1 && lastIndex !== -1) {
        const start = Math.min(currentIndex, lastIndex);
        const end = Math.max(currentIndex, lastIndex);
        threads.slice(start, end + 1).forEach(thread => {
          if (checked) {
            next.add(thread.id);
          } else {
            next.delete(thread.id);
          }
        });
      } else if (checked) {
        next.add(threadId);
      } else {
        next.delete(threadId);
      }

      return next;
    });
    setLastSelectedThreadId(threadId);
  };

  const handleToggleThreadSelection = (
    threadId: string,
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    event.stopPropagation();
    toggleThreadSelection(
      threadId,
      event.target.checked,
      (event.nativeEvent as MouseEvent).shiftKey
    );
  };

  const handleThreadRowClick = (thread: Thread, event: React.MouseEvent) => {
    if (!isSelectionMode) {
      onThreadSelect(thread);
      return;
    }

    event.preventDefault();
    toggleThreadSelection(
      thread.id,
      !selectedThreadIds.has(thread.id),
      event.shiftKey
    );
  };

  const handleToggleAllThreads = (event: React.ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();
    if (event.target.checked) {
      setSelectedThreadIds(new Set(threads.map(thread => thread.id)));
      setLastSelectedThreadId(threads[threads.length - 1]?.id ?? null);
    } else {
      setSelectedThreadIds(new Set());
      setLastSelectedThreadId(null);
    }
  };

  const clearThreadSelection = () => {
    setSelectedThreadIds(new Set());
    setLastSelectedThreadId(null);
    setIsSelectionMode(false);
  };

  const enterThreadSelectionMode = () => {
    setIsSelectionMode(true);
  };

  const handleBulkDeleteThreads = async () => {
    const threadIds = Array.from(selectedThreadIds);
    if (threadIds.length === 0) return;
    if (!confirm(`Delete ${threadIds.length} threads and all their messages?`)) return;

    try {
      setIsBulkDeleting(true);
      const result = await bulkDeleteThreads(threadIds);
      const deletedIds = new Set(result.deleted_thread_ids);
      const remainingSelectedIds = new Set<string>();
      result.not_found_thread_ids.forEach(threadId => remainingSelectedIds.add(threadId));
      result.failed_thread_ids.forEach(failure => remainingSelectedIds.add(failure.thread_id));

      setThreads(prev => prev.filter(thread => !deletedIds.has(thread.id)));
      setSelectedThreadIds(remainingSelectedIds);
      if (remainingSelectedIds.size === 0) {
        setIsSelectionMode(false);
      }
      setLastSelectedThreadId(null);

      if (activeThreadId && deletedIds.has(activeThreadId)) {
        onThreadSelect(null);
      }

      const failedCount = result.failed_thread_ids.length;
      const notFoundCount = result.not_found_thread_ids.length;
      if (failedCount > 0 || notFoundCount > 0) {
        console.error('Bulk thread delete completed with issues:', result);
        alert(`Deleted ${result.deleted_thread_ids.length} threads. ${failedCount + notFoundCount} could not be deleted.`);
      }
    } catch (error) {
      console.error('Failed to delete selected threads:', error);
      alert('Failed to delete selected threads.');
    } finally {
      setIsBulkDeleting(false);
    }
  };

  const handleEditThread = async (threadId: string) => {
    if (!editingName.trim()) return;

    try {
      const updated = await updateThread(threadId, editingName.trim());
      setThreads(prev => prev.map(t => t.id === threadId ? { ...t, name: updated.name } : t));
      setEditingThreadId(null);
      setEditingName('');
    } catch (error) {
      console.error('Failed to update thread:', error);
    }
  };

  const startEditing = (thread: Thread, event: React.MouseEvent) => {
    event.stopPropagation();
    setEditingThreadId(thread.id);
    setEditingName(thread.name);
  };

  const handleForkThread = async (thread: Thread, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      setForkingThreadId(thread.id);
      const forked = await forkThread(thread.id);
      setThreads(prev => [forked, ...prev]);
      onThreadForked?.(forked);
      if (!onThreadForked) {
        onThreadSelect(forked);
      }
    } catch (error) {
      console.error('Failed to fork thread:', error);
      alert('Failed to fork thread.');
    } finally {
      setForkingThreadId(null);
    }
  };

  // Add validation check when embedding model changes
  useEffect(() => {
    if (!newThreadEmbedModel) {
      setIsEmbedModelValid(null);
      setIsCheckingEmbedModel(false);
      return;
    }

    const validateEmbedModel = async () => {
      setIsCheckingEmbedModel(true);
      setIsEmbedModelValid(null);
      try {
        const isReady = await checkEmbedModelReady(newThreadEmbedModel);
        setIsEmbedModelValid(isReady);
      } catch (error) {
        setIsEmbedModelValid(false);
      } finally {
        setIsCheckingEmbedModel(false);
      }
    };

    validateEmbedModel();
  }, [newThreadEmbedModel]);

  const handleOpenCreateDialog = () => {
    const now = new Date();
    setNewThreadName(`Thread ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`);
    setCreateDialogOpen(true);
  };

  const focusThreadInList = (threadId: string, event?: React.MouseEvent) => {
    event?.preventDefault();
    event?.stopPropagation();

    const row = threadRowRefs.current[threadId];
    if (!row) return;

    row.scrollIntoView({ block: 'center', behavior: 'smooth' });
    setFocusedThreadId(threadId);
  };

  useEffect(() => {
    if (!focusedThreadId) return;

    const clearFocusedThread = () => {
      setFocusedThreadId(null);
    };

    document.addEventListener('click', clearFocusedThread);
    return () => {
      document.removeEventListener('click', clearFocusedThread);
    };
  }, [focusedThreadId]);

  const renderThreadReference = (
    threadId: string | null | undefined,
    fallbackName?: string | null
  ) => {
    if (!threadId) {
      return <Typography variant="caption" color="text.secondary">None</Typography>;
    }

    const target = threadsById.get(threadId);
    if (!target) {
      return (
        <Typography variant="caption" color="text.secondary">
          {fallbackName || threadId}
        </Typography>
      );
    }

    return (
      <Chip
        label={target.name}
        size="small"
        clickable
        onClick={(event) => focusThreadInList(threadId, event)}
        onMouseDown={(event) => event.stopPropagation()}
        sx={{
          height: 22,
          maxWidth: '100%',
          color: 'primary.contrastText',
          bgcolor: 'primary.main',
          justifyContent: 'flex-start',
          '& .MuiChip-label': {
            px: 0.75,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          },
          '&:hover, &:focus-visible': {
            bgcolor: 'primary.dark',
          },
        }}
      />
    );
  };

  const renderThreadTooltip = (thread: Thread) => {
    const forkInfo = thread.thread_metadata?.fork;
    const childIds = Array.isArray(thread.thread_metadata?.fork_children)
      ? thread.thread_metadata.fork_children.filter((id): id is string => typeof id === 'string' && id.length > 0)
      : [];
    const documents = Object.entries(thread.documents_meta || {})
      .filter((entry): entry is [string, NonNullable<Thread['documents_meta']>[string]] => {
        const meta = entry[1];
        return !!meta && typeof meta === 'object' && !Array.isArray(meta);
      })
      .filter(([, meta]) =>
        Boolean(meta.file_name || meta.page_count || meta.document_available_in_thread_at)
      );
    const sectionSx = {
      pt: 0.75,
      mt: 0.75,
      borderTop: 1,
      borderColor: 'divider',
      '&:first-of-type': {
        pt: 0,
        mt: 0,
        borderTop: 0,
      },
    };

    return (
      <Box
        sx={{
          p: 0.5,
          pr: 0.75,
          minWidth: 220,
          maxWidth: 320,
          maxHeight: 'min(360px, calc(100vh - 96px))',
          overflowY: 'auto',
        }}
        onClick={(event) => event.stopPropagation()}
      >
        <Box sx={sectionSx}>
          <Typography variant="caption" color="text.secondary" component="div">
            Created
          </Typography>
          <Typography variant="caption" component="div">
            {new Date(thread.created_at).toLocaleString()}
          </Typography>
        </Box>
        <Box sx={sectionSx}>
          <Typography variant="caption" color="text.secondary" component="div">
            Embedding model
          </Typography>
          <Typography variant="caption" component="div" sx={{ wordBreak: 'break-word' }}>
            {thread.embed_model}
          </Typography>
        </Box>
        {forkInfo?.parent_thread_id && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Parent
            </Typography>
            {renderThreadReference(forkInfo.parent_thread_id, forkInfo.parent_thread_name)}
          </Box>
        )}
        {childIds.length > 0 && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Children
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
              {childIds.map(childId => (
                <Box key={childId}>
                  {renderThreadReference(childId)}
                </Box>
              ))}
            </Box>
          </Box>
        )}
        {documents.length > 0 && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Documents
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
              {documents.map(([fileHash, meta]) => (
                <Box key={fileHash} sx={{ minWidth: 0 }}>
                  {meta.file_name && (
                    <Typography variant="caption" component="div" sx={{ fontWeight: 600, lineHeight: 1.25, wordBreak: 'break-word' }}>
                      {meta.file_name}
                    </Typography>
                  )}
                  {meta.page_count !== undefined && meta.page_count !== null && meta.page_count !== '' && (
                    <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                      Pages: {meta.page_count}
                    </Typography>
                  )}
                  {meta.document_available_in_thread_at && (
                    <Typography variant="caption" color="text.secondary" component="div" sx={{ lineHeight: 1.25 }}>
                      Added: {new Date(meta.document_available_in_thread_at).toLocaleString()}
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          </Box>
        )}
        {forkInfo?.forked_at && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Forked at
            </Typography>
            <Typography variant="caption" component="div">
              {new Date(forkInfo.forked_at).toLocaleString()}
            </Typography>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Paper
      elevation={0}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: theme.palette.background.default,
        color: theme.palette.text.primary
      }}
    >
      {/* Header */}
      <Box sx={{
        p: 1.5,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: theme.palette.background.paper
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            {expanded ? <SpeakerNotesIcon fontSize="small" /> : <SpeakerNotesOffIcon fontSize="small" />}
          </IconButton>
          <Typography variant="subtitle2" fontWeight="bold">
            Threads
          </Typography>
          <Chip label={threads.length} size="small" />
          <Tooltip title="Create new thread">
            <IconButton
              size="small"
              color="primary"
              onClick={handleOpenCreateDialog}
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        {isSelectionMode ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Tooltip title={allThreadsSelected ? "Clear selection" : "Select all threads. Shift-click a thread to select a range."}>
              <Checkbox
                size="small"
                checked={allThreadsSelected}
                indeterminate={someThreadsSelected}
                onChange={handleToggleAllThreads}
                disabled={isBulkDeleting}
                sx={{ p: 0.5 }}
              />
            </Tooltip>
            <Tooltip title="Shift-click threads to select a range">
              <Chip label={`${selectedCount} selected`} size="small" color="primary" />
            </Tooltip>
            <Tooltip title="Clear selection">
              <IconButton size="small" onClick={clearThreadSelection} disabled={isBulkDeleting}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Delete selected threads">
              <span>
                <IconButton
                  size="small"
                  color="error"
                  onClick={handleBulkDeleteThreads}
                  disabled={isBulkDeleting || selectedCount === 0}
                >
                  {isBulkDeleting ? <CircularProgress size={16} /> : <DeleteIcon fontSize="small" />}
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Tooltip title="Select threads to delete. Shift-click a thread to select a range.">
              <span>
                <IconButton
                  size="small"
                  color="error"
                  onClick={enterThreadSelectionMode}
                  disabled={threads.length === 0}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </span>
            </Tooltip>
          </Box>
        )}
      </Box>

      {/* Thread List */}
      <Collapse in={expanded} sx={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : threads.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No threads yet
            </Typography>
            <Button
              size="small"
              startIcon={<AddIcon />}
              onClick={handleOpenCreateDialog}
              sx={{ mt: 1 }}
            >
              Create Thread
            </Button>
          </Box>
        ) : (
          <List dense sx={{ p: 0 }}>
            {threads.map((thread) => (
              <ListItem
                key={thread.id}
                ref={(node) => {
                  threadRowRefs.current[thread.id] = node;
                }}
                disablePadding
                sx={{
                  bgcolor: activeThreadId === thread.id
                    ? theme.palette.mode === 'dark'
                      ? theme.palette.primary.dark
                      : theme.palette.primary.light
                    : focusedThreadId === thread.id
                      ? theme.palette.action.focus
                    : isSelectionMode && selectedThreadIds.has(thread.id)
                      ? theme.palette.action.selected
                    : 'transparent',
                  boxShadow: focusedThreadId === thread.id
                    ? `inset 3px 0 0 ${theme.palette.primary.main}`
                    : 'none',
                  transition: 'background-color 160ms ease, box-shadow 160ms ease',
                  '&:hover': {
                    bgcolor: activeThreadId === thread.id
                      ? theme.palette.mode === 'dark'
                        ? theme.palette.primary.dark
                        : theme.palette.primary.light
                      : focusedThreadId === thread.id
                        ? theme.palette.action.focus
                      : isSelectionMode && selectedThreadIds.has(thread.id)
                        ? theme.palette.action.selected
                      : theme.palette.mode === 'dark'
                        ? theme.palette.background.paper
                        : theme.palette.grey[100]
                  }
                }}
              >
                <Tooltip
                  title={editingThreadId === thread.id ? "" : renderThreadTooltip(thread)}
                  placement="left"
                  arrow
                  enterDelay={500}
                  leaveDelay={150}
                  disableHoverListener={editingThreadId === thread.id}
                  disableInteractive={false}
                >
                  <ListItemButton
                    onClick={(e) => handleThreadRowClick(thread, e)}
                    selected={activeThreadId === thread.id}
                    sx={{ py: 1, pr: isSelectionMode ? 1 : 12 }}
                  >
                    {isSelectionMode && (
                      <Checkbox
                        edge="start"
                        size="small"
                        checked={selectedThreadIds.has(thread.id)}
                        onChange={(e) => handleToggleThreadSelection(thread.id, e)}
                        onClick={(e) => e.stopPropagation()}
                        disabled={isBulkDeleting}
                        inputProps={{ 'aria-label': `Select ${thread.name}` }}
                        sx={{ p: 0.5, mr: 0.5 }}
                      />
                    )}

                    {editingThreadId === thread.id ? (
                      <TextField
                        size="small"
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleEditThread(thread.id);
                          if (e.key === 'Escape') setEditingThreadId(null);
                        }}
                        onBlur={() => handleEditThread(thread.id)}
                        autoFocus
                        fullWidth
                        onClick={(e) => e.stopPropagation()}
                      />
                    ) : (
                      <ListItemText
                        primary={
                          <Typography
                            variant="body2"
                            fontWeight={activeThreadId === thread.id ? 'bold' : 'normal'}
                            noWrap
                          >
                            {thread.name}
                          </Typography>
                        }
                        secondaryTypographyProps={{ component: 'span' }}
                        secondary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">
                              {formatDate(thread.created_at)}
                            </Typography>
                            {thread.message_count !== undefined && thread.message_count > 0 && (
                              <Chip
                                label={`${thread.message_count} msgs`}
                                size="small"
                                sx={{ height: 16, fontSize: '0.65rem' }}
                              />
                            )}
                            {thread.file_count !== undefined && thread.file_count > 0 && (
                              <Chip
                                icon={<DescriptionIcon sx={{ fontSize: '0.7rem !important' }} />}
                                label={thread.file_count}
                                size="small"
                                sx={{ height: 16, fontSize: '0.65rem' }}
                              />
                            )}
                          </Box>
                        }
                      />
                    )}
                  </ListItemButton>
                </Tooltip>

                {!isSelectionMode && (
                  <ListItemSecondaryAction>
                    <Tooltip title="Fork thread">
                      <span>
                        <IconButton
                          size="small"
                          onClick={(e) => handleForkThread(thread, e)}
                          disabled={forkingThreadId === thread.id}
                          sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                        >
                          {forkingThreadId === thread.id ? <CircularProgress size={16} /> : <CallSplitIcon fontSize="small" />}
                        </IconButton>
                      </span>
                    </Tooltip>
                    <IconButton
                      size="small"
                      onClick={(e) => startEditing(thread, e)}
                      sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                    >
                      <EditIcon fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                )}
              </ListItem>
            ))}
          </List>
        )}
      </Collapse>

      {/* Create Thread Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Create New Thread</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Thread Name"
            fullWidth
            value={newThreadName}
            onChange={(e) => setNewThreadName(e.target.value)}
            placeholder="e.g., Research Paper Analysis"
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={newThreadEmbedModel}
              label="Embedding Model"
              onChange={(e) => setNewThreadEmbedModel(e.target.value)}
            >
              {[...availableEmbedModels.embedding_models, ...availableEmbedModels.local_embedding_models, ...availableEmbedModels.not_embedding_models].map((model) => (
                <MenuItem key={model} value={model}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getModelIcon(model)}
                    <Box>
                      {model}
                      {availableEmbedModels.local_embedding_models.includes(model) && (
                        <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                          (slower)
                        </Typography>
                      )}
                      {availableEmbedModels.not_embedding_models.includes(model) && (
                        <Typography variant="caption" sx={{ ml: 1, color: 'text.secondary' }}>
                          (may not work)
                        </Typography>
                      )}
                    </Box>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {isCheckingEmbedModel && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Checking embedding model...</Typography>
            </Box>
          )}
          {isEmbedModelValid === false && !isCheckingEmbedModel && (
            <Typography color="error" variant="body2" sx={{ mt: 1 }}>
              The selected model is not an embedding model. Please choose a valid model.
            </Typography>
          )}
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'warning.light', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <LockIcon fontSize="small" />
              <Typography variant="caption" color="text.secondary">
                The embedding model is locked once a thread is created.
                To use a different model, create a new thread.
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateThread}
            variant="contained"
            disabled={!newThreadName.trim() || !newThreadEmbedModel || creating || isEmbedModelValid === false || isCheckingEmbedModel}
          >
            {creating ? <CircularProgress size={20} /> : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default ThreadSidebar;
