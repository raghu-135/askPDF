import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { alpha, useTheme } from '@mui/material/styles';
import { useVirtualizer } from '@tanstack/react-virtual';

declare const process: {
  env: Record<string, string | undefined>;
};
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
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
  FormControlLabel,
  Switch,
  Alert,
  Divider,
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
import EmbeddingModelReadinessIndicator from './EmbeddingModelReadinessIndicator';
import ClearIcon from '@mui/icons-material/Clear';
import CallSplitIcon from '@mui/icons-material/CallSplit';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';

import {
  Project,
  Thread,
  createProject,
  createThread,
  listProjects,
  listThreads,
  bulkDeleteThreads,
  forkThread,
  updateThread,
  updateProject,
  cloneProject,
  deleteProject,
  getProjectLifecycleSummary,
  type ProjectLifecycleSummary,
} from '../lib/api';
import { fetchAvailableEmbeddingModels, checkEmbeddingModelReady } from '../lib/models-api';
import { formatDate } from '../lib/date-utils';
import {
  defaultProjectCloneName,
  projectDeletionConfirmed,
} from '../lib/project-lifecycle';
import {
  sidebarDeletionTarget,
  threadsEligibleForProjectDeletion,
} from '../lib/sidebar-deletion';
import ThreadReferenceChip from './ThreadReferenceChip';
import ThreadForkDialog, { MemoryCopyMode } from './ThreadForkDialog';
import { flexTruncateSx, singleLineTruncateSx } from '../lib/truncation';


export interface ThreadSidebarHeaderState {
  projectCount: number;
  threadCount: number;
  activeProjectThreadCount: number;
  deletionTarget: 'projects' | 'threads';
  hasDeletableItems: boolean;
  isSelectionMode: boolean;
  selectedCount: number;
  allItemsSelected: boolean;
  someItemsSelected: boolean;
  isBulkDeleting: boolean;
  openCreateProjectDialog: () => void;
  enterSelectionMode: () => void;
  clearSelection: () => void;
  deleteSelected: () => void;
  toggleAllItems: (checked: boolean) => void;
}

interface ThreadSidebarProps {
  activeThreadId: string | null;
  activeProjectId?: string | null;
  onThreadSelect: (thread: Thread | null) => void;
  onProjectSelect?: (project: Project) => void;
  onProjectReadinessChange?: (projectId: string, ready: boolean | null) => void;
  onProjectUpdated?: (project: Project) => void;
  onProjectCloned?: (project: Project) => void;
  onProjectDeleted?: (projectId: string) => void;
  onThreadForked?: (thread: Thread) => void;
  onEmbeddingModelChange?: (model: string) => void;
  hideHeader?: boolean;
  onHeaderStateChange?: (state: ThreadSidebarHeaderState | null) => void;
  darkMode?: boolean;
  selectionOnly?: boolean;
}

const ThreadSidebar: React.FC<ThreadSidebarProps> = ({
  activeThreadId,
  activeProjectId,
  onThreadSelect,
  onProjectSelect,
  onProjectReadinessChange,
  onProjectUpdated,
  onProjectCloned,
  onProjectDeleted,
  onThreadForked,
  onEmbeddingModelChange,
  hideHeader = false,
  onHeaderStateChange,
  darkMode = false,
  selectionOnly = false,
}) => {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const theme = useTheme();
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [createProjectDialogOpen, setCreateProjectDialogOpen] = useState(false);
  const [newThreadName, setNewThreadName] = useState(() => {
    const now = new Date();
    return `Thread ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;
  });
  const [newThreadProjectId, setNewThreadProjectId] = useState<string>('');
  const [isThreadProjectLocked, setIsThreadProjectLocked] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [newProjectDescription, setNewProjectDescription] = useState('');
  const [newProjectEmbeddingModel, setNewProjectEmbeddingModel] = useState('');
  const [newProjectReadsUserMemory, setNewProjectReadsUserMemory] = useState(false);
  const [settingsProject, setSettingsProject] = useState<Project | null>(null);
  const [settingsProjectName, setSettingsProjectName] = useState('');
  const [settingsProjectReadsUserMemory, setSettingsProjectReadsUserMemory] = useState(false);
  const [projectLifecycle, setProjectLifecycle] = useState<ProjectLifecycleSummary | null>(null);
  const [projectLifecycleLoading, setProjectLifecycleLoading] = useState(false);
  const [projectActionError, setProjectActionError] = useState('');
  const [cloneProjectDialog, setCloneProjectDialog] = useState<{
    project: Project;
    includeThreads: boolean;
  } | null>(null);
  const [cloneProjectName, setCloneProjectName] = useState('');
  const [deleteProjectDialog, setDeleteProjectDialog] = useState<Project | null>(null);
  const [deleteProjectConfirmation, setDeleteProjectConfirmation] = useState('');
  const [availableEmbeddingModels, setAvailableEmbeddingModels] = useState<{
    local_embedding_models: string[];
    embedding_models: string[];
    not_embedding_models: string[];
  }>({ local_embedding_models: [], embedding_models: [], not_embedding_models: [] });
  const [creating, setCreating] = useState(false);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [expanded, setExpanded] = useState(true);
  const [isEmbeddingModelValid, setIsEmbeddingModelValid] = useState<boolean | null>(null);
  const [isCheckingEmbeddingModel, setIsCheckingEmbeddingModel] = useState(false);
  const [selectedThreadIds, setSelectedThreadIds] = useState<Set<string>>(new Set());
  const [lastSelectedThreadId, setLastSelectedThreadId] = useState<string | null>(null);
  const [selectedProjectIds, setSelectedProjectIds] = useState<Set<string>>(new Set());
  const [bulkProjectDeleteOpen, setBulkProjectDeleteOpen] = useState(false);
  const [bulkProjectDeleteConfirmation, setBulkProjectDeleteConfirmation] = useState('');
  const [bulkProjectDeleteSummaries, setBulkProjectDeleteSummaries] = useState<
    Array<{ project: Project; summary: ProjectLifecycleSummary | null; error?: string }>
  >([]);
  const [bulkProjectDeleteLoading, setBulkProjectDeleteLoading] = useState(false);
  const [bulkProjectDeleteError, setBulkProjectDeleteError] = useState('');
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);
  const [forkingThreadId, setForkingThreadId] = useState<string | null>(null);
  const [forkDialogThread, setForkDialogThread] = useState<Thread | null>(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [focusedThreadId, setFocusedThreadId] = useState<string | null>(null);
  const [projectReadiness, setProjectReadiness] = useState<Record<string, boolean | null>>({});
  const threadRowRefs = useRef<Record<string, HTMLLIElement | null>>({});
  const threadListRef = useRef<HTMLDivElement | null>(null);

  const deletionTarget = sidebarDeletionTarget(activeProjectId);
  const eligibleThreads = useMemo(
    () => threadsEligibleForProjectDeletion(threads, activeProjectId),
    [activeProjectId, threads]
  );
  const selectedCount = deletionTarget === 'projects'
    ? selectedProjectIds.size
    : selectedThreadIds.size;
  const eligibleItemCount = deletionTarget === 'projects'
    ? projects.length
    : eligibleThreads.length;
  const allItemsSelected = eligibleItemCount > 0 && selectedCount === eligibleItemCount;
  const someItemsSelected = selectedCount > 0 && !allItemsSelected;
  const threadsById = useMemo(
    () => new Map(threads.map(thread => [thread.id, thread])),
    [threads]
  );
  const projectsById = useMemo(
    () => new Map(projects.map(project => [project.id, project])),
    [projects]
  );
  const groupedThreads = useMemo(() => {
    const groups = new Map<string, { project: Project | null; threads: Thread[] }>();
    for (const project of projects) {
      groups.set(project.id, { project, threads: [] });
    }
    for (const thread of threads) {
      const key = thread.project_id || 'unassigned';
      const current = groups.get(key) || { project: projectsById.get(key) || null, threads: [] };
      current.threads.push(thread);
      groups.set(key, current);
    }
    return Array.from(groups.values());
  }, [projects, projectsById, threads]);
  const virtualThreadRows = useMemo(() => (
    groupedThreads.flatMap((group) => [
      { kind: 'group' as const, id: `group-${group.project?.id || 'unassigned'}`, group },
      ...group.threads.map((thread) => ({ kind: 'thread' as const, id: thread.id, thread })),
    ])
  ), [groupedThreads]);
  const threadVirtualizer = useVirtualizer({
    count: virtualThreadRows.length,
    getScrollElement: () => threadListRef.current,
    estimateSize: (index) => virtualThreadRows[index]?.kind === 'group' ? 40 : 64,
    overscan: 10,
    getItemKey: (index) => virtualThreadRows[index]?.id ?? index,
  });

  // Helper function to get icon and color for model type
  const getModelIcon = (modelName: string) => {
    if (availableEmbeddingModels.embedding_models.includes(modelName)) {
      return <CheckCircleIcon fontSize="inherit" color="primary" />;
    } else if (availableEmbeddingModels.local_embedding_models.includes(modelName)) {
      return <CheckCircleIcon fontSize="inherit" sx={{ color: 'orange' }} />;
    } else if (availableEmbeddingModels.not_embedding_models.includes(modelName)) {
      return <ErrorIcon fontSize="inherit" color="error" />;
    }
    return null;
  };

  // Load threads and embedding models on mount
  useEffect(() => {
    loadSidebarData();
    fetchAvailableEmbeddingModels().then((models) => {
      setAvailableEmbeddingModels(models);
      const allModels = [...models.embedding_models, ...models.local_embedding_models, ...models.not_embedding_models];
      const defaultModel = allModels[0] || '';
      if (!newProjectEmbeddingModel && defaultModel) {
        setNewProjectEmbeddingModel(defaultModel);
      }
    });
  }, []);

  const loadSidebarData = async () => {
    try {
      setLoading(true);
      const [threadResponse, projectResponse] = await Promise.all([
        listThreads(),
        listProjects(),
      ]);
      setThreads(threadResponse.threads);
      setProjects(projectResponse.projects);
      if (!newThreadProjectId && projectResponse.projects[0]?.id) {
        setNewThreadProjectId(projectResponse.projects[0].id);
      }
    } catch (error) {
      console.error('Failed to load sidebar data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateThread = async () => {
    if (!newThreadName.trim() || !newThreadProjectId) return;

    try {
      setCreating(true);
      const thread = await createThread(newThreadName.trim(), newThreadProjectId);
      setThreads(prev => [thread, ...prev]);
      onThreadSelect(thread);
      if (onEmbeddingModelChange) {
        onEmbeddingModelChange(thread.embeddingModel);
      }
      setCreateDialogOpen(false);
      setIsThreadProjectLocked(false);
      setNewThreadName('');
    } catch (error) {
      console.error('Failed to create thread:', error);
    } finally {
      setCreating(false);
    }
  };

  const handleCreateProject = async () => {
    if (!newProjectName.trim() || !newProjectEmbeddingModel) return;
    try {
      setCreating(true);
      const project = await createProject(
        newProjectName.trim(),
        newProjectEmbeddingModel,
        newProjectDescription.trim(),
        {
          memory: {
            project_reads_user_memory: newProjectReadsUserMemory,
          },
        }
      );
      setProjects(prev => [...prev, project]);
      setNewThreadProjectId(project.id);
      setCreateProjectDialogOpen(false);
      setNewProjectName('');
      setNewProjectDescription('');
      setNewProjectReadsUserMemory(false);
    } catch (error) {
      console.error('Failed to create project:', error);
    } finally {
      setCreating(false);
    }
  };

  const loadProjectLifecycle = async (project: Project) => {
    setProjectLifecycleLoading(true);
    setProjectActionError('');
    try {
      const [summary, ready] = await Promise.all([
        getProjectLifecycleSummary(project.id),
        checkEmbeddingModelReady(project.embeddingModel),
      ]);
      setProjectLifecycle(summary);
      setProjectReadiness((current) => ({ ...current, [project.id]: ready }));
      onProjectReadinessChange?.(project.id, ready);
    } catch (error: any) {
      setProjectActionError(error?.message || 'Unable to load project lifecycle details.');
      setProjectLifecycle(null);
    } finally {
      setProjectLifecycleLoading(false);
    }
  };

  const handleOpenProjectSettings = (project: Project) => {
    setSettingsProject(project);
    setSettingsProjectName(project.name);
    setProjectLifecycle(null);
    setProjectActionError('');
    setSettingsProjectReadsUserMemory(
      project.settings_json?.memory?.project_reads_user_memory === true
    );
    void loadProjectLifecycle(project);
  };

  const handleSaveProjectSettings = async () => {
    const projectName = settingsProjectName.trim();
    if (!settingsProject || !projectName) return;
    try {
      setCreating(true);
      setProjectActionError('');
      const updated = await updateProject(settingsProject.id, {
        name: projectName,
        settings_json: {
          memory: {
            project_reads_user_memory: settingsProjectReadsUserMemory,
          },
        },
      });
      setProjects((current) => current.map((project) => (
        project.id === updated.id ? updated : project
      )));
      onProjectUpdated?.(updated);
      setSettingsProject(null);
    } catch (error) {
      console.error('Failed to update project memory settings:', error);
      setProjectActionError(
        error instanceof Error ? error.message : 'Failed to update project settings.'
      );
    } finally {
      setCreating(false);
    }
  };

  const handleOpenCloneProject = (project: Project, includeThreads: boolean) => {
    setCloneProjectName(defaultProjectCloneName(project.name));
    setCloneProjectDialog({ project, includeThreads });
    setProjectActionError('');
  };

  const handleCloneProject = async () => {
    if (!cloneProjectDialog || !cloneProjectName.trim()) return;
    try {
      setCreating(true);
      setProjectActionError('');
      const result = await cloneProject(
        cloneProjectDialog.project.id,
        cloneProjectName.trim(),
        cloneProjectDialog.includeThreads,
      );
      await loadSidebarData();
      setSettingsProject(null);
      setCloneProjectDialog(null);
      onProjectCloned?.(result.project);
    } catch (error: any) {
      setProjectActionError(error?.message || 'Failed to clone project.');
    } finally {
      setCreating(false);
    }
  };

  const handleOpenDeleteProject = (project: Project) => {
    setDeleteProjectConfirmation('');
    setDeleteProjectDialog(project);
    setProjectActionError('');
  };

  const handleDeleteProject = async () => {
    if (
      !deleteProjectDialog
      || !projectDeletionConfirmed(deleteProjectConfirmation, deleteProjectDialog.name)
    ) return;
    try {
      setCreating(true);
      setProjectActionError('');
      const projectId = deleteProjectDialog.id;
      await deleteProject(projectId);
      setProjects((current) => current.filter((project) => project.id !== projectId));
      setThreads((current) => current.filter((thread) => thread.project_id !== projectId));
      setSettingsProject(null);
      setDeleteProjectDialog(null);
      onProjectDeleted?.(projectId);
    } catch (error: any) {
      setProjectActionError(error?.message || 'Failed to delete project.');
    } finally {
      setCreating(false);
    }
  };

  const toggleThreadSelection = (
    threadId: string,
    checked: boolean,
    isShiftClick: boolean
  ) => {
    if (!activeProjectId || !eligibleThreads.some((thread) => thread.id === threadId)) return;
    setSelectedThreadIds(prev => {
      const next = new Set(prev);
      const currentIndex = eligibleThreads.findIndex(thread => thread.id === threadId);
      const lastIndex = lastSelectedThreadId
        ? eligibleThreads.findIndex(thread => thread.id === lastSelectedThreadId)
        : -1;

      if (isShiftClick && currentIndex !== -1 && lastIndex !== -1) {
        const start = Math.min(currentIndex, lastIndex);
        const end = Math.max(currentIndex, lastIndex);
        eligibleThreads.slice(start, end + 1).forEach(thread => {
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

  const handleThreadRowClick = (thread: Thread) => {
    onThreadSelect(thread);
  };

  const handleProjectClick = async (project: Project) => {
    if (selectionOnly) return;
    if (activeProjectId === project.id) {
      return;
    }
    onProjectSelect?.(project);
    setProjectReadiness((current) => ({ ...current, [project.id]: null }));
    onProjectReadinessChange?.(project.id, null);
    try {
      const ready = await checkEmbeddingModelReady(project.embeddingModel);
      setProjectReadiness((current) => ({ ...current, [project.id]: ready }));
      onProjectReadinessChange?.(project.id, ready);
    } catch {
      setProjectReadiness((current) => ({ ...current, [project.id]: false }));
      onProjectReadinessChange?.(project.id, false);
    }
  };

  const handleToggleAllItemsChecked = useCallback((checked: boolean) => {
    if (deletionTarget === 'projects') {
      setSelectedProjectIds(checked ? new Set(projects.map((project) => project.id)) : new Set());
      return;
    }
    setSelectedThreadIds(checked ? new Set(eligibleThreads.map((thread) => thread.id)) : new Set());
    setLastSelectedThreadId(checked ? eligibleThreads[eligibleThreads.length - 1]?.id ?? null : null);
  }, [deletionTarget, eligibleThreads, projects]);

  const handleToggleAllThreads = (event: React.ChangeEvent<HTMLInputElement>) => {
    event.stopPropagation();
    handleToggleAllItemsChecked(event.target.checked);
  };

  const clearThreadSelection = useCallback(() => {
    setSelectedThreadIds(new Set());
    setSelectedProjectIds(new Set());
    setLastSelectedThreadId(null);
    setIsSelectionMode(false);
  }, []);

  const enterThreadSelectionMode = useCallback(() => {
    setIsSelectionMode(true);
  }, []);

  const handleBulkDeleteThreads = useCallback(async () => {
    const eligibleIds = new Set(eligibleThreads.map((thread) => thread.id));
    const threadIds = Array.from(selectedThreadIds).filter((threadId) => eligibleIds.has(threadId));
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
  }, [activeThreadId, eligibleThreads, onThreadSelect, selectedThreadIds]);

  const handleRequestBulkDeleteProjects = useCallback(async () => {
    const selectedProjects = projects.filter((project) => selectedProjectIds.has(project.id));
    if (selectedProjects.length === 0) return;
    setBulkProjectDeleteOpen(true);
    setBulkProjectDeleteConfirmation('');
    setBulkProjectDeleteError('');
    setBulkProjectDeleteLoading(true);
    try {
      const summaries = await Promise.all(selectedProjects.map(async (project) => {
        try {
          return {
            project,
            summary: await getProjectLifecycleSummary(project.id),
          };
        } catch (error) {
          return {
            project,
            summary: null,
            error: error instanceof Error ? error.message : 'Unable to inspect project',
          };
        }
      }));
      setBulkProjectDeleteSummaries(summaries);
    } finally {
      setBulkProjectDeleteLoading(false);
    }
  }, [projects, selectedProjectIds]);

  const handleBulkDeleteProjects = useCallback(async () => {
    const deletable = bulkProjectDeleteSummaries.filter((item) => item.summary?.can_delete);
    const confirmation = `DELETE ${deletable.length} ${deletable.length === 1 ? 'PROJECT' : 'PROJECTS'}`;
    if (deletable.length === 0 || bulkProjectDeleteConfirmation !== confirmation) return;

    setIsBulkDeleting(true);
    setBulkProjectDeleteError('');
    const deletedIds = new Set<string>();
    const failures: string[] = [];
    for (const item of deletable) {
      try {
        await deleteProject(item.project.id);
        deletedIds.add(item.project.id);
      } catch (error) {
        failures.push(
          `${item.project.name}: ${error instanceof Error ? error.message : 'delete failed'}`
        );
      }
    }

    setProjects((current) => current.filter((project) => !deletedIds.has(project.id)));
    setThreads((current) => current.filter((thread) => !deletedIds.has(thread.project_id || '')));
    setSelectedProjectIds((current) => new Set(
      Array.from(current).filter((projectId) => !deletedIds.has(projectId))
    ));
    if (deletedIds.size > 0) {
      onProjectDeleted?.(deletedIds.values().next().value as string);
    }
    if (failures.length === 0) {
      setBulkProjectDeleteOpen(false);
      clearThreadSelection();
    } else {
      setBulkProjectDeleteError(failures.join('\n'));
    }
    setIsBulkDeleting(false);
  }, [
    bulkProjectDeleteConfirmation,
    bulkProjectDeleteSummaries,
    clearThreadSelection,
    onProjectDeleted,
  ]);

  useEffect(() => {
    setSelectedThreadIds(new Set());
    setSelectedProjectIds(new Set());
    setLastSelectedThreadId(null);
    setIsSelectionMode(false);
  }, [activeProjectId]);

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

  const openForkDialog = (thread: Thread, event: React.MouseEvent) => {
    event.stopPropagation();
    setForkDialogThread(thread);
  };

  const handleForkThread = async (options: { name?: string; targetProjectId?: string; memoryCopyMode?: MemoryCopyMode }) => {
    if (!forkDialogThread) return;
    try {
      setForkingThreadId(forkDialogThread.id);
      const forked = await forkThread(forkDialogThread.id, options);
      setThreads(prev => [forked, ...prev]);
      setForkDialogThread(null);
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

  // Project model selection is immutable, so surface availability before creation.
  useEffect(() => {
    if (!newProjectEmbeddingModel) {
      setIsEmbeddingModelValid(null);
      setIsCheckingEmbeddingModel(false);
      return;
    }

    const validateEmbeddingModel = async () => {
      setIsCheckingEmbeddingModel(true);
      setIsEmbeddingModelValid(null);
      try {
        const isReady = await checkEmbeddingModelReady(newProjectEmbeddingModel);
        setIsEmbeddingModelValid(isReady);
      } catch (error) {
        setIsEmbeddingModelValid(false);
      } finally {
        setIsCheckingEmbeddingModel(false);
      }
    };

    validateEmbeddingModel();
  }, [newProjectEmbeddingModel]);

  const handleOpenCreateThreadDialog = useCallback((
    projectId?: string,
    lockProject = false,
  ) => {
    const now = new Date();
    setNewThreadName(`Thread ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`);
    if (projectId) setNewThreadProjectId(projectId);
    setIsThreadProjectLocked(Boolean(projectId) && lockProject);
    setCreateDialogOpen(true);
  }, []);

  const handleOpenCreateProjectDialog = useCallback(() => {
    setNewProjectReadsUserMemory(false);
    setCreateProjectDialogOpen(true);
  }, []);

  const headerState = useMemo<ThreadSidebarHeaderState>(() => ({
    projectCount: projects.length,
    threadCount: threads.length,
    activeProjectThreadCount: activeProjectId
      ? threads.filter((thread) => thread.project_id === activeProjectId).length
      : 0,
    deletionTarget,
    hasDeletableItems: eligibleItemCount > 0,
    isSelectionMode,
    selectedCount,
    allItemsSelected,
    someItemsSelected,
    isBulkDeleting,
    openCreateProjectDialog: selectionOnly ? () => undefined : handleOpenCreateProjectDialog,
    enterSelectionMode: selectionOnly ? () => undefined : enterThreadSelectionMode,
    clearSelection: clearThreadSelection,
    deleteSelected: deletionTarget === 'projects'
      ? handleRequestBulkDeleteProjects
      : handleBulkDeleteThreads,
    toggleAllItems: handleToggleAllItemsChecked,
  }), [
    allItemsSelected,
    clearThreadSelection,
    deletionTarget,
    eligibleItemCount,
    enterThreadSelectionMode,
    handleBulkDeleteThreads,
    handleOpenCreateProjectDialog,
    handleRequestBulkDeleteProjects,
    handleToggleAllItemsChecked,
    isBulkDeleting,
    isSelectionMode,
    selectedCount,
    someItemsSelected,
    projects.length,
    threads,
    activeProjectId,
    selectionOnly,
  ]);

  useEffect(() => {
    onHeaderStateChange?.(headerState);
  }, [headerState, onHeaderStateChange]);

  useEffect(() => {
    return () => {
      onHeaderStateChange?.(null);
    };
  }, [onHeaderStateChange]);

  const focusThreadInList = (threadId: string, event?: React.MouseEvent) => {
    event?.preventDefault();
    event?.stopPropagation();

    const row = threadRowRefs.current[threadId];
    if (!row) {
      const virtualIndex = virtualThreadRows.findIndex((item) => item.kind === 'thread' && item.thread.id === threadId);
      if (virtualIndex >= 0) {
        threadVirtualizer.scrollToIndex(virtualIndex, { align: 'center', behavior: 'smooth' });
        setFocusedThreadId(threadId);
      }
      return;
    }

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
            Project
          </Typography>
          <Typography variant="caption" component="div" sx={{ wordBreak: 'break-word' }}>
            {thread.project_id ? projectsById.get(thread.project_id)?.name || thread.project_id : 'Unassigned'}
          </Typography>
        </Box>
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
            {thread.embeddingModel}
          </Typography>
        </Box>
        {forkInfo?.parent_thread_id && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Parent
            </Typography>
            <ThreadReferenceChip
              threadId={forkInfo.parent_thread_id}
              fallbackName={forkInfo.parent_thread_name}
              threadsById={threadsById}
              onOpenThread={(target) => focusThreadInList(target.id)}
            />
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
                  <ThreadReferenceChip
                    threadId={childId}
                    threadsById={threadsById}
                    onOpenThread={(target) => focusThreadInList(target.id)}
                  />
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
        {forkInfo?.memory_copy_mode && (
          <Box sx={sectionSx}>
            <Typography variant="caption" color="text.secondary" component="div">
              Memory copy
            </Typography>
            <Typography variant="caption" component="div" sx={{ wordBreak: 'break-word' }}>
              {forkInfo.memory_copy_mode.replace(/_/g, ' ')}
              {Array.isArray(forkInfo.copied_memory_ids) ? ` (${forkInfo.copied_memory_ids.length})` : ''}
            </Typography>
          </Box>
        )}
      </Box>
    );
  };

  const threadActionButtonSx = {
    width: 32,
    height: 32,
    flex: '0 0 32px',
    color: 'text.secondary',
    bgcolor: 'transparent',
    opacity: 0.6,
    transition: 'color 160ms ease, opacity 160ms ease, outline-color 160ms ease',
    '&:hover, &.Mui-focusVisible': {
      color: 'text.primary',
      bgcolor: 'transparent',
      opacity: 1,
    },
    '&.Mui-focusVisible': {
      outline: `1px solid ${theme.palette.action.focus}`,
      outlineOffset: -2,
    },
    '&.Mui-disabled': {
      bgcolor: 'transparent',
      opacity: 0.45,
    },
  };
  const bulkDeletableProjects = bulkProjectDeleteSummaries.filter(
    (item) => item.summary?.can_delete
  );
  const bulkBlockedProjects = bulkProjectDeleteSummaries.filter(
    (item) => !item.summary?.can_delete
  );
  const bulkProjectConfirmationPhrase = (
    `DELETE ${bulkDeletableProjects.length} ${
      bulkDeletableProjects.length === 1 ? 'PROJECT' : 'PROJECTS'
    }`
  );

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
      {!hideHeader && (
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
            {!selectionOnly && <Tooltip title="Create new project">
              <IconButton
                size="small"
                color="primary"
                onClick={handleOpenCreateProjectDialog}
              >
                <AddIcon fontSize="small" />
              </IconButton>
            </Tooltip>}
          </Box>
          {!selectionOnly && (isSelectionMode ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Tooltip title={allItemsSelected ? "Clear selection" : `Select all ${deletionTarget}`}>
                <Checkbox
                  size="small"
                  checked={allItemsSelected}
                  indeterminate={someItemsSelected}
                  onChange={handleToggleAllThreads}
                  disabled={isBulkDeleting}
                  sx={{ p: 0.5 }}
                />
              </Tooltip>
              <Tooltip title={`${selectedCount} ${deletionTarget} selected`}>
                <Chip label={`${selectedCount} selected`} size="small" color="primary" />
              </Tooltip>
              <Tooltip title="Clear selection">
                <IconButton size="small" onClick={clearThreadSelection} disabled={isBulkDeleting}>
                  <ClearIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title={`Delete selected ${deletionTarget}`}>
                <span>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={
                      deletionTarget === 'projects'
                        ? handleRequestBulkDeleteProjects
                        : handleBulkDeleteThreads
                    }
                    disabled={isBulkDeleting || selectedCount === 0}
                  >
                    {isBulkDeleting ? <CircularProgress size={16} /> : <DeleteIcon fontSize="small" />}
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Tooltip title={
                deletionTarget === 'projects'
                  ? 'Select projects to delete'
                  : 'Select threads in this project to delete'
              }>
                <span>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={enterThreadSelectionMode}
                    disabled={eligibleItemCount === 0}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          ))}
        </Box>
      )}

      {/* Thread List */}
      <Collapse in={hideHeader || expanded} sx={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : virtualThreadRows.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              {activeProjectId ? 'No threads in this project' : 'No projects yet'}
            </Typography>
            {!selectionOnly && activeProjectId && (
              <Button size="small" startIcon={<AddIcon />} onClick={() => handleOpenCreateThreadDialog(activeProjectId || undefined)} sx={{ mt: 1 }}>
                Create Thread
              </Button>
            )}
          </Box>
        ) : (
          <List component="div" ref={threadListRef} dense sx={{ p: 0, overflow: 'auto', flex: 1, minHeight: 0 }}>
            <Box sx={{ height: `${threadVirtualizer.getTotalSize()}px`, position: 'relative', width: '100%' }}>
              {threadVirtualizer.getVirtualItems().map((virtualItem) => {
                const row = virtualThreadRows[virtualItem.index];
                if (!row) return null;
                if (row.kind === 'group') {
                  return (
                    <Box
                      key={virtualItem.key}
                      data-index={virtualItem.index}
                      ref={threadVirtualizer.measureElement}
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        transform: `translateY(${virtualItem.start}px)`,
                        px: 1.5,
                        py: 0.75,
                        bgcolor: activeProjectId === row.group.project?.id
                          ? alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.24 : 0.14)
                          : row.group.project && selectedProjectIds.has(row.group.project.id)
                            ? 'action.selected'
                            : 'action.hover',
                        transition: 'background-color 160ms ease',
                        '&:hover': {
                          bgcolor: activeProjectId === row.group.project?.id
                            ? alpha(theme.palette.primary.main, theme.palette.mode === 'dark' ? 0.24 : 0.14)
                            : 'action.selected',
                        },
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', ...flexTruncateSx }}>
                        {isSelectionMode && deletionTarget === 'projects' && row.group.project && (
                          <Checkbox
                            size="small"
                            checked={selectedProjectIds.has(row.group.project.id)}
                            onChange={(event) => {
                              event.stopPropagation();
                              const projectId = row.group.project!.id;
                              setSelectedProjectIds((current) => {
                                const next = new Set(current);
                                if (event.target.checked) next.add(projectId);
                                else next.delete(projectId);
                                return next;
                              });
                            }}
                            inputProps={{ 'aria-label': `Select project ${row.group.project.name}` }}
                            sx={{ p: 0.5, mr: 0.5 }}
                          />
                        )}
                        <Button
                          size="small"
                          disableRipple
                          color={activeProjectId === row.group.project?.id ? 'primary' : 'inherit'}
                          onClick={() => row.group.project && handleProjectClick(row.group.project)}
                          sx={{
                            flex: 1,
                            ...flexTruncateSx,
                            px: 0.5,
                            textTransform: 'none',
                            justifyContent: 'flex-start',
                            borderRadius: 0,
                            '&:hover, &:active, &.Mui-focusVisible': {
                              bgcolor: 'transparent',
                            },
                          }}
                        >
                          <Typography variant="caption" fontWeight={700} noWrap sx={singleLineTruncateSx}>
                            {row.group.project?.name || 'Unassigned'}
                          </Typography>
                        </Button>
                        {row.group.project && !selectionOnly ? (
                          <Box sx={{ display: 'flex', alignItems: 'center', flex: '0 0 auto' }}>
                          {activeProjectId === row.group.project.id && (
                            <Box sx={{ display: 'flex', mr: 0.25 }}>
                              <EmbeddingModelReadinessIndicator
                                model={row.group.project.embeddingModel}
                                ready={projectReadiness[row.group.project.id]}
                                size={16}
                              />
                            </Box>
                          )}
                          <Tooltip title={`Create thread in ${row.group.project.name}`}>
                            <IconButton
                              size="small"
                              aria-label={`Create thread in ${row.group.project.name}`}
                              onClick={() => handleOpenCreateThreadDialog(row.group.project!.id, true)}
                              sx={{ width: 24, height: 24 }}
                            >
                              <AddIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Project settings">
                            <IconButton
                              size="small"
                              onClick={() => handleOpenProjectSettings(row.group.project as Project)}
                              sx={{ width: 24, height: 24 }}
                            >
                              <SettingsIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                          </Tooltip>
                          </Box>
                        ) : null}
                      </Box>
                    </Box>
                  );
                }

                const thread = row.thread;
                return (
                  <Box
                    key={virtualItem.key}
                    data-index={virtualItem.index}
                    ref={(node: HTMLDivElement | null) => threadVirtualizer.measureElement(node)}
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      transform: `translateY(${virtualItem.start}px)`,
                    }}
                  >
                    <ListItem
                      ref={(node) => {
                        threadRowRefs.current[thread.id] = node;
                      }}
                      disablePadding
                      sx={{
                        display: 'flex',
                        alignItems: 'stretch',
                        bgcolor: activeThreadId === thread.id
                          ? theme.palette.mode === 'dark'
                            ? theme.palette.primary.dark
                            : theme.palette.primary.light
                          : focusedThreadId === thread.id
                            ? theme.palette.action.focus
                          : isSelectionMode
                            && deletionTarget === 'threads'
                            && selectedThreadIds.has(thread.id)
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
                            : isSelectionMode
                              && deletionTarget === 'threads'
                              && selectedThreadIds.has(thread.id)
                              ? theme.palette.action.selected
                            : theme.palette.mode === 'dark'
                              ? theme.palette.background.paper
                              : theme.palette.grey[100]
                        }
                      }}
                    >
                      <ListItemButton
                        onClick={() => handleThreadRowClick(thread)}
                        selected={activeThreadId === thread.id}
                        sx={{
                          flex: 1,
                          ...flexTruncateSx,
                          py: 1,
                          pr: 1,
                          bgcolor: 'transparent',
                          '&:hover, &.Mui-selected, &.Mui-selected:hover, &.Mui-focusVisible': {
                            bgcolor: 'transparent',
                          },
                        }}
                      >
                        {isSelectionMode
                          && deletionTarget === 'threads'
                          && thread.project_id === activeProjectId
                          && !selectionOnly && (
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
                          <Tooltip title={renderThreadTooltip(thread)} placement="left" arrow enterDelay={500} leaveDelay={150} disableInteractive={false}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', flex: '1 1 auto', ...flexTruncateSx }}>
                              <ListItemText
                                primary={
                                  <Typography variant="body2" fontWeight={activeThreadId === thread.id ? 'bold' : 'normal'} noWrap sx={singleLineTruncateSx}>
                                    {thread.name}
                                  </Typography>
                                }
                                secondaryTypographyProps={{ component: 'span' }}
                                secondary={
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, ...flexTruncateSx }}>
                                    <Typography variant="caption" color="text.secondary" noWrap sx={{ flex: '0 1 auto', ...singleLineTruncateSx }}>
                                      {formatDate(thread.created_at)}
                                    </Typography>
                                    {thread.message_count !== undefined && thread.message_count > 0 && (
                                      <Chip label={`${thread.message_count} msgs`} size="small" sx={{ height: 16, fontSize: '0.65rem' }} />
                                    )}
                                    {thread.file_count !== undefined && thread.file_count > 0 && (
                                      <Chip icon={<DescriptionIcon sx={{ fontSize: '0.7rem !important' }} />} label={thread.file_count} size="small" sx={{ height: 16, fontSize: '0.65rem' }} />
                                    )}
                                  </Box>
                                }
                                sx={{ m: 0, ...flexTruncateSx }}
                              />
                            </Box>
                          </Tooltip>
                        )}
                      </ListItemButton>

                      {!selectionOnly && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, flex: '0 0 auto', px: 1 }}>
                          <Tooltip title="Fork thread">
                            <span>
                              <IconButton size="small" onClick={(e) => openForkDialog(thread, e)} disabled={forkingThreadId === thread.id} sx={threadActionButtonSx}>
                                {forkingThreadId === thread.id ? <CircularProgress size={16} /> : <CallSplitIcon fontSize="small" />}
                              </IconButton>
                            </span>
                          </Tooltip>
                          <Tooltip title="Rename thread">
                            <IconButton size="small" onClick={(e) => startEditing(thread, e)} sx={threadActionButtonSx}>
                              <EditIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      )}
                    </ListItem>
                  </Box>
                );
              })}
            </Box>
            {false && groupedThreads.map((group) => (
              <Box key={group.project?.id || 'unassigned'}>
                <Box sx={{ px: 1.5, py: 0.75, bgcolor: 'action.hover', borderTop: 1, borderColor: 'divider' }}>
                  <Typography variant="caption" color="text.secondary" fontWeight={700} noWrap>
                    {group.project?.name || 'Unassigned'}
                  </Typography>
                </Box>
                {group.threads.map((thread) => (
                  <ListItem
                    key={thread.id}
                    ref={(node) => {
                      threadRowRefs.current[thread.id] = node;
                    }}
                    disablePadding
                    sx={{
                      display: 'flex',
                      alignItems: 'stretch',
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
                    <ListItemButton
                      onClick={() => handleThreadRowClick(thread)}
                      selected={activeThreadId === thread.id}
                      sx={{
                        flex: 1,
                        minWidth: 0,
                        py: 1,
                        pr: 1,
                        bgcolor: 'transparent',
                        '&:hover, &.Mui-selected, &.Mui-selected:hover, &.Mui-focusVisible': {
                          bgcolor: 'transparent',
                        },
                      }}
                    >
                      {isSelectionMode && !selectionOnly && (
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
                        <Tooltip title={renderThreadTooltip(thread)} placement="left" arrow enterDelay={500} leaveDelay={150} disableInteractive={false}>
                          <Box sx={{ display: 'inline-flex', flexDirection: 'column', minWidth: 0, maxWidth: '100%' }}>
                            <ListItemText
                              primary={
                                <Typography variant="body2" fontWeight={activeThreadId === thread.id ? 'bold' : 'normal'} noWrap>
                                  {thread.name}
                                </Typography>
                              }
                              secondaryTypographyProps={{ component: 'span' }}
                              secondary={
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, minWidth: 0, maxWidth: '100%' }}>
                                  <Typography variant="caption" color="text.secondary" noWrap>
                                    {formatDate(thread.created_at)}
                                  </Typography>
                                  {thread.message_count !== undefined && thread.message_count > 0 && (
                                    <Chip label={`${thread.message_count} msgs`} size="small" sx={{ height: 16, fontSize: '0.65rem' }} />
                                  )}
                                  {thread.file_count !== undefined && thread.file_count > 0 && (
                                    <Chip icon={<DescriptionIcon sx={{ fontSize: '0.7rem !important' }} />} label={thread.file_count} size="small" sx={{ height: 16, fontSize: '0.65rem' }} />
                                  )}
                                </Box>
                              }
                              sx={{ m: 0, minWidth: 0 }}
                            />
                          </Box>
                        </Tooltip>
                      )}
                    </ListItemButton>

                    {!isSelectionMode && !selectionOnly && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25, flex: '0 0 auto', px: 1 }}>
                        <Tooltip title="Fork thread">
                          <span>
                            <IconButton size="small" onClick={(e) => openForkDialog(thread, e)} disabled={forkingThreadId === thread.id} sx={threadActionButtonSx}>
                              {forkingThreadId === thread.id ? <CircularProgress size={16} /> : <CallSplitIcon fontSize="small" />}
                            </IconButton>
                          </span>
                        </Tooltip>
                        <Tooltip title="Rename thread">
                          <IconButton size="small" onClick={(e) => startEditing(thread, e)} sx={threadActionButtonSx}>
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    )}
                  </ListItem>
                ))}
              </Box>
            ))}
          </List>
        )}
      </Collapse>

      <Dialog
        open={bulkProjectDeleteOpen}
        onClose={() => {
          if (isBulkDeleting) return;
          setBulkProjectDeleteOpen(false);
          setBulkProjectDeleteError('');
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete selected projects</DialogTitle>
        <DialogContent>
          {bulkProjectDeleteLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Checking project contents and active runs...</Typography>
            </Box>
          ) : (
            <>
              {bulkProjectDeleteError && (
                <Alert severity="error" sx={{ mb: 2, whiteSpace: 'pre-line' }}>
                  {bulkProjectDeleteError}
                </Alert>
              )}
              <Typography variant="body2" color="text.secondary">
                {bulkDeletableProjects.length} selected {
                  bulkDeletableProjects.length === 1 ? 'project' : 'projects'
                } will be permanently deleted with {
                  bulkDeletableProjects.length === 1 ? 'its' : 'their'
                } threads, project memory, and unshared files.
              </Typography>
              {bulkDeletableProjects.length > 0 && (
                <Box sx={{ mt: 1.5 }}>
                  {bulkDeletableProjects.map(({ project }) => (
                    <Typography key={project.id} variant="body2" noWrap title={project.name}>
                      {project.name}
                    </Typography>
                  ))}
                </Box>
              )}
              {bulkBlockedProjects.length > 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  <Typography variant="body2" fontWeight={700} sx={{ mb: 0.5 }}>
                    These projects will be skipped
                  </Typography>
                  {bulkBlockedProjects.map(({ project, summary, error }) => (
                    <Typography key={project.id} variant="body2">
                      {project.name}: {error || (
                        summary?.protected
                          ? 'default project is protected'
                          : summary?.active_run_count
                            ? 'has active or awaiting-human runs'
                            : 'cannot be deleted'
                      )}
                    </Typography>
                  ))}
                </Alert>
              )}
              {bulkDeletableProjects.length > 0 && (
                <TextField
                  fullWidth
                  label={`Type "${bulkProjectConfirmationPhrase}" to confirm`}
                  value={bulkProjectDeleteConfirmation}
                  onChange={(event) => setBulkProjectDeleteConfirmation(event.target.value)}
                  disabled={isBulkDeleting}
                  sx={{ mt: 2 }}
                />
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setBulkProjectDeleteOpen(false)}
            disabled={isBulkDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleBulkDeleteProjects}
            disabled={
              bulkProjectDeleteLoading
              || isBulkDeleting
              || bulkDeletableProjects.length === 0
              || bulkProjectDeleteConfirmation !== bulkProjectConfirmationPhrase
            }
          >
            {isBulkDeleting
              ? <CircularProgress size={20} />
              : `Delete ${bulkDeletableProjects.length === 1 ? 'project' : 'projects'}`}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Create Thread Dialog */}
      <Dialog
        open={createDialogOpen}
        onClose={() => {
          setCreateDialogOpen(false);
          setIsThreadProjectLocked(false);
        }}
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
            <InputLabel>Project</InputLabel>
            <Select
              value={newThreadProjectId}
              label="Project"
              disabled={isThreadProjectLocked}
              onChange={(e) => setNewThreadProjectId(e.target.value)}
            >
              {projects.map((project) => (
                <MenuItem key={project.id} value={project.id}>{project.name}</MenuItem>
              ))}
            </Select>
          </FormControl>
          {!isThreadProjectLocked && (
            <Button
              size="small"
              startIcon={<AddIcon />}
              onClick={() => {
                setNewProjectReadsUserMemory(false);
                setCreateProjectDialogOpen(true);
              }}
              sx={{ mt: 1 }}
            >
              Create project
            </Button>
          )}
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'warning.light', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <LockIcon fontSize="small" />
              <Typography variant="caption" color="text.secondary">
                This thread inherits the project model
                {projectsById.get(newThreadProjectId)?.embeddingModel
                  ? ` (${projectsById.get(newThreadProjectId)?.embeddingModel})`
                  : ''}.
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setCreateDialogOpen(false);
            setIsThreadProjectLocked(false);
          }}>Cancel</Button>
          <Button
            onClick={handleCreateThread}
            variant="contained"
            disabled={!newThreadName.trim() || !newThreadProjectId || creating}
          >
            {creating ? <CircularProgress size={20} /> : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={Boolean(settingsProject)}
        onClose={() => !creating && setSettingsProject(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Project settings</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            margin="dense"
            label="Project name"
            value={settingsProjectName}
            onChange={(event) => setSettingsProjectName(event.target.value)}
            disabled={creating}
            inputProps={{ maxLength: 200 }}
            sx={{ mb: 1 }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={settingsProjectReadsUserMemory}
                onChange={(event) => setSettingsProjectReadsUserMemory(event.target.checked)}
              />
            }
            label="Allow global memory"
          />
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Applies immediately. Each thread keeps its own global-memory preference.
          </Typography>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Project actions</Typography>
          {projectActionError && (
            <Alert severity="error" sx={{ mb: 1.5 }}>{projectActionError}</Alert>
          )}
          {projectLifecycleLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
              <CircularProgress size={18} />
              <Typography variant="body2">Loading project details...</Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'grid', gap: 1 }}>
              <Button
                variant="outlined"
                startIcon={<ContentCopyIcon />}
                onClick={() => settingsProject && handleOpenCloneProject(settingsProject, false)}
                disabled={
                  creating
                  || !settingsProject
                  || !projectLifecycle?.can_clone
                  || projectReadiness[settingsProject.id] !== true
                }
                sx={{ justifyContent: 'flex-start' }}
              >
                Clone project
              </Button>
              <Button
                variant="outlined"
                startIcon={<ContentCopyIcon />}
                onClick={() => settingsProject && handleOpenCloneProject(settingsProject, true)}
                disabled={
                  creating
                  || !settingsProject
                  || !projectLifecycle?.can_clone
                  || projectReadiness[settingsProject.id] !== true
                }
                sx={{ justifyContent: 'flex-start' }}
              >
                Clone with threads
              </Button>
              <Button
                variant="outlined"
                color="error"
                startIcon={<DeleteForeverIcon />}
                onClick={() => settingsProject && handleOpenDeleteProject(settingsProject)}
                disabled={creating || !settingsProject || !projectLifecycle?.can_delete}
                sx={{ justifyContent: 'flex-start' }}
              >
                Delete project
              </Button>
              {projectLifecycle?.blocked_reason === 'active_agent_runs' && (
                <Typography variant="caption" color="warning.main">
                  Finish or cancel active agent runs before cloning or deleting this project.
                </Typography>
              )}
              {projectLifecycle?.protected && (
                <Typography variant="caption" color="text.secondary">
                  The default project cannot be deleted.
                </Typography>
              )}
              {settingsProject && projectReadiness[settingsProject.id] === false && (
                <Typography variant="caption" color="warning.main">
                  Cloning is unavailable while the locked embedding model is offline.
                </Typography>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsProject(null)} disabled={creating}>Cancel</Button>
          <Button
            onClick={handleSaveProjectSettings}
            variant="contained"
            disabled={creating || !settingsProjectName.trim()}
          >
            {creating ? <CircularProgress size={20} /> : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={Boolean(cloneProjectDialog)}
        onClose={() => !creating && setCloneProjectDialog(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>
          {cloneProjectDialog?.includeThreads ? 'Clone with threads' : 'Clone project'}
        </DialogTitle>
        <DialogContent>
          {projectActionError && (
            <Alert severity="error" sx={{ mb: 1.5 }}>{projectActionError}</Alert>
          )}
          <TextField
            autoFocus
            fullWidth
            label="Project name"
            value={cloneProjectName}
            onChange={(event) => setCloneProjectName(event.target.value)}
            disabled={creating}
            sx={{ mt: 0.5 }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
            {cloneProjectDialog?.includeThreads
              ? 'Copies shared knowledge, project and thread memory, completed conversations, annotations, and read-only historical debug traces.'
              : 'Copies shared knowledge, project settings, and active project memory into a new project.'}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCloneProjectDialog(null)} disabled={creating}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleCloneProject}
            disabled={creating || !cloneProjectName.trim()}
          >
            {creating ? <CircularProgress size={20} /> : 'Clone'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={Boolean(deleteProjectDialog)}
        onClose={() => !creating && setDeleteProjectDialog(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Delete project</DialogTitle>
        <DialogContent>
          {projectActionError && (
            <Alert severity="error" sx={{ mb: 1.5 }}>{projectActionError}</Alert>
          )}
          <Typography variant="body2" color="text.secondary">
            This permanently deletes {projectLifecycle?.thread_count || 0} threads,{' '}
            and {projectLifecycle?.memory_count || 0} memories.
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {projectLifecycle?.shared_file_count || 0} shared files will remain available to
            other projects. {projectLifecycle?.orphan_file_count || 0} unreferenced files will
            be permanently removed.
          </Typography>
          <TextField
            fullWidth
            label={`Type "${deleteProjectDialog?.name || ''}" to confirm`}
            value={deleteProjectConfirmation}
            onChange={(event) => setDeleteProjectConfirmation(event.target.value)}
            disabled={creating}
            sx={{ mt: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteProjectDialog(null)} disabled={creating}>Cancel</Button>
          <Button
            variant="contained"
            color="error"
            onClick={handleDeleteProject}
            disabled={
              creating
              || !deleteProjectDialog
              || !projectDeletionConfirmed(
                deleteProjectConfirmation,
                deleteProjectDialog.name,
              )
            }
          >
            {creating ? <CircularProgress size={20} /> : 'Delete permanently'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog
        open={createProjectDialogOpen}
        onClose={() => {
          setCreateProjectDialogOpen(false);
          setNewProjectReadsUserMemory(false);
        }}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Create Project</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Project Name"
            fullWidth
            value={newProjectName}
            onChange={(event) => setNewProjectName(event.target.value)}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            value={newProjectDescription}
            onChange={(event) => setNewProjectDescription(event.target.value)}
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={newProjectEmbeddingModel}
              label="Embedding Model"
              onChange={(event) => setNewProjectEmbeddingModel(String(event.target.value))}
            >
              {[...availableEmbeddingModels.embedding_models, ...availableEmbeddingModels.local_embedding_models].map((model) => (
                <MenuItem key={model} value={model}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getModelIcon(model)}
                    {model}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControlLabel
            sx={{ mt: 1 }}
            control={
              <Switch
                checked={newProjectReadsUserMemory}
                onChange={(event) => setNewProjectReadsUserMemory(event.target.checked)}
              />
            }
            label="Allow global memory"
          />
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
            Threads must also opt in before this project can recall global memory.
          </Typography>
          {isCheckingEmbeddingModel && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
              <CircularProgress size={20} />
              <Typography variant="body2">Checking embedding model...</Typography>
            </Box>
          )}
          {isEmbeddingModelValid === false && !isCheckingEmbeddingModel && (
            <Typography color="warning.main" variant="body2" sx={{ mt: 1 }}>
              This model is currently unavailable. The project can be created, but its threads will be read-only until the model is available.
            </Typography>
          )}
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'warning.light', borderRadius: 1, display: 'flex', gap: 1 }}>
            <LockIcon fontSize="small" />
            <Typography variant="caption" color="text.secondary">
              The project embedding model is permanent. Every thread and project-scoped memory uses this model.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setCreateProjectDialogOpen(false);
            setNewProjectReadsUserMemory(false);
          }}>Cancel</Button>
          <Button
            onClick={handleCreateProject}
            variant="contained"
            disabled={!newProjectName.trim() || !newProjectEmbeddingModel || creating || isCheckingEmbeddingModel}
          >
            {creating ? <CircularProgress size={20} /> : 'Create Project'}
          </Button>
        </DialogActions>
      </Dialog>
      <ThreadForkDialog
        open={Boolean(forkDialogThread)}
        sourceThread={forkDialogThread}
        projects={projects}
        submitting={Boolean(forkingThreadId)}
        onClose={() => setForkDialogThread(null)}
        onSubmit={handleForkThread}
      />
    </Paper>
  );
};

export default ThreadSidebar;
