import React, { useState } from 'react';
import {
  Box,
  Checkbox,
  Chip,
  CircularProgress,
  IconButton,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ClearIcon from '@mui/icons-material/Clear';
import DeleteIcon from '@mui/icons-material/Delete';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import FolderOutlinedIcon from '@mui/icons-material/FolderOutlined';
import ThreadSidebar, { type ThreadSidebarHeaderState } from './ThreadSidebar';
import type { Project, Thread } from '../lib/api';

const countLabel = (count: number, noun: string) => (
  `${count} ${noun}${count === 1 ? '' : 's'}`
);

export default function ThreadSecondaryPanel({
  activeThread,
  activeProject,
  threadProject,
  activeThreadId,
  activeProjectId,
  sidebarKey,
  selectionOnly = false,
  darkMode = false,
  onThreadSelect,
  onProjectSelect,
  onProjectReadinessChange,
  onProjectUpdated,
  onProjectCloned,
  onProjectDeleted,
  onThreadForked,
  onBackToProject,
  renderConversation,
  renderSelectedTitle,
  selectedActions,
}: {
  activeThread: Thread | null;
  activeProject?: Project | null;
  threadProject?: Project | null;
  activeThreadId?: string | null;
  activeProjectId?: string | null;
  sidebarKey?: React.Key;
  selectionOnly?: boolean;
  darkMode?: boolean;
  onThreadSelect: (thread: Thread | null) => void;
  onProjectSelect?: (project: Project) => void;
  onProjectReadinessChange?: (projectId: string, ready: boolean | null) => void;
  onProjectUpdated?: (project: Project) => void;
  onProjectCloned?: (project: Project) => void;
  onProjectDeleted?: (projectId: string) => void;
  onThreadForked?: (thread: Thread) => void;
  onBackToProject: () => void;
  renderConversation?: (thread: Thread) => React.ReactNode;
  renderSelectedTitle?: (thread: Thread) => React.ReactNode;
  selectedActions?: React.ReactNode;
}) {
  const [headerState, setHeaderState] = useState<ThreadSidebarHeaderState | null>(null);
  const parentLabel = threadProject?.name || (activeThread?.project_id ? 'Project' : 'Threads');

  const renderListActions = () => {
    if (!headerState || selectionOnly) return null;

    if (headerState.isSelectionMode) {
      const itemLabel = headerState.deletionTarget === 'projects' ? 'projects' : 'threads';
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Tooltip title={headerState.allItemsSelected ? 'Clear selection' : `Select all ${itemLabel}`}>
            <Checkbox
              size="small"
              checked={headerState.allItemsSelected}
              indeterminate={headerState.someItemsSelected}
              onChange={(event) => headerState.toggleAllItems(event.target.checked)}
              disabled={headerState.isBulkDeleting}
              sx={{ p: 0.5 }}
            />
          </Tooltip>
          <Tooltip title={`${headerState.selectedCount} ${itemLabel} selected`}>
            <Chip label={`${headerState.selectedCount} selected`} size="small" color="primary" />
          </Tooltip>
          <Tooltip title="Clear selection">
            <IconButton size="small" onClick={headerState.clearSelection} disabled={headerState.isBulkDeleting}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title={`Delete selected ${itemLabel}`}>
            <span>
              <IconButton
                size="small"
                color="error"
                onClick={headerState.deleteSelected}
                disabled={headerState.isBulkDeleting || headerState.selectedCount === 0}
              >
                {headerState.isBulkDeleting ? <CircularProgress size={16} /> : <DeleteIcon fontSize="small" />}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      );
    }

    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <Tooltip title="Create new project">
          <span>
            <IconButton
              size="small"
              color="primary"
              aria-label="Create new project"
              onClick={headerState.openCreateProjectDialog}
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={
          headerState.deletionTarget === 'projects'
            ? 'Select projects to delete'
            : 'Select threads in this project to delete'
        }>
          <span>
            <IconButton
              size="small"
              color="error"
              onClick={headerState.enterSelectionMode}
              disabled={!headerState.hasDeletableItems}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
    );
  };

  const header = (
    <Box
      sx={{
        minHeight: 49,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        display: 'flex',
        alignItems: 'stretch',
        flexShrink: 0,
      }}
    >
      {activeThread && renderConversation ? (
        <>
          <Box
            sx={{
              flex: '0 1 auto',
              maxWidth: '42%',
              px: 1,
              display: 'flex',
              alignItems: 'center',
              minWidth: 0,
            }}
          >
            <Tooltip title={`Back to ${parentLabel}`}>
              <Chip
                size="small"
                icon={<ArrowBackIcon fontSize="small" />}
                label={parentLabel}
                onClick={onBackToProject}
                color="primary"
                sx={{
                  minWidth: 0,
                  maxWidth: '100%',
                  '& .MuiChip-label': {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  },
                }}
              />
            </Tooltip>
          </Box>
          <Box
            sx={{
              flex: 1,
              minWidth: 0,
              px: 1.5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 1,
            }}
          >
            {renderSelectedTitle ? renderSelectedTitle(activeThread) : (
              <Typography variant="subtitle2" fontWeight={700} noWrap sx={{ minWidth: 0 }}>
                {activeThread.name}
              </Typography>
            )}
            {selectedActions}
          </Box>
        </>
      ) : (
        <>
          <Box
            sx={{
              flex: 1,
              minWidth: 0,
              px: 1.5,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <FolderOutlinedIcon fontSize="small" color="primary" />
            <Tooltip title={activeProject?.name || 'Projects'}>
              <Typography variant="subtitle1" fontWeight={700} noWrap>
                {activeProject?.name || 'Projects'}
              </Typography>
            </Tooltip>
            {activeProject ? (
              <Chip
                label={countLabel(headerState?.activeProjectThreadCount ?? 0, 'thread')}
                size="small"
              />
            ) : (
              <>
                <Chip label={countLabel(headerState?.projectCount ?? 0, 'project')} size="small" />
                <Chip label={countLabel(headerState?.threadCount ?? 0, 'thread')} size="small" />
              </>
            )}
          </Box>
          <Box
            sx={{
              flex: '0 0 auto',
              px: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-end',
            }}
          >
            {renderListActions()}
          </Box>
        </>
      )}
    </Box>
  );

  return (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)' }}>
      {header}
      <Box sx={{ minHeight: 0, overflow: 'hidden' }}>
        {activeThread && renderConversation ? (
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {renderConversation(activeThread)}
          </Box>
        ) : (
          <Box sx={{ height: '100%', overflow: 'auto' }}>
            <ThreadSidebar
              key={sidebarKey}
              activeThreadId={activeThreadId ?? activeThread?.id ?? null}
              activeProjectId={activeProjectId}
              onThreadSelect={onThreadSelect}
              onProjectSelect={onProjectSelect}
              onProjectReadinessChange={onProjectReadinessChange}
              onProjectUpdated={onProjectUpdated}
              onProjectCloned={onProjectCloned}
              onProjectDeleted={onProjectDeleted}
              onThreadForked={onThreadForked}
              hideHeader
              onHeaderStateChange={setHeaderState}
              darkMode={darkMode}
              selectionOnly={selectionOnly}
            />
          </Box>
        )}
      </Box>
    </Box>
  );
}
