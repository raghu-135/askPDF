import React, { useState } from 'react';
import {
  Box,
  Button,
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
import ForumIcon from '@mui/icons-material/Forum';
import ThreadSidebar, { type ThreadSidebarHeaderState } from './ThreadSidebar';
import type { Project, Thread } from '../lib/api';

export default function ThreadSecondaryPanel({
  activeThread,
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
  onClearThread,
  renderConversation,
  renderSelectedTitle,
  selectedActions,
}: {
  activeThread: Thread | null;
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
  onClearThread: () => void;
  renderConversation?: (thread: Thread) => React.ReactNode;
  renderSelectedTitle?: (thread: Thread) => React.ReactNode;
  selectedActions?: React.ReactNode;
}) {
  const [headerState, setHeaderState] = useState<ThreadSidebarHeaderState | null>(null);

  const renderListActions = () => {
    if (!headerState || selectionOnly) return null;

    if (headerState.isSelectionMode) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Tooltip title={headerState.allThreadsSelected ? 'Clear selection' : 'Select all threads. Shift-click a thread to select a range.'}>
            <Checkbox
              size="small"
              checked={headerState.allThreadsSelected}
              indeterminate={headerState.someThreadsSelected}
              onChange={(event) => headerState.toggleAllThreads(event.target.checked)}
              disabled={headerState.isBulkDeleting}
              sx={{ p: 0.5 }}
            />
          </Tooltip>
          <Tooltip title="Shift-click threads to select a range">
            <Chip label={`${headerState.selectedCount} selected`} size="small" color="primary" />
          </Tooltip>
          <Tooltip title="Clear selection">
            <IconButton size="small" onClick={headerState.clearSelection} disabled={headerState.isBulkDeleting}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete selected threads">
            <span>
              <IconButton
                size="small"
                color="error"
                onClick={headerState.deleteSelectedThreads}
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
        <Tooltip title="Create new thread">
          <span>
            <IconButton
              size="small"
              color="primary"
              onClick={headerState.openCreateDialog}
              disabled={!headerState.hasThreads && !headerState.openCreateDialog}
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Select threads to delete. Shift-click a thread to select a range.">
          <span>
            <IconButton
              size="small"
              color="error"
              onClick={headerState.enterSelectionMode}
              disabled={!headerState.hasThreads}
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
              width: 132,
              flex: '0 0 132px',
              px: 1,
              display: 'flex',
              alignItems: 'center',
              minWidth: 0,
            }}
          >
            <Button
              size="small"
              startIcon={<ForumIcon fontSize="small" />}
              onClick={onClearThread}
              sx={{ textTransform: 'none', minWidth: 0 }}
            >
              All threads
            </Button>
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
            <ForumIcon fontSize="small" color="primary" />
            <Typography variant="subtitle1" fontWeight={700} noWrap>
              Threads
            </Typography>
            <Chip label={headerState?.threadCount ?? 0} size="small" />
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
