import React from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  FormControlLabel,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import LockIcon from '@mui/icons-material/Lock';
import type { Project, ProjectLifecycleSummary } from '../../lib/api';

export default function ProjectSettingsForm({
  projectName,
  onProjectNameChange,
  allowGlobalMemory,
  onAllowGlobalMemoryChange,
  embeddingModel,
  disabled = false,
  lifecycle,
  lifecycleLoading = false,
  lifecycleError = '',
  projectReady = true,
  onCloneProject,
  onCloneProjectWithThreads,
  onDeleteProject,
}: {
  projectName: string;
  onProjectNameChange: (value: string) => void;
  allowGlobalMemory: boolean;
  onAllowGlobalMemoryChange: (value: boolean) => void;
  embeddingModel?: string;
  disabled?: boolean;
  lifecycle?: ProjectLifecycleSummary | null;
  lifecycleLoading?: boolean;
  lifecycleError?: string;
  projectReady?: boolean;
  onCloneProject?: () => void;
  onCloneProjectWithThreads?: () => void;
  onDeleteProject?: () => void;
}) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
        Project settings
      </Typography>
      <TextField
        fullWidth
        label="Project name"
        value={projectName}
        onChange={(event) => onProjectNameChange(event.target.value)}
        disabled={disabled}
        inputProps={{ maxLength: 200 }}
      />
      <FormControlLabel
        control={
          <Switch
            checked={allowGlobalMemory}
            onChange={(event) => onAllowGlobalMemoryChange(event.target.checked)}
            disabled={disabled}
          />
        }
        label="Allow global memory"
      />
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
        Applies immediately. Each thread keeps its own global-memory preference.
      </Typography>
      {embeddingModel ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LockIcon fontSize="small" color="action" />
          <Typography variant="body2" color="text.secondary">
            Embedding model: <strong>{embeddingModel}</strong> (locked at creation)
          </Typography>
        </Box>
      ) : null}
      {(onCloneProject || onCloneProjectWithThreads || onDeleteProject) ? (
        <>
          <Divider />
          <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Project actions</Typography>
          {lifecycleError ? <Alert severity="error">{lifecycleError}</Alert> : null}
          {lifecycleLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1 }}>
              <CircularProgress size={18} />
              <Typography variant="body2">Loading project details...</Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'grid', gap: 1 }}>
              {onCloneProject ? (
                <Button
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={onCloneProject}
                  disabled={disabled || !lifecycle?.can_clone || !projectReady}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Clone project
                </Button>
              ) : null}
              {onCloneProjectWithThreads ? (
                <Button
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={onCloneProjectWithThreads}
                  disabled={disabled || !lifecycle?.can_clone || !projectReady}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Clone with threads
                </Button>
              ) : null}
              {onDeleteProject ? (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<DeleteForeverIcon />}
                  onClick={onDeleteProject}
                  disabled={disabled || !lifecycle?.can_delete}
                  sx={{ justifyContent: 'flex-start' }}
                >
                  Delete project
                </Button>
              ) : null}
              {lifecycle?.blocked_reason === 'active_agent_runs' ? (
                <Typography variant="caption" color="warning.main">
                  Finish or cancel active agent runs before cloning or deleting this project.
                </Typography>
              ) : null}
              {lifecycle?.protected ? (
                <Typography variant="caption" color="text.secondary">
                  The default project cannot be deleted.
                </Typography>
              ) : null}
              {!projectReady ? (
                <Typography variant="caption" color="warning.main">
                  Cloning is unavailable while the locked embedding model is offline.
                </Typography>
              ) : null}
            </Box>
          )}
        </>
      ) : null}
    </Box>
  );
}
