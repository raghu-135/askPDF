import React, { useEffect, useMemo, useState } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import type { Project, Thread } from '../lib/api';

export type MemoryCopyMode = 'thread_snapshot' | 'project_snapshot' | 'all';

interface ThreadForkDialogProps {
  open: boolean;
  sourceThread: Thread | null;
  projects: Project[];
  fromMessageId?: string | null;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (options: {
    name?: string;
    targetProjectId?: string;
    memoryCopyMode?: MemoryCopyMode;
  }) => void;
}

const memoryModeLabels: Record<MemoryCopyMode, string> = {
  thread_snapshot: 'Thread snapshot',
  project_snapshot: 'Project snapshot',
  all: 'Thread and project snapshots',
};

const ThreadForkDialog: React.FC<ThreadForkDialogProps> = ({
  open,
  sourceThread,
  projects,
  fromMessageId,
  submitting = false,
  onClose,
  onSubmit,
}) => {
  const [name, setName] = useState('');
  const [targetProjectId, setTargetProjectId] = useState('');
  const [memoryCopyMode, setMemoryCopyMode] = useState<MemoryCopyMode>('thread_snapshot');
  const sourceProjectId = sourceThread?.project_id || '';
  const isCrossProjectFork = Boolean(targetProjectId && sourceProjectId && targetProjectId !== sourceProjectId);

  useEffect(() => {
    if (!open || !sourceThread) return;
    setName(`${sourceThread.name} (Fork)`);
    setTargetProjectId(sourceThread.project_id || projects[0]?.id || '');
    setMemoryCopyMode('thread_snapshot');
  }, [open, projects, sourceThread]);

  useEffect(() => {
    if (!open || !sourceThread) return;
    setMemoryCopyMode(isCrossProjectFork ? 'project_snapshot' : 'thread_snapshot');
  }, [isCrossProjectFork, open, sourceThread]);

  const selectableModes = useMemo<MemoryCopyMode[]>(
    () => (isCrossProjectFork ? ['project_snapshot', 'all'] : ['thread_snapshot', 'all']),
    [isCrossProjectFork]
  );

  if (!sourceThread) return null;

  return (
    <Dialog open={open} onClose={submitting ? undefined : onClose} fullWidth maxWidth="xs">
      <DialogTitle>{fromMessageId ? 'Fork from message' : 'Fork thread'}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <TextField
            label="Thread name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            fullWidth
            autoFocus
          />
          <FormControl fullWidth>
            <InputLabel>Project</InputLabel>
            <Select
              value={targetProjectId}
              label="Project"
              onChange={(event) => setTargetProjectId(String(event.target.value))}
            >
              {projects.map((project) => (
                <MenuItem key={project.id} value={project.id}>
                  {project.name}
                </MenuItem>
              ))}
              {!projects.length && (
                <MenuItem value={sourceThread.project_id || ''}>
                  {sourceThread.project_id || 'Unassigned'}
                </MenuItem>
              )}
            </Select>
          </FormControl>
          <FormControl fullWidth>
            <InputLabel>Memory copy</InputLabel>
            <Select
              value={memoryCopyMode}
              label="Memory copy"
              onChange={(event) => setMemoryCopyMode(event.target.value as MemoryCopyMode)}
            >
              {selectableModes.map((mode) => (
                <MenuItem key={mode} value={mode}>
                  {memoryModeLabels[mode]}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Typography variant="caption" color="text.secondary">
            {isCrossProjectFork
              ? 'Project memories are copied into the selected project as snapshots.'
              : 'Project memory stays shared; thread memory is copied as a branch snapshot.'}
          </Typography>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={submitting}>Cancel</Button>
        <Button
          variant="contained"
          disabled={submitting || !name.trim()}
          onClick={() => onSubmit({
            name: name.trim(),
            targetProjectId: targetProjectId || undefined,
            memoryCopyMode,
          })}
        >
          Fork
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ThreadForkDialog;
