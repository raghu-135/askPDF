import React from 'react';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import SaveIcon from '@mui/icons-material/Save';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import type { AgentPatternTemplate, AgentPatternVersion } from '../../lib/api';

export interface BuilderPersistenceState {
  templateId: string;
  name: string;
  description: string;
}

export interface BuilderPersistedPattern {
  template: AgentPatternTemplate;
  version: AgentPatternVersion;
}

export interface BuilderBoundaryMessage {
  severity: 'info' | 'warning' | 'error';
  message: string;
}

export default function BuilderPersistencePanel({
  form,
  onFormChange,
  persisted,
  busyAction,
  statusMessage,
  errorMessage,
  canSave,
  authoringDisabled,
  boundaryMessages,
  onSave,
  onDelete,
}: {
  form: BuilderPersistenceState;
  onFormChange: (patch: Partial<BuilderPersistenceState>) => void;
  persisted: BuilderPersistedPattern | null;
  busyAction: 'save' | 'delete' | null;
  statusMessage: string | null;
  errorMessage: string | null;
  canSave: boolean;
  authoringDisabled?: boolean;
  boundaryMessages?: BuilderBoundaryMessage[];
  onSave: () => void;
  onDelete: () => void;
}) {
  const savedTemplateId = persisted?.template.id;
  const saveDisabled = authoringDisabled || !canSave || busyAction === 'save';
  const deleteDisabled = authoringDisabled || !persisted || Boolean(persisted.template.is_builtin) || Boolean(busyAction);
  const formDisabled = authoringDisabled || Boolean(busyAction);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Save Workflow
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Custom workflow available to every thread after saving
        </Typography>
      </Box>
      <Divider />
      {(boundaryMessages || []).map((message) => (
        <Alert key={`${message.severity}-${message.message}`} severity={message.severity}>
          {message.message}
        </Alert>
      ))}
      {statusMessage ? <Alert severity="success">{statusMessage}</Alert> : null}
      {errorMessage ? <Alert severity="error">{errorMessage}</Alert> : null}
      {persisted ? (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
          <Chip size="small" color="success" label="Saved" />
          <Chip size="small" variant="outlined" label={savedTemplateId} />
        </Box>
      ) : null}
      <TextField
        size="small"
        label="Name"
        value={form.name}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ name: event.target.value })}
      />
      <TextField
        size="small"
        label="Description"
        value={form.description}
        multiline
        minRows={2}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ description: event.target.value })}
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 1 }}>
        <Button
          size="small"
          variant="contained"
          startIcon={<SaveIcon />}
          disabled={saveDisabled}
          onClick={onSave}
          sx={{ borderRadius: 1 }}
        >
          {busyAction === 'save' ? 'Saving' : 'Save Workflow'}
        </Button>
        <Tooltip title={persisted?.template.is_builtin ? 'Built-in workflows cannot be deleted' : 'Delete custom workflow'}>
          <span>
            <Button
              size="small"
              variant="outlined"
              color="error"
              startIcon={<DeleteOutlineIcon />}
              disabled={deleteDisabled}
              onClick={onDelete}
              sx={{ borderRadius: 1, minHeight: 30 }}
            >
              {busyAction === 'delete' ? 'Deleting' : 'Delete'}
            </Button>
          </span>
        </Tooltip>
      </Box>
      <Divider />
      <Typography variant="caption" color="text.secondary">
        Saved compatible workflows appear in the Agent workflow menu for all threads.
      </Typography>
    </Box>
  );
}
