import React from 'react';
import DriveFileRenameOutlineIcon from '@mui/icons-material/DriveFileRenameOutline';
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
  ownerId: string;
  changelog: string;
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
  onGenerateTemplateId,
  onSave,
}: {
  form: BuilderPersistenceState;
  onFormChange: (patch: Partial<BuilderPersistenceState>) => void;
  persisted: BuilderPersistedPattern | null;
  busyAction: 'save' | null;
  statusMessage: string | null;
  errorMessage: string | null;
  canSave: boolean;
  authoringDisabled?: boolean;
  boundaryMessages?: BuilderBoundaryMessage[];
  onGenerateTemplateId: () => void;
  onSave: () => void;
}) {
  const savedTemplateId = persisted?.template.id;
  const savedVersion = persisted?.version.version;
  const saveDisabled = authoringDisabled || !canSave || busyAction === 'save';
  const formDisabled = authoringDisabled || Boolean(busyAction);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Save Pattern
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Internal custom pattern version available to every thread
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
          <Chip size="small" variant="outlined" label={`v${savedVersion}`} />
        </Box>
      ) : null}
      <TextField
        size="small"
        label="Name"
        value={form.name}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ name: event.target.value })}
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 1, alignItems: 'start' }}>
        <TextField
          size="small"
          label="Template ID"
          value={form.templateId}
          disabled={formDisabled}
          onChange={(event) => onFormChange({ templateId: event.target.value })}
          helperText="Internal ID; must not collide with built-ins."
        />
        <Tooltip title="Generate from name">
          <span>
            <Button
              size="small"
              variant="outlined"
              startIcon={<DriveFileRenameOutlineIcon />}
              onClick={onGenerateTemplateId}
              disabled={formDisabled}
              sx={{ borderRadius: 1, minHeight: 40 }}
            >
              Slug
            </Button>
          </span>
        </Tooltip>
      </Box>
      <TextField
        size="small"
        label="Description"
        value={form.description}
        multiline
        minRows={2}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ description: event.target.value })}
      />
      <TextField
        size="small"
        label="Owner ID"
        value={form.ownerId}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ ownerId: event.target.value })}
      />
      <TextField
        size="small"
        label="Changelog"
        value={form.changelog}
        multiline
        minRows={2}
        disabled={formDisabled}
        onChange={(event) => onFormChange({ changelog: event.target.value })}
      />
      <Button
        size="small"
        variant="contained"
        startIcon={<SaveIcon />}
        disabled={saveDisabled}
        onClick={onSave}
        sx={{ borderRadius: 1 }}
      >
        {busyAction === 'save' ? 'Saving' : 'Save Internal Version'}
      </Button>
      <Divider />
      <Typography variant="caption" color="text.secondary">
        Saved compatible patterns appear in the Agent pattern menu for all threads.
      </Typography>
    </Box>
  );
}
