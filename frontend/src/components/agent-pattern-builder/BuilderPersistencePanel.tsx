import React from 'react';
import ArchiveIcon from '@mui/icons-material/Archive';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DriveFileRenameOutlineIcon from '@mui/icons-material/DriveFileRenameOutline';
import PublishIcon from '@mui/icons-material/Publish';
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
  selectThreadId: string;
}

export interface BuilderPersistedPattern {
  template: AgentPatternTemplate;
  version: AgentPatternVersion;
}

export default function BuilderPersistencePanel({
  form,
  onFormChange,
  persisted,
  busyAction,
  statusMessage,
  errorMessage,
  canSave,
  onGenerateTemplateId,
  onSave,
  onPublish,
  onArchive,
  onSelectForThread,
}: {
  form: BuilderPersistenceState;
  onFormChange: (patch: Partial<BuilderPersistenceState>) => void;
  persisted: BuilderPersistedPattern | null;
  busyAction: 'save' | 'publish' | 'archive' | 'select' | null;
  statusMessage: string | null;
  errorMessage: string | null;
  canSave: boolean;
  onGenerateTemplateId: () => void;
  onSave: () => void;
  onPublish: () => void;
  onArchive: () => void;
  onSelectForThread: () => void;
}) {
  const savedTemplateId = persisted?.template.id;
  const savedVersion = persisted?.version.version;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Save And Select
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Internal custom pattern version
        </Typography>
      </Box>
      <Divider />
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
        onChange={(event) => onFormChange({ name: event.target.value })}
      />
      <Box sx={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) auto', gap: 1, alignItems: 'start' }}>
        <TextField
          size="small"
          label="Template ID"
          value={form.templateId}
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
        onChange={(event) => onFormChange({ description: event.target.value })}
      />
      <TextField
        size="small"
        label="Owner ID"
        value={form.ownerId}
        onChange={(event) => onFormChange({ ownerId: event.target.value })}
      />
      <TextField
        size="small"
        label="Changelog"
        value={form.changelog}
        multiline
        minRows={2}
        onChange={(event) => onFormChange({ changelog: event.target.value })}
      />
      <Button
        size="small"
        variant="contained"
        startIcon={<SaveIcon />}
        disabled={!canSave || busyAction === 'save'}
        onClick={onSave}
        sx={{ borderRadius: 1 }}
      >
        {busyAction === 'save' ? 'Saving' : 'Save Internal Version'}
      </Button>
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<PublishIcon />}
          disabled={!persisted || busyAction === 'publish'}
          onClick={onPublish}
          sx={{ borderRadius: 1 }}
        >
          Publish
        </Button>
        <Button
          size="small"
          variant="outlined"
          color="warning"
          startIcon={<ArchiveIcon />}
          disabled={!persisted || busyAction === 'archive'}
          onClick={onArchive}
          sx={{ borderRadius: 1 }}
        >
          Archive
        </Button>
      </Box>
      <Divider />
      <TextField
        size="small"
        label="Thread ID"
        value={form.selectThreadId}
        onChange={(event) => onFormChange({ selectThreadId: event.target.value })}
        helperText="Selects the saved current internal pattern for this thread."
      />
      <Button
        size="small"
        variant="contained"
        color="secondary"
        startIcon={<CloudUploadIcon />}
        disabled={!persisted || !form.selectThreadId.trim() || busyAction === 'select'}
        onClick={onSelectForThread}
        sx={{ borderRadius: 1 }}
      >
        {busyAction === 'select' ? 'Selecting' : 'Select For Thread'}
      </Button>
    </Box>
  );
}

