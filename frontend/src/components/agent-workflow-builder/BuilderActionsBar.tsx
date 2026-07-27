import React from 'react';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import UndoIcon from '@mui/icons-material/Undo';
import RedoIcon from '@mui/icons-material/Redo';
import SaveIcon from '@mui/icons-material/Save';
import ScienceIcon from '@mui/icons-material/Science';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import {
  Box,
  Button,
  Chip,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import type { AgentWorkflow, AgentWorkflowValidationReport } from '../../lib/api';
import type { AgentWorkflowStarter } from '../../lib/agent-workflow-builder';

export default function BuilderActionsBar({
  starter,
  customWorkflows,
  disabled,
  onStarterChange,
  onReset,
  onValidate,
  validating,
  validation,
  dirty,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onOpenSave,
  saveBusy,
  savedWorkflowId,
  workflowName,
  testMode,
  onToggleTest,
  hasTestSession,
  onClearTestSession,
}: {
  starter: string;
  customWorkflows?: AgentWorkflow[];
  disabled?: boolean;
  onStarterChange: (starter: AgentWorkflowStarter | string) => void;
  onReset: () => void;
  onValidate: () => void;
  validating: boolean;
  validation: AgentWorkflowValidationReport | null;
  dirty?: boolean;
  canUndo?: boolean;
  canRedo?: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onOpenSave: () => void;
  saveBusy?: boolean;
  savedWorkflowId?: string;
  workflowName?: string;
  testMode?: boolean;
  onToggleTest: () => void;
  hasTestSession?: boolean;
  onClearTestSession?: () => void;
}) {
  const validationChip = validation ? (
    <Chip
      size="small"
      color={validation.valid ? 'success' : 'error'}
      icon={validation.valid ? <CheckCircleIcon /> : <ErrorOutlineIcon />}
      label={validation.valid ? 'Valid' : `${validation.errors?.length || 0} errors`}
    />
  ) : null;
  const customOptions = customWorkflows || [];

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: { xs: 'stretch', md: 'center' },
        justifyContent: 'space-between',
        gap: 1,
        flexWrap: 'wrap',
        px: 2,
        py: 1.25,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
          Agent Workflow Builder
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Build, connect, validate, and test an agent workflow {dirty ? '· Unsaved changes' : '· Saved'}
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        {validationChip}
        <FormControl size="small" disabled={disabled} sx={{ minWidth: 190 }}>
          <InputLabel id="builder-starter-label">Workflow</InputLabel>
          <Select
            labelId="builder-starter-label"
            label="Workflow"
            value={starter}
            onChange={(event: SelectChangeEvent) => onStarterChange(event.target.value)}
          >
            <MenuItem value="router">Router</MenuItem>
            <MenuItem value="plan_execute">Plan Execute</MenuItem>
            <MenuItem value="evaluator_replanner">Evaluator/Replanner</MenuItem>
            {customOptions.length ? <Divider /> : null}
            {customOptions.map((pattern) => (
              <MenuItem key={pattern.id} value={`custom:${pattern.id}`}>
                {pattern.name || pattern.id}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Button size="small" startIcon={<UndoIcon />} onClick={onUndo} disabled={disabled || !canUndo}>Undo</Button>
        <Button size="small" startIcon={<RedoIcon />} onClick={onRedo} disabled={disabled || !canRedo}>Redo</Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<RestartAltIcon />}
          onClick={onReset}
          disabled={disabled}
          sx={{ borderRadius: 1 }}
        >
          Reset
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<PlayArrowIcon />}
          onClick={onValidate}
          disabled={validating}
          sx={{ borderRadius: 1 }}
        >
          {validating ? 'Validating' : 'Validate'}
        </Button>
        <Box sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' }, maxWidth: 180 }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }} noWrap>
            {workflowName || 'Untitled workflow'}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
            {savedWorkflowId ? `Saved · ${savedWorkflowId}` : dirty ? 'Unsaved changes' : 'Not saved yet'}
          </Typography>
        </Box>
        <Button
          size="small"
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={onOpenSave}
          disabled={disabled || saveBusy}
          sx={{ borderRadius: 1 }}
        >
          {saveBusy ? 'Saving' : 'Save'}
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={testMode ? <ArrowBackIcon /> : <ScienceIcon />}
          onClick={onToggleTest}
          sx={{ borderRadius: 1 }}
        >
          {testMode ? 'Back to Builder' : 'Test'}
        </Button>
        {testMode && hasTestSession && onClearTestSession && (
          <Button
            size="small"
            startIcon={<DeleteSweepIcon />}
            onClick={onClearTestSession}
          >
            Clear test
          </Button>
        )}
      </Box>
    </Box>
  );
}
