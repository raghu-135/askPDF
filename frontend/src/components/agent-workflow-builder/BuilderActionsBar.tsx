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
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import {
  Box,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Tooltip,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import type { AgentWorkflow, AgentWorkflowValidationReport } from '../../lib/api';
import type { AgentWorkflowStarter } from '../../lib/agent-workflow-builder';
import { WorkbenchToolbarTrailingActions } from '../workbench/WorkbenchToolbar';

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
  darkMode,
  onToggleDarkMode,
  layoutControl,
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
  darkMode: boolean;
  onToggleDarkMode: () => void;
  layoutControl?: React.ReactNode;
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
        alignItems: 'center',
        gap: 0.75,
        minHeight: 48,
        px: 1,
        py: 0.5,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        overflowX: 'auto',
        overflowY: 'hidden',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'nowrap', minWidth: 0, flex: '1 1 auto' }}>
        {validationChip}
        <FormControl size="small" disabled={disabled} sx={{ minWidth: 180 }}>
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
        <Tooltip title="Undo">
          <span><IconButton size="small" aria-label="Undo" onClick={onUndo} disabled={disabled || !canUndo}><UndoIcon fontSize="small" /></IconButton></span>
        </Tooltip>
        <Tooltip title="Redo">
          <span><IconButton size="small" aria-label="Redo" onClick={onRedo} disabled={disabled || !canRedo}><RedoIcon fontSize="small" /></IconButton></span>
        </Tooltip>
        <Tooltip title="Reset workflow">
          <span><IconButton size="small" aria-label="Reset workflow" onClick={onReset} disabled={disabled}><RestartAltIcon fontSize="small" /></IconButton></span>
        </Tooltip>
        <Tooltip title={validating ? 'Validating workflow' : 'Validate workflow'}>
          <span>
            <IconButton size="small" aria-label="Validate workflow" onClick={onValidate} disabled={validating}>
              {validating ? <CircularProgress size={18} /> : <PlayArrowIcon fontSize="small" />}
            </IconButton>
          </span>
        </Tooltip>
        <Box sx={{ minWidth: 0, display: { xs: 'none', sm: 'block' }, maxWidth: 180 }}>
          <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }} noWrap>
            {workflowName || 'Untitled workflow'}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }} noWrap>
            {savedWorkflowId ? `Saved · ${savedWorkflowId}` : dirty ? 'Unsaved changes' : 'Not saved yet'}
          </Typography>
        </Box>
        <Tooltip title={saveBusy ? 'Saving workflow' : 'Save workflow'}>
          <span>
            <IconButton color="primary" size="small" aria-label="Save workflow" onClick={onOpenSave} disabled={disabled || saveBusy}>
              {saveBusy ? <CircularProgress size={18} /> : <SaveIcon fontSize="small" />}
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={testMode ? 'Back to Builder' : 'Test workflow'}>
          <IconButton size="small" color={testMode ? 'default' : 'primary'} aria-label={testMode ? 'Back to Builder' : 'Test workflow'} onClick={onToggleTest}>
            {testMode ? <ArrowBackIcon fontSize="small" /> : <ScienceIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
        {testMode && hasTestSession && onClearTestSession && (
          <Tooltip title="Clear test session">
            <IconButton size="small" aria-label="Clear test session" onClick={onClearTestSession}>
              <DeleteSweepIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <WorkbenchToolbarTrailingActions>
        <Tooltip title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
          <IconButton
            size="small"
            color={darkMode ? 'primary' : 'default'}
            onClick={onToggleDarkMode}
            aria-label={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {darkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
        {layoutControl}
      </WorkbenchToolbarTrailingActions>
    </Box>
  );
}
