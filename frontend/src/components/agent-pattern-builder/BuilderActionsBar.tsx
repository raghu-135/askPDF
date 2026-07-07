import React from 'react';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
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
import type { AgentWorkflow, AgentPatternValidationReport } from '../../lib/api';
import type { AgentPatternStarter } from '../../lib/agent-pattern-builder';

export default function BuilderActionsBar({
  starter,
  customWorkflows,
  disabled,
  onStarterChange,
  onReset,
  onValidate,
  validating,
  validation,
}: {
  starter: string;
  customWorkflows?: AgentWorkflow[];
  disabled?: boolean;
  onStarterChange: (starter: AgentPatternStarter | string) => void;
  onReset: () => void;
  onValidate: () => void;
  validating: boolean;
  validation: AgentPatternValidationReport | null;
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
          Internal graph authoring surface
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
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={onValidate}
          disabled={validating}
          sx={{ borderRadius: 1 }}
        >
          {validating ? 'Validating' : 'Validate'}
        </Button>
      </Box>
    </Box>
  );
}
