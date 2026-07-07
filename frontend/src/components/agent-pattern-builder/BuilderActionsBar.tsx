import React from 'react';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import {
  Box,
  Button,
  Chip,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import type { AgentPatternValidationReport } from '../../lib/api';
import type { AgentPatternStarter } from '../../lib/agent-pattern-builder';

export default function BuilderActionsBar({
  starter,
  onStarterChange,
  onReset,
  onValidate,
  validating,
  validation,
}: {
  starter: AgentPatternStarter;
  onStarterChange: (starter: AgentPatternStarter) => void;
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
          Agent Pattern Builder
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Internal graph authoring surface
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        {validationChip}
        <FormControl size="small" sx={{ minWidth: 190 }}>
          <InputLabel id="builder-starter-label">Starter</InputLabel>
          <Select
            labelId="builder-starter-label"
            label="Starter"
            value={starter}
            onChange={(event: SelectChangeEvent) => onStarterChange(event.target.value as AgentPatternStarter)}
          >
            <MenuItem value="router">Router</MenuItem>
            <MenuItem value="plan_execute">Plan Execute</MenuItem>
            <MenuItem value="evaluator_replanner">Evaluator/Replanner</MenuItem>
          </Select>
        </FormControl>
        <Button
          size="small"
          variant="outlined"
          startIcon={<RestartAltIcon />}
          onClick={onReset}
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

