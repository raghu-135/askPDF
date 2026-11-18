import React from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ReplayIcon from '@mui/icons-material/Replay';
import type { ContextBudgetDefaults, PromptDefaults } from '../lib/api';

type AdvancedFieldSpec = NonNullable<PromptDefaults['advanced_field_specs']>[number];

export type ThreadAdvancedSettingsValue = {
  contextWindow: number;
  contextBudget: ContextBudgetDefaults;
  coverageIterations: Record<string, number>;
};

export type ThreadAdvancedSettingsSectionProps = {
  defaults: ThreadAdvancedSettingsValue;
  value: ThreadAdvancedSettingsValue;
  fieldSpecs: AdvancedFieldSpec[];
  onChange: (next: ThreadAdvancedSettingsValue) => void;
  onResetField: (fieldKey: string) => void;
  onResetSection: () => void;
};

function labelForSpec(spec: AdvancedFieldSpec): string {
  return spec.key.replace(/_/g, ' ');
}

const ThreadAdvancedSettingsSection: React.FC<ThreadAdvancedSettingsSectionProps> = ({
  defaults,
  value,
  fieldSpecs,
  onChange,
  onResetField,
  onResetSection,
}) => {
  const ratioFields = fieldSpecs.filter((spec) => spec.type === 'ratio');
  const intFields = fieldSpecs.filter((spec) => spec.type === 'integer');

  const updateBudget = (key: keyof ContextBudgetDefaults, raw: string) => {
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) return;
    onChange({
      ...value,
      contextBudget: { ...value.contextBudget, [key]: parsed },
    });
  };

  const updateCoverage = (key: string, raw: string) => {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed)) return;
    onChange({
      ...value,
      coverageIterations: { ...value.coverageIterations, [key]: parsed },
    });
  };

  return (
    <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: 'divider', '&:before': { display: 'none' } }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ width: '100%', pr: 1 }}>
          <Typography variant="subtitle2">Advanced</Typography>
          <Tooltip title="Reset entire Advanced section to defaults">
            <IconButton
              size="small"
              onClick={(event) => {
                event.stopPropagation();
                onResetSection();
              }}
            >
              <ReplayIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </AccordionSummary>
      <AccordionDetails>
        <Stack spacing={2}>
          <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
            <TextField
              fullWidth
              type="number"
              label="Context window (tokens)"
              value={value.contextWindow}
              onChange={(e) => onChange({ ...value, contextWindow: Number(e.target.value) })}
              inputProps={{ min: 256, max: 2_000_000 }}
            />
            <Tooltip title="Reset context window to default">
              <IconButton size="small" sx={{ mt: 1 }} onClick={() => onResetField('context_window')}>
                <ReplayIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          <Typography variant="caption" color="text.secondary">
            Context packing ratios (defaults from environment)
          </Typography>
          {ratioFields.map((spec) => {
            const key = spec.key as keyof ContextBudgetDefaults;
            return (
              <Box key={spec.key} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                <TextField
                  fullWidth
                  type="number"
                  label={labelForSpec(spec)}
                  value={value.contextBudget[key] ?? defaults.contextBudget[key]}
                  onChange={(e) => updateBudget(key, e.target.value)}
                  inputProps={{ min: 0.01, max: 1, step: 0.01 }}
                />
                <Tooltip title={`Reset ${labelForSpec(spec)} to default`}>
                  <IconButton size="small" sx={{ mt: 1 }} onClick={() => onResetField(spec.key)}>
                    <ReplayIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            );
          })}
          <Typography variant="caption" color="text.secondary">
            Coverage iteration limits
          </Typography>
          {intFields
            .filter((spec) => spec.key !== 'context_window')
            .map((spec) => (
              <Box key={spec.key} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                <TextField
                  fullWidth
                  type="number"
                  label={labelForSpec(spec)}
                  value={value.coverageIterations[spec.key] ?? defaults.coverageIterations[spec.key]}
                  onChange={(e) => updateCoverage(spec.key, e.target.value)}
                  inputProps={{ min: spec.minimum, max: spec.maximum }}
                />
                <Tooltip title={`Reset ${labelForSpec(spec)} to default`}>
                  <IconButton size="small" sx={{ mt: 1 }} onClick={() => onResetField(spec.key)}>
                    <ReplayIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
};

export default ThreadAdvancedSettingsSection;