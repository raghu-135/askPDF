import React from 'react';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  Typography,
} from '@mui/material';
import type {
  AgentWorkflowValidationReport,
} from '../../lib/api';
import type { BuilderSelection, BuilderValidationIssue } from './types';

export default function BuilderValidationPanel({
  validation,
  issues,
  onSelectIssue,
  onApplyFix,
}: {
  validation: AgentWorkflowValidationReport | null;
  issues: BuilderValidationIssue[];
  onSelectIssue: (selection: BuilderSelection) => void;
  onApplyFix: (issue: BuilderValidationIssue) => void;
}) {
  const hasErrors = issues.some((issue) => issue.severity === 'error');
  const hasWarnings = issues.some((issue) => issue.severity === 'warning');

  return (
    <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mb: 1, bgcolor: 'background.paper' }}>
      <Box sx={{ p: 1 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {hasErrors ? <ErrorOutlineIcon fontSize="small" color="error" /> : validation?.valid ? <CheckCircleIcon fontSize="small" color="success" /> : null}
            Validation
          </Typography>
            {!validation ? (
              <Alert severity="info">Run validation to check the assembled graph against the backend validator.</Alert>
            ) : validation.valid && issues.length === 0 ? (
              <Alert severity="success">Backend validation passed.</Alert>
            ) : (
              <>
                <Alert severity={hasErrors ? 'error' : hasWarnings ? 'warning' : 'info'}>
                  {hasErrors ? 'Backend validation found errors.' : 'Backend validation returned warnings.'}
                </Alert>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
                  {issues.map((issue) => (
                    <Box
                      key={issue.id}
                      sx={{
                        display: 'grid',
                        gridTemplateColumns: 'auto minmax(0, 1fr) auto',
                        gap: 1,
                        alignItems: 'center',
                        p: 0.75,
                        border: 1,
                        borderColor: issue.severity === 'error' ? 'error.light' : 'warning.light',
                        borderRadius: 1,
                      }}
                    >
                      {issue.severity === 'error' ? <ErrorOutlineIcon color="error" fontSize="small" /> : <WarningAmberIcon color="warning" fontSize="small" />}
                      <Box sx={{ minWidth: 0 }}>
                        {issue.code ? (
                          <Chip
                            size="small"
                            variant="outlined"
                            label={issue.code.replaceAll('_', ' ')}
                            sx={{ mb: 0.5, height: 20, fontSize: '0.65rem' }}
                          />
                        ) : null}
                        <Typography variant="caption" sx={{ display: 'block', wordBreak: 'break-word' }}>
                          {issue.message}
                        </Typography>
                      </Box>
                      {issue.fix ? (
                        <Button size="small" onClick={() => onApplyFix(issue)} sx={{ whiteSpace: 'nowrap' }}>
                          Fix
                        </Button>
                      ) : issue.selection ? (
                          <Button size="small" onClick={() => onSelectIssue(issue.selection)} sx={{ whiteSpace: 'nowrap' }}>
                            Select
                          </Button>
                      ) : (
                        <Chip size="small" variant="outlined" label="global" />
                      )}
                    </Box>
                  ))}
                </Box>
              </>
            )}
        </Box>
      </Box>
    </Box>
  );
}
