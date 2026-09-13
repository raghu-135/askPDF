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
  workflowIsValid,
  serviceError,
  onSelectIssue,
}: {
  validation: AgentWorkflowValidationReport | null;
  issues: BuilderValidationIssue[];
  workflowIsValid: boolean;
  serviceError?: string | null;
  onSelectIssue: (selection: BuilderSelection) => void;
}) {
  const hasErrors = issues.some((issue) => issue.severity === 'error');
  const hasWarnings = issues.some((issue) => issue.severity === 'warning');

  return (
    <Box sx={{ flexShrink: 0, bgcolor: 'background.paper' }}>
      <Box sx={{ px: 1.25, py: 1 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Typography variant="subtitle2" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {hasErrors ? <ErrorOutlineIcon fontSize="small" color="error" /> : workflowIsValid ? <CheckCircleIcon fontSize="small" color="success" /> : null}
            Validation
          </Typography>
          {serviceError ? <Alert severity="error">Validation service error: {serviceError}</Alert> : null}
            {!validation ? (
              <Alert severity="info">Run validation to check the assembled graph against the backend validator.</Alert>
            ) : workflowIsValid ? (
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
                      {issue.selection ? (
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
