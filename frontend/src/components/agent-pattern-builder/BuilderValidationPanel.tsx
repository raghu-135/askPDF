import React, { useState } from 'react';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import PreviewIcon from '@mui/icons-material/Preview';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  Tab,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import dynamic from 'next/dynamic';
import type {
  AgentPatternCatalogResponse,
  AgentPatternValidationReport,
  ThreadAgentConfigPreviewResponse,
} from '../../lib/api';
import type { AgentPatternBuilderSpec } from '../../lib/api';
import { JsonPreview } from '../agent-graph/AgentGraphInspectorPrimitives';
import type { BuilderSelection, BuilderValidationIssue } from './types';

const AgentGraphCanvas = dynamic(() => import('../agent-graph/AgentGraphCanvas'), { ssr: false });

export default function BuilderValidationPanel({
  catalog,
  spec,
  validation,
  issues,
  threadPreviewId,
  onThreadPreviewIdChange,
  onPreviewThread,
  previewing,
  previewResult,
  previewError,
  onSelectIssue,
}: {
  catalog: AgentPatternCatalogResponse;
  spec: AgentPatternBuilderSpec;
  validation: AgentPatternValidationReport | null;
  issues: BuilderValidationIssue[];
  threadPreviewId: string;
  onThreadPreviewIdChange: (threadId: string) => void;
  onPreviewThread: () => void;
  previewing: boolean;
  previewResult: ThreadAgentConfigPreviewResponse | null;
  previewError: string | null;
  onSelectIssue: (selection: BuilderSelection) => void;
}) {
  const [tab, setTab] = useState<'validation' | 'spec' | 'thread-preview'>('validation');
  const hasErrors = issues.some((issue) => issue.severity === 'error');
  const hasWarnings = issues.some((issue) => issue.severity === 'warning');
  const previewSpec = previewResult?.resolved_spec_json || spec;
  const previewPrompt = previewResult?.prompt_preview || previewResult?.prompt;

  return (
    <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mb: 1, bgcolor: 'background.paper' }}>
      <Box sx={{ px: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Tabs
          value={tab}
          onChange={(_, value) => setTab(value)}
          sx={{ minHeight: 34, '& .MuiTab-root': { minHeight: 34, py: 0, fontSize: '0.78rem' } }}
        >
          <Tab
            value="validation"
            label={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                {hasErrors ? <ErrorOutlineIcon fontSize="small" color="error" /> : validation?.valid ? <CheckCircleIcon fontSize="small" color="success" /> : null}
                Validation
              </Box>
            }
          />
          <Tab value="spec" label="Spec Preview" />
          <Tab value="thread-preview" label="Thread Preview" />
        </Tabs>
      </Box>
      <Box sx={{ p: 1 }}>
        {tab === 'validation' ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
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
                      <Typography variant="caption" sx={{ wordBreak: 'break-word' }}>
                        {issue.message}
                      </Typography>
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
        ) : null}
        {tab === 'spec' ? (
          <Box>
            <Typography variant="caption" color="text.secondary">
              Assembled schema v2 custom workflow spec sent to validation/save endpoints.
            </Typography>
            <JsonPreview value={spec} maxHeight={360} />
          </Box>
        ) : null}
        {tab === 'thread-preview' ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto' }, gap: 1 }}>
              <TextField
                size="small"
                label="Thread ID"
                value={threadPreviewId}
                onChange={(event) => onThreadPreviewIdChange(event.target.value)}
              />
              <Button
                size="small"
                variant="outlined"
                startIcon={<PreviewIcon />}
                disabled={!threadPreviewId.trim() || previewing}
                onClick={onPreviewThread}
                sx={{ borderRadius: 1, whiteSpace: 'nowrap' }}
              >
                {previewing ? 'Previewing' : 'Preview Thread'}
              </Button>
            </Box>
            {previewError ? <Alert severity="error">{previewError}</Alert> : null}
            {previewResult ? (
              <Alert severity={previewResult.validation?.valid === false ? 'warning' : 'success'}>
                Thread preview returned {previewResult.template_id || 'a workflow'}{previewResult.template_version ? ` v${previewResult.template_version}` : ''}.
              </Alert>
            ) : (
              <Alert severity="info">
                Thread preview uses the thread-specific backend endpoint. If that endpoint is not available in this checkout, the error is shown here.
              </Alert>
            )}
            {previewResult?.validation?.errors?.length ? (
              <Box>
                {previewResult.validation.errors.map((message, index) => (
                  <Typography key={`${message}-${index}`} variant="caption" color="error" sx={{ display: 'block' }}>
                    {message}
                  </Typography>
                ))}
              </Box>
            ) : null}
            <Divider />
            <Typography variant="caption" sx={{ fontWeight: 700 }}>
              Graph Preview
            </Typography>
            <AgentGraphCanvas
              resolvedSpec={previewSpec}
              nodeCatalog={catalog.node_catalog}
              mode="builder"
              showInspector={false}
            />
            {previewPrompt ? (
              <TextField
                label="Prompt Preview"
                value={previewPrompt}
                multiline
                minRows={8}
                maxRows={16}
                slotProps={{ input: { readOnly: true } }}
              />
            ) : null}
          </Box>
        ) : null}
      </Box>
    </Box>
  );
}
