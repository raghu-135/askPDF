import React, { useMemo, useState } from 'react';
import {
  Alert, Box, Button, Chip, Collapse, Stack, TextField, Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import type { AgentRunPendingInterrupt, AgentTaskSummary } from '../../lib/api';
import {
  budgetUsageFromInterrupt,
  buildDecisionLeadLine,
  countReviewIssues,
  formatBudgetMeterLine,
  RESEARCH_BUDGET_TOOLTIP,
} from '../../lib/deep-research-decision-ui';
import { ConversationMarkdown } from './ConversationMarkdown';

function warningSummaryLines(pending: AgentRunPendingInterrupt): string[] {
  const lines: string[] = [];
  const gaps = Array.isArray(pending.gaps) ? pending.gaps.map(String) : [];
  gaps.forEach((gap) => { if (gap.trim()) lines.push(gap.trim()); });
  const warnings = Array.isArray(pending.warnings) ? pending.warnings : [];
  warnings.forEach((entry) => {
    if (!entry || typeof entry !== 'object') return;
    const code = String((entry as { code?: string }).code || 'warning');
    const details = (entry as { details?: Record<string, unknown> }).details;
    const issues = details && Array.isArray(details.issues) ? details.issues : [];
    if (issues.length) {
      issues.forEach((issue) => {
        if (typeof issue === 'string') lines.push(issue);
        else if (issue && typeof issue === 'object') {
          const message = String((issue as { message?: string }).message || (issue as { summary?: string }).summary || code);
          lines.push(message);
        }
      });
    } else {
      lines.push(code.replaceAll('_', ' '));
    }
  });
  return lines;
}

export function DeepResearchBudgetSummary({
  pending,
  task,
}: {
  pending: AgentRunPendingInterrupt;
  task: AgentTaskSummary | null | undefined;
}) {
  const snapshot = useMemo(() => budgetUsageFromInterrupt(pending, task), [pending, task]);
  if (!snapshot) return null;
  return (
    <Stack direction="row" spacing={0.75} flexWrap="wrap" alignItems="center">
      <Chip
        size="small"
        variant="outlined"
        title={RESEARCH_BUDGET_TOOLTIP}
        label={`Tranche ${snapshot.trancheIndex} · ${formatBudgetMeterLine(snapshot)}`}
      />
    </Stack>
  );
}

export function DeepResearchIssueList({
  pending,
  defaultExpanded,
}: {
  pending: AgentRunPendingInterrupt;
  defaultExpanded?: boolean;
}) {
  const lines = useMemo(() => warningSummaryLines(pending), [pending]);
  const [open, setOpen] = useState(Boolean(defaultExpanded));
  if (!lines.length) return null;
  return (
    <Box>
      <Button
        size="small"
        color="inherit"
        endIcon={<ExpandMoreIcon sx={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />}
        onClick={() => setOpen((value) => !value)}
        sx={{ textTransform: 'none', px: 0 }}
      >
        {lines.length} limitation{lines.length === 1 ? '' : 's'} identified
      </Button>
      <Collapse in={open}>
        <Box component="ul" sx={{ m: 0, pl: 2.5, typography: 'body2', color: 'text.secondary' }}>
          {lines.map((line, index) => <li key={`${index}-${line.slice(0, 24)}`}>{line}</li>)}
        </Box>
      </Collapse>
    </Box>
  );
}

export function DeepResearchDecisionLead({
  pending,
  task,
  isBudgetReview,
}: {
  pending: AgentRunPendingInterrupt;
  task: AgentTaskSummary | null | undefined;
  isBudgetReview: boolean;
}) {
  const snapshot = useMemo(() => budgetUsageFromInterrupt(pending, task), [pending, task]);
  const { warningCount, gapCount } = countReviewIssues(pending);
  const text = buildDecisionLeadLine({
    isBudgetReview,
    snapshot,
    issueCount: warningCount,
    gapCount,
    fallbackBody: pending.body,
  });
  return <Typography variant="body2">{text}</Typography>;
}

export function DeepResearchDecisionStickyFooter({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <Box sx={{
      position: 'sticky',
      bottom: 0,
      zIndex: 1,
      bgcolor: 'background.default',
      borderTop: 1,
      borderColor: 'divider',
      pt: 1,
      mt: 0.5,
      display: 'flex',
      flexDirection: 'column',
      gap: 1,
    }}>
      {children}
    </Box>
  );
}

export function DeepResearchGuidanceField({
  label,
  helperText,
  value,
  onChange,
  required,
}: {
  label: string;
  helperText?: string;
  value: string;
  onChange: (value: string) => void;
  required?: boolean;
}) {
  return (
    <TextField
      fullWidth
      multiline
      minRows={2}
      maxRows={6}
      label={label}
      helperText={helperText}
      required={required}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  );
}

export function DeepResearchProvisionalAnswer({
  content,
  emptyMessage,
}: {
  content?: string | null;
  emptyMessage?: string;
}) {
  if (content?.trim()) {
    return <ConversationMarkdown content={String(content)} />;
  }
  if (emptyMessage) {
    return <Alert severity="warning">{emptyMessage}</Alert>;
  }
  return null;
}
