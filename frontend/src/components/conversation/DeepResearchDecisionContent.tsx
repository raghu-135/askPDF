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
  formatBudgetDimensionChip,
  RESEARCH_BUDGET_DIMENSIONS,
  RESEARCH_BUDGET_TOOLTIP,
  reviewSummaryLines,
} from '../../lib/deep-research-decision-ui';
import { buildCitationDisplayMap, rewriteDocCitationTokens } from '../../lib/research-citation-display';
import type { CitationDisplaySource } from '../../lib/research-citation-display';
import { ConversationDisclosure } from './ConversationDisclosure';
import { ConversationMarkdown } from './ConversationMarkdown';

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
    <Stack direction="row" spacing={0.75} flexWrap="wrap" alignItems="center" useFlexGap>
      <Chip size="small" variant="outlined" label={`Tranche ${snapshot.trancheIndex}`} title={RESEARCH_BUDGET_TOOLTIP} />
      {RESEARCH_BUDGET_DIMENSIONS.map((key) => {
        const limit = snapshot.trancheLimits[key];
        if (limit == null) return null;
        const used = snapshot.trancheUsage[key] || 0;
        return (
          <Chip
            key={key}
            size="small"
            variant="outlined"
            title={RESEARCH_BUDGET_TOOLTIP}
            label={formatBudgetDimensionChip(key, used, limit)}
            sx={{ maxWidth: '100%', height: 'auto', '& .MuiChip-label': { whiteSpace: 'normal', py: 0.25 } }}
          />
        );
      })}
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
  const lines = useMemo(() => reviewSummaryLines(pending), [pending]);
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
  citationSources,
  defaultCollapsed = true,
}: {
  content?: string | null;
  emptyMessage?: string;
  citationSources?: CitationDisplaySource[];
  defaultCollapsed?: boolean;
}) {
  const displayContent = useMemo(() => {
    const raw = String(content || '');
    if (!raw.trim()) return raw;
    const map = buildCitationDisplayMap(citationSources);
    return rewriteDocCitationTokens(raw, map);
  }, [content, citationSources]);

  if (displayContent?.trim()) {
    if (defaultCollapsed) {
      return (
        <ConversationDisclosure label="Provisional report" defaultExpanded={false}>
          <ConversationMarkdown content={displayContent} />
        </ConversationDisclosure>
      );
    }
    return <ConversationMarkdown content={displayContent} />;
  }
  if (emptyMessage) {
    return <Alert severity="warning">{emptyMessage}</Alert>;
  }
  return null;
}
