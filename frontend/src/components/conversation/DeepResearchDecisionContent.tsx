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
  reviewLimitationItems,
} from '../../lib/deep-research-decision-ui';
import { DeepResearchBudgetChips } from './DeepResearchBudgetChips';
import { buildCitationDisplayMap, rewriteDocCitationTokens } from '../../lib/research-citation-display';
import type { CitationDisplaySource } from '../../lib/research-citation-display';
import { ConversationMarkdownWithCitations } from './ConversationMarkdownWithCitations';

const RUNTIME_REPORT_SECTION_PATTERN = /\n##\s+Unresolved required work[\s\S]*$/i;

function stripRuntimeReportSections(markdown: string): string {
  return markdown.replace(RUNTIME_REPORT_SECTION_PATTERN, '').trimEnd();
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
  return <DeepResearchBudgetChips snapshot={snapshot} />;
}

export function DeepResearchIssueList({
  pending,
  defaultExpanded,
  todoTitles,
}: {
  pending: AgentRunPendingInterrupt;
  defaultExpanded?: boolean;
  todoTitles?: Record<string, string>;
}) {
  const items = useMemo(() => reviewLimitationItems(pending, todoTitles), [pending, todoTitles]);
  const [open, setOpen] = useState(Boolean(defaultExpanded));
  if (!items.length) return null;
  return (
    <Box sx={{ flexShrink: 0 }}>
      <Button
        size="small"
        color="inherit"
        endIcon={<ExpandMoreIcon sx={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />}
        onClick={() => setOpen((value) => !value)}
        sx={{ textTransform: 'none', px: 0 }}
      >
        {items.length} limitation{items.length === 1 ? '' : 's'} identified
      </Button>
      <Collapse in={open}>
        <Box component="ul" sx={{ m: 0, pl: 0, listStyle: 'none', typography: 'body2', color: 'text.secondary', pb: 0.5 }}>
          {items.map((item, index) => (
            <Box component="li" key={`${index}-${item.text.slice(0, 24)}`} sx={{ display: 'flex', gap: 0.75, alignItems: 'flex-start', mb: 0.75 }}>
              <Chip
                size="small"
                variant="outlined"
                label={item.kind === 'gap' ? 'Gap' : 'Review'}
                sx={{ height: 20, flexShrink: 0, '& .MuiChip-label': { px: 0.75, fontSize: '0.65rem' } }}
              />
              <Typography component="span" variant="body2" color="text.secondary">{item.text}</Typography>
            </Box>
          ))}
        </Box>
      </Collapse>
    </Box>
  );
}

export function DeepResearchDecisionLead({
  pending,
  task,
  isBudgetReview,
  todoTitles,
}: {
  pending: AgentRunPendingInterrupt;
  task: AgentTaskSummary | null | undefined;
  isBudgetReview: boolean;
  todoTitles?: Record<string, string>;
}) {
  const snapshot = useMemo(() => budgetUsageFromInterrupt(pending, task), [pending, task]);
  const counts = countReviewIssues(pending, todoTitles);
  const text = buildDecisionLeadLine({
    isBudgetReview,
    snapshot,
    gapCount: counts.gapCount,
    reviewFindingCount: counts.reviewFindingCount,
    limitationCount: counts.limitationCount,
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
  const fieldId = 'deep-research-guidance-input';
  return (
    <TextField
      fullWidth
      multiline
      minRows={2}
      maxRows={6}
      id={fieldId}
      name="research-guidance"
      label={label}
      helperText={helperText}
      required={required}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      inputProps={{ 'aria-label': label }}
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
    const raw = stripRuntimeReportSections(String(content || ''));
    if (!raw.trim()) return raw;
    const map = buildCitationDisplayMap(citationSources);
    return rewriteDocCitationTokens(raw, map);
  }, [content, citationSources]);

  const [open, setOpen] = useState(!defaultCollapsed);
  if (displayContent?.trim()) {
    if (defaultCollapsed) {
      return (
        <Box sx={{ flexShrink: 0 }}>
          <Button
            size="small"
            color="inherit"
            endIcon={<ExpandMoreIcon sx={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />}
            onClick={() => setOpen((value) => !value)}
            sx={{ textTransform: 'none', px: 0 }}
          >
            Provisional report
          </Button>
          <Collapse in={open}>
            <Stack spacing={0.5} sx={{ pt: 0.5 }}>
              <ConversationMarkdownWithCitations content={displayContent} citationSources={citationSources} />
            </Stack>
          </Collapse>
        </Box>
      );
    }
    return <ConversationMarkdownWithCitations content={displayContent} citationSources={citationSources} />;
  }
  if (emptyMessage) {
    return <Alert severity="warning">{emptyMessage}</Alert>;
  }
  return null;
}
