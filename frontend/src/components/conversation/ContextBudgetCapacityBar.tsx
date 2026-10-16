import React, { useMemo } from 'react';
import { Box, Slider, Tooltip, Typography } from '@mui/material';
import type { ContextBudgetRatios } from '../../lib/context-budget';
import { contextWindowBreakdown, type ContextWindowBreakdown } from '../../lib/context-budget';
import {
  BAR_SEGMENT_REPLY_BODY,
  BAR_SEGMENT_REPLY_TITLE,
  BAR_SEGMENT_RETRIEVAL_BODY,
  BAR_SEGMENT_RETRIEVAL_TITLE,
  BAR_SEGMENT_SAFETY_BODY,
  BAR_SEGMENT_SAFETY_TITLE,
  CONTEXT_PACKING_WINDOW_HELP,
  CONTEXT_WINDOW_SPLIT_HELP,
} from '../../lib/context-budget-help';
import { ContextBudgetHelpTip, contextBudgetTooltipSlotProps } from './ContextBudgetHelpTip';

type ContextBudgetCapacityBarProps = {
  contextWindow: number;
  contextBudget: ContextBudgetRatios;
  disabled?: boolean;
  showUsableSlider?: boolean;
  onUsableRatioChange?: (usableRatio: number) => void;
};

const COLORS = {
  retrieval: 'warning.main',
  reply: 'info.main',
  safety: 'success.main',
} as const;

type WindowSegmentConfig = {
  color: string;
  title: string;
  body: string;
  legendTitle: string;
  tokens: (breakdown: ContextWindowBreakdown) => number;
};

const WINDOW_SEGMENTS: WindowSegmentConfig[] = [
  {
    color: COLORS.retrieval,
    title: BAR_SEGMENT_RETRIEVAL_TITLE,
    body: BAR_SEGMENT_RETRIEVAL_BODY,
    legendTitle: 'Max retrieval (your sliders)',
    tokens: (breakdown) => breakdown.prefetchCap,
  },
  {
    color: COLORS.reply,
    title: BAR_SEGMENT_REPLY_TITLE,
    body: BAR_SEGMENT_REPLY_BODY,
    legendTitle: 'System + answer (automatic)',
    tokens: (breakdown) => breakdown.answerReserve,
  },
  {
    color: COLORS.safety,
    title: BAR_SEGMENT_SAFETY_TITLE,
    body: BAR_SEGMENT_SAFETY_BODY,
    legendTitle: 'Safety margin (automatic)',
    tokens: (breakdown) => breakdown.headroom,
  },
];

type BudgetBarSegment = WindowSegmentConfig & {
  share: number;
  tokenCount: number;
};

function LegendDot({ color }: { color: string }) {
  return (
    <Box
      component="span"
      sx={{
        width: 8,
        height: 8,
        borderRadius: '50%',
        bgcolor: color,
        flex: '0 0 auto',
        mt: '3px',
      }}
    />
  );
}

function LegendRow({
  color,
  title,
  detail,
}: {
  color: string;
  title: string;
  detail: string;
}) {
  return (
    <Box sx={{ display: 'flex', gap: 0.75, alignItems: 'flex-start' }}>
      <LegendDot color={color} />
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="caption" sx={{ display: 'block', lineHeight: 1.35 }}>
          {title}
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.35 }}>
          {detail}
        </Typography>
      </Box>
    </Box>
  );
}

function SegmentTooltipContent({ title, body, tokenCount, share }: BudgetBarSegment) {
  return (
    <Box>
      <Typography variant="caption" sx={{ display: 'block', fontWeight: 700 }}>
        {title}
      </Typography>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.25 }}>
        {tokenCount.toLocaleString()} tok · {(share * 100).toFixed(0)}% of ctx
      </Typography>
      <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
        {body}
      </Typography>
    </Box>
  );
}

function BudgetBar({
  segments,
  height,
  ariaLabel,
}: {
  segments: BudgetBarSegment[];
  height: number;
  ariaLabel: string;
}) {
  return (
    <Box
      role="img"
      aria-label={ariaLabel}
      sx={{
        display: 'flex',
        height,
        width: '100%',
        borderRadius: 1,
        overflow: 'hidden',
        bgcolor: 'action.hover',
      }}
    >
      {segments.map((segment, index) => (
        segment.share > 0 ? (
          <Tooltip
            key={index}
            describeChild
            title={<SegmentTooltipContent {...segment} />}
            placement="top"
            enterDelay={250}
            slotProps={contextBudgetTooltipSlotProps}
          >
            <Box
              component="span"
              sx={{
                display: 'block',
                width: `${segment.share * 100}%`,
                bgcolor: segment.color,
                minWidth: 6,
                height: '100%',
              }}
            />
          </Tooltip>
        ) : null
      ))}
    </Box>
  );
}

function segmentDetail(tokenCount: number, share: number): string {
  return `${tokenCount.toLocaleString()} tok · ${(share * 100).toFixed(0)}% of ctx`;
}

export function ContextBudgetCapacityBar({
  contextWindow,
  contextBudget,
  disabled = false,
  showUsableSlider = false,
  onUsableRatioChange,
}: ContextBudgetCapacityBarProps) {
  const breakdown = useMemo(
    () => contextWindowBreakdown(contextWindow, contextBudget),
    [contextBudget, contextWindow],
  );
  const windowTokens = Math.max(1, contextWindow);

  const barSegments = useMemo((): BudgetBarSegment[] => {
    return WINDOW_SEGMENTS.map((config) => {
      const tokenCount = config.tokens(breakdown);
      return {
        ...config,
        tokenCount,
        share: tokenCount / windowTokens,
      };
    });
  }, [breakdown, windowTokens]);

  const detailedAria = barSegments
    .map((segment) => `${segment.tokenCount} tokens ${segment.legendTitle.toLowerCase()}`)
    .join(', ');

  return (
    <Box sx={{ mt: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.25, mb: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
          How your {contextWindow.toLocaleString()} token target is split. Hover each color on the bar for what it means.
        </Typography>
        <ContextBudgetHelpTip text={CONTEXT_WINDOW_SPLIT_HELP} ariaLabel="Window split help" />
      </Box>
      <BudgetBar height={12} ariaLabel={detailedAria} segments={barSegments} />
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1.25 }}>
        {barSegments.map((segment) => (
          <LegendRow
            key={segment.legendTitle}
            color={segment.color}
            title={segment.legendTitle}
            detail={segmentDetail(segment.tokenCount, segment.share)}
          />
        ))}
      </Box>

      {showUsableSlider && onUsableRatioChange && (
        <Box sx={{ mt: 2, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, mb: 0.25, alignItems: 'center' }}>
            <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
              <Typography variant="caption" fontWeight={600}>
                Packing window
              </Typography>
              <ContextBudgetHelpTip text={CONTEXT_PACKING_WINDOW_HELP} ariaLabel="Packing window help" />
            </Box>
            <Typography variant="caption" color="text.secondary">
              {(contextBudget.usable_ratio * 100).toFixed(0)}% of ctx size
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.75 }}>
            How much of ctx size AskPDF uses when computing the slices above. The rest stays empty as safety margin.
          </Typography>
          <Slider
            size="small"
            min={0.5}
            max={1}
            step={0.01}
            value={contextBudget.usable_ratio}
            disabled={disabled}
            onChange={(_, value) => {
              const next = Array.isArray(value) ? value[0] : value;
              onUsableRatioChange(next);
            }}
          />
        </Box>
      )}
    </Box>
  );
}
