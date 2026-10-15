import React, { useMemo } from 'react';
import { Box, Slider, Tooltip, Typography } from '@mui/material';
import type { ContextBudgetRatios } from '../../lib/context-budget';
import { contextWindowBreakdown } from '../../lib/context-budget';
import {
  CONTEXT_COMPACT_BAR_HELP,
  CONTEXT_PACKING_WINDOW_HELP,
  CONTEXT_WINDOW_SPLIT_HELP,
} from '../../lib/context-budget-help';
import { ContextBudgetHelpTip } from './ContextBudgetHelpTip';

type ContextBudgetCapacityBarProps = {
  contextWindow: number;
  contextBudget: ContextBudgetRatios;
  disabled?: boolean;
  compact?: boolean;
  showUsableSlider?: boolean;
  onUsableRatioChange?: (usableRatio: number) => void;
};

const COLORS = {
  retrieval: 'warning.main',
  reply: 'info.main',
  safety: 'success.main',
  remainder: 'success.dark',
} as const;

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

function BudgetBar({
  segments,
  height,
  ariaLabel,
}: {
  segments: Array<{ share: number; color: string }>;
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
          <Box
            key={index}
            sx={{
              width: `${segment.share * 100}%`,
              bgcolor: segment.color,
              minWidth: 2,
            }}
          />
        ) : null
      ))}
    </Box>
  );
}

export function ContextBudgetCapacityBar({
  contextWindow,
  contextBudget,
  disabled = false,
  compact = false,
  showUsableSlider = false,
  onUsableRatioChange,
}: ContextBudgetCapacityBarProps) {
  const breakdown = useMemo(
    () => contextWindowBreakdown(contextWindow, contextBudget),
    [contextBudget, contextWindow],
  );
  const windowTokens = Math.max(1, contextWindow);
  const retrievalTokens = breakdown.prefetchCap;
  const remainderTokens = breakdown.llmFree;
  const retrievalShare = retrievalTokens / windowTokens;
  const remainderShare = remainderTokens / windowTokens;

  if (compact) {
    return (
      <Box sx={{ px: 0.25, py: 0.5 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', gap: 1, mb: 0.35, alignItems: 'center' }}>
          <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25, minWidth: 0 }}>
            <Typography variant="caption" color="text.secondary">
              Ctx {contextWindow.toLocaleString()} split
            </Typography>
            <ContextBudgetHelpTip text={CONTEXT_COMPACT_BAR_HELP} ariaLabel="Context split help" />
          </Box>
            <Typography variant="caption" color="text.secondary">
              <Box component="span" sx={{ color: 'warning.main', fontWeight: 600 }}>
                {retrievalTokens.toLocaleString()}
              </Box>
              {' retrieval · '}
              <Box component="span" sx={{ color: 'success.dark', fontWeight: 600 }}>
                {remainderTokens.toLocaleString()}
              </Box>
              {' left'}
            </Typography>
        </Box>
        <Tooltip
          title={CONTEXT_COMPACT_BAR_HELP}
          placement="top"
          enterDelay={500}
          slotProps={{ tooltip: { sx: { maxWidth: 300 } } }}
        >
          <Box sx={{ cursor: 'help' }}>
            <BudgetBar
              height={6}
              ariaLabel={`${retrievalTokens} tokens max retrieval, ${remainderTokens} tokens remaining`}
              segments={[
                { share: retrievalShare, color: COLORS.retrieval },
                { share: remainderShare, color: COLORS.remainder },
              ]}
            />
          </Box>
        </Tooltip>
      </Box>
    );
  }

  const detailedAria = [
    `${retrievalTokens} tokens max retrieval`,
    `${breakdown.answerReserve} tokens for system and answer`,
    `${breakdown.headroom} tokens safety margin`,
  ].join(', ');

  return (
    <Box sx={{ mt: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 0.25, mb: 1 }}>
        <Typography variant="caption" color="text.secondary" sx={{ flex: 1 }}>
          How your {contextWindow.toLocaleString()} token target is split. The sliders above only change the orange
          retrieval slice; blue and green are computed automatically.
        </Typography>
        <ContextBudgetHelpTip text={CONTEXT_WINDOW_SPLIT_HELP} ariaLabel="Window split help" />
      </Box>
      <BudgetBar
        height={10}
        ariaLabel={detailedAria}
        segments={[
          { share: retrievalShare, color: COLORS.retrieval },
          { share: breakdown.answerReserve / windowTokens, color: COLORS.reply },
          { share: breakdown.headroom / windowTokens, color: COLORS.safety },
        ]}
      />
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1.25 }}>
        <Tooltip
          title="Sum of the four retrieval sliders above—maximum combined history, memory, and document text."
          placement="left"
          slotProps={{ tooltip: { sx: { maxWidth: 280 } } }}
        >
          <Box>
            <LegendRow
              color={COLORS.retrieval}
              title="Max retrieval (your sliders)"
              detail={`${retrievalTokens.toLocaleString()} tok · ${(retrievalShare * 100).toFixed(0)}% of ctx`}
            />
          </Box>
        </Tooltip>
        <Tooltip
          title="Reserved inside the packing window when your slice sliders total less than 100%. Used for system prompt, tools, and the model’s reply."
          placement="left"
          slotProps={{ tooltip: { sx: { maxWidth: 280 } } }}
        >
          <Box>
            <LegendRow
              color={COLORS.reply}
              title="System + answer (automatic)"
              detail={`${breakdown.answerReserve.toLocaleString()} tok · ${((breakdown.answerReserve / windowTokens) * 100).toFixed(0)}% of ctx`}
            />
          </Box>
        </Tooltip>
        <Tooltip
          title="Intentionally unused share of ctx size so AskPDF does not pack all the way to the provider limit."
          placement="left"
          slotProps={{ tooltip: { sx: { maxWidth: 280 } } }}
        >
          <Box>
            <LegendRow
              color={COLORS.safety}
              title="Safety margin (automatic)"
              detail={`${breakdown.headroom.toLocaleString()} tok · ${((breakdown.headroom / windowTokens) * 100).toFixed(0)}% of ctx`}
            />
          </Box>
        </Tooltip>
      </Box>
      <Typography variant="caption" sx={{ display: 'block', mt: 1.25, fontWeight: 600 }}>
        {remainderTokens.toLocaleString()} tokens not counted toward retrieval caps
        {' '}
        ({(remainderShare * 100).toFixed(0)}% of ctx)
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.25 }}>
        That is system + answer ({breakdown.answerReserve.toLocaleString()}) plus safety margin ({breakdown.headroom.toLocaleString()}).
      </Typography>

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
