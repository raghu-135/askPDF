import React from 'react';
import { Chip, Stack, Tooltip } from '@mui/material';
import type { BudgetUsageSnapshot } from '../../lib/deep-research-decision-ui';
import {
  formatBudgetDimensionChip,
  RESEARCH_BUDGET_DIMENSIONS,
  RESEARCH_BUDGET_TOOLTIP,
  researchBudgetDimensionTooltip,
} from '../../lib/deep-research-decision-ui';

export function DeepResearchBudgetChips({
  snapshot,
  tranchePrefix = true,
}: {
  snapshot: BudgetUsageSnapshot;
  tranchePrefix?: boolean;
}) {
  return (
    <Stack direction="row" spacing={0.75} flexWrap="wrap" alignItems="center" useFlexGap>
      {tranchePrefix ? (
        <Tooltip title="Research budget tranche index for this task attempt.">
          <Chip
            size="small"
            variant="outlined"
            label={`Tranche ${snapshot.trancheIndex}`}
          />
        </Tooltip>
      ) : null}
      {RESEARCH_BUDGET_DIMENSIONS.map((key) => {
        const limit = snapshot.trancheLimits[key];
        if (limit == null) return null;
        const used = snapshot.trancheUsage[key] || 0;
        return (
          <Tooltip key={key} title={researchBudgetDimensionTooltip(key)}>
            <Chip
              size="small"
              variant="outlined"
              label={formatBudgetDimensionChip(key, used, limit)}
              sx={{ maxWidth: '100%', height: 'auto', '& .MuiChip-label': { whiteSpace: 'normal', py: 0.25 } }}
            />
          </Tooltip>
        );
      })}
    </Stack>
  );
}

export { RESEARCH_BUDGET_TOOLTIP };
