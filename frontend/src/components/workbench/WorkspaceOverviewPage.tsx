import React from 'react';
import { Box, Stack, Typography } from '@mui/material';
import OverviewSection from './OverviewSection';
import InstructionGuide from './InstructionGuide';
import type { HomeInstructionSection } from '../../lib/home-instructions';

export default function WorkspaceOverviewPage({
  title,
  subtitle,
  actions,
  guideSections,
  guideTitle = 'How this works',
  guideDefaultExpanded = false,
  stats,
  settings,
  settingsTitle = 'Settings',
  darkMode = false,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  guideSections: HomeInstructionSection[];
  guideTitle?: string;
  guideDefaultExpanded?: boolean;
  stats?: React.ReactNode;
  settings?: React.ReactNode;
  settingsTitle?: string;
  darkMode?: boolean;
}) {
  return (
    <Box
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: darkMode ? '#222' : 'grey.50',
        color: darkMode ? '#eee' : 'inherit',
        p: { xs: 2, md: 3 },
      }}
    >
      <Box sx={{ maxWidth: 1080, mx: 'auto' }}>
        <Stack spacing={2}>
          <Stack spacing={1.25}>
            <Typography variant="h4" sx={{ fontWeight: 800 }}>
              {title}
            </Typography>
            {subtitle ? (
              <Typography color="text.secondary" sx={{ maxWidth: 760 }}>
                {subtitle}
              </Typography>
            ) : null}
            {actions ? <Box>{actions}</Box> : null}
          </Stack>

          <OverviewSection
            title={guideTitle}
            collapsible
            defaultExpanded={guideDefaultExpanded}
          >
            <InstructionGuide sections={guideSections} />
          </OverviewSection>

          {stats ? (
            <OverviewSection title="Quick stats">
              {stats}
            </OverviewSection>
          ) : null}

          {settings ? (
            <OverviewSection title={settingsTitle}>
              {settings}
            </OverviewSection>
          ) : null}
        </Stack>
      </Box>
    </Box>
  );
}
