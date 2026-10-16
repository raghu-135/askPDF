import React from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Paper,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

const panelSx = {
  border: 1,
  borderColor: 'divider',
  bgcolor: 'background.paper',
  borderRadius: 1,
  minWidth: 0,
};

export const overviewSeparatedItemSx = {
  pt: 1.25,
  mt: 1.25,
  borderTop: 1,
  borderColor: 'divider',
  '&:first-of-type': {
    pt: 0,
    mt: 0,
    borderTop: 0,
  },
};

export function OverviewSeparatedItem({
  label,
  children,
}: {
  label?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <Box sx={overviewSeparatedItemSx}>
      {label ? (
        <Typography
          variant="caption"
          color="text.secondary"
          component="div"
          sx={{ mb: 0.75 }}
        >
          {label}
        </Typography>
      ) : null}
      {children}
    </Box>
  );
}

export default function OverviewSection({
  title,
  children,
  collapsible = false,
  defaultExpanded = true,
  expanded,
  onExpandedChange,
}: {
  title: string;
  children: React.ReactNode;
  collapsible?: boolean;
  defaultExpanded?: boolean;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
}) {
  if (collapsible) {
    const controlled = expanded !== undefined && onExpandedChange;
    return (
      <Accordion
        disableGutters
        elevation={0}
        {...(controlled
          ? {
            expanded,
            onChange: (_: React.SyntheticEvent, nextExpanded: boolean) => onExpandedChange(nextExpanded),
          }
          : { defaultExpanded })}
        sx={{
          ...panelSx,
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          sx={{
            px: 2,
            minHeight: 48,
            '& .MuiAccordionSummary-content': { my: 1 },
          }}
        >
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            {title}
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ px: 2, pt: 0, pb: 2 }}>
          {children}
        </AccordionDetails>
      </Accordion>
    );
  }

  return (
    <Paper elevation={0} sx={{ ...panelSx, p: 2 }}>
      <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5 }}>
        {title}
      </Typography>
      {children}
    </Paper>
  );
}
