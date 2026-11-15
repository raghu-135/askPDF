import React, { useLayoutEffect, useMemo, useRef } from 'react';
import { Box, Tab, Tabs } from '@mui/material';
import type { TabsActions } from '@mui/material/Tabs';
import { selectedInstanceTabValue } from '../../lib/instance-tabs.ts';

export type NestedInstanceTab = {
  id: string;
  label: React.ReactNode;
};

const nestedTabsSx = (minHeight: number) => ({
  flex: '1 1 auto',
  minWidth: 0,
  width: 0,
  minHeight,
  '& .MuiTabs-scroller': {
    minWidth: 0,
    overflow: 'auto !important',
  },
  '& .MuiTabs-flexContainer': {
    minWidth: 'max-content',
  },
  '& .MuiTabs-scrollButtons': { flex: '0 0 32px' },
  '& .MuiTabs-scrollButtons.Mui-disabled': {
    width: 0,
    flexBasis: 0,
    opacity: 0,
    overflow: 'hidden',
  },
  '& .MuiTab-root': {
    minHeight,
    maxHeight: minHeight,
    textTransform: 'none',
    py: 0,
    px: 1,
    minWidth: 'max-content',
    maxWidth: 240,
  },
  '& .MuiTab-root.Mui-selected': {
    color: 'primary.main',
    fontWeight: 600,
  },
  '& .MuiTabs-indicator': {
    height: 2,
  },
});

export default function NestedInstanceTabs({
  tabs,
  activeId,
  onActiveChange,
  ariaLabel,
  trailingAction,
  minHeight = 38,
}: {
  tabs: NestedInstanceTab[];
  activeId: string | null;
  onActiveChange: (id: string) => void;
  ariaLabel: string;
  trailingAction?: React.ReactNode;
  minHeight?: number;
}) {
  const selectedTabValue = selectedInstanceTabValue(tabs, activeId);
  const tabIds = useMemo(() => tabs.map((tab) => tab.id).join('\0'), [tabs]);
  const tabsActionsRef = useRef<TabsActions>(null);
  const stripRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (tabs.length === 0) return undefined;

    const syncIndicator = () => {
      tabsActionsRef.current?.updateIndicator();
    };

    syncIndicator();
    let raf1 = 0;
    let raf2 = 0;
    raf1 = requestAnimationFrame(() => {
      syncIndicator();
      raf2 = requestAnimationFrame(syncIndicator);
    });

    const strip = stripRef.current;
    const flexContainer = strip?.querySelector('.MuiTabs-flexContainer');
    if (!flexContainer) {
      return () => {
        cancelAnimationFrame(raf1);
        cancelAnimationFrame(raf2);
      };
    }

    const resizeObserver = new ResizeObserver(syncIndicator);
    resizeObserver.observe(flexContainer);
    flexContainer.querySelectorAll('[role="tab"]').forEach((tab) => {
      resizeObserver.observe(tab);
    });

    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      resizeObserver.disconnect();
    };
  }, [selectedTabValue, tabIds, tabs.length]);

  if (tabs.length === 0 && !trailingAction) return null;

  return (
    <Box
      ref={stripRef}
      sx={{
        display: 'flex',
        alignItems: 'stretch',
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        minHeight,
        minWidth: 0,
        width: '100%',
        overflow: 'hidden',
      }}
    >
      {tabs.length > 0 ? (
        <Tabs
          action={tabsActionsRef}
          value={selectedTabValue}
          onChange={(_, tabId) => typeof tabId === 'string' && onActiveChange(tabId)}
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile
          textColor="primary"
          indicatorColor="primary"
          aria-label={ariaLabel}
          sx={nestedTabsSx(minHeight)}
        >
          {tabs.map((tab) => (
            <Tab key={tab.id} value={tab.id} label={tab.label} />
          ))}
        </Tabs>
      ) : (
        <Box sx={{ flex: '1 1 auto', minWidth: 0 }} />
      )}
      {trailingAction ? (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            px: 0.5,
            flex: '0 0 auto',
            borderLeft: tabs.length > 0 ? 1 : 0,
            borderColor: 'divider',
          }}
        >
          {trailingAction}
        </Box>
      ) : null}
    </Box>
  );
}
