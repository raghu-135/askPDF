import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, IconButton, Tooltip } from '@mui/material';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import {
  isWorkspaceChromeSeparator,
  splitWorkspaceChromeRows,
  type WorkspaceChromeEntry,
} from '../../lib/workspace-chrome';
import useStoredLayoutState from './useStoredLayoutState';

const itemSx = {
  display: 'inline-flex',
  alignItems: 'center',
  flexShrink: 0,
  maxWidth: '100%',
};

function ChromeItem({ children }: { children: React.ReactNode }) {
  return <Box component="span" sx={itemSx}>{children}</Box>;
}

function normalizeExpanded(value?: unknown): boolean {
  return value === true;
}

export default function CollapsibleWorkspaceChrome({
  items,
  storageKey,
  defaultExpanded = false,
  ariaLabel = 'Workspace tools',
}: {
  items: readonly WorkspaceChromeEntry[];
  storageKey: string;
  defaultExpanded?: boolean;
  ariaLabel?: string;
}) {
  const [expanded, setExpanded] = useStoredLayoutState(
    storageKey,
    defaultExpanded,
    normalizeExpanded,
  );

  const visibleItems = useMemo(
    () => items.filter((entry) => !isWorkspaceChromeSeparator(entry) && entry != null && entry !== false),
    [items],
  );

  const expandedRows = useMemo(() => splitWorkspaceChromeRows(items), [items]);
  const collapsedRowItems = expandedRows[0] ?? [];
  const hasHiddenRows = expandedRows.length > 1;
  const collapsedItemsRef = useRef<HTMLDivElement>(null);
  const [collapsedOverflows, setCollapsedOverflows] = useState(false);

  useEffect(() => {
    if (expanded) return undefined;

    const node = collapsedItemsRef.current;
    if (!node) return undefined;

    const measureOverflow = () => {
      setCollapsedOverflows(node.scrollWidth > node.clientWidth + 1);
    };

    measureOverflow();
    const observer = new ResizeObserver(measureOverflow);
    observer.observe(node);
    return () => observer.disconnect();
  }, [collapsedRowItems, expanded]);

  const showChevron = expanded || collapsedOverflows || hasHiddenRows;

  const toggleExpanded = useCallback(() => {
    setExpanded((current) => !current);
  }, [setExpanded]);

  const chevron = showChevron ? (
    <Tooltip title={expanded ? 'Collapse tools' : 'Expand tools'}>
      <IconButton
        size="small"
        aria-label={expanded ? 'Collapse tools' : 'Expand tools'}
        aria-expanded={expanded}
        onClick={toggleExpanded}
        sx={{ flexShrink: 0 }}
      >
        {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
      </IconButton>
    </Tooltip>
  ) : null;

  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <Box
      role="toolbar"
      aria-label={ariaLabel}
      sx={{
        flexShrink: 0,
        px: 1,
        py: 0.5,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      {expanded ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          {expandedRows.map((row, rowIndex) => (
            <Box
              key={`chrome-row-${rowIndex}`}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                minWidth: 0,
                width: '100%',
              }}
            >
              <Box
              sx={{
                display: 'flex',
                alignItems: 'flex-start',
                flexWrap: 'wrap',
                gap: 0.5,
                flex: 1,
                minWidth: 0,
                width: '100%',
              }}
              >
                {row.map((item, itemIndex) => (
                  <ChromeItem key={`chrome-item-${rowIndex}-${itemIndex}`}>{item}</ChromeItem>
                ))}
              </Box>
              {rowIndex === 0 ? chevron : null}
            </Box>
          ))}
        </Box>
      ) : (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            minWidth: 0,
            width: '100%',
          }}
        >
          <Box
            ref={collapsedItemsRef}
            sx={{
              display: 'flex',
              alignItems: 'center',
              flexWrap: 'nowrap',
              gap: 0.5,
              flex: 1,
              minWidth: 0,
              overflowX: 'hidden',
              overflowY: 'hidden',
            }}
          >
            {collapsedRowItems.map((item, itemIndex) => (
              <ChromeItem key={`chrome-collapsed-${itemIndex}`}>{item}</ChromeItem>
            ))}
          </Box>
          {chevron}
        </Box>
      )}
    </Box>
  );
}
