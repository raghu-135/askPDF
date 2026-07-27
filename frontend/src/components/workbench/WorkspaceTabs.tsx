import React from 'react';
import {
  Badge,
  Box,
  CircularProgress,
  IconButton,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import CodeIcon from '@mui/icons-material/Code';
import DeleteIcon from '@mui/icons-material/Delete';
import LanguageIcon from '@mui/icons-material/Language';
import OpenInBrowserIcon from '@mui/icons-material/OpenInBrowser';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import RouteIcon from '@mui/icons-material/Route';
import { truncateFileName } from '../../lib/pdf-utils';
import type { BackendSentence, BBox } from '../../lib/bbox-derivation';
import {
  ProcessStatus,
  ThreadFileSourceType,
  type ProcessStatus as ProcessStatusValue,
  type ThreadFileSourceType as ThreadFileSourceTypeValue,
} from '../../lib/enums';

type Sentence = Omit<BackendSentence, 'bboxes'> & { bboxes: BBox[] };

export type DocumentWorkspaceTab = {
  kind: 'document';
  id: string;
  fileName: string;
  fileHash: string;
  downloadUrl: string;
  sentences: Sentence[] | null;
  text?: string;
  sourceType?: ThreadFileSourceTypeValue;
  sourceUrl?: string;
  parsingStatus?: Extract<ProcessStatusValue, typeof ProcessStatus.Pending | typeof ProcessStatus.Completed | typeof ProcessStatus.Failed>;
};

export type BrowserWorkspaceTab = { kind: 'browser'; id: 'browser-tab'; label: string };
export type CanvasWorkspaceTab = { kind: 'canvas'; id: 'canvas-tab'; label: string; issueCount?: number };
export type SpecWorkspaceTab = { kind: 'spec'; id: 'spec-tab'; label: string; dirty?: boolean };
export type TraceWorkspaceTab = {
  kind: 'trace';
  id: 'trace-tab';
  label: string;
  status?: 'idle' | 'running' | 'failed' | 'review';
  count?: number;
};
export type WorkspaceTab = DocumentWorkspaceTab | BrowserWorkspaceTab | CanvasWorkspaceTab | SpecWorkspaceTab | TraceWorkspaceTab;
export type WorkspaceRenderer = Partial<Record<WorkspaceTab['kind'], (tab: WorkspaceTab) => React.ReactNode>>;

const statusColor = (status?: TraceWorkspaceTab['status']) => {
  if (status === 'failed') return 'error';
  if (status === 'running') return 'primary';
  if (status === 'review') return 'warning';
  return 'default';
};

export default React.memo(function WorkspaceTabs({
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onDocumentRemove,
  onAddBrowserToThread,
  isBrowserCapturing = false,
}: {
  tabs: WorkspaceTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose?: (tabId: string) => void;
  onDocumentRemove?: (tabId: string) => void;
  onAddBrowserToThread?: () => void;
  isBrowserCapturing?: boolean;
}) {
  if (tabs.length === 0) return null;
  const activeIndex = Math.max(0, tabs.findIndex((tab) => tab.id === activeTabId));

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', minHeight: 40 }}>
      <Tabs
        value={activeIndex}
        onChange={(_, index) => tabs[index] && onTabChange(tabs[index].id)}
        variant="scrollable"
        scrollButtons="auto"
        aria-label="Workspace content"
        sx={{ minHeight: 40, '& .MuiTab-root': { minHeight: 40, textTransform: 'none', fontSize: '0.875rem', py: 0, px: 1.5 } }}
      >
        {tabs.map((tab) => {
          if (tab.kind === 'canvas') {
            return <Tab key={tab.id} icon={<Badge color="error" badgeContent={tab.issueCount || 0}><AccountTreeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} />;
          }
          if (tab.kind === 'spec') {
            return <Tab key={tab.id} icon={<Badge color="primary" variant={tab.dirty ? 'dot' : 'standard'}><CodeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} />;
          }
          if (tab.kind === 'trace') {
            return (
              <Tab
                key={tab.id}
                icon={<Badge color={statusColor(tab.status)} variant={tab.status === 'running' ? 'dot' : 'standard'} badgeContent={tab.status === 'running' ? undefined : tab.count} max={99}><RouteIcon fontSize="small" /></Badge>}
                iconPosition="start"
                label={tab.label}
              />
            );
          }
          if (tab.kind === 'browser') {
            const active = tab.id === activeTabId;
            return (
              <Tab
                key={tab.id}
                label={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <OpenInBrowserIcon fontSize="small" />
                    <Typography component="span">{tab.label}</Typography>
                    {active && onAddBrowserToThread && (
                      <Tooltip title="Add current page to thread">
                        <span>
                          <IconButton size="small" disabled={isBrowserCapturing} onClick={(event) => { event.stopPropagation(); onAddBrowserToThread(); }} sx={{ p: 0.3 }}>
                            {isBrowserCapturing ? <CircularProgress size={14} /> : <AddIcon fontSize="small" />}
                          </IconButton>
                        </span>
                      </Tooltip>
                    )}
                  </Box>
                }
              />
            );
          }

          const isBrowserDocument = tab.sourceType === ThreadFileSourceType.Browser;
          let label = truncateFileName(tab.fileName);
          if (isBrowserDocument && tab.sourceUrl) {
            try {
              label = new URL(tab.sourceUrl).hostname;
            } catch {
              label = tab.fileName;
            }
          }
          const fullTitle = isBrowserDocument ? (tab.sourceUrl || tab.fileName) : tab.fileName;
          return (
            <Tab
              key={tab.id}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  {isBrowserDocument ? <LanguageIcon fontSize="small" color="primary" /> : <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7 }} />}
                  <Tooltip title={fullTitle}><Typography component="span" noWrap sx={{ maxWidth: 120 }}>{label}</Typography></Tooltip>
                  {onDocumentRemove && (
                    <Tooltip title="Remove from thread">
                      <IconButton size="small" onClick={(event) => { event.stopPropagation(); onDocumentRemove(tab.id); }} sx={{ p: 0.2, color: 'error.main', opacity: 0.7 }}>
                        <DeleteIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </Tooltip>
                  )}
                  {onTabClose && (
                    <Tooltip title="Close tab">
                      <IconButton size="small" onClick={(event) => { event.stopPropagation(); onTabClose(tab.id); }} sx={{ p: 0.2, opacity: 0.65 }}>
                        <CloseIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </Tooltip>
                  )}
                </Box>
              }
            />
          );
        })}
      </Tabs>
    </Box>
  );
});
