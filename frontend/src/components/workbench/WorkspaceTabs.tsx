import React from 'react';
import {
  Badge,
  Box,
  CircularProgress,
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
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import LanguageIcon from '@mui/icons-material/Language';
import HomeIcon from '@mui/icons-material/Home';
import MemoryIcon from '@mui/icons-material/Memory';
import OpenInBrowserIcon from '@mui/icons-material/OpenInBrowser';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import RouteIcon from '@mui/icons-material/Route';
import ErrorIcon from '@mui/icons-material/Error';
import ReplayIcon from '@mui/icons-material/Replay';
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
  associationScope?: 'thread' | 'project';
  isProjectKnowledge?: boolean;
  processingError?: string;
};

export type BrowserWorkspaceTab = { kind: 'browser'; id: 'browser-tab'; label: string };
export type HomeWorkspaceTab = { kind: 'home'; id: 'home-tab'; label: string };
export type MemoryWorkspaceTab = { kind: 'memory'; id: 'memory-tab'; label: string };
export type CanvasWorkspaceTab = { kind: 'canvas'; id: 'canvas-tab'; label: string; issueCount?: number };
export type SpecWorkspaceTab = { kind: 'spec'; id: 'spec-tab'; label: string; dirty?: boolean };
export type TraceWorkspaceTab = {
  kind: 'trace';
  id: 'trace-tab';
  label: string;
  status?: 'idle' | 'running' | 'failed' | 'review';
  count?: number;
};
export type WorkspaceTab = DocumentWorkspaceTab | BrowserWorkspaceTab | HomeWorkspaceTab | MemoryWorkspaceTab | CanvasWorkspaceTab | SpecWorkspaceTab | TraceWorkspaceTab;

const statusColor = (status?: TraceWorkspaceTab['status']) => {
  if (status === 'failed') return 'error';
  if (status === 'running') return 'primary';
  if (status === 'review') return 'warning';
  return 'default';
};

const commonTabSx = {
  '&.Mui-selected': {
    bgcolor: 'action.selected',
  },
};

export default React.memo(function WorkspaceTabs({
  tabs,
  activeTabId,
  onTabChange,
  onTabClose,
  onDocumentRemove,
  onDocumentPromote,
  onDocumentRetry,
  onAddBrowserToThread,
  isBrowserCapturing = false,
  documentContext = 'thread',
}: {
  tabs: WorkspaceTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
  onTabClose?: (tabId: string) => void;
  onDocumentRemove?: (tabId: string) => void;
  onDocumentPromote?: (tabId: string) => void;
  onDocumentRetry?: (tabId: string) => void;
  onAddBrowserToThread?: () => void;
  isBrowserCapturing?: boolean;
  documentContext?: 'thread' | 'project';
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
        sx={{
          minHeight: 40,
          '& .MuiTab-root': {
            minHeight: 40,
            textTransform: 'none',
            fontSize: '0.875rem',
            py: 0,
            px: 1.5,
          },
        }}
      >
        {tabs.map((tab) => {
          if (tab.kind === 'home') {
            return <Tab key={tab.id} icon={<HomeIcon fontSize="small" />} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'memory') {
            return <Tab key={tab.id} icon={<MemoryIcon fontSize="small" />} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'canvas') {
            return <Tab key={tab.id} icon={<Badge color="error" badgeContent={tab.issueCount || 0}><AccountTreeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'spec') {
            return <Tab key={tab.id} icon={<Badge color="primary" variant={tab.dirty ? 'dot' : 'standard'}><CodeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'trace') {
            return (
              <Tab
                key={tab.id}
                icon={<Badge color={statusColor(tab.status)} variant={tab.status === 'running' ? 'dot' : 'standard'} badgeContent={tab.status === 'running' ? undefined : tab.count} max={99}><RouteIcon fontSize="small" /></Badge>}
                iconPosition="start"
                label={tab.label}
                sx={commonTabSx}
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
                      <Tooltip title={documentContext === 'project' ? 'Add current page to project' : 'Add current page to thread'}>
                        <span>
                          <Box
                            component="span"
                            role="button"
                            tabIndex={isBrowserCapturing ? -1 : 0}
                            aria-disabled={isBrowserCapturing}
                            onClick={(event) => {
                              event.stopPropagation();
                              if (!isBrowserCapturing) onAddBrowserToThread();
                            }}
                            onKeyDown={(event) => {
                              if (!isBrowserCapturing && (event.key === 'Enter' || event.key === ' ')) {
                                event.preventDefault();
                                event.stopPropagation();
                                onAddBrowserToThread();
                              }
                            }}
                            sx={{ p: 0.3, display: 'inline-flex', borderRadius: 1 }}
                          >
                            {isBrowserCapturing ? <CircularProgress size={14} /> : <AddIcon fontSize="small" />}
                          </Box>
                        </span>
                      </Tooltip>
                    )}
                  </Box>
                }
                sx={commonTabSx}
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
                  {isBrowserDocument ? (
                    <Tooltip title="Open source webpage">
                      <Box
                        component="span"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (tab.sourceUrl) {
                            window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                          }
                        }}
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          cursor: tab.sourceUrl ? 'pointer' : 'default',
                          p: '2px',
                          borderRadius: 1,
                          color: 'primary.main',
                        }}
                      >
                        <LanguageIcon fontSize="small" />
                      </Box>
                    </Tooltip>
                  ) : <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7 }} />}
                  <Tooltip title={fullTitle} placement="bottom">
                    <Typography
                      component="span"
                      sx={{
                        maxWidth: 120,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {label}
                    </Typography>
                  </Tooltip>
                  {tab.parsingStatus === ProcessStatus.Pending && <CircularProgress size={13} />}
                  {tab.parsingStatus === ProcessStatus.Failed && (
                    <Tooltip title={tab.processingError || 'Processing failed'}>
                      <ErrorIcon color="error" sx={{ fontSize: 15 }} />
                    </Tooltip>
                  )}
                  {tab.parsingStatus === ProcessStatus.Failed && onDocumentRetry && (
                    <Tooltip title="Retry processing">
                      <Box
                        component="span"
                        role="button"
                        tabIndex={0}
                        onClick={(event) => { event.stopPropagation(); onDocumentRetry(tab.id); }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            event.stopPropagation();
                            onDocumentRetry(tab.id);
                          }
                        }}
                        sx={{ p: 0.2, display: 'inline-flex', borderRadius: 1 }}
                      >
                        <ReplayIcon sx={{ fontSize: 15 }} />
                      </Box>
                    </Tooltip>
                  )}
                  {tab.associationScope === 'project' && (
                    <Tooltip title="Project knowledge"><MemoryIcon sx={{ fontSize: 14, color: 'primary.main' }} /></Tooltip>
                  )}
                  {onDocumentPromote && tab.associationScope === 'thread' && !tab.isProjectKnowledge && (
                    <Tooltip title="Add to project knowledge">
                      <Box
                        component="span"
                        role="button"
                        tabIndex={0}
                        onClick={(event) => { event.stopPropagation(); onDocumentPromote(tab.id); }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            event.stopPropagation();
                            onDocumentPromote(tab.id);
                          }
                        }}
                        sx={{ p: 0.2, display: 'inline-flex', borderRadius: 1 }}
                      >
                        <CreateNewFolderIcon sx={{ fontSize: 15 }} />
                      </Box>
                    </Tooltip>
                  )}
                  {onDocumentRemove && (documentContext === 'project' || tab.associationScope !== 'project') && (
                    <Tooltip title={documentContext === 'project' ? 'Remove from project' : 'Remove from thread'}>
                      <Box
                        component="span"
                        role="button"
                        tabIndex={0}
                        className="tab-remove-btn"
                        onClick={(event) => { event.stopPropagation(); onDocumentRemove(tab.id); }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            event.stopPropagation();
                            onDocumentRemove(tab.id);
                          }
                        }}
                        sx={{ p: 0.2, color: 'error.main', opacity: 0, display: 'inline-flex', borderRadius: 1 }}
                      >
                        <DeleteIcon sx={{ fontSize: 14 }} />
                      </Box>
                    </Tooltip>
                  )}
                  {onTabClose && (
                    <Tooltip title="Close tab">
                      <Box
                        component="span"
                        role="button"
                        tabIndex={0}
                        onClick={(event) => { event.stopPropagation(); onTabClose(tab.id); }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            event.preventDefault();
                            event.stopPropagation();
                            onTabClose(tab.id);
                          }
                        }}
                        sx={{ p: 0.2, opacity: 0.65, display: 'inline-flex', borderRadius: 1 }}
                      >
                        <CloseIcon sx={{ fontSize: 16 }} />
                      </Box>
                    </Tooltip>
                  )}
                  </Box>
                }
                sx={{
                  ...commonTabSx,
                  '&:hover .tab-remove-btn': {
                    opacity: 0.7,
                  },
                }}
              />
            );
        })}
      </Tabs>
    </Box>
  );
});
