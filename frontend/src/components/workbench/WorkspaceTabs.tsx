import React from 'react';
import {
  Badge,
  Box,
  CircularProgress,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import CodeIcon from '@mui/icons-material/Code';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteIcon from '@mui/icons-material/Delete';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import FolderIcon from '@mui/icons-material/Folder';
import LanguageIcon from '@mui/icons-material/Language';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import OpenInBrowserIcon from '@mui/icons-material/OpenInBrowser';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PublicIcon from '@mui/icons-material/Public';
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

const documentTabWidth = 168;

const stopTabAction = (event: React.SyntheticEvent) => {
  event.preventDefault();
  event.stopPropagation();
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
  const [documentMenuAnchor, setDocumentMenuAnchor] = React.useState<HTMLElement | null>(null);
  const [documentMenuTabId, setDocumentMenuTabId] = React.useState<string | null>(null);

  if (tabs.length === 0) return null;
  const activeIndex = tabs.findIndex((tab) => tab.id === activeTabId);
  const activeDocumentMenuTab = tabs.find((tab): tab is DocumentWorkspaceTab => (
    tab.kind === 'document' && tab.id === documentMenuTabId
  )) || null;

  const closeDocumentMenu = () => {
    setDocumentMenuAnchor(null);
    setDocumentMenuTabId(null);
  };

  const handleCopySourceUrl = async (sourceUrl: string) => {
    try {
      await navigator.clipboard?.writeText(sourceUrl);
    } catch {
      window.prompt('Copy source URL', sourceUrl);
    }
  };

  const renderSystemTab = ({
    tab,
    icon,
    tooltip,
    label,
  }: {
    tab: BrowserWorkspaceTab | MemoryWorkspaceTab | TraceWorkspaceTab;
    icon: React.ReactElement;
    tooltip: string;
    label?: string | React.ReactElement;
  }) => {
    const active = tab.id === activeTabId;
    return (
      <Tooltip key={tab.id} title={tooltip}>
        <Tab
          aria-label={tooltip}
          icon={icon}
          iconPosition="start"
          label={active ? (label ?? tab.label) : undefined}
          onClick={active && tab.kind === 'memory' ? () => onTabChange(tab.id) : undefined}
          sx={{
            ...commonTabSx,
            minWidth: active ? undefined : 44,
            px: active ? 1.5 : 1,
          }}
        />
      </Tooltip>
    );
  };

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', minHeight: 40, minWidth: 0, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
      <Tabs
        value={activeIndex >= 0 ? activeIndex : false}
        onChange={(_, index) => tabs[index] && onTabChange(tabs[index].id)}
        variant="scrollable"
        scrollButtons="auto"
        allowScrollButtonsMobile
        aria-label="Workspace content"
        sx={{
          minHeight: 40,
          minWidth: 0,
          width: '100%',
          maxWidth: '100%',
          display: 'flex',
          '& .MuiTabs-scroller': {
            minWidth: 0,
            flex: '1 1 auto',
          },
          '& .MuiTabs-flexContainer': { minWidth: 0 },
          '& .MuiTabs-scrollButtons': { flex: '0 0 36px' },
          '& .MuiTabs-scrollButtons.Mui-disabled': {
            width: 0,
            flexBasis: 0,
            opacity: 0,
            overflow: 'hidden',
          },
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
            const active = tab.id === activeTabId;
            return (
              <Tooltip key={tab.id} title="Home">
                <Tab
                  aria-label="Home"
                  label={tab.label}
                  sx={{
                    ...commonTabSx,
                    minWidth: active ? undefined : 44,
                    px: active ? 1.5 : 1,
                  }}
                />
              </Tooltip>
            );
          }
          if (tab.kind === 'memory') {
            return renderSystemTab({
              tab,
              icon: <PsychologyIcon fontSize="small" />,
              tooltip: 'Memory',
            });
          }
          if (tab.kind === 'canvas') {
            return <Tab key={tab.id} icon={<Badge color="error" badgeContent={tab.issueCount || 0}><AccountTreeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'spec') {
            return <Tab key={tab.id} icon={<Badge color="primary" variant={tab.dirty ? 'dot' : 'standard'}><CodeIcon fontSize="small" /></Badge>} iconPosition="start" label={tab.label} sx={commonTabSx} />;
          }
          if (tab.kind === 'trace') {
            return renderSystemTab({
              tab,
              icon: <Badge color={statusColor(tab.status)} variant={tab.status === 'running' ? 'dot' : 'standard'} badgeContent={tab.status === 'running' ? undefined : tab.count} max={99}><RouteIcon fontSize="small" /></Badge>,
              tooltip: 'Debug Trace',
            });
          }
          if (tab.kind === 'browser') {
            const active = tab.id === activeTabId;
            return renderSystemTab({
              tab,
              icon: <PublicIcon fontSize="small" />,
              tooltip: 'Browser',
              label: (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
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
                              stopTabAction(event);
                              if (!isBrowserCapturing) onAddBrowserToThread();
                            }}
                            onKeyDown={(event) => {
                              if (!isBrowserCapturing && (event.key === 'Enter' || event.key === ' ')) {
                                stopTabAction(event);
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
              ),
            });
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
              aria-label={fullTitle}
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0, width: '100%' }}>
                  {isBrowserDocument ? (
                    <Tooltip title="Open source webpage">
                      <span>
                        <Box
                          component="span"
                          role={tab.sourceUrl ? 'button' : undefined}
                          tabIndex={tab.sourceUrl ? 0 : -1}
                          onClick={(event) => {
                            stopTabAction(event);
                            if (tab.sourceUrl) {
                              window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                            }
                          }}
                          onKeyDown={(event) => {
                            if (tab.sourceUrl && (event.key === 'Enter' || event.key === ' ')) {
                              stopTabAction(event);
                              window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                            }
                          }}
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flex: '0 0 auto',
                            cursor: tab.sourceUrl ? 'pointer' : 'default',
                            p: '2px',
                            borderRadius: 1,
                            color: 'primary.main',
                          }}
                        >
                          <LanguageIcon fontSize="small" />
                        </Box>
                      </span>
                    </Tooltip>
                  ) : <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7, flex: '0 0 auto' }} />}
                  <Tooltip title={fullTitle} placement="bottom">
                    <Typography
                      component="span"
                      sx={{
                        minWidth: 0,
                        flex: '1 1 auto',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        textAlign: 'left',
                      }}
                    >
                      {label}
                    </Typography>
                  </Tooltip>
                  {tab.parsingStatus === ProcessStatus.Pending && <CircularProgress size={13} sx={{ flex: '0 0 auto' }} />}
                  {tab.parsingStatus === ProcessStatus.Failed && (
                    <Tooltip title={tab.processingError || 'Processing failed'}>
                      <ErrorIcon color="error" sx={{ fontSize: 15, flex: '0 0 auto' }} />
                    </Tooltip>
                  )}
                  {tab.associationScope === 'project' && (
                    <Tooltip title="Project knowledge"><FolderIcon sx={{ fontSize: 14, color: 'primary.main', flex: '0 0 auto' }} /></Tooltip>
                  )}
                  <Tooltip title="Document actions">
                    <span>
                      <Box
                        component="span"
                        role="button"
                        tabIndex={0}
                        aria-label={`Document actions for ${fullTitle}`}
                        onClick={(event) => {
                          stopTabAction(event);
                          setDocumentMenuAnchor(event.currentTarget);
                          setDocumentMenuTabId(tab.id);
                        }}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter' || event.key === ' ') {
                            stopTabAction(event);
                            setDocumentMenuAnchor(event.currentTarget);
                            setDocumentMenuTabId(tab.id);
                          }
                        }}
                        sx={{ p: 0.2, display: 'inline-flex', borderRadius: 1, opacity: 0.72, flex: '0 0 auto' }}
                      >
                        <MoreVertIcon sx={{ fontSize: 16 }} />
                      </Box>
                    </span>
                  </Tooltip>
                  </Box>
                }
                sx={{
                  ...commonTabSx,
                  width: documentTabWidth,
                  minWidth: documentTabWidth,
                  maxWidth: documentTabWidth,
                  overflow: 'hidden',
                }}
              />
            );
        })}
      </Tabs>
      <Menu
        anchorEl={documentMenuAnchor}
        open={Boolean(documentMenuAnchor && activeDocumentMenuTab)}
        onClose={closeDocumentMenu}
        onClick={(event) => event.stopPropagation()}
      >
        {activeDocumentMenuTab?.sourceType === ThreadFileSourceType.Browser && activeDocumentMenuTab.sourceUrl && (
          <MenuItem
            onClick={() => {
              window.open(activeDocumentMenuTab.sourceUrl, '_blank', 'noopener,noreferrer');
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><OpenInBrowserIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Open source webpage</ListItemText>
          </MenuItem>
        )}
        {activeDocumentMenuTab?.sourceType === ThreadFileSourceType.Browser && activeDocumentMenuTab.sourceUrl && (
          <MenuItem
            onClick={() => {
              void handleCopySourceUrl(activeDocumentMenuTab.sourceUrl!);
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><ContentCopyIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Copy source URL</ListItemText>
          </MenuItem>
        )}
        {activeDocumentMenuTab && onDocumentPromote && activeDocumentMenuTab.associationScope === 'thread' && !activeDocumentMenuTab.isProjectKnowledge && (
          <MenuItem
            onClick={() => {
              onDocumentPromote(activeDocumentMenuTab.id);
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><CreateNewFolderIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Add to project knowledge</ListItemText>
          </MenuItem>
        )}
        {activeDocumentMenuTab?.parsingStatus === ProcessStatus.Failed && onDocumentRetry && (
          <MenuItem
            onClick={() => {
              onDocumentRetry(activeDocumentMenuTab.id);
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><ReplayIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Retry processing</ListItemText>
          </MenuItem>
        )}
        {activeDocumentMenuTab && onDocumentRemove && (documentContext === 'project' || activeDocumentMenuTab.associationScope !== 'project') && (
          <MenuItem
            onClick={() => {
              onDocumentRemove(activeDocumentMenuTab.id);
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><DeleteIcon color="error" fontSize="small" /></ListItemIcon>
            <ListItemText>{documentContext === 'project' ? 'Remove from project' : 'Delete from thread'}</ListItemText>
          </MenuItem>
        )}
        {activeDocumentMenuTab && onTabClose && (
          <MenuItem
            onClick={() => {
              onTabClose(activeDocumentMenuTab.id);
              closeDocumentMenu();
            }}
          >
            <ListItemIcon><CloseIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Close tab</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </Box>
  );
});
