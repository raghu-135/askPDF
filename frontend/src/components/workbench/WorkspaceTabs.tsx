import React from 'react';
import {
  Badge,
  Box,
  Tab,
  Tabs,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import BugReportIcon from '@mui/icons-material/BugReport';
import CodeIcon from '@mui/icons-material/Code';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import FolderIcon from '@mui/icons-material/Folder';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PublicIcon from '@mui/icons-material/Public';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import ForumIcon from '@mui/icons-material/Forum';
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
export type ProjectWorkspaceTab = { kind: 'project'; id: 'project-tab'; label: string };
export type ThreadWorkspaceTab = { kind: 'thread'; id: 'thread-tab'; label: string };
export type MemoryWorkspaceTab = { kind: 'memory'; id: 'memory-tab'; label: string };
export type DocumentsWorkspaceTab = { kind: 'documents'; id: 'documents-tab'; label: string; count?: number };
export type CanvasWorkspaceTab = { kind: 'canvas'; id: 'canvas-tab'; label: string; issueCount?: number };
export type ResearchCanvasWorkspaceTab = { kind: 'research_canvas'; id: 'research-canvas-tab'; label: string; count?: number };
export type SpecWorkspaceTab = { kind: 'spec'; id: 'spec-tab'; label: string; dirty?: boolean };
export type TraceWorkspaceTab = {
  kind: 'trace';
  id: 'trace-tab';
  label: string;
  status?: 'idle' | 'running' | 'failed' | 'review';
  count?: number;
};
export type EmbeddingsWorkspaceTab = {
  kind: 'embeddings';
  id: 'embeddings-tab';
  label: string;
};
export type WorkspaceTab =
  | BrowserWorkspaceTab
  | HomeWorkspaceTab
  | ProjectWorkspaceTab
  | ThreadWorkspaceTab
  | MemoryWorkspaceTab
  | DocumentsWorkspaceTab
  | CanvasWorkspaceTab
  | ResearchCanvasWorkspaceTab
  | SpecWorkspaceTab
  | TraceWorkspaceTab
  | EmbeddingsWorkspaceTab;

const statusColor = (status?: TraceWorkspaceTab['status']) => {
  if (status === 'failed') return 'error';
  if (status === 'running') return 'primary';
  if (status === 'review') return 'warning';
  return 'default';
};

const workspaceTabsSx = {
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
    color: 'text.secondary',
  },
  '& .MuiTab-root.Mui-selected': {
    color: 'primary.main',
    fontWeight: 600,
    bgcolor: 'action.selected',
  },
  '& .MuiTabs-indicator': {
    height: 2,
  },
};

const commonTabSx = {
  textTransform: 'none',
  minHeight: 40,
};

export default React.memo(function WorkspaceTabs({
  tabs,
  activeTabId,
  onTabChange,
}: {
  tabs: WorkspaceTab[];
  activeTabId: string | null;
  onTabChange: (tabId: string) => void;
}) {
  if (tabs.length === 0) return null;
  const selectedTabValue = activeTabId && tabs.some((tab) => tab.id === activeTabId) ? activeTabId : false;

  const renderLabeledTab = ({
    tabId,
    label,
    icon,
    tooltip,
  }: {
    tabId: string;
    label: string;
    icon?: React.ReactElement;
    tooltip: string;
  }) => (
    <Tab
      key={tabId}
      value={tabId}
      aria-label={tooltip}
      title={tooltip}
      icon={icon}
      iconPosition={icon ? 'start' : undefined}
      label={label}
      sx={commonTabSx}
    />
  );

  return (
    <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', minHeight: 40, minWidth: 0, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
      <Tabs
        value={selectedTabValue}
        onChange={(_, tabId) => typeof tabId === 'string' && onTabChange(tabId)}
        variant="scrollable"
        scrollButtons="auto"
        allowScrollButtonsMobile
        textColor="primary"
        indicatorColor="primary"
        aria-label="Workspace content"
        sx={workspaceTabsSx}
      >
        {tabs.map((tab) => {
          if (tab.kind === 'home') {
            return renderLabeledTab({ tabId: tab.id, label: tab.label, tooltip: 'Home' });
          }
          if (tab.kind === 'project') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <FolderIcon fontSize="small" />,
              tooltip: 'Project',
            });
          }
          if (tab.kind === 'thread') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <ForumIcon fontSize="small" />,
              tooltip: 'Thread',
            });
          }
          if (tab.kind === 'memory') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <PsychologyIcon fontSize="small" />,
              tooltip: 'Memory',
            });
          }
          if (tab.kind === 'documents') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <Badge color="default" badgeContent={tab.count || undefined} max={99}><DescriptionOutlinedIcon fontSize="small" /></Badge>,
              tooltip: 'Documents',
            });
          }
          if (tab.kind === 'browser') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <PublicIcon fontSize="small" />,
              tooltip: 'Browser',
            });
          }
          if (tab.kind === 'canvas') {
            return (
              <Tab
                key={tab.id}
                value={tab.id}
                icon={<Badge color="error" badgeContent={tab.issueCount || 0}><AccountTreeIcon fontSize="small" /></Badge>}
                iconPosition="start"
                label={tab.label}
                sx={commonTabSx}
              />
            );
          }
          if (tab.kind === 'spec') {
            return (
              <Tab
                key={tab.id}
                value={tab.id}
                icon={<Badge color="primary" variant={tab.dirty ? 'dot' : 'standard'}><CodeIcon fontSize="small" /></Badge>}
                iconPosition="start"
                label={tab.label}
                sx={commonTabSx}
              />
            );
          }
          if (tab.kind === 'research_canvas') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <Badge color="default" badgeContent={tab.count} max={99}><DashboardOutlinedIcon fontSize="small" /></Badge>,
              tooltip: 'Canvas',
            });
          }
          if (tab.kind === 'trace') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: (
                <Badge
                  color={statusColor(tab.status)}
                  variant={tab.status === 'running' ? 'dot' : 'standard'}
                  badgeContent={tab.status === 'running' ? undefined : tab.count}
                  max={99}
                >
                  <BugReportIcon fontSize="small" />
                </Badge>
              ),
              tooltip: 'Debug Trace',
            });
          }
          if (tab.kind === 'embeddings') {
            return renderLabeledTab({
              tabId: tab.id,
              label: tab.label,
              icon: <ScatterPlotIcon fontSize="small" />,
              tooltip: 'Embeddings',
            });
          }
          return null;
        })}
      </Tabs>
    </Box>
  );
});
