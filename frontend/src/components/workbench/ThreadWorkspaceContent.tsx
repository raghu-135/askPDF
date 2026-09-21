import dynamic from 'next/dynamic';
import { Box, CircularProgress, Typography } from '@mui/material';
import { EMBEDDINGS_TAB_ID, PROJECT_OVERVIEW_TAB_ID, isBrowserWorkspaceActive, type PdfTab } from '../../lib/document-tabs';
import EmbeddingSpaceViewer from '../embeddings/EmbeddingSpaceViewer';
import TraceWorkspace, { type TraceRunTab } from './TraceWorkspace';
import ResearchCanvasWorkspace from './ResearchCanvasWorkspace';
import BrowserWorkspaceFrame from './BrowserWorkspaceFrame';
import MemoryWorkspace from './MemoryWorkspace';
import HomeInstructions from './HomeInstructions';
import ProjectOverview from './ProjectOverview';
import type { Project, Thread } from '../../lib/api';
import type { MemoryManagerIntent } from '../../lib/memory-manager';
import type { DocumentCanvasCitationTarget } from '../../lib/canvas-spec';
import { RESEARCH_CANVAS_TAB_ID } from '../../lib/canvas-spec';

const PdfViewer = dynamic(() => import('../PdfViewer'), { ssr: false });

export default function ThreadWorkspaceContent({
  activeTabId,
  activeDocument,
  documentSentences,
  documentDownloadUrl,
  traceTabs,
  activeTraceId,
  onActiveTraceChange,
  onCloseTrace,
  isLoading = false,
  isResizing = false,
  darkMode = false,
  currentDocumentSentenceId = null,
  onDocumentJump,
  autoScroll = false,
  highlightEnabled = true,
  threadId,
  activeThread = null,
  activeProject = null,
  projectInventoryVersion = 0,
  curatorRefreshVersion = 0,
  inventoryLoading = false,
  hasProjects = false,
  onOpenMemoryCurator,
  onCreateProject,
  onCreateThread,
  onCapturePage,
  onRequestUpload,
  documentCount = 0,
  emptyTitle,
  emptyDescription,
  activeCanvasId = null,
  onActiveCanvasChange,
  onOpenDocumentCitation,
  canvasRefreshVersion = 0,
  documents = [],
}: {
  activeTabId: string | null;
  activeDocument: PdfTab | null;
  documentSentences: any[];
  documentDownloadUrl: string | null;
  traceTabs: TraceRunTab[];
  activeTraceId: string | null;
  onActiveTraceChange: (runId: string) => void;
  onCloseTrace: (runId: string) => void;
  isLoading?: boolean;
  isResizing?: boolean;
  darkMode?: boolean;
  currentDocumentSentenceId?: number | null;
  onDocumentJump: (id: number) => void;
  autoScroll?: boolean;
  highlightEnabled?: boolean;
  threadId?: string | null;
  activeThread?: Thread | null;
  activeProject?: Project | null;
  projectInventoryVersion?: number;
  curatorRefreshVersion?: number;
  inventoryLoading?: boolean;
  hasProjects?: boolean;
  onOpenMemoryCurator?: (intent: MemoryManagerIntent) => void;
  onCreateProject?: () => void;
  onCreateThread?: () => void;
  onCapturePage?: () => void;
  onRequestUpload?: () => void;
  documentCount?: number;
  emptyTitle: string;
  emptyDescription: string;
  activeCanvasId?: string | null;
  onActiveCanvasChange?: (canvasId: string) => void;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
  canvasRefreshVersion?: number;
  documents?: readonly PdfTab[];
}) {
  return (
    <Box sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
      {activeTabId === 'home-tab' ? (
        <HomeInstructions
          darkMode={darkMode}
          inventoryLoading={inventoryLoading}
          hasProjects={hasProjects}
          onCreateProject={onCreateProject}
        />
      ) : activeTabId === PROJECT_OVERVIEW_TAB_ID ? (
        isLoading ? (
          <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: darkMode ? '#222' : 'grey.50', color: darkMode ? '#eee' : 'inherit' }}>
            <CircularProgress color={darkMode ? 'inherit' : 'primary'} />
            <Typography sx={{ ml: 2 }}>Loading documents...</Typography>
          </Box>
        ) : (
          <ProjectOverview
            projectName={activeProject?.name || 'Project'}
            documentCount={documentCount}
            darkMode={darkMode}
            onCreateThread={onCreateThread || (() => undefined)}
            onCapturePage={onCapturePage || (() => undefined)}
            onRequestUpload={onRequestUpload || (() => undefined)}
          />
        )
      ) : activeTabId === RESEARCH_CANVAS_TAB_ID ? (
        <ResearchCanvasWorkspace
          threadId={threadId ?? null}
          activeCanvasId={activeCanvasId}
          onActiveCanvasChange={onActiveCanvasChange || (() => undefined)}
          onOpenDocumentCitation={onOpenDocumentCitation}
          refreshVersion={canvasRefreshVersion}
        />
      ) : activeTabId === EMBEDDINGS_TAB_ID ? (
        threadId ? (
          <EmbeddingSpaceViewer
            threadId={threadId}
            documents={documents || []}
            onOpenDocumentCitation={onOpenDocumentCitation}
          />
        ) : (
          <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 4 }}>
            <Typography color="text.secondary">Select a thread to inspect embeddings.</Typography>
          </Box>
        )
      ) : activeTabId === 'trace-tab' ? (
        <TraceWorkspace
          tabs={traceTabs}
          activeRunId={activeTraceId}
          onActiveRunChange={onActiveTraceChange}
          onClose={onCloseTrace}
          suspendHeavyContent={isResizing}
        />
      ) : activeTabId === 'memory-tab' ? (
        <MemoryWorkspace
          activeThread={activeThread}
          activeProject={activeProject}
          projectInventoryVersion={projectInventoryVersion}
          curatorRefreshVersion={curatorRefreshVersion}
          onOpenCurator={onOpenMemoryCurator}
        />
      ) : isBrowserWorkspaceActive({ activeTabId }) ? (
        <BrowserWorkspaceFrame />
      ) : isLoading ? (
        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: darkMode ? '#222' : 'grey.50', color: darkMode ? '#eee' : 'inherit' }}>
          <CircularProgress color={darkMode ? 'inherit' : 'primary'} />
          <Typography sx={{ ml: 2 }}>Loading documents...</Typography>
        </Box>
      ) : documentDownloadUrl ? (
        <PdfViewer
          downloadUrl={documentDownloadUrl}
          sentences={documentSentences}
          currentId={currentDocumentSentenceId}
          onJump={onDocumentJump}
          autoScroll={autoScroll}
          isResizing={isResizing}
          highlightEnabled={highlightEnabled}
          darkMode={darkMode}
          threadId={threadId ?? null}
          fileHash={activeDocument?.fileHash ?? null}
          mode={threadId ? 'thread-editable' : 'source-readonly'}
        />
      ) : (
        <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', bgcolor: darkMode ? '#222' : 'grey.50', color: darkMode ? '#eee' : 'inherit', p: 4 }}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h5" gutterBottom>{emptyTitle}</Typography>
            <Typography color="text.secondary">{emptyDescription}</Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
}
