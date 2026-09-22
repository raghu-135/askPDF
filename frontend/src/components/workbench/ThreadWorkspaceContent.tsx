import dynamic from 'next/dynamic';
import { Box, CircularProgress, Typography } from '@mui/material';
import {
  DOCUMENTS_TAB_ID,
  EMBEDDINGS_TAB_ID,
  PROJECT_OVERVIEW_TAB_ID,
  isBrowserWorkspaceActive,
  isDocumentsWorkspaceActive,
  type PdfTab,
} from '../../lib/document-tabs';
import TraceWorkspace, { type TraceRunTab } from './TraceWorkspace';
import ResearchCanvasWorkspace from './ResearchCanvasWorkspace';
import BrowserWorkspaceFrame from './BrowserWorkspaceFrame';
import MemoryWorkspace from './MemoryWorkspace';
import HomeInstructions from './HomeInstructions';
import ProjectOverview from './ProjectOverview';
import DocumentsWorkspace from './DocumentsWorkspace';
import type { PlayerControlsProps } from '../PlayerControls';
import type { Project, Thread } from '../../lib/api';
import type { MemoryManagerIntent } from '../../lib/memory-manager';
import type { DocumentCanvasCitationTarget } from '../../lib/canvas-spec';
import { RESEARCH_CANVAS_TAB_ID } from '../../lib/canvas-spec';
import type { DocumentWorkspaceTab } from './WorkspaceTabs';

const EmbeddingSpaceViewer = dynamic(() => import('../embeddings/EmbeddingSpaceViewer'), { ssr: false });

export default function ThreadWorkspaceContent({
  activeTabId,
  activeDocumentId,
  onActiveDocumentChange,
  activeDocument,
  documentSentences,
  documentDownloadUrl,
  cachedDocumentId,
  cachedDocument,
  cachedSentences,
  cachedDownloadUrl,
  traceTabs,
  activeTraceId,
  onActiveTraceChange,
  onCloseTrace,
  onCloseDocument,
  onDocumentRemove,
  onDocumentPromote,
  onDocumentRetry,
  onInspectChunks,
  documentContext = 'thread',
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
  onAddCapture,
  onRequestUpload,
  isBrowserCapturing = false,
  documentCount = 0,
  activeCanvasId = null,
  onActiveCanvasChange,
  onOpenDocumentCitation,
  canvasRefreshVersion = 0,
  documents = [],
  playerControlProps = null,
}: {
  activeTabId: string | null;
  activeDocumentId: string | null;
  onActiveDocumentChange: (documentId: string) => void;
  activeDocument: PdfTab | null;
  documentSentences: any[];
  documentDownloadUrl: string | null;
  cachedDocumentId?: string | null;
  cachedDocument?: PdfTab | null;
  cachedSentences?: any[];
  cachedDownloadUrl?: string | null;
  traceTabs: TraceRunTab[];
  activeTraceId: string | null;
  onActiveTraceChange: (runId: string) => void;
  onCloseTrace: (runId: string) => void;
  onCloseDocument?: (documentId: string) => void;
  onDocumentRemove?: (documentId: string) => void;
  onDocumentPromote?: (documentId: string) => void;
  onDocumentRetry?: (documentId: string) => void;
  onInspectChunks?: (tab: DocumentWorkspaceTab) => void;
  documentContext?: 'thread' | 'project';
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
  onAddCapture?: () => void;
  onRequestUpload?: () => void;
  isBrowserCapturing?: boolean;
  documentCount?: number;
  activeCanvasId?: string | null;
  onActiveCanvasChange?: (canvasId: string) => void;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
  canvasRefreshVersion?: number;
  documents?: readonly PdfTab[];
  playerControlProps?: PlayerControlsProps | null;
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
            isBrowserCapturing={isBrowserCapturing}
          />
        )
      ) : isDocumentsWorkspaceActive(activeTabId) ? (
        <DocumentsWorkspace
          documents={documents}
          activeDocumentId={activeDocumentId}
          onActiveDocumentChange={onActiveDocumentChange}
          onCloseDocument={onCloseDocument}
          onDocumentRemove={onDocumentRemove}
          onDocumentPromote={onDocumentPromote}
          onDocumentRetry={onDocumentRetry}
          onInspectChunks={onInspectChunks}
          documentContext={documentContext}
          isLoading={isLoading}
          isResizing={isResizing}
          darkMode={darkMode}
          threadId={threadId ?? null}
          activeDocument={activeDocument}
          documentSentences={documentSentences}
          documentDownloadUrl={documentDownloadUrl}
          cachedDocumentId={cachedDocumentId}
          cachedDocument={cachedDocument}
          cachedSentences={cachedSentences}
          cachedDownloadUrl={cachedDownloadUrl}
          currentDocumentSentenceId={currentDocumentSentenceId}
          onDocumentJump={onDocumentJump}
          autoScroll={autoScroll}
          highlightEnabled={highlightEnabled}
          onRequestUpload={onRequestUpload}
          onCapturePage={onCapturePage}
          onAddCapture={onAddCapture}
          isBrowserCapturing={isBrowserCapturing}
          emptyTitle={threadId ? 'Add sources to this thread' : 'Add project knowledge'}
          emptyDescription={threadId
            ? 'Upload a PDF or capture a page from the browser to start asking questions.'
            : 'Upload a PDF or capture a page to add shared project knowledge.'}
          showCreateThread={Boolean(activeProject && !threadId)}
          onCreateThread={onCreateThread}
          playerControlProps={playerControlProps}
        />
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
      ) : (
        <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', bgcolor: darkMode ? '#222' : 'grey.50', color: darkMode ? '#eee' : 'inherit', p: 4 }}>
          <Typography color="text.secondary">Choose a workspace tab to get started.</Typography>
        </Box>
      )}
    </Box>
  );
}
