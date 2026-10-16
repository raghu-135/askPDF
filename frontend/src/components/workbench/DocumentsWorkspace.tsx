import dynamic from 'next/dynamic';
import React, { useState } from 'react';
import { Box, CircularProgress, Stack, Typography } from '@mui/material';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import type { PdfTab } from '../../lib/document-tabs';
import type { DocumentWorkspaceTab } from './WorkspaceTabs';
import NestedInstanceTabs from './NestedInstanceTabs';
import DocumentInstanceTabLabel from './DocumentInstanceTabLabel';
import WorkspaceEmptyState from './WorkspaceEmptyState';
import AddSourcesMenu from './AddSourcesMenu';
import AddSourcesActions from './AddSourcesActions';
import {
  PlayerControlsProvider,
  PlayerExtrasChrome,
  PlayerPlaybackChrome,
  type PlayerControlsProps,
} from '../PlayerControls';

const PdfViewer = dynamic(() => import('../PdfViewer'), { ssr: false });

export default function DocumentsWorkspace({
  documents,
  activeDocumentId,
  onActiveDocumentChange,
  onCloseDocument,
  onDocumentRemove,
  onDocumentPromote,
  onDocumentRetry,
  onInspectChunks,
  documentContext = 'thread',
  isLoading = false,
  isResizing = false,
  darkMode = false,
  threadId = null,
  activeDocument,
  documentSentences,
  documentDownloadUrl,
  cachedDocumentId = null,
  cachedDocument,
  cachedSentences = [],
  cachedDownloadUrl = null,
  currentDocumentSentenceId = null,
  onDocumentJump,
  autoScroll = false,
  highlightEnabled = true,
  onRequestUpload,
  onCapturePage,
  onAddCapture,
  isBrowserCapturing = false,
  emptyTitle = 'Add sources to this thread',
  emptyDescription = 'Upload a PDF or capture a page from the browser to start asking questions.',
  showCreateThread = false,
  onCreateThread,
  playerControlProps = null,
}: {
  documents: readonly PdfTab[];
  activeDocumentId: string | null;
  onActiveDocumentChange: (documentId: string) => void;
  onCloseDocument?: (documentId: string) => void;
  onDocumentRemove?: (documentId: string) => void;
  onDocumentPromote?: (documentId: string) => void;
  onDocumentRetry?: (documentId: string) => void;
  onInspectChunks?: (tab: DocumentWorkspaceTab) => void;
  documentContext?: 'thread' | 'project';
  isLoading?: boolean;
  isResizing?: boolean;
  darkMode?: boolean;
  threadId?: string | null;
  activeDocument: PdfTab | null;
  documentSentences: any[];
  documentDownloadUrl: string | null;
  cachedDocumentId?: string | null;
  cachedDocument?: PdfTab | null;
  cachedSentences?: any[];
  cachedDownloadUrl?: string | null;
  currentDocumentSentenceId?: number | null;
  onDocumentJump: (id: number) => void;
  autoScroll?: boolean;
  highlightEnabled?: boolean;
  onRequestUpload?: () => void;
  onCapturePage?: () => void;
  onAddCapture?: () => void;
  isBrowserCapturing?: boolean;
  emptyTitle?: string;
  emptyDescription?: string;
  showCreateThread?: boolean;
  onCreateThread?: () => void;
  playerControlProps?: PlayerControlsProps | null;
}) {
  const [chromeHost, setChromeHost] = useState<HTMLDivElement | null>(null);
  const documentTabs = documents.map((tab) => ({ ...tab, kind: 'document' as const }));

  const nestedTabs = documentTabs.map((tab) => ({
    id: tab.id,
    label: (
      <DocumentInstanceTabLabel
        tab={tab}
        documentContext={documentContext}
        onClose={onCloseDocument}
        onDocumentRemove={onDocumentRemove}
        onDocumentPromote={onDocumentPromote}
        onDocumentRetry={onDocumentRetry}
        onInspectChunks={onInspectChunks}
      />
    ),
  }));

  const addSourcesAction = onRequestUpload && onAddCapture ? (
    <AddSourcesMenu
      onUpload={onRequestUpload}
      onCapturePage={onAddCapture}
      capturing={isBrowserCapturing}
      tooltip="Add source"
    />
  ) : null;

  if (isLoading) {
    return (
      <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: darkMode ? '#222' : 'grey.50', color: darkMode ? '#eee' : 'inherit' }}>
        <CircularProgress color={darkMode ? 'inherit' : 'primary'} />
        <Typography sx={{ ml: 2 }}>Loading documents...</Typography>
      </Box>
    );
  }

  if (documents.length === 0) {
    return (
      <WorkspaceEmptyState
        darkMode={darkMode}
        icon={<PictureAsPdfIcon sx={{ fontSize: 48, opacity: 0.4 }} />}
        title={emptyTitle}
        description={emptyDescription}
        actions={onRequestUpload && onCapturePage ? (
          <AddSourcesActions
            onRequestUpload={onRequestUpload}
            onCapturePage={onCapturePage}
            onCreateThread={showCreateThread ? onCreateThread : undefined}
            isBrowserCapturing={isBrowserCapturing}
          />
        ) : undefined}
      />
    );
  }

  const showCachedViewer = cachedDocumentId
    && cachedDocumentId !== activeDocument?.fileHash
    && cachedDocument
    && cachedDownloadUrl;

  const workspaceChromeSlots = playerControlProps ? {
    playback: <PlayerPlaybackChrome />,
    extras: <PlayerExtrasChrome />,
  } : undefined;

  const viewerBody = (
    <Box sx={{ height: '100%', minHeight: 0, display: 'grid', gridTemplateRows: 'auto auto minmax(0, 1fr)' }}>
      <NestedInstanceTabs
        tabs={nestedTabs}
        activeId={activeDocumentId}
        onActiveChange={onActiveDocumentChange}
        ariaLabel="Open documents"
        trailingAction={addSourcesAction}
      />
      <Box ref={setChromeHost} sx={{ minHeight: 0 }} />
      <Box sx={{ minHeight: 0, overflow: 'hidden', position: 'relative' }}>
        {showCachedViewer ? (
          <Box sx={{ position: 'absolute', inset: 0, visibility: 'hidden', pointerEvents: 'none' }} aria-hidden>
            <PdfViewer
              downloadUrl={cachedDownloadUrl}
              sentences={cachedSentences}
              currentId={null}
              onJump={() => undefined}
              autoScroll={false}
              isResizing={isResizing}
              highlightEnabled={false}
              darkMode={darkMode}
              threadId={threadId}
              fileHash={cachedDocument?.fileHash ?? null}
              mode={threadId ? 'thread-editable' : 'source-readonly'}
            />
          </Box>
        ) : null}
        {documentDownloadUrl && activeDocument ? (
          <PdfViewer
            downloadUrl={documentDownloadUrl}
            sentences={documentSentences}
            currentId={currentDocumentSentenceId}
            onJump={onDocumentJump}
            autoScroll={autoScroll}
            isResizing={isResizing}
            highlightEnabled={highlightEnabled}
            darkMode={darkMode}
            threadId={threadId}
            fileHash={activeDocument.fileHash}
            mode={threadId ? 'thread-editable' : 'source-readonly'}
            chromeHost={chromeHost}
            workspaceChromeSlots={workspaceChromeSlots}
          />
        ) : (
          <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 4 }}>
            <Stack spacing={1} alignItems="center">
              <CircularProgress size={24} />
              <Typography color="text.secondary">Loading document…</Typography>
            </Stack>
          </Box>
        )}
      </Box>
    </Box>
  );

  if (!playerControlProps) {
    return viewerBody;
  }

  return (
    <PlayerControlsProvider {...playerControlProps}>
      {viewerBody}
    </PlayerControlsProvider>
  );
}
