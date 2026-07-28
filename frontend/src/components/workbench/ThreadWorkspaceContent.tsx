import dynamic from 'next/dynamic';
import { Box, CircularProgress, Typography } from '@mui/material';
import { isBrowserWorkspaceActive, type PdfTab } from '../../lib/document-tabs';
import TraceWorkspace, { type TraceRunTab } from './TraceWorkspace';
import BrowserWorkspaceFrame from './BrowserWorkspaceFrame';
import MemoryWorkspace from './MemoryWorkspace';
import type { Thread } from '../../lib/api';

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
  isBrowserActive = false,
  isLoading = false,
  isResizing = false,
  darkMode = false,
  currentDocumentSentenceId = null,
  onDocumentJump,
  autoScroll = false,
  highlightEnabled = true,
  threadId,
  activeThread = null,
  emptyTitle,
  emptyDescription,
}: {
  activeTabId: string | null;
  activeDocument: PdfTab | null;
  documentSentences: any[];
  documentDownloadUrl: string | null;
  traceTabs: TraceRunTab[];
  activeTraceId: string | null;
  onActiveTraceChange: (runId: string) => void;
  onCloseTrace: (runId: string) => void;
  isBrowserActive?: boolean;
  isLoading?: boolean;
  isResizing?: boolean;
  darkMode?: boolean;
  currentDocumentSentenceId?: number | null;
  onDocumentJump: (id: number) => void;
  autoScroll?: boolean;
  highlightEnabled?: boolean;
  threadId?: string | null;
  activeThread?: Thread | null;
  emptyTitle: string;
  emptyDescription: string;
}) {
  return (
    <Box sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
      {activeTabId === 'trace-tab' ? (
        <TraceWorkspace
          tabs={traceTabs}
          activeRunId={activeTraceId}
          onActiveRunChange={onActiveTraceChange}
          onClose={onCloseTrace}
          suspendHeavyContent={isResizing}
        />
      ) : activeTabId === 'memory-tab' ? (
        <MemoryWorkspace activeThread={activeThread} />
      ) : isBrowserWorkspaceActive({ activeTabId, isBrowserActive }) ? (
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
