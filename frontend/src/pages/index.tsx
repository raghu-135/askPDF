import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import React, { useState, useEffect, useCallback, useMemo } from "react";
import { Typography, Box, Button, CssBaseline, IconButton, Tooltip, CircularProgress, Chip, Checkbox } from "@mui/material";
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import ForumIcon from '@mui/icons-material/Forum';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import ClearIcon from '@mui/icons-material/Clear';
import AutoAwesomeSharpIcon from '@mui/icons-material/AutoAwesomeSharp';

declare const process: {
  env: Record<string, string | undefined>;
};
import PdfUploader from "../components/PdfUploader";

import PlayerControls from "../components/PlayerControls";
import ChatInterface, { type ChatTraceDescriptor } from "../components/ChatInterface";
import ThreadSidebar, { ThreadSidebarHeaderState } from "../components/ThreadSidebar";
import { buildDocumentWorkspaceTabs, type PdfTab } from "../lib/document-tabs";
import WorkbenchShell, { useWorkbenchLayout } from '../components/workbench/WorkbenchShell';
import DockMenuButton from '../components/workbench/DockMenuButton';
import { WorkbenchToolbarTrailingActions } from '../components/workbench/WorkbenchToolbar';
import WorkspaceTabs from '../components/workbench/WorkspaceTabs';
import DocumentWorkspaceContent from '../components/workbench/DocumentWorkspaceContent';
import useTraceTabs from '../components/workbench/useTraceTabs';
import ThreadLineageTooltipContent from "../components/ThreadLineageTooltipContent";
import { Thread, removeSourceFromThread, getParsedSentences, captureBrowserPage, pollForFileReady, getThread, deleteThread, listThreads } from "../lib/api";
import { loadThreadTabs, createPdfTabFromUpload, extractTextFromSentences } from "../lib/thread-utils";
import { handleTabCloseUtil, getActiveTab, getActiveTabData } from "../lib/pdf-utils";
import { transformSentences } from "../lib/bbox-derivation";
import { ProcessStatus } from "../lib/enums";
import type { ResolvedWorkbenchPlacement } from '../lib/workbench-layout';

export default function Home() {
  // Multiple PDF tabs state
  const [pdfTabs, setPdfTabs] = useState<PdfTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [isPdfLoading, setIsPdfLoading] = useState(false);

  // Get active tab and its data using utility
  const activeTab = getActiveTab(pdfTabs, activeTabId);
  const { pdfSentences, downloadUrl, fileHash, fileName } = getActiveTabData(activeTab);

  const [activeSource, setActiveSource] = useState<'pdf' | 'chat'>('pdf');
  const [currentPdfId, setCurrentPdfId] = useState<number | null>(null);
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [chatSentences, setChatSentences] = useState<any[]>([]);

  // Highlight toggle
  const [highlightEnabled, setHighlightEnabled] = useState(true);
  // PDF dark mode toggle, SSR-safe: initialize as undefined, set after mount
  const [pdfDarkMode, setPdfDarkMode] = useState<boolean | undefined>(undefined);

  // On mount, set dark mode from browser preference
  useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      setPdfDarkMode(window.matchMedia('(prefers-color-scheme: dark)').matches);
    } else {
      setPdfDarkMode(false);
    }
  }, []);

  // Listen for browser color scheme changes
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => {
      setPdfDarkMode(e.matches);
    };
    media.addEventListener('change', handler);
    return () => media.removeEventListener('change', handler);
  }, []);

  // Thread state
  const [activeThread, setActiveThread] = useState<Thread | null>(null);

  // Right panel tab state (0 = Threads, 1 = Chat)
  const [rightPanelTab, setRightPanelTab] = useState(0);

  // Sidebar refresh trigger
  const [sidebarVersion, setSidebarVersion] = useState(0);
  const [threadListHeaderState, setThreadListHeaderState] = useState<ThreadSidebarHeaderState | null>(null);
  const [isDeletingActiveThread, setIsDeletingActiveThread] = useState(false);
  const [rightPanelLineageThreads, setRightPanelLineageThreads] = useState<Thread[]>([]);

  // Browser tab state
  const [showBrowserTab, setShowBrowserTab] = useState(false);
  const [isBrowserActive, setIsBrowserActive] = useState(false);
  const [isBrowserCapturing, setIsBrowserCapturing] = useState(false);

  const [workbenchLayout, setWorkbenchLayout] = useWorkbenchLayout('askpdf.workbench.normal');
  const [resolvedPlacement, setResolvedPlacement] = useState<ResolvedWorkbenchPlacement>('right');
  const [isResizing, setIsResizing] = useState(false);
  const {
    traceTabs,
    activeTraceId,
    setActiveTraceId,
    openTrace,
    closeTrace,
    clearTraces,
  } = useTraceTabs();


  // Handle thread selection
  const handleThreadSelect = useCallback(async (thread: Thread | null) => {
    // Clear current state
    setPdfTabs([]);
    setActiveTabId(null);
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    setChatSentences([]);
    clearTraces();
    
    // Reset browser state when leaving thread context
    setIsBrowserActive(false);

    if (thread) {
      try {
        setIsPdfLoading(true);
        // Always fetch the latest thread data to ensure we have current files and stats
        const detailedThread = await import("../lib/api").then(m => m.getThread(thread.id));
        setActiveThread(detailedThread);

        const loadedTabs = await loadThreadTabs(detailedThread);
        if (loadedTabs.length > 0) {
          setPdfTabs(loadedTabs);
          setActiveTabId(loadedTabs[0].id);
        } else {
          setActiveTabId('browser-tab');
          setIsBrowserActive(true);
        }
      } catch (err) {
        console.error('Failed to load thread files:', err);
      } finally {
        setIsPdfLoading(false);
      }
    } else {
      setActiveThread(null);
    }
  }, [clearTraces]);

  const handleThreadForked = useCallback(async (thread: Thread) => {
    setSidebarVersion(v => v + 1);
    await handleThreadSelect(thread);
    setRightPanelTab(1);
  }, [handleThreadSelect]);

  const handleThreadSelectFromList = useCallback((thread: Thread | null) => {
    handleThreadSelect(thread);
    if (thread) {
      setRightPanelTab(1);
    }
  }, [handleThreadSelect]);

  const handleOpenThreadInChat = useCallback((thread: Thread) => {
    handleThreadSelect(thread);
    setRightPanelTab(1);
  }, [handleThreadSelect]);

  const handleThreadUpdated = async () => {
    setSidebarVersion(v => v + 1);

    if (!activeThread) return;
    try {
      const updatedThread = await getThread(activeThread.id);
      setActiveThread(updatedThread);
    } catch (error) {
      console.error('Failed to refresh thread after chat update:', error);
    }
  };

  const handleShowAllThreads = useCallback(async () => {
    setRightPanelTab(0);
    await handleThreadSelect(null);
  }, [handleThreadSelect]);

  const handleDeleteActiveThread = useCallback(async () => {
    if (!activeThread || isDeletingActiveThread) return;
    if (!confirm(`Delete "${activeThread.name}" and all its messages?`)) return;

    try {
      setIsDeletingActiveThread(true);
      await deleteThread(activeThread.id);
      setSidebarVersion(v => v + 1);
      await handleThreadSelect(null);
      setRightPanelTab(0);
    } catch (error) {
      console.error('Failed to delete active thread:', error);
      alert('Failed to delete thread.');
    } finally {
      setIsDeletingActiveThread(false);
    }
  }, [activeThread, handleThreadSelect, isDeletingActiveThread]);


  // Handle PDF upload - create new tab or focus existing
  const handlePdfUploaded = async (data: any) => {
    const fileHash = data?.fileHash;

    // Check if tab already exists for this file
    const existingTab = pdfTabs.find(tab => tab.fileHash === fileHash);
    if (existingTab) {
      // Focus existing tab instead of creating duplicate
      setActiveTabId(existingTab.id);
      setIsBrowserActive(false);
      setCurrentPdfId(null);
      setCurrentChatId(null);
      setPlayRequestId(null);
      setActiveSource('pdf');
      return;
    }

    const newTab = createPdfTabFromUpload(data);

    setPdfTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);

    if (activeThread && fileHash) {
      try {
        const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
        setActiveThread(updatedThread);
        setSidebarVersion(v => v + 1);
      } catch (error) {
        console.error('Failed to refresh thread after upload:', error);
      }
    }

    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
  };

  // Handle parsing completion - update tab with fetched sentences
  const handleParsingComplete = async (fileHash: string, sentences: any[]) => {
    const transformedSentences = transformSentences(sentences);
    setPdfTabs(prev => prev.map(tab => {
      if (tab.fileHash === fileHash) {
        return {
          ...tab,
          sentences: transformedSentences,
          text: extractTextFromSentences(transformedSentences),
          parsingStatus: ProcessStatus.Completed,
        };
      }
      return tab;
    }));
  };

  const handleIndexingComplete = async (_fileHash: string) => {
    if (!activeThread) return;
    try {
      const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
      setActiveThread(updatedThread);
      setSidebarVersion(v => v + 1);
    } catch (error) {
      console.error('Failed to refresh thread after indexing completed:', error);
    }
  };

  // Poll for parsing status when active tab is pending
  useEffect(() => {
    if (!activeTab || activeTab.parsingStatus !== ProcessStatus.Pending || !activeThread) {
      return;
    }

    let pollInterval: NodeJS.Timeout | null = null;

    const pollSentences = async () => {
      try {
        // Single endpoint returns both status and sentences
        const parsedData = await getParsedSentences(activeTab.fileHash, activeThread.id);
        if (parsedData?.sentences !== null && Array.isArray(parsedData.sentences) && parsedData.sentences.length > 0) {
          // Parsing complete - sentences is an array
          handleParsingComplete(activeTab.fileHash, parsedData.sentences);
          if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
          }
        }
        // If sentences is null, undefined, not an array, or empty, parsing is still pending - continue polling
      } catch (error: any) {
        // Log error but don't crash - file may not be ready yet
        if (error?.message?.includes('not attached')) {
          console.log(`File ${activeTab.fileHash} not yet attached to thread, will retry...`);
        } else {
          console.error("Failed to fetch parsed sentences:", error);
        }
        // Continue polling - don't throw
      }
    };

    // Run immediately
    pollSentences();

    // Then set up interval
    pollInterval = setInterval(pollSentences, 5000);

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [activeTab?.fileHash, activeTab?.parsingStatus, activeThread?.id]);

  // Handle remove source from thread (deletes from DB + Weaviate, closes tab)
  const handleTabRemove = async (tabId: string) => {
    if (!activeThread) return;
    const tab = pdfTabs.find(t => t.id === tabId);
    if (!tab) return;

    try {
      await removeSourceFromThread(activeThread.id, tab.fileHash);
    } catch (error) {
      console.error('Failed to remove source from thread:', error);
    }

    // Close the tab and refresh sidebar
    handleTabClose(tabId);
    try {
      const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
      setActiveThread(updatedThread);
      setSidebarVersion(v => v + 1);
    } catch (error) {
      console.error('Failed to refresh thread after source removal:', error);
    }
  };

  // Handle tab change
  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId);
    setIsBrowserActive(tabId === 'browser-tab');
  };

  // Handle tab close
  const handleTabClose = (tabId: string) => {
    handleTabCloseUtil(
      tabId,
      pdfTabs,
      activeTabId,
      setPdfTabs,
      setActiveTabId,
      setCurrentPdfId,
      setPlayRequestId
    );
  };

  // Handle adding browser page to thread
  const handleAddBrowserToThread = async () => {
    if (!activeThread || isBrowserCapturing) return;

    setIsBrowserCapturing(true);
    try {
      const result = await captureBrowserPage(activeThread.id);

      // Pre-verify file is accessible before creating tab
      const isReady = await pollForFileReady(activeThread.id, result.fileHash, {
        maxAttempts: 10,
        intervalMs: 500,
        timeoutMs: 5000,
      });

      if (!isReady) {
        console.error("Browser capture: File not ready after polling");
        alert("Failed to load captured page. The file may still be processing. Please try again in a moment.");
        return;
      }

      // Backend returns combined "title - url", extract just the title for display
      const displayTitle = result.title.includes(' - ')
        ? result.title.split(' - ')[0]
        : result.title;

      // Transform to match PDF upload format and reuse handler for consistent behavior
      const uploadData = {
        fileHash: result.fileHash,
        fileName: displayTitle,
        downloadUrl: `/threads/${activeThread.id}/files/${result.fileHash}/download`,
        sentences: null,
      };

      await handlePdfUploaded(uploadData);
      setIsBrowserActive(false);

    } catch (err: any) {
      console.error("Failed to capture browser page:", err);
      alert(`Failed to capture page: ${err.message}`);
    } finally {
      setIsBrowserCapturing(false);
    }
  };

  useEffect(() => {
    if (!activeThread) {
      setRightPanelLineageThreads([]);
      return;
    }

    let cancelled = false;
    listThreads()
      .then(response => {
        if (!cancelled) {
          setRightPanelLineageThreads(response.threads);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setRightPanelLineageThreads([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeThread?.id, sidebarVersion]);

  const rightPanelLineageThreadsById = useMemo(
    () => new Map(rightPanelLineageThreads.map(thread => [thread.id, thread])),
    [rightPanelLineageThreads]
  );
  const activeThreadForTooltip = activeThread
    ? rightPanelLineageThreadsById.get(activeThread.id) || activeThread
    : null;

  const renderAllThreadsHeaderActions = () => {
    if (!threadListHeaderState) return null;

    if (threadListHeaderState.isSelectionMode) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Tooltip title={threadListHeaderState.allThreadsSelected ? "Clear selection" : "Select all threads. Shift-click a thread to select a range."}>
            <Checkbox
              size="small"
              checked={threadListHeaderState.allThreadsSelected}
              indeterminate={threadListHeaderState.someThreadsSelected}
              onChange={(event) => threadListHeaderState.toggleAllThreads(event.target.checked)}
              disabled={threadListHeaderState.isBulkDeleting}
              sx={{ p: 0.5 }}
            />
          </Tooltip>
          <Tooltip title="Shift-click threads to select a range">
            <Chip label={`${threadListHeaderState.selectedCount} selected`} size="small" color="primary" />
          </Tooltip>
          <Tooltip title="Clear selection">
            <IconButton size="small" onClick={threadListHeaderState.clearSelection} disabled={threadListHeaderState.isBulkDeleting}>
              <ClearIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete selected threads">
            <span>
              <IconButton
                size="small"
                color="error"
                onClick={threadListHeaderState.deleteSelectedThreads}
                disabled={threadListHeaderState.isBulkDeleting || threadListHeaderState.selectedCount === 0}
              >
                {threadListHeaderState.isBulkDeleting ? <CircularProgress size={16} /> : <DeleteIcon fontSize="small" />}
              </IconButton>
            </span>
          </Tooltip>
        </Box>
      );
    }

    return (
      <Tooltip title="Select threads to delete. Shift-click a thread to select a range.">
        <span>
          <IconButton
            size="small"
            color="error"
            onClick={threadListHeaderState.enterSelectionMode}
            disabled={!threadListHeaderState.hasThreads}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
    );
  };

  const renderRightPanelHeader = () => {
    const isThreadMode = rightPanelTab === 1 && activeThread;

    return (
      <Box
        sx={{
          minHeight: 49,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
          display: 'flex',
          alignItems: 'stretch',
          flexShrink: 0,
        }}
      >
        {isThreadMode ? (
          <>
            <Box
              sx={{
                width: 132,
                flex: '0 0 132px',
                px: 1,
                display: 'flex',
                alignItems: 'center',
                minWidth: 0,
              }}
            >
              <Button
                size="small"
                startIcon={<ForumIcon fontSize="small" />}
                onClick={handleShowAllThreads}
                sx={{ textTransform: 'none', minWidth: 0 }}
              >
                All threads
              </Button>
            </Box>
            <Box
              sx={{
                flex: 1,
                minWidth: 0,
                px: 1.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 1,
              }}
            >
              <Tooltip
                title={
                  <ThreadLineageTooltipContent
                    thread={activeThreadForTooltip || activeThread}
                    threadsById={rightPanelLineageThreadsById}
                    onOpenThread={handleOpenThreadInChat}
                  />
                }
                arrow
                enterDelay={300}
                leaveDelay={150}
                disableInteractive={false}
              >
                <Box
                  sx={{
                    flex: 1,
                    minWidth: 0,
                    alignSelf: 'stretch',
                    display: 'flex',
                    alignItems: 'center',
                    cursor: 'default',
                  }}
                >
                  <Typography
                    variant="subtitle2"
                    fontWeight={700}
                    noWrap
                    sx={{ minWidth: 0 }}
                  >
                    {activeThread.name}
                  </Typography>
                </Box>
              </Tooltip>
              <Tooltip title="Delete current thread">
                <span>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={handleDeleteActiveThread}
                    disabled={isDeletingActiveThread}
                  >
                    {isDeletingActiveThread ? <CircularProgress size={16} /> : <DeleteIcon fontSize="small" />}
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
          </>
        ) : (
          <>
            <Box
              sx={{
                flex: 1,
                minWidth: 0,
                px: 1.5,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <ForumIcon fontSize="small" color="primary" />
              <Typography variant="subtitle1" fontWeight={700} noWrap>
                Threads
              </Typography>
              <Chip label={threadListHeaderState?.threadCount ?? 0} size="small" />
              <Tooltip title="Create new thread">
                <span>
                  <IconButton
                    size="small"
                    color="primary"
                    onClick={threadListHeaderState?.openCreateDialog}
                    disabled={!threadListHeaderState}
                  >
                    <AddIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </Box>
            <Box
              sx={{
                flex: '0 0 auto',
                px: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
              }}
            >
              {renderAllThreadsHeaderActions()}
            </Box>
          </>
        )}
      </Box>
    );
  };

  const workspaceTabs = useMemo(() => buildDocumentWorkspaceTabs({
    enabled: Boolean(activeThread),
    documents: pdfTabs,
    traces: traceTabs,
  }), [activeThread, pdfTabs, traceTabs]);

  const handleWorkspaceTabChange = useCallback((tabId: string) => {
    setActiveTabId(tabId);
    setIsBrowserActive(tabId === 'browser-tab');
  }, []);

  const handleOpenTrace = useCallback((trace: ChatTraceDescriptor) => {
    openTrace(trace);
    setActiveTabId('trace-tab');
    setIsBrowserActive(false);
  }, [openTrace]);

  // Memoize theme to prevent recreation on every render
  const theme = useMemo(() => getTheme(pdfDarkMode), [pdfDarkMode]);

  // Don't render until pdfDarkMode is determined (prevents hydration mismatch)
  if (pdfDarkMode === undefined) return null;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', overflow: 'hidden', bgcolor: 'background.default' }}>
        <WorkbenchShell
          layout={workbenchLayout}
          onLayoutChange={setWorkbenchLayout}
          onResolvedPlacementChange={setResolvedPlacement}
          onResizingChange={setIsResizing}
          secondaryLabel="Threads and chat"
          primaryToolbar={
            <Box sx={{ px: 1.5, py: 0.75, minHeight: 49, borderBottom: 1, borderColor: 'divider', bgcolor: pdfDarkMode ? '#222' : 'background.paper', color: pdfDarkMode ? '#eee' : 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', minWidth: 0, flex: '1 1 auto' }}>
                <PdfUploader
                  threadId={activeThread?.id ?? null}
                  onUploaded={handlePdfUploaded}
                  onIndexingComplete={handleIndexingComplete}
                  onParsingComplete={handleParsingComplete}
                  disabled={!activeThread}
                  tooltipText={!activeThread ? 'Select or create a thread first' : undefined}
                />
                <Tooltip title="Agent workflow builder">
                  <IconButton color="primary" size="small" onClick={() => window.open('/agent-workflow-builder', '_blank', 'noopener,noreferrer')}>
                    <AutoAwesomeSharpIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                {activeThread && rightPanelTab === 1 && (
                  <PlayerControls
                    sentences={activeSource === 'pdf' ? pdfSentences : chatSentences}
                    currentId={activeSource === 'pdf' ? currentPdfId : currentChatId}
                    onCurrentChange={(id) => {
                      if (activeSource === 'pdf') setCurrentPdfId(id);
                      else setCurrentChatId(id);
                      setPlayRequestId(null);
                    }}
                    playRequestId={playRequestId}
                    autoScroll={autoScroll}
                    onAutoScrollChange={setAutoScroll}
                    highlightEnabled={highlightEnabled}
                    onHighlightEnabledChange={setHighlightEnabled}
                  />
                )}
              </Box>
              <WorkbenchToolbarTrailingActions>
                <Tooltip title={pdfDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
                  <IconButton color={pdfDarkMode ? 'primary' : 'default'} onClick={() => setPdfDarkMode((value) => !value)} size="small">
                    {pdfDarkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                  </IconButton>
                </Tooltip>
                <DockMenuButton value={workbenchLayout} resolvedPlacement={resolvedPlacement} onChange={setWorkbenchLayout} label="Threads and chat layout" />
              </WorkbenchToolbarTrailingActions>
            </Box>
          }
          primaryTabs={
            activeThread ? (
              <WorkspaceTabs
                tabs={workspaceTabs}
                activeTabId={activeTabId}
                onTabChange={handleWorkspaceTabChange}
                onTabClose={handleTabClose}
                onDocumentRemove={handleTabRemove}
                onAddBrowserToThread={handleAddBrowserToThread}
                isBrowserCapturing={isBrowserCapturing}
              />
            ) : null
          }
          primaryContent={
            <DocumentWorkspaceContent
              activeTabId={activeTabId}
              activeDocument={activeTab}
              documentSentences={pdfSentences}
              documentDownloadUrl={downloadUrl}
              traceTabs={traceTabs}
              activeTraceId={activeTraceId}
              onActiveTraceChange={setActiveTraceId}
              onCloseTrace={closeTrace}
              isBrowserActive={isBrowserActive}
              isLoading={isPdfLoading}
              isResizing={isResizing}
              darkMode={pdfDarkMode}
              currentDocumentSentenceId={activeSource === 'pdf' ? currentPdfId : null}
              onDocumentJump={(id) => { setActiveSource('pdf'); setCurrentPdfId(id); setPlayRequestId(id); }}
              autoScroll={autoScroll}
              highlightEnabled={highlightEnabled}
              threadId={activeThread?.id ?? null}
              emptyTitle="Welcome to AskPDF"
              emptyDescription="Select or create a thread, then upload a PDF or open the browser."
            />
          }
          secondaryHeader={renderRightPanelHeader()}
          secondaryContent={
            <Box sx={{ height: '100%', overflow: 'hidden' }}>
              <Box sx={{ height: '100%', display: rightPanelTab === 0 ? 'block' : 'none', overflow: 'auto' }}>
                <ThreadSidebar
                  key={sidebarVersion}
                  activeThreadId={activeThread?.id || null}
                  onThreadSelect={handleThreadSelectFromList}
                  onThreadForked={handleThreadForked}
                  hideHeader
                  onHeaderStateChange={setThreadListHeaderState}
                  darkMode={pdfDarkMode}
                />
              </Box>
              <Box sx={{ height: '100%', display: rightPanelTab === 1 ? 'flex' : 'none', flexDirection: 'column' }}>
                {activeThread ? (
                  <ChatInterface
                    activeThread={activeThread}
                    chatSentences={chatSentences}
                    setChatSentences={setChatSentences}
                    currentChatId={currentChatId}
                    activeSource={activeSource}
                    onJump={(id) => { setActiveSource('chat'); setCurrentChatId(id); setPlayRequestId(id); }}
                    onResetChatId={() => { setCurrentChatId(null); setPlayRequestId(null); }}
                    onThreadForked={handleThreadForked}
                    onThreadUpdate={handleThreadUpdated}
                    onOpenThread={handleOpenThreadInChat}
                    onOpenTrace={handleOpenTrace}
                    hideInlineLineage
                    darkMode={pdfDarkMode}
                    autoScroll={autoScroll}
                    isPanelResizing={isResizing}
                  />
                ) : (
                  <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 3 }}>
                    <Typography color="text.secondary" textAlign="center">Select or create a thread to start chatting</Typography>
                  </Box>
                )}
              </Box>
            </Box>
          }
        />
      </Box>
    </ThemeProvider>
  );
}
