import VerticalAlignCenterIcon from '@mui/icons-material/VerticalAlignCenter';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import SpeakerNotesIcon from '@mui/icons-material/SpeakerNotes';
import SpeakerNotesOffIcon from '@mui/icons-material/SpeakerNotesOff';
import React, { useState, useEffect, useRef, useCallback } from "react";
import { Container, Stack, Typography, Box, Button, FormControl, InputLabel, Select, MenuItem, CssBaseline, IconButton, Tooltip, Tabs, Tab, CircularProgress } from "@mui/material";
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import FluorescentIcon from '@mui/icons-material/Fluorescent';
import ChatIcon from '@mui/icons-material/Chat';
import ForumIcon from '@mui/icons-material/Forum';

declare const process: {
  env: Record<string, string | undefined>;
};
import PdfUploader from "../components/PdfUploader";
import WebUploader from "../components/WebUploader";
import PdfViewer from "../components/PdfViewer";
import WebViewer from "../components/WebViewer";
import PlayerControls from "../components/PlayerControls";
import ChatInterface from "../components/ChatInterface";
import ThreadSidebar from "../components/ThreadSidebar";
import PdfTabs, { PdfTab } from "../components/PdfTabs";
import { Thread, DocumentSource, addFileToThread, removeSourceFromThread } from "../lib/api";
import { 
  loadThreadTabs, 
  handleThreadSelectUtil, 
  handlePdfUploadedUtil, 
  handleWebIndexedUtil, 
  handleTabRemoveUtil 
} from "../lib/thread-utils";
import { handleTabChangeUtil, handleTabCloseUtil, getActiveTab, getActiveTabData, handleHighlightSourcesUtil, clearSourceHighlightsUtil } from "../lib/pdf-utils";

type Sentence = { id: number; text: string; bboxes: any[] };

export default function Home() {
  // Multiple PDF tabs state
  const [pdfTabs, setPdfTabs] = useState<PdfTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>(null);
  const [isPdfLoading, setIsPdfLoading] = useState(false);

  // Get active tab and its data using utility
  const activeTab = getActiveTab(pdfTabs, activeTabId);
  const { pdfSentences, pdfUrl, fileHash, fileName } = getActiveTabData(activeTab);

  const [activeSource, setActiveSource] = useState<'pdf' | 'chat'>('pdf');
  const [currentPdfId, setCurrentPdfId] = useState<number | null>(null);
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [chatSentences, setChatSentences] = useState<any[]>([]);
  const [highlightedSourceSentenceIds, setHighlightedSourceSentenceIds] = useState<number[]>([]);
  const [highlightedSourceFileHash, setHighlightedSourceFileHash] = useState<string | null>(null);
  const [highlightedSourceMessageId, setHighlightedSourceMessageId] = useState<string | null>(null);
  const [highlightedSourceMode, setHighlightedSourceMode] = useState<'off' | 'focused' | 'all'>('off');

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


  // Resizable chat panel
  const [chatWidth, setChatWidth] = useState(450);
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);
  const [isResizing, setIsResizing] = useState(false);
  const chatWidthRef = useRef(450);
  const rafIdRef = useRef<number | null>(null);

  // Thread sidebar width
  const rightPanelMinWidth = 350;


  const handleThreadSelect = async (thread: Thread | null) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    await handleThreadSelectUtil(
      thread,
      apiBase,
      setPdfTabs,
      setActiveTabId,
      setCurrentPdfId,
      setCurrentChatId,
      setPlayRequestId,
      setActiveSource,
      setChatSentences,
      setHighlightedSourceSentenceIds,
      setHighlightedSourceFileHash,
      setHighlightedSourceMessageId,
      setActiveThread,
      setIsPdfLoading
    );
  };


  const handlePdfUploaded = async (data: any) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    await handlePdfUploadedUtil(
      data,
      apiBase,
      activeThread,
      setPdfTabs,
      setActiveTabId,
      setActiveThread,
      setSidebarVersion,
      setCurrentPdfId,
      setCurrentChatId,
      setPlayRequestId,
      setActiveSource
    );
  };

  const handleWebIndexed = async (data: { fileHash: string; url: string; status: string; message?: string }) => {
    await handleWebIndexedUtil(
      data,
      activeThread,
      setPdfTabs,
      setActiveTabId,
      setActiveThread,
      setSidebarVersion,
      setCurrentPdfId,
      setCurrentChatId,
      setPlayRequestId,
      setActiveSource
    );
  };

  const handleTabRemove = async (tabId: string) => {
    await handleTabRemoveUtil(
      tabId,
      pdfTabs,
      activeThread,
      handleTabClose,
      setActiveThread,
      setSidebarVersion
    );
  };

  // Handle tab change
  const handleTabChange = (tabId: string) => {
    handleTabChangeUtil(tabId, setActiveTabId, setCurrentPdfId, setPlayRequestId, setActiveSource);
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




  const clearSourceHighlights = () => {
    clearSourceHighlightsUtil(setHighlightedSourceSentenceIds, setHighlightedSourceFileHash, setHighlightedSourceMessageId);
    setHighlightedSourceMode('off');
  };

  const handleHighlightSources = (payload: { messageId: string; documentSources: DocumentSource[]; matchedSentenceIds?: number[] }, threshold?: number) => {
    let nextMode: 'all' | 'focused' | 'off' = 'focused';

    if (highlightedSourceMessageId === payload.messageId) {
      if (highlightedSourceMode === 'focused') {
        nextMode = 'all';
      } else if (highlightedSourceMode === 'all') {
        nextMode = 'off';
      }
    }

    if (nextMode === 'off') {
      clearSourceHighlights();
    } else {
      handleHighlightSourcesUtil(
        payload,
        highlightedSourceMessageId,
        pdfTabs,
        activeTab,
        activeTabId,
        setPdfTabs,
        setActiveTabId,
        setHighlightedSourceSentenceIds,
        setHighlightedSourceFileHash,
        setHighlightedSourceMessageId,
        setActiveSource,
        setCurrentPdfId,
        nextMode,
        threshold ?? 0.0
      );
      setHighlightedSourceMode(nextMode);
    }
  };

  // Handle resize with optimized performance
  const handleMouseDown = useCallback(() => {
    setIsResizing(true);
    chatWidthRef.current = chatWidth;
  }, [chatWidth]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }

    rafIdRef.current = requestAnimationFrame(() => {
      const newWidth = window.innerWidth - e.clientX;
      const minWidth = window.innerWidth * 0.2;
      const maxWidth = window.innerWidth * 0.8;
      const constrainedWidth = Math.max(minWidth, Math.min(maxWidth, newWidth));
      chatWidthRef.current = constrainedWidth;
      document.documentElement.style.setProperty('--chat-width', `${constrainedWidth}px`);
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsResizing(false);
    setChatWidth(chatWidthRef.current);
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }
  }, []);

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [isResizing, handleMouseMove, handleMouseUp]);

  const canDisplayRightPanel = true; // Always show right panel for threads
  const rightPanelWidth = isRightPanelOpen
    ? (isResizing ? 'var(--chat-width, 450px)' : chatWidth)
    : 0;

  useEffect(() => {
    if (!highlightedSourceFileHash) return;
    if (!activeTab?.fileHash) {
      clearSourceHighlights();
      return;
    }
    if (highlightedSourceFileHash !== activeTab.fileHash) {
      clearSourceHighlights();
    }
  }, [activeTab?.fileHash, highlightedSourceFileHash]);


  // Don't render until pdfDarkMode is determined (prevents hydration mismatch)
  if (pdfDarkMode === undefined) return null;

  return (
    <ThemeProvider theme={getTheme(pdfDarkMode)}>
      <CssBaseline />
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "row", overflow: "hidden", bgcolor: 'background.default' }}>

        {/* Left Column: PDF Content & Controls */}
        <Box sx={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, borderRight: 1, borderColor: 'divider' }}>
          {/* Top Controls Bar */}
          <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider', bgcolor: pdfDarkMode ? '#222' : 'background.paper', color: pdfDarkMode ? '#eee' : 'inherit' }}>
            <Stack direction="row" spacing={2} alignItems="center" justifyContent="center" flexWrap="wrap" useFlexGap>

              {/* PDF Uploader */}
              <PdfUploader
                onUploaded={handlePdfUploaded}
                disabled={!activeThread}
                tooltipText={!activeThread ? "Select or create a thread first" : undefined}
              />

              {/* Web Uploader */}
              <WebUploader
                threadId={activeThread?.id ?? null}
                onIndexed={handleWebIndexed}
                disabled={!activeThread}
                tooltipText={!activeThread ? "Select or create a thread first" : undefined}
              />

              {/* Auto-scroll Toggle */}
              <Tooltip title={autoScroll ? "Disable Auto-Scroll" : "Enable Auto-Scroll"}>
                <IconButton
                  color={autoScroll ? "primary" : "default"}
                  onClick={() => setAutoScroll(a => !a)}
                  sx={{ border: autoScroll ? 1 : 0, borderColor: autoScroll ? 'primary.main' : 'transparent' }}
                  size="small"
                >
                  <VerticalAlignCenterIcon />
                </IconButton>
              </Tooltip>

              {/* Highlight Toggle */}
              <Tooltip title={highlightEnabled ? "Disable Highlighting" : "Enable Highlighting"}>
                <IconButton
                  color={highlightEnabled ? "primary" : "default"}
                  onClick={() => setHighlightEnabled(h => !h)}
                  sx={{ border: highlightEnabled ? 1 : 0, borderColor: highlightEnabled ? 'primary.main' : 'transparent' }}
                  size="small"
                >
                  <FluorescentIcon />
                </IconButton>
              </Tooltip>

              {/* PDF Dark Mode Toggle */}
              <Tooltip title={pdfDarkMode ? "Disable PDF Dark Mode" : "Enable PDF Dark Mode"}>
                <IconButton
                  color={pdfDarkMode ? "primary" : "default"}
                  onClick={() => setPdfDarkMode(d => !d)}
                  sx={{ border: pdfDarkMode ? 1 : 0, borderColor: pdfDarkMode ? 'primary.main' : 'transparent' }}
                  size="small"
                >
                  <DarkModeIcon />
                </IconButton>
              </Tooltip>

              {/* Right Panel Toggle */}
              <Tooltip title={isRightPanelOpen ? "Hide Threads" : "Show Threads"}>
                <IconButton
                  color="primary"
                  size="small"
                  onClick={() => setIsRightPanelOpen(open => !open)}
                  sx={{ border: 1, borderColor: 'primary.main' }}
                >
                  {isRightPanelOpen ? <SpeakerNotesOffIcon fontSize="small" /> : <SpeakerNotesIcon fontSize="small" />}
                </IconButton>
              </Tooltip>

              {/* Player Controls */}
              {(((pdfSentences.length > 0 && pdfUrl) || chatSentences.length > 0) && activeThread && rightPanelTab === 1) && (
                <PlayerControls
                  sentences={activeSource === 'pdf' ? pdfSentences : chatSentences}
                  currentId={activeSource === 'pdf' ? currentPdfId : currentChatId}
                  onCurrentChange={(id) => {
                    if (activeSource === 'pdf') {
                      setCurrentPdfId(id);
                    } else {
                      setCurrentChatId(id);
                    }
                    setPlayRequestId(null);
                  }}
                  playRequestId={playRequestId}
                />
              )}
            </Stack>
          </Box>

          {/* PDF Tabs */}
          {pdfTabs.length > 0 && (
            <PdfTabs
              tabs={pdfTabs}
              activeTabId={activeTabId}
              onTabChange={handleTabChange}
              onTabClose={handleTabClose}
              onTabRemove={activeThread ? handleTabRemove : undefined}
              darkMode={pdfDarkMode}
            />
          )}

          {/* PDF / Web Viewer Area */}
          <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            {isPdfLoading ? (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: pdfDarkMode ? '#222' : 'grey.50', color: pdfDarkMode ? '#eee' : 'inherit' }}>
                <CircularProgress color={pdfDarkMode ? 'inherit' : 'primary'} />
                <Typography sx={{ ml: 2 }}>Loading documents...</Typography>
              </Box>
            ) : activeTab?.sourceType === 'web' && activeTab.sourceUrl && activeTab.fileHash ? (
              /* Web source view — rendered as a self-contained saved HTML page */
              <WebViewer
                url={activeTab.sourceUrl}
                fileHash={activeTab.fileHash}
                threadId={activeThread?.id}
                darkMode={pdfDarkMode}
                isResizing={isResizing}
              />
            ) : (pdfSentences?.length ?? 0) > 0 && pdfUrl ? (
              <PdfViewer
                pdfUrl={pdfUrl}
                sentences={pdfSentences}
                currentId={activeSource === 'pdf' ? currentPdfId : null}
                highlightIds={highlightedSourceFileHash === fileHash ? highlightedSourceSentenceIds : []}
                onJump={(id) => {
                  setActiveSource('pdf');
                  setCurrentPdfId(id);
                  setPlayRequestId(id);
                }}
                autoScroll={autoScroll}
                isResizing={isResizing}
                highlightEnabled={highlightEnabled}
                darkMode={pdfDarkMode}
              />
            ) : (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: pdfDarkMode ? '#222' : 'grey.50', color: pdfDarkMode ? '#eee' : 'inherit', p: 4 }}>
                <Box>
                  <Typography variant="h5" sx={{ color: pdfDarkMode ? '#eee' : 'textSecondary' }} gutterBottom>
                    Welcome to AskPDF
                  </Typography>
                  <Typography sx={{ mb: 2, color: pdfDarkMode ? '#ccc' : 'textSecondary' }}>
                    To get started:
                  </Typography>
                  <ul style={{ color: pdfDarkMode ? '#bbb' : '#888', margin: 0, paddingLeft: 20, fontSize: 16 }}>
                    <li>Use the <b>Threads</b> tab on the right to create a new thread with an embedding model.</li>
                    <li>Click <b>Upload PDF</b> to add PDF documents to your thread.</li>
                    <li>Enter a URL and click <b>Add Webpage</b> to index a website.</li>
                    <li>Switch to the <b>Chat</b> tab to ask questions about your sources using AI.</li>
                    <li>The AI remembers your conversations - relevant past Q&A pairs are recalled automatically.</li>
                    <li>Double-click any text to start audio playback from that point.</li>
                  </ul>
                  <Typography sx={{ mt: 2, mb: 1, color: pdfDarkMode ? '#ccc' : 'textSecondary' }}>
                    Settings tips:
                  </Typography>
                  <ul style={{ color: pdfDarkMode ? '#bbb' : '#888', margin: 0, paddingLeft: 20, fontSize: 16 }}>
                    <li>Open the <b>Chat</b> tab and click the <b>gear icon</b> to configure AI prompt settings for the current thread.</li>
                    <li>Toggle <b>Reasoning mode</b> for deeper multi-step answers on reasoning-capable models.</li>
                    <li>Use <b>Intent Agent</b> (requires Reasoning mode) to rewrite follow-up questions into standalone queries.</li>
                    <li>Enable <b>Reranker</b> to improve ordering of retrieved chunks.</li>
                    <li>Customize <b>Tools</b> to control which capabilities the assistant can use.</li>
                    <li>Edit the <b>System role</b> to change the assistant’s behavior and tone.</li>
                    <li>Use <b>Prompt preview</b> to see the exact prompt that will be sent to the model.</li>
                  </ul>
                  <Typography sx={{ mt: 2, fontSize: 14, color: pdfDarkMode ? '#aaa' : 'textSecondary' }}>
                    <b>Note:</b> The embedding model is locked once a thread is created. Create a new thread to use a different model.
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
        </Box>

        {/* Resizable Divider */}
        {canDisplayRightPanel && isRightPanelOpen && (
          <Box
            onMouseDown={handleMouseDown}
            sx={{
              width: '12px',
              mx: '-6px',
              cursor: 'col-resize',
              position: 'relative',
              zIndex: 10,
              display: 'flex',
              justifyContent: 'center',
              '&:hover .divider-line, &:active .divider-line': {
                backgroundColor: 'primary.main',
                width: '4px',
              },
            }}
          >
            <Box className="divider-line" sx={{
              width: '2px',
              height: '100%',
              backgroundColor: isResizing ? 'primary.main' : 'divider',
              transition: 'all 0.2s',
            }} />
          </Box>
        )}

        {/* Right Column: Threads & Chat Interface */}
        {canDisplayRightPanel && (
          <Box sx={{
            width: rightPanelWidth,
            minWidth: 0,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            transition: isResizing || !isRightPanelOpen ? 'none' : 'width 0.1s ease-out',
            bgcolor: 'background.paper',
            visibility: isRightPanelOpen ? 'visible' : 'hidden',
            pointerEvents: isRightPanelOpen ? 'auto' : 'none',
            overflow: 'hidden'
          }}>
            {/* Tabs Header */}
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs
                value={rightPanelTab}
                onChange={(_, newValue) => {
                  setRightPanelTab(newValue);
                  if (newValue === 0) {
                    handleThreadSelect(null);
                  }
                }}
                variant="fullWidth"
              >
                <Tab
                  icon={<ForumIcon fontSize="small" />}
                  iconPosition="start"
                  label="Threads"
                  sx={{ minHeight: 56, textTransform: 'none', flex: 2 }}
                  onClick={() => handleThreadSelect(null)}
                />
                <Tab
                  icon={<ChatIcon fontSize="small" />}
                  iconPosition="start"
                  label={activeThread ? activeThread.name : "Chat"}
                  disabled={!activeThread}
                  sx={{ minHeight: 56, textTransform: 'none', flex: 8 }}
                />
              </Tabs>
            </Box>

            {/* Tab Content */}
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              {/* Threads Tab */}
              <Box sx={{
                height: '100%',
                display: rightPanelTab === 0 ? 'block' : 'none',
                overflow: 'auto'
              }}>
                <ThreadSidebar
                  key={sidebarVersion}
                  activeThreadId={activeThread?.id || null}
                  onThreadSelect={(thread) => {
                    handleThreadSelect(thread);
                    // Switch to Chat tab when a thread is selected
                    if (thread) {
                      setRightPanelTab(1);
                    }
                  }}
                  darkMode={pdfDarkMode}
                />
              </Box>

              {/* Chat Tab */}
              <Box sx={{
                height: '100%',
                display: rightPanelTab === 1 ? 'flex' : 'none',
                flexDirection: 'column'
              }}>
                {activeThread ? (
                  <ChatInterface
                    activeThread={activeThread}
                    chatSentences={chatSentences}
                    setChatSentences={setChatSentences}
                    currentChatId={currentChatId}
                    activeSource={activeSource}
                    onJump={(id) => {
                      setActiveSource('chat');
                      setCurrentChatId(id);
                      setPlayRequestId(id);
                    }}
                    onResetChatId={() => {
                      setCurrentChatId(null);
                      setPlayRequestId(null);
                    }}
                    onHighlightSources={handleHighlightSources}
                    activeHighlightMessageId={highlightedSourceMessageId}
                    activeHighlightMode={highlightedSourceMode}
                    darkMode={pdfDarkMode}
                    autoScroll={autoScroll}
                  />
                ) : (
                  <Box sx={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    p: 3
                  }}>
                    <Typography color="text.secondary" textAlign="center">
                      Select or create a thread to start chatting
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          </Box>
        )}

        {/* Global Drag Mask */}
        {isResizing && (
          <Box sx={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            cursor: 'col-resize',
            userSelect: 'none',
            backgroundColor: 'transparent',
          }} />
        )}
      </Box>
    </ThemeProvider>
  );
}
