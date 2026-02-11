import VerticalAlignCenterIcon from '@mui/icons-material/VerticalAlignCenter';
import DarkModeIcon from '@mui/icons-material/DarkMode';
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
import PdfViewer from "../components/PdfViewer";
import PlayerControls from "../components/PlayerControls";
import ChatInterface from "../components/ChatInterface";
import ThreadSidebar from "../components/ThreadSidebar";
import PdfTabs, { PdfTab } from "../components/PdfTabs";
import { Thread, addFileToThread } from "../lib/api";
import { loadThreadTabs, createPdfTabFromUpload, extractTextFromSentences } from "../lib/thread-utils";
import { handleTabChangeUtil, handleTabCloseUtil, getActiveTab, getActiveTabData } from "../lib/pdf-utils";

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

  // Highlight toggle
  const [highlightEnabled, setHighlightEnabled] = useState(true);
  // PDF dark mode toggle, autodetect from browser on first load and listen for changes
  const [pdfDarkMode, setPdfDarkMode] = useState(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

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


  // Handle thread selection
  const handleThreadSelect = async (thread: Thread | null) => {
    // Clear current state
    setPdfTabs([]);
    setActiveTabId(null);
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    
    if (thread) {
      try {
        setIsPdfLoading(true);
        // Always fetch the latest thread data to ensure we have current files and stats
        const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const detailedThread = await import("../lib/api").then(m => m.getThread(thread.id));
        setActiveThread(detailedThread);
        
        const loadedTabs = await loadThreadTabs(detailedThread, apiBase);
        if (loadedTabs.length > 0) {
          setPdfTabs(loadedTabs);
          setActiveTabId(loadedTabs[0].id);
        }
      } catch (err) {
        console.error('Failed to load thread files:', err);
      } finally {
        setIsPdfLoading(false);
      }
    } else {
      setActiveThread(null);
    }
  };


  // Handle PDF upload - create new tab
  const handlePdfUploaded = async (data: any) => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const newTab = createPdfTabFromUpload(data, apiBase);

    setPdfTabs(prev => [...prev, newTab]);
    setActiveTabId(newTab.id);

    // If we have an active thread, add the file to it
    if (activeThread && data?.fileHash && data?.fileName) {
      try {
        await addFileToThread(
          activeThread.id,
          data.fileHash,
          data.fileName,
          newTab.text
        );

        // Refresh active thread to trigger UI updates (like indexing status in ChatInterface)
        // Refresh full thread data to ensure we have updated file_count
        const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
        setActiveThread(updatedThread);
        // Trigger sidebar refresh to update file counts in the list
        setSidebarVersion(v => v + 1);
      } catch (error) {
        console.error('Failed to add file to thread:', error);
      }
    }

    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
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
              <Button
                variant="outlined"
                size="small"
                onClick={() => setIsRightPanelOpen(open => !open)}
              >
                {isRightPanelOpen ? "Hide Threads" : "Show Threads"}
              </Button>

              {/* Player Controls */}
              {pdfSentences.length > 0 && pdfUrl && (
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
              darkMode={pdfDarkMode}
            />
          )}

          {/* PDF Viewer Area */}
          <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
            {isPdfLoading ? (
              <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: pdfDarkMode ? '#222' : 'grey.50', color: pdfDarkMode ? '#eee' : 'inherit' }}>
                <CircularProgress color={pdfDarkMode ? 'inherit' : 'primary'} />
                <Typography sx={{ ml: 2 }}>Loading PDF documents...</Typography>
              </Box>
            ) : (pdfSentences?.length ?? 0) > 0 && pdfUrl ? (
              <PdfViewer
                pdfUrl={pdfUrl}
                sentences={pdfSentences}
                currentId={activeSource === 'pdf' ? currentPdfId : null}
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
                    <li>Click <b>Upload PDF</b> to add documents to your thread.</li>
                    <li>Switch to the <b>Chat</b> tab to ask questions about your PDFs using AI.</li>
                    <li>The AI remembers your conversations - relevant past Q&A pairs are recalled automatically.</li>
                    <li>Double-click any text to start audio playback from that point.</li>
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
                onChange={(_, newValue) => setRightPanelTab(newValue)}
                variant="fullWidth"
              >
                <Tab 
                  icon={<ForumIcon fontSize="small" />} 
                  iconPosition="start" 
                  label="Threads" 
                  sx={{ minHeight: 56, textTransform: 'none', flex: 2 }}
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
                    darkMode={pdfDarkMode}
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
