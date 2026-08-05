import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { Typography, Box, CssBaseline, IconButton, Tooltip, CircularProgress } from "@mui/material";
import { ThemeProvider } from '@mui/material/styles';
import { getTheme } from '../theme';
import { useAppThemeMode } from '../hooks/useAppThemeMode';
import DeleteIcon from '@mui/icons-material/Delete';
import AutoAwesomeSharpIcon from '@mui/icons-material/AutoAwesomeSharp';
import HomeIcon from '@mui/icons-material/Home';

declare const process: {
  env: Record<string, string | undefined>;
};
import PdfUploader from "../components/PdfUploader";

import PlayerControls from "../components/PlayerControls";
import ChatInterface, { type ChatTraceDescriptor } from "../components/ChatInterface";
import ThreadSecondaryPanel from "../components/ThreadSecondaryPanel";
import MemoryManagerPanel from "../components/MemoryManagerPanel";
import { buildDocumentWorkspaceTabs, buildHomeWorkspaceTabs, buildProjectWorkspaceTabs, type PdfTab } from "../lib/document-tabs";
import WorkbenchShell, { useWorkbenchLayout } from '../components/workbench/WorkbenchShell';
import DockMenuButton from '../components/workbench/DockMenuButton';
import { WorkbenchToolbarTrailingActions } from '../components/workbench/WorkbenchToolbar';
import WorkspaceTabs from '../components/workbench/WorkspaceTabs';
import ThreadWorkspaceContent from '../components/workbench/ThreadWorkspaceContent';
import useTraceTabs from '../components/workbench/useTraceTabs';
import ThreadLineageTooltipContent from "../components/ThreadLineageTooltipContent";
import { Project, Thread, removeSourceFromThread, removeSourceFromProject, promoteFileToProject, retryTargetFile, getParsedSentencesForTarget, captureBrowserPageForTarget, pollForTargetFileReady, getThread, getProject, deleteThread, listThreads, type KnowledgeTarget } from "../lib/api";
import { loadThreadTabs, loadProjectTabs, hydrateThreadPdfTab, createPdfTabFromUpload, extractTextFromSentences } from "../lib/thread-utils";
import { handleTabCloseUtil, getActiveTab, getActiveTabData } from "../lib/pdf-utils";
import { transformSentences } from "../lib/bbox-derivation";
import { ProcessStatus, ThreadFileSourceType } from "../lib/enums";
import type { ResolvedWorkbenchPlacement } from '../lib/workbench-layout';
import { checkEmbeddingModelReady } from '../lib/models-api';
import { flexTruncateSx, singleLineTruncateSx } from '../lib/truncation';
import { defaultMemoryManagerIntent, reviewManagerIntent, type MemoryManagerIntent } from '../lib/memory-manager';

export default function Home() {
  // Multiple PDF tabs state
  const [pdfTabs, setPdfTabs] = useState<PdfTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>('home-tab');
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
  const { darkMode: pdfDarkMode, toggleDarkMode, hydrated: themeHydrated } = useAppThemeMode();

  // Thread state
  const [activeThread, setActiveThread] = useState<Thread | null>(null);
  const [activeProject, setActiveProject] = useState<Project | null>(null);
  const [threadProject, setThreadProject] = useState<Project | null>(null);
  const [projectModelReady, setProjectModelReady] = useState<boolean | null>(null);

  // Sidebar refresh trigger
  const [sidebarVersion, setSidebarVersion] = useState(0);
  const [isDeletingActiveThread, setIsDeletingActiveThread] = useState(false);
  const [rightPanelLineageThreads, setRightPanelLineageThreads] = useState<Thread[]>([]);
  const [memoryManagerIntent, setMemoryManagerIntent] = useState<MemoryManagerIntent | null>(null);
  const [memoryManagerDirty, setMemoryCuratorDirty] = useState(false);
  const memoryManagerDirtyRef = useRef(false);
  memoryManagerDirtyRef.current = memoryManagerDirty;
  const [memoryRefreshVersion, setMemoryRefreshVersion] = useState(0);
  const [lastNonMemoryTabByContext, setLastNonMemoryTabByContext] = useState<Record<string, string>>({});

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

  const confirmDiscardMemoryCurator = useCallback(() => (
    !memoryManagerDirtyRef.current
    || window.confirm('Discard the unconfirmed memory proposal?')
  ), []);

  const workspaceContextKey = activeThread
    ? `thread:${activeThread.id}`
    : activeProject
      ? `project:${activeProject.id}`
      : 'home';

  const rememberNonMemoryTab = useCallback((tabId: string | null, contextKey = workspaceContextKey) => {
    if (!tabId || tabId === 'memory-tab') return;
    setLastNonMemoryTabByContext((current) => ({ ...current, [contextKey]: tabId }));
  }, [workspaceContextKey]);

  const fallbackNonMemoryTab = useCallback(() => {
    const firstDocument = pdfTabs[0]?.id;
    if (firstDocument) return firstDocument;
    if (activeThread || activeProject) return 'browser-tab';
    return 'home-tab';
  }, [activeProject, activeThread, pdfTabs]);

  // Handle thread selection
  const handleThreadSelect = useCallback(async (thread: Thread | null) => {
    if (memoryManagerIntent && !confirmDiscardMemoryCurator()) return;
    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    // Clear current state
    setPdfTabs([]);
    setActiveTabId(thread ? 'browser-tab' : 'home-tab');
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    setChatSentences([]);
    setActiveProject(null);
    setThreadProject(null);
    clearTraces();
    
    // Reset browser state when leaving thread context
    setIsBrowserActive(false);

    if (thread) {
      setActiveProject(null);
      try {
        setIsPdfLoading(true);
        // Always fetch the latest thread data to ensure we have current files and stats
        const detailedThread = await import("../lib/api").then(m => m.getThread(thread.id));
        setActiveThread(detailedThread);

        const [loadedTabs, parentProject] = await Promise.all([
          loadThreadTabs(detailedThread),
          detailedThread.project_id
            ? getProject(detailedThread.project_id).catch(() => null)
            : Promise.resolve(null),
        ]);
        setThreadProject(parentProject);
        if (loadedTabs.length > 0) {
          setPdfTabs(loadedTabs);
          setActiveTabId(loadedTabs[0].id);
          window.setTimeout(() => {
            detailedThread.files.slice(1).forEach(async (threadFile) => {
              try {
                const hydrated = await hydrateThreadPdfTab(detailedThread.id, threadFile);
                setPdfTabs(prev => prev.map(tab => tab.fileHash === hydrated.fileHash ? hydrated : tab));
              } catch (error) {
                console.warn(`Failed to hydrate background PDF tab ${threadFile.fileHash}:`, error);
              }
            });
          }, 0);
        } else {
          setActiveTabId('browser-tab');
        }
      } catch (err) {
        console.error('Failed to load thread files:', err);
      } finally {
        setIsPdfLoading(false);
      }
    } else {
      setActiveThread(null);
      setActiveTabId('home-tab');
    }
  }, [clearTraces, confirmDiscardMemoryCurator, memoryManagerIntent]);

  const handleProjectSelect = useCallback(async (project: Project) => {
    if (memoryManagerIntent && !confirmDiscardMemoryCurator()) return;
    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    setActiveThread(null);
    setThreadProject(null);
    setActiveProject(project);
    setPdfTabs([]);
    clearTraces();
    setIsBrowserActive(false);
    setActiveTabId('browser-tab');
    setIsPdfLoading(true);
    setProjectModelReady(null);
    try {
      const tabs = await loadProjectTabs(project);
      setPdfTabs(tabs);
      setActiveTabId(tabs[0]?.id || 'browser-tab');
      setIsBrowserActive(false);
    } catch (error) {
      console.error('Failed to open project knowledge:', error);
      setProjectModelReady(false);
    } finally {
      setIsPdfLoading(false);
    }
  }, [clearTraces, confirmDiscardMemoryCurator, memoryManagerIntent]);

  const handleThreadForked = useCallback(async (thread: Thread) => {
    setSidebarVersion(v => v + 1);
    await handleThreadSelect(thread);
  }, [handleThreadSelect]);

  const handleProjectCloned = useCallback(async (project: Project) => {
    setSidebarVersion(v => v + 1);
    await handleProjectSelect(project);
  }, [handleProjectSelect]);

  const handleProjectUpdated = useCallback((project: Project) => {
    setSidebarVersion((version) => version + 1);
    setActiveProject((current) => current?.id === project.id ? project : current);
    setThreadProject((current) => current?.id === project.id ? project : current);
  }, []);

  const handleThreadSelectFromList = useCallback((thread: Thread | null) => {
    handleThreadSelect(thread);
  }, [handleThreadSelect]);

  const handleOpenThreadInChat = useCallback((thread: Thread) => {
    handleThreadSelect(thread);
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

  const handleOpenHome = useCallback(() => {
    if (memoryManagerIntent && !confirmDiscardMemoryCurator()) return;
    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    setActiveThread(null);
    setActiveProject(null);
    setThreadProject(null);
    setProjectModelReady(null);
    setPdfTabs([]);
    setActiveTabId('home-tab');
    setIsBrowserActive(false);
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    setChatSentences([]);
    clearTraces();
  }, [clearTraces, confirmDiscardMemoryCurator, memoryManagerIntent]);

  const handleBackToProject = useCallback(async () => {
    try {
      const project = threadProject || (
        activeThread?.project_id ? await getProject(activeThread.project_id) : null
      );
      if (project) {
        await handleProjectSelect(project);
        return;
      }
    } catch (error) {
      console.error('Failed to open thread project:', error);
    }
    handleOpenHome();
  }, [activeThread?.project_id, handleOpenHome, handleProjectSelect, threadProject]);

  const handleProjectDeleted = useCallback((projectId: string) => {
    setSidebarVersion(v => v + 1);
    if (
      activeProject?.id === projectId
      || activeThread?.project_id === projectId
    ) {
      handleOpenHome();
    }
  }, [activeProject?.id, activeThread?.project_id, handleOpenHome]);

  const handleDeleteActiveThread = useCallback(async () => {
    if (!activeThread || isDeletingActiveThread) return;
    if (!confirm(`Delete "${activeThread.name}" and all its messages?`)) return;

    try {
      setIsDeletingActiveThread(true);
      await deleteThread(activeThread.id);
      setSidebarVersion(v => v + 1);
      await handleThreadSelect(null);
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
    setIsBrowserActive(false);

    if (activeThread && fileHash) {
      try {
        const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
        setActiveThread(updatedThread);
        setSidebarVersion(v => v + 1);
      } catch (error) {
        console.error('Failed to refresh thread after upload:', error);
      }
    } else if (activeProject && fileHash) {
      setPdfTabs(await loadProjectTabs(activeProject));
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
    if (!activeThread && !activeProject) return;
    try {
      if (activeProject) {
        setPdfTabs(await loadProjectTabs(activeProject));
        return;
      }
      const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
      setActiveThread(updatedThread);
      setSidebarVersion(v => v + 1);
    } catch (error) {
      console.error('Failed to refresh thread after indexing completed:', error);
    }
  };

  // Poll for parsing status when active tab is pending
  useEffect(() => {
    if (!activeTab || activeTab.parsingStatus !== ProcessStatus.Pending || (!activeThread && !activeProject)) {
      return;
    }

    let pollInterval: NodeJS.Timeout | null = null;

    const pollSentences = async () => {
      try {
        // Single endpoint returns both status and sentences
        const target: KnowledgeTarget = activeThread
          ? { scope: 'thread', id: activeThread.id }
          : { scope: 'project', id: activeProject!.id };
        const parsedData = await getParsedSentencesForTarget(activeTab.fileHash, target);
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
        // Don't crash while a newly attached file is still becoming visible.
        if (!error?.message?.includes('not attached')) {
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
  }, [activeTab?.fileHash, activeTab?.parsingStatus, activeThread?.id, activeProject?.id]);

  // Handle remove source from thread (deletes from DB + Weaviate, closes tab)
  const handleTabRemove = async (tabId: string) => {
    const tab = pdfTabs.find(t => t.id === tabId);
    if (!tab) return;

    try {
      if (activeThread) {
        if (tab.associationScope !== 'thread') return;
        await removeSourceFromThread(activeThread.id, tab.fileHash);
      } else if (activeProject) {
        await removeSourceFromProject(activeProject.id, tab.fileHash);
      } else {
        return;
      }
    } catch (error) {
      console.error('Failed to remove source from thread:', error);
    }

    // Close the tab and refresh sidebar
    handleTabClose(tabId);
    try {
      if (activeProject) {
        setPdfTabs(await loadProjectTabs(activeProject));
        return;
      }
      if (!activeThread) return;
      const updatedThread = await import("../lib/api").then(m => m.getThread(activeThread.id));
      setActiveThread(updatedThread);
      setSidebarVersion(v => v + 1);
    } catch (error) {
      console.error('Failed to refresh thread after source removal:', error);
    }
  };

  const handlePromoteDocument = async (tabId: string) => {
    if (!activeThread) return;
    const tab = pdfTabs.find((item) => item.id === tabId);
    if (!tab || tab.associationScope !== 'thread' || tab.isProjectKnowledge || !activeThread.project_id) return;
    try {
      if (!await checkEmbeddingModelReady(activeThread.embeddingModel)) return;
      await promoteFileToProject(activeThread.project_id, {
        fileHash: tab.fileHash,
        fileName: tab.fileName,
        filePath: tab.sourceUrl,
      });
      const updated = await getThread(activeThread.id);
      setActiveThread(updated);
      setPdfTabs(await loadThreadTabs(updated));
    } catch (error) {
      console.error('Failed to promote source:', error);
    }
  };

  const handleRetryDocument = async (tabId: string) => {
    const tab = pdfTabs.find((item) => item.id === tabId);
    const target: KnowledgeTarget | null = activeThread
      ? { scope: 'thread', id: activeThread.id }
      : activeProject ? { scope: 'project', id: activeProject.id } : null;
    if (!tab || !target) return;
    try {
      await retryTargetFile(target, tab.fileHash);
      setPdfTabs((current) => current.map((item) => (
        item.id === tabId ? { ...item, parsingStatus: ProcessStatus.Pending, processingError: undefined } : item
      )));
    } catch (error) {
      console.error('Failed to retry document processing:', error);
    }
  };

  // Handle tab change
  const handleTabChange = (tabId: string) => {
    setActiveTabId(tabId);
    setIsBrowserActive(tabId === 'browser-tab');
    const tab = pdfTabs.find(item => item.id === tabId);
    if (!tab || tabId === 'browser-tab' || tab.sentences) return;
    if (activeProject) {
      void loadProjectTabs(activeProject).then((tabs) => {
        setPdfTabs(tabs);
      });
      return;
    }
    if (!activeThread) return;
    void hydrateThreadPdfTab(activeThread.id, {
      fileHash: tab.fileHash,
      fileName: tab.fileName,
      sourceType: tab.sourceType,
      associationScope: tab.associationScope,
      isProjectKnowledge: tab.isProjectKnowledge,
    }).then((hydrated) => {
      setPdfTabs(prev => prev.map(item => item.fileHash === hydrated.fileHash ? hydrated : item));
    }).catch((error) => {
      console.warn(`Failed to hydrate selected PDF tab ${tab.fileHash}:`, error);
    });
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
    const target: KnowledgeTarget | null = activeThread
      ? { scope: 'thread', id: activeThread.id }
      : activeProject ? { scope: 'project', id: activeProject.id } : null;
    if (!target || isBrowserCapturing || (target.scope === 'project' && projectModelReady !== true)) return;

    setIsBrowserCapturing(true);
    try {
      const result = await captureBrowserPageForTarget(target);

      // Pre-verify file is accessible before creating tab
      const isReady = await pollForTargetFileReady(target, result.fileHash, {
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
        downloadUrl: `/${target.scope}s/${target.id}/files/${result.fileHash}/download`,
        sentences: null,
        sourceType: ThreadFileSourceType.Browser,
        sourceUrl: result.url,
        filePath: result.url,
        addedAt: new Date().toISOString(),
        associationScope: target.scope === 'thread' ? 'thread' : 'project',
        isProjectKnowledge: target.scope === 'project',
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

  const workspaceTabs = useMemo(
    () => activeThread
      ? buildDocumentWorkspaceTabs({
          enabled: true,
          documents: pdfTabs,
          traces: traceTabs,
        })
      : activeProject ? buildProjectWorkspaceTabs(pdfTabs) : buildHomeWorkspaceTabs(),
    [activeThread, activeProject, pdfTabs, traceTabs],
  );

  const handleWorkspaceTabChange = useCallback((tabId: string) => {
    if (memoryManagerIntent && tabId !== 'memory-tab') {
      if (!confirmDiscardMemoryCurator()) return;
      setMemoryManagerIntent(null);
      setMemoryCuratorDirty(false);
    }
    if (tabId !== 'memory-tab') {
      rememberNonMemoryTab(tabId);
    }
    setActiveTabId(tabId);
    setIsBrowserActive(tabId === 'browser-tab');
    if (tabId === 'memory-tab') {
      const memoryProject = activeProject || threadProject;
      if (!memoryManagerIntent) {
        setMemoryCuratorDirty(false);
        setMemoryManagerIntent(defaultMemoryManagerIntent({
          thread: activeThread,
          project: memoryProject,
        }));
      }
    }
  }, [activeProject, activeThread, confirmDiscardMemoryCurator, memoryManagerIntent, rememberNonMemoryTab, threadProject]);

  const handleOpenMemoryCurator = useCallback((intent: MemoryManagerIntent) => {
    if (memoryManagerIntent && memoryManagerDirtyRef.current && !confirmDiscardMemoryCurator()) return;
    rememberNonMemoryTab(activeTabId);
    setActiveTabId('memory-tab');
    setIsBrowserActive(false);
    setMemoryCuratorDirty(false);
    setMemoryManagerIntent(intent);
  }, [activeTabId, confirmDiscardMemoryCurator, memoryManagerIntent, rememberNonMemoryTab]);

  const handleMemoryBack = useCallback(() => {
    if (!confirmDiscardMemoryCurator()) return;
    setMemoryCuratorDirty(false);
    setMemoryManagerIntent(null);
    const target = lastNonMemoryTabByContext[workspaceContextKey] || fallbackNonMemoryTab();
    if (target === 'home-tab') {
      handleOpenHome();
      return;
    }
    setActiveTabId(target);
    setIsBrowserActive(target === 'browser-tab');
  }, [confirmDiscardMemoryCurator, fallbackNonMemoryTab, handleOpenHome, lastNonMemoryTabByContext, workspaceContextKey]);

  const handleOpenConversationReview = useCallback(() => {
    if (!activeThread) return;
    handleOpenMemoryCurator(reviewManagerIntent(activeThread));
  }, [activeThread, handleOpenMemoryCurator]);

  const handleOpenTrace = useCallback((trace: ChatTraceDescriptor) => {
    rememberNonMemoryTab('trace-tab');
    openTrace(trace);
    setActiveTabId('trace-tab');
    setIsBrowserActive(false);
  }, [openTrace, rememberNonMemoryTab]);

  const isMemoryWorkspaceActive = activeTabId === 'memory-tab';
  const activeMemoryIntent = isMemoryWorkspaceActive
    ? memoryManagerIntent || defaultMemoryManagerIntent({
      thread: activeThread,
      project: activeProject || threadProject,
    })
    : null;
  const memoryContextSubtitle = activeMemoryIntent
    ? activeMemoryIntent.scopeType === 'thread'
      ? activeThread?.name || 'Thread'
      : activeMemoryIntent.scopeType === 'project'
        ? (activeProject || threadProject)?.name || 'Project'
        : 'Home'
    : undefined;
  const memoryBackLabel = activeMemoryIntent
    ? activeMemoryIntent.scopeType === 'user'
      ? 'Back to Home'
      : activeMemoryIntent.scopeType === 'thread'
        ? 'Back to Thread'
        : 'Back to Project'
    : 'Back';

  // Memoize theme to prevent recreation on every render
  const theme = useMemo(() => getTheme(pdfDarkMode), [pdfDarkMode]);

  // Don't render until theme mode is determined (prevents hydration mismatch)
  if (!themeHydrated) return null;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: '100vh', overflow: 'hidden', bgcolor: 'background.default' }}>
        <WorkbenchShell
          layout={workbenchLayout}
          onLayoutChange={setWorkbenchLayout}
          onResolvedPlacementChange={setResolvedPlacement}
          onResizingChange={setIsResizing}
          secondaryLabel={isMemoryWorkspaceActive ? 'Memory curator' : 'Threads and chat'}
          primaryToolbar={
            <Box sx={{ px: 1.5, py: 0.75, minHeight: 49, borderBottom: 1, borderColor: 'divider', bgcolor: pdfDarkMode ? '#222' : 'background.paper', color: pdfDarkMode ? '#eee' : 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', minWidth: 0, flex: '1 1 auto' }}>
                <Tooltip title="Home">
                  <IconButton
                    color="default"
                    size="small"
                    aria-label="Home"
                    onClick={handleOpenHome}
                  >
                    <HomeIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                <PdfUploader
                  target={activeThread
                    ? { scope: 'thread', id: activeThread.id }
                    : activeProject ? { scope: 'project', id: activeProject.id } : null}
                  onUploaded={handlePdfUploaded}
                  onIndexingComplete={handleIndexingComplete}
                  onParsingComplete={handleParsingComplete}
                  disabled={!activeThread && (!activeProject || projectModelReady !== true)}
                  tooltipText={!activeThread && !activeProject
                    ? 'Select a thread or project first'
                    : activeProject && projectModelReady !== true ? 'Project embedding model is unavailable' : undefined}
                />
                <Tooltip title="Agent workflow builder">
                  <IconButton color="primary" size="small" onClick={() => window.open('/agent-workflow-builder', '_blank', 'noopener,noreferrer')}>
                    <AutoAwesomeSharpIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
                {activeThread && (
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
                  <IconButton color={pdfDarkMode ? 'primary' : 'default'} onClick={toggleDarkMode} size="small">
                    {pdfDarkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                  </IconButton>
                </Tooltip>
                <DockMenuButton value={workbenchLayout} resolvedPlacement={resolvedPlacement} onChange={setWorkbenchLayout} label="Threads and chat layout" />
              </WorkbenchToolbarTrailingActions>
            </Box>
          }
          primaryTabs={
            <WorkspaceTabs
              tabs={workspaceTabs}
              activeTabId={activeTabId}
              onTabChange={handleWorkspaceTabChange}
              onTabClose={handleTabClose}
              onDocumentRemove={handleTabRemove}
              onDocumentPromote={handlePromoteDocument}
              onDocumentRetry={handleRetryDocument}
              documentContext={activeProject ? 'project' : 'thread'}
              onAddBrowserToThread={handleAddBrowserToThread}
              isBrowserCapturing={isBrowserCapturing}
            />
          }
          primaryContent={
            <ThreadWorkspaceContent
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
              activeThread={activeThread}
              activeProject={activeProject}
              projectInventoryVersion={sidebarVersion}
              curatorRefreshVersion={memoryRefreshVersion}
              onOpenMemoryCurator={handleOpenMemoryCurator}
              emptyTitle="Welcome to AskPDF"
              emptyDescription="Select or create a thread, then upload a PDF or open the browser."
            />
          }
          secondaryContent={
            activeMemoryIntent ? (
              <MemoryManagerPanel
                key={`${activeMemoryIntent.mode}:${activeMemoryIntent.memory?.id || activeMemoryIntent.scopeType}:${activeMemoryIntent.scopeId}`}
                intent={activeMemoryIntent}
                onBack={handleMemoryBack}
                backLabel={memoryBackLabel}
                contextSubtitle={memoryContextSubtitle}
                onDirtyChange={setMemoryCuratorDirty}
                onApplied={() => setMemoryRefreshVersion((version) => version + 1)}
              />
            ) : (
            <ThreadSecondaryPanel
              activeThread={activeThread}
              activeProject={activeProject}
              threadProject={threadProject}
              activeProjectId={activeProject?.id ?? null}
              sidebarKey={sidebarVersion}
              onThreadSelect={handleThreadSelectFromList}
              onProjectSelect={handleProjectSelect}
              onProjectReadinessChange={(_projectId, ready) => setProjectModelReady(ready)}
              onProjectUpdated={handleProjectUpdated}
              onProjectCloned={handleProjectCloned}
              onProjectDeleted={handleProjectDeleted}
              onThreadForked={handleThreadForked}
              onBackToProject={handleBackToProject}
              darkMode={pdfDarkMode}
              renderSelectedTitle={(thread) => (
                <Tooltip
                  title={
                    <ThreadLineageTooltipContent
                      thread={activeThreadForTooltip || thread}
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
                      ...flexTruncateSx,
                      alignSelf: 'stretch',
                      display: 'flex',
                      alignItems: 'center',
                      cursor: 'default',
                    }}
                  >
                    <Typography variant="subtitle2" fontWeight={700} noWrap sx={singleLineTruncateSx}>
                      {thread.name}
                    </Typography>
                  </Box>
                </Tooltip>
              )}
              selectedActions={(
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
              )}
              renderConversation={(thread) => (
                <ChatInterface
                  activeThread={thread}
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
                  onOpenMemoryReview={handleOpenConversationReview}
                  hideInlineLineage
                  darkMode={pdfDarkMode}
                  autoScroll={autoScroll}
                  isPanelResizing={isResizing}
                />
              )}
            />
            )
          }
        />
      </Box>
    </ThemeProvider>
  );
}
