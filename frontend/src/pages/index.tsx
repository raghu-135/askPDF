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

import ChatInterface, { type ChatTraceDescriptor } from "../components/ChatInterface";
import ThreadSecondaryPanel from "../components/ThreadSecondaryPanel";
import type { ThreadSidebarHeaderState } from "../components/ThreadSidebar";
import MemoryManagerPanel from "../components/MemoryManagerPanel";
import {
  buildDocumentWorkspaceTabs,
  buildHomeWorkspaceTabs,
  buildProjectWorkspaceTabs,
  DOCUMENTS_TAB_ID,
  isDocumentsWorkspaceActive,
  PROJECT_OVERVIEW_TAB_ID,
  THREAD_OVERVIEW_TAB_ID,
  type PdfTab,
} from "../lib/document-tabs";
import {
  readWorkspaceResume,
  resolveWorkspaceResume,
  writeWorkspaceResume,
  type WorkspaceResumeState,
} from '../lib/workspace-resume-state';
import { RESEARCH_CANVAS_TAB_ID } from "../lib/canvas-spec";
import type { CanvasRef, DocumentCanvasCitationTarget } from "../lib/canvas-spec";
import WorkbenchShell, { useWorkbenchLayout } from '../components/workbench/WorkbenchShell';
import DockMenuButton from '../components/workbench/DockMenuButton';
import { WorkbenchToolbar, WorkbenchToolbarTrailingActions } from '../components/workbench/WorkbenchToolbar';
import WorkspaceTabs, { type DocumentWorkspaceTab } from '../components/workbench/WorkspaceTabs';
import DocumentChunkInspectorDialog from '../components/document/DocumentChunkInspectorDialog';
import ThreadWorkspaceContent from '../components/workbench/ThreadWorkspaceContent';
import useTraceTabs from '../components/workbench/useTraceTabs';
import ThreadLineageTooltipContent from "../components/ThreadLineageTooltipContent";
import { API_BASE, Project, Thread, removeSourceFromThread, removeSourceFromProject, promoteFileToProject, retryTargetFile, getParsedSentencesForTarget, captureBrowserPageForTarget, pollForTargetFileReady, getThread, getProject, getPdfForTarget, deleteThread, listThreads, type KnowledgeTarget } from "../lib/api";
import { loadThreadTabs, loadProjectTabs, hydrateThreadPdfTab, createPdfTabFromUpload, extractTextFromSentences } from "../lib/thread-utils";
import { closeDocumentTabUtil, getActiveTab, getActiveTabData } from "../lib/pdf-utils";
import { isParsedSentencePayload, transformSentences } from "../lib/bbox-derivation";
import { ProcessStatus, ThreadFileSourceType } from "../lib/enums";
import type { ResolvedWorkbenchPlacement } from '../lib/workbench-layout';
import { checkEmbeddingModelReady } from '../lib/models-api';
import { flexTruncateSx, singleLineTruncateSx } from '../lib/truncation';
import { defaultMemoryManagerIntent, reviewManagerIntent, type MemoryManagerIntent } from '../lib/memory-manager';
import type { ConversationSentence } from '../lib/chat-sentence-cache';
import { ThreadChatSettingsProvider } from '../lib/thread-chat-settings-context';

export default function Home() {
  // Multiple PDF tabs state
  const [pdfTabs, setPdfTabs] = useState<PdfTab[]>([]);
  const [activeTabId, setActiveTabId] = useState<string | null>('home-tab');
  const [activeDocumentId, setActiveDocumentId] = useState<string | null>(null);
  const [cachedPdfFileHash, setCachedPdfFileHash] = useState<string | null>(null);
  const previousDocumentFileHashRef = useRef<string | null>(null);
  const [isPdfLoading, setIsPdfLoading] = useState(false);

  const activeDocument = getActiveTab(pdfTabs, activeDocumentId);
  const cachedDocument = cachedPdfFileHash
    ? pdfTabs.find((tab) => tab.fileHash === cachedPdfFileHash) || null
    : null;
  const { pdfSentences, downloadUrl, fileHash, fileName } = getActiveTabData(activeDocument);
  const cachedDocumentData = getActiveTabData(cachedDocument);

  const [activeSource, setActiveSource] = useState<'pdf' | 'chat'>('pdf');
  const [currentPdfId, setCurrentPdfId] = useState<number | null>(null);
  const [currentChatId, setCurrentChatId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [chatSentences, setChatSentences] = useState<ConversationSentence[]>([]);
  const [chatPlaybackSourceKey, setChatPlaybackSourceKey] = useState('chat:none');

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
  const [sidebarHeaderState, setSidebarHeaderState] = useState<ThreadSidebarHeaderState | null>(null);
  const workspaceNavRef = useRef(0);

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
  const [activeCanvasId, setActiveCanvasId] = useState<string | null>(null);
  const [canvasRefreshVersion, setCanvasRefreshVersion] = useState(0);
  const [chunkInspectorTab, setChunkInspectorTab] = useState<DocumentWorkspaceTab | null>(null);

  const confirmDiscardMemoryCurator = useCallback(() => (
    !memoryManagerDirtyRef.current
    || window.confirm('Discard the unconfirmed memory proposal?')
  ), []);

  const workspaceContextKey = activeThread
    ? `thread:${activeThread.id}`
    : activeProject
      ? `project:${activeProject.id}`
      : 'home';

  const chunkInspectorTarget = useMemo(() => {
    if (activeProject) return { scope: 'project' as const, id: activeProject.id };
    if (activeThread) return { scope: 'thread' as const, id: activeThread.id };
    return null;
  }, [activeProject, activeThread]);

  const handleInspectChunks = useCallback((tab: DocumentWorkspaceTab) => {
    if (!chunkInspectorTarget) return;
    setChunkInspectorTab(tab);
  }, [chunkInspectorTarget]);

  const persistWorkspaceResume = useCallback((
    contextKey: string,
    state: WorkspaceResumeState,
  ) => {
    writeWorkspaceResume(contextKey, state);
  }, []);

  const hydrateResumedDocumentTab = useCallback((
    tab: PdfTab,
    target: { kind: 'thread'; threadId: string } | { kind: 'project'; projectId: string },
  ) => {
    if (!tab || tab.sentences) return;
    if (target.kind === 'thread') {
      void hydrateThreadPdfTab(target.threadId, {
        fileHash: tab.fileHash,
        fileName: tab.fileName,
        sourceType: tab.sourceType,
        associationScope: tab.associationScope,
        isProjectKnowledge: tab.isProjectKnowledge,
      }).then((hydrated) => {
        setPdfTabs((prev) => prev.map((item) => (item.fileHash === hydrated.fileHash ? hydrated : item)));
      }).catch((error) => {
        console.warn(`Failed to hydrate resumed document ${tab.fileHash}:`, error);
      });
      return;
    }
    void getPdfForTarget(tab.fileHash, { scope: 'project', id: target.projectId }).then((pdfData) => {
      const sentences = transformSentences(pdfData.sentences);
      setPdfTabs((prev) => prev.map((item) => (
        item.id === tab.id
          ? {
            ...item,
            downloadUrl: `${API_BASE}/api${pdfData.downloadUrl}?t=${Date.now()}`,
            sentences,
            text: extractTextFromSentences(sentences),
            parsingStatus: ProcessStatus.Completed,
          }
          : item
      )));
    }).catch((error) => {
      console.warn(`Failed to hydrate resumed document ${tab.fileHash}:`, error);
    });
  }, []);

  // Handle thread selection
  const handleThreadSelect = useCallback(async (thread: Thread | null) => {
    if (memoryManagerIntent && !confirmDiscardMemoryCurator()) return;
    const nav = ++workspaceNavRef.current;
    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    // Clear current state
    setPdfTabs([]);
    setCachedPdfFileHash(null);
    previousDocumentFileHashRef.current = null;
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    setChatSentences([]);
    setActiveProject(null);
    setThreadProject(null);
    clearTraces();
    setActiveCanvasId(null);
    setActiveDocumentId(null);
    setActiveTraceId(null);

    if (thread) {
      const contextKey = `thread:${thread.id}`;
      const cached = readWorkspaceResume(contextKey);
      const initialTabId = cached?.tabId && cached.tabId !== 'memory-tab'
        ? cached.tabId
        : THREAD_OVERVIEW_TAB_ID;
      setActiveTabId(initialTabId);
      setActiveDocumentId(
        isDocumentsWorkspaceActive(initialTabId) && cached?.documentId ? cached.documentId : null,
      );
      setActiveCanvasId(
        initialTabId === RESEARCH_CANVAS_TAB_ID && cached?.canvasId ? cached.canvasId : null,
      );
      setIsBrowserActive(initialTabId === 'browser-tab');

      setActiveProject(null);
      setActiveThread(thread);
      try {
        setIsPdfLoading(true);
        const detailedThread = await getThread(thread.id);
        if (nav !== workspaceNavRef.current) return;

        const [loadedTabs, parentProject] = await Promise.all([
          loadThreadTabs(detailedThread, { eagerCount: 0 }),
          detailedThread.project_id
            ? getProject(detailedThread.project_id).catch(() => null)
            : Promise.resolve(null),
        ]);
        if (nav !== workspaceNavRef.current) return;
        setActiveThread(detailedThread);
        setThreadProject(parentProject);
        setPdfTabs(loadedTabs);

        const resolved = resolveWorkspaceResume({
          contextKey,
          state: cached,
          availableTabs: buildDocumentWorkspaceTabs({
            enabled: true,
            documentCount: loadedTabs.length,
            traces: [],
            includeResearchCanvas: true,
          }),
          pdfTabs: loadedTabs,
          traceIds: [],
          canvasIds: [],
        });
        setActiveTabId(resolved.tabId);
        setActiveDocumentId(resolved.documentId ?? null);
        setActiveTraceId(resolved.traceId ?? null);
        setActiveCanvasId(resolved.canvasId ?? null);
        setIsBrowserActive(resolved.tabId === 'browser-tab');
        persistWorkspaceResume(contextKey, resolved);

        if (resolved.tabId === DOCUMENTS_TAB_ID && resolved.documentId) {
          const tab = loadedTabs.find((item) => item.id === resolved.documentId);
          if (tab) hydrateResumedDocumentTab(tab, { kind: 'thread', threadId: detailedThread.id });
        }
      } catch (err) {
        if (nav !== workspaceNavRef.current) return;
        console.error('Failed to load thread files:', err);
        setActiveTabId(THREAD_OVERVIEW_TAB_ID);
      } finally {
        if (nav === workspaceNavRef.current) {
          setIsPdfLoading(false);
        }
      }
    } else {
      setActiveThread(null);
      const homeTabs = buildHomeWorkspaceTabs();
      const resolved = resolveWorkspaceResume({
        contextKey: 'home',
        state: readWorkspaceResume('home'),
        availableTabs: homeTabs,
      });
      setActiveTabId(resolved.tabId);
      setIsBrowserActive(false);
    }
  }, [clearTraces, confirmDiscardMemoryCurator, hydrateResumedDocumentTab, memoryManagerIntent, persistWorkspaceResume, setActiveTraceId]);

  const handleProjectSelect = useCallback(async (project: Project) => {
    if (memoryManagerIntent && !confirmDiscardMemoryCurator()) return;
    const nav = ++workspaceNavRef.current;
    const contextKey = `project:${project.id}`;
    const cached = readWorkspaceResume(contextKey);
    const initialTabId = cached?.tabId && cached.tabId !== 'memory-tab'
      ? cached.tabId
      : PROJECT_OVERVIEW_TAB_ID;

    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    setActiveThread(null);
    setThreadProject(null);
    setActiveProject(project);
    setPdfTabs([]);
    setCachedPdfFileHash(null);
    previousDocumentFileHashRef.current = null;
    clearTraces();
    setActiveCanvasId(null);
    setActiveTraceId(null);
    setActiveTabId(initialTabId);
    setActiveDocumentId(
      isDocumentsWorkspaceActive(initialTabId) && cached?.documentId ? cached.documentId : null,
    );
    setIsBrowserActive(initialTabId === 'browser-tab');
    setIsPdfLoading(true);
    setProjectModelReady(null);
    try {
      const tabs = await loadProjectTabs(project, { eagerCount: 0 });
      if (nav !== workspaceNavRef.current) return;
      setPdfTabs(tabs);

      const resolved = resolveWorkspaceResume({
        contextKey,
        state: cached,
        availableTabs: buildProjectWorkspaceTabs(tabs.length),
        pdfTabs: tabs,
      });
      setActiveTabId(resolved.tabId);
      setActiveDocumentId(resolved.documentId ?? null);
      setIsBrowserActive(resolved.tabId === 'browser-tab');
      persistWorkspaceResume(contextKey, resolved);

      if (resolved.tabId === DOCUMENTS_TAB_ID && resolved.documentId) {
        const tab = tabs.find((item) => item.id === resolved.documentId);
        if (tab) hydrateResumedDocumentTab(tab, { kind: 'project', projectId: project.id });
      }
    } catch (error) {
      if (nav !== workspaceNavRef.current) return;
      console.error('Failed to open project knowledge:', error);
      setProjectModelReady(false);
      setActiveTabId(PROJECT_OVERVIEW_TAB_ID);
    } finally {
      if (nav === workspaceNavRef.current) {
        setIsPdfLoading(false);
      }
    }
  }, [clearTraces, confirmDiscardMemoryCurator, hydrateResumedDocumentTab, memoryManagerIntent, persistWorkspaceResume, setActiveTraceId]);

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
    workspaceNavRef.current += 1;
    setMemoryManagerIntent(null);
    setMemoryCuratorDirty(false);
    setActiveThread(null);
    setActiveProject(null);
    setThreadProject(null);
    setProjectModelReady(null);
    setPdfTabs([]);
    const resolved = resolveWorkspaceResume({
      contextKey: 'home',
      state: readWorkspaceResume('home'),
      availableTabs: buildHomeWorkspaceTabs(),
    });
    setActiveTabId(resolved.tabId);
    setActiveDocumentId(null);
    setCachedPdfFileHash(null);
    previousDocumentFileHashRef.current = null;
    setIsBrowserActive(false);
    setCurrentPdfId(null);
    setCurrentChatId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    setChatSentences([]);
    clearTraces();
    setActiveCanvasId(null);
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
      setActiveTabId(DOCUMENTS_TAB_ID);
      setActiveDocumentId(existingTab.id);
      persistWorkspaceResume(workspaceContextKey, {
        tabId: DOCUMENTS_TAB_ID,
        documentId: existingTab.id,
        traceId: null,
        canvasId: null,
      });
      setIsBrowserActive(false);
      setCurrentPdfId(null);
      setCurrentChatId(null);
      setPlayRequestId(null);
      setActiveSource('pdf');
      return;
    }

    // Upload responses intentionally contain only file data. Add the local
    // association metadata immediately so actions work before the next reload.
    const newTab = {
      ...createPdfTabFromUpload(data),
      associationScope: activeThread ? 'thread' as const : 'project' as const,
      isProjectKnowledge: Boolean(activeProject),
    };

    setPdfTabs(prev => [...prev, newTab]);
    setActiveTabId(DOCUMENTS_TAB_ID);
    setActiveDocumentId(newTab.id);
    persistWorkspaceResume(workspaceContextKey, {
      tabId: DOCUMENTS_TAB_ID,
      documentId: newTab.id,
      traceId: null,
      canvasId: null,
    });
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
    if (!Array.isArray(sentences)) {
      return;
    }
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

  useEffect(() => {
    if (!activeDocument?.fileHash) return;
    if (previousDocumentFileHashRef.current && previousDocumentFileHashRef.current !== activeDocument.fileHash) {
      setCachedPdfFileHash(previousDocumentFileHashRef.current);
    }
    previousDocumentFileHashRef.current = activeDocument.fileHash;
  }, [activeDocument?.fileHash]);

  // Poll for parsing status when active document is pending
  useEffect(() => {
    if (!activeDocument || activeDocument.parsingStatus !== ProcessStatus.Pending || (!activeThread && !activeProject)) {
      return;
    }

    let pollInterval: NodeJS.Timeout | null = null;

    const pollSentences = async () => {
      try {
        // Single endpoint returns both status and sentences
        const target: KnowledgeTarget = activeThread
          ? { scope: 'thread', id: activeThread.id }
          : { scope: 'project', id: activeProject!.id };
        const parsedData = await getParsedSentencesForTarget(activeDocument.fileHash, target);
        if (isParsedSentencePayload(parsedData?.sentences)) {
          handleParsingComplete(activeDocument.fileHash, parsedData.sentences);
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
  }, [activeDocument?.fileHash, activeDocument?.parsingStatus, activeThread?.id, activeProject?.id]);

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

    handleDocumentClose(tabId);
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

  const hydrateDocumentOnSelect = useCallback((tab: PdfTab) => {
    if (!tab || tab.sentences) return;
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
      setPdfTabs((prev) => prev.map((item) => item.fileHash === hydrated.fileHash ? hydrated : item));
    }).catch((error) => {
      console.warn(`Failed to hydrate selected PDF tab ${tab.fileHash}:`, error);
    });
  }, [activeProject, activeThread]);

  const handleActiveDocumentChange = useCallback((documentId: string) => {
    setActiveDocumentId(documentId);
    setCurrentPdfId(null);
    setPlayRequestId(null);
    setActiveSource('pdf');
    const tab = pdfTabs.find((item) => item.id === documentId);
    if (tab) hydrateDocumentOnSelect(tab);
    if (isDocumentsWorkspaceActive(activeTabId)) {
      persistWorkspaceResume(workspaceContextKey, {
        tabId: DOCUMENTS_TAB_ID,
        documentId,
        traceId: null,
        canvasId: null,
      });
    }
  }, [activeTabId, hydrateDocumentOnSelect, pdfTabs, persistWorkspaceResume, workspaceContextKey]);

  const handleOpenOverviewDocument = useCallback((documentId: string) => {
    setActiveTabId(DOCUMENTS_TAB_ID);
    setActiveDocumentId(documentId);
    setIsBrowserActive(false);
    setActiveSource('pdf');
    const tab = pdfTabs.find((item) => item.id === documentId);
    if (tab) hydrateDocumentOnSelect(tab);
    persistWorkspaceResume(workspaceContextKey, {
      tabId: DOCUMENTS_TAB_ID,
      documentId,
      traceId: null,
      canvasId: null,
    });
  }, [hydrateDocumentOnSelect, pdfTabs, persistWorkspaceResume, workspaceContextKey]);

  const handleDocumentClose = useCallback((tabId: string) => {
    closeDocumentTabUtil(
      tabId,
      pdfTabs,
      activeDocumentId,
      setPdfTabs,
      setActiveDocumentId,
      setCurrentPdfId,
      setPlayRequestId,
    );
  }, [activeDocumentId, pdfTabs]);

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
          documentCount: pdfTabs.length,
          traces: traceTabs,
          includeResearchCanvas: true,
        })
      : activeProject ? buildProjectWorkspaceTabs(pdfTabs.length) : buildHomeWorkspaceTabs(),
    [activeThread, activeProject, pdfTabs.length, traceTabs],
  );

  const handleWorkspaceTabChange = useCallback((tabId: string) => {
    if (memoryManagerIntent && tabId !== 'memory-tab') {
      if (!confirmDiscardMemoryCurator()) return;
      setMemoryManagerIntent(null);
      setMemoryCuratorDirty(false);
    }
    let nextDocumentId = activeDocumentId;
    let nextTraceId = activeTraceId;
    let nextCanvasId = activeCanvasId;
    if (tabId === DOCUMENTS_TAB_ID && !activeDocumentId && pdfTabs[0]) {
      nextDocumentId = pdfTabs[0].id;
      hydrateDocumentOnSelect(pdfTabs[0]);
    }
    if (tabId !== DOCUMENTS_TAB_ID) nextDocumentId = null;
    if (tabId !== 'trace-tab') nextTraceId = null;
    if (tabId !== RESEARCH_CANVAS_TAB_ID) nextCanvasId = null;
    setActiveTabId(tabId);
    setActiveDocumentId(nextDocumentId);
    setActiveTraceId(nextTraceId);
    setActiveCanvasId(nextCanvasId);
    setIsBrowserActive(tabId === 'browser-tab');
    if (tabId !== 'memory-tab') {
      persistWorkspaceResume(workspaceContextKey, {
        tabId,
        documentId: nextDocumentId,
        traceId: nextTraceId,
        canvasId: nextCanvasId,
      });
    }
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
  }, [activeCanvasId, activeDocumentId, activeProject, activeThread, activeTraceId, confirmDiscardMemoryCurator, hydrateDocumentOnSelect, memoryManagerIntent, pdfTabs, persistWorkspaceResume, threadProject, workspaceContextKey]);

  const handleOpenMemoryCurator = useCallback((intent: MemoryManagerIntent) => {
    if (memoryManagerIntent && memoryManagerDirtyRef.current && !confirmDiscardMemoryCurator()) return;
    setActiveTabId('memory-tab');
    setIsBrowserActive(false);
    setMemoryCuratorDirty(false);
    setMemoryManagerIntent(intent);
  }, [confirmDiscardMemoryCurator, memoryManagerIntent]);

  const handleMemoryBack = useCallback(() => {
    if (!confirmDiscardMemoryCurator()) return;
    setMemoryCuratorDirty(false);
    setMemoryManagerIntent(null);
    const availableTabs = activeThread
      ? buildDocumentWorkspaceTabs({
        enabled: true,
        documentCount: pdfTabs.length,
        traces: traceTabs,
        includeResearchCanvas: true,
      })
      : activeProject
        ? buildProjectWorkspaceTabs(pdfTabs.length)
        : buildHomeWorkspaceTabs();
    const resolved = resolveWorkspaceResume({
      contextKey: workspaceContextKey,
      state: readWorkspaceResume(workspaceContextKey),
      availableTabs,
      pdfTabs,
      traceIds: traceTabs.map((trace) => trace.id),
      canvasIds: activeCanvasId ? [activeCanvasId] : [],
    });
    if (resolved.tabId === 'home-tab' && workspaceContextKey === 'home') {
      setActiveTabId(resolved.tabId);
      setIsBrowserActive(false);
      return;
    }
    setActiveTabId(resolved.tabId);
    setActiveDocumentId(resolved.documentId ?? null);
    setActiveTraceId(resolved.traceId ?? null);
    setActiveCanvasId(resolved.canvasId ?? null);
    setIsBrowserActive(resolved.tabId === 'browser-tab');
  }, [activeCanvasId, activeProject, activeThread, confirmDiscardMemoryCurator, pdfTabs, traceTabs, workspaceContextKey]);

  const handleOpenConversationReview = useCallback((draftContent?: string) => {
    if (!activeThread) return;
    if (draftContent) {
      handleOpenMemoryCurator({
        ...defaultMemoryManagerIntent({ thread: activeThread, project: activeProject || threadProject }),
        draftContent,
      });
      return;
    }
    handleOpenMemoryCurator(reviewManagerIntent(activeThread));
  }, [activeProject, activeThread, handleOpenMemoryCurator, threadProject]);

  const handleOpenCanvas = useCallback((canvas: CanvasRef) => {
    setActiveCanvasId(canvas.id);
    setActiveTabId(RESEARCH_CANVAS_TAB_ID);
    setIsBrowserActive(false);
    setCanvasRefreshVersion((value) => value + 1);
    persistWorkspaceResume(workspaceContextKey, {
      tabId: RESEARCH_CANVAS_TAB_ID,
      documentId: null,
      traceId: null,
      canvasId: canvas.id,
    });
  }, [persistWorkspaceResume, workspaceContextKey]);

  const handleOpenDocumentCitation = useCallback((target: DocumentCanvasCitationTarget) => {
    const documentTab = pdfTabs.find((tab) => tab.fileHash === target.fileHash);
    if (!documentTab) return;
    setActiveTabId(DOCUMENTS_TAB_ID);
    setActiveDocumentId(documentTab.id);
    setIsBrowserActive(false);
    hydrateDocumentOnSelect(documentTab);
    setActiveSource('pdf');
    persistWorkspaceResume(workspaceContextKey, {
      tabId: DOCUMENTS_TAB_ID,
      documentId: documentTab.id,
      traceId: null,
      canvasId: null,
    });
    if (target.sentenceId != null) {
      setCurrentPdfId(target.sentenceId);
      if (target.play !== false) {
        setPlayRequestId(target.sentenceId);
      }
    }
  }, [hydrateDocumentOnSelect, pdfTabs, persistWorkspaceResume, workspaceContextKey]);

  const handleOpenTrace = useCallback((trace: ChatTraceDescriptor) => {
    if (trace.activate !== false) {
      setActiveTabId('trace-tab');
      setIsBrowserActive(false);
      persistWorkspaceResume(workspaceContextKey, {
        tabId: 'trace-tab',
        documentId: null,
        traceId: trace.id,
        canvasId: null,
      });
    }
    openTrace(trace);
  }, [openTrace, persistWorkspaceResume, workspaceContextKey]);

  const handleActiveTraceChange = useCallback((traceId: string | null) => {
    setActiveTraceId(traceId);
    if (activeTabId === 'trace-tab' && traceId) {
      persistWorkspaceResume(workspaceContextKey, {
        tabId: 'trace-tab',
        documentId: null,
        traceId,
        canvasId: null,
      });
    }
  }, [activeTabId, persistWorkspaceResume, setActiveTraceId, workspaceContextKey]);

  const handleActiveCanvasChange = useCallback((canvasId: string | null) => {
    setActiveCanvasId(canvasId);
    if (activeTabId === RESEARCH_CANVAS_TAB_ID && canvasId) {
      persistWorkspaceResume(workspaceContextKey, {
        tabId: RESEARCH_CANVAS_TAB_ID,
        documentId: null,
        traceId: null,
        canvasId,
      });
    }
  }, [activeTabId, persistWorkspaceResume, workspaceContextKey]);

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
      <ThreadChatSettingsProvider>
      <Box sx={{ height: '100vh', overflow: 'hidden', bgcolor: 'background.default' }}>
        <WorkbenchShell
          layout={workbenchLayout}
          onLayoutChange={setWorkbenchLayout}
          onResolvedPlacementChange={setResolvedPlacement}
          onResizingChange={setIsResizing}
          secondaryLabel={isMemoryWorkspaceActive ? 'Memory curator' : 'Threads and chat'}
          primaryToolbar={
            <WorkbenchToolbar
              sx={{
                px: 1.5,
                py: 0.75,
                minHeight: 49,
                borderBottom: 1,
                borderColor: 'divider',
                bgcolor: pdfDarkMode ? '#222' : 'background.paper',
                color: pdfDarkMode ? '#eee' : 'inherit',
              }}
              trailing={(
                <WorkbenchToolbarTrailingActions>
                  <Tooltip title={pdfDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
                    <IconButton color={pdfDarkMode ? 'primary' : 'default'} onClick={toggleDarkMode} size="small">
                      {pdfDarkMode ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
                    </IconButton>
                  </Tooltip>
                  <DockMenuButton value={workbenchLayout} resolvedPlacement={resolvedPlacement} onChange={setWorkbenchLayout} label="Threads and chat layout" />
                </WorkbenchToolbarTrailingActions>
              )}
            >
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
              {(activeThread || activeProject) && (
                <PdfUploader
                  target={activeThread
                    ? { scope: 'thread', id: activeThread.id }
                    : { scope: 'project', id: activeProject!.id }}
                  onUploaded={handlePdfUploaded}
                  onIndexingComplete={handleIndexingComplete}
                  onParsingComplete={handleParsingComplete}
                  showButton={false}
                />
              )}
              <Tooltip title="Agent workflow builder">
                <IconButton color="primary" size="small" onClick={() => window.open('/agent-workflow-builder', '_blank', 'noopener,noreferrer')}>
                  <AutoAwesomeSharpIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </WorkbenchToolbar>
          }
          primaryTabs={
            <WorkspaceTabs
              tabs={workspaceTabs}
              activeTabId={activeTabId}
              onTabChange={handleWorkspaceTabChange}
            />
          }
          primaryContent={
            <ThreadWorkspaceContent
              activeTabId={activeTabId}
              activeDocumentId={activeDocumentId}
              onActiveDocumentChange={handleActiveDocumentChange}
              activeDocument={activeDocument}
              documentSentences={pdfSentences}
              documentDownloadUrl={downloadUrl}
              cachedDocumentId={cachedPdfFileHash}
              cachedDocument={cachedDocument}
              cachedSentences={cachedDocumentData.pdfSentences}
              cachedDownloadUrl={cachedDocumentData.downloadUrl}
              traceTabs={traceTabs}
              activeTraceId={activeTraceId}
              onActiveTraceChange={handleActiveTraceChange}
              onCloseTrace={closeTrace}
              onCloseDocument={handleDocumentClose}
              onDocumentRemove={handleTabRemove}
              onDocumentPromote={handlePromoteDocument}
              onDocumentRetry={handleRetryDocument}
              onInspectChunks={chunkInspectorTarget ? handleInspectChunks : undefined}
              documentContext={activeProject ? 'project' : 'thread'}
              isLoading={isPdfLoading}
              isResizing={isResizing}
              darkMode={pdfDarkMode}
              currentDocumentSentenceId={activeSource === 'pdf' ? currentPdfId : null}
              onDocumentJump={(id) => { setActiveSource('pdf'); setCurrentPdfId(id); setPlayRequestId(id); }}
              autoScroll={autoScroll}
              highlightEnabled={highlightEnabled}
              threadId={activeThread?.id ?? null}
              activeThread={activeThread}
              activeCanvasId={activeCanvasId}
              onActiveCanvasChange={handleActiveCanvasChange}
              onOpenDocumentCitation={handleOpenDocumentCitation}
              canvasRefreshVersion={canvasRefreshVersion}
              documents={pdfTabs}
              activeProject={activeProject}
              projectInventoryVersion={sidebarVersion}
              curatorRefreshVersion={memoryRefreshVersion}
              inventoryLoading={sidebarHeaderState?.isLoading ?? true}
              hasProjects={(sidebarHeaderState?.projectCount ?? 0) > 0}
              onOpenMemoryCurator={handleOpenMemoryCurator}
              onCreateProject={sidebarHeaderState?.openCreateProjectDialog}
              onCreateThread={sidebarHeaderState?.openCreateThreadDialog}
              onCapturePage={() => {
                setActiveTabId('browser-tab');
                setIsBrowserActive(true);
                persistWorkspaceResume(workspaceContextKey, {
                  tabId: 'browser-tab',
                  documentId: null,
                  traceId: null,
                  canvasId: null,
                });
              }}
              onAddCapture={() => { void handleAddBrowserToThread(); }}
              onRequestUpload={() => {
                const input = document.getElementById('pdf-upload-input');
                if (input instanceof HTMLInputElement) input.click();
              }}
              isBrowserCapturing={isBrowserCapturing}
              documentCount={pdfTabs.length}
              threadProject={threadProject}
              threadsById={rightPanelLineageThreadsById}
              onOpenThread={handleOpenThreadInChat}
              onOpenDocument={handleOpenOverviewDocument}
              onProjectUpdated={handleProjectUpdated}
              projectReady={activeProject ? projectModelReady !== false : true}
              onCloneProject={sidebarHeaderState?.openCloneProjectDialog
                ? () => sidebarHeaderState.openCloneProjectDialog?.(false)
                : undefined}
              onCloneProjectWithThreads={sidebarHeaderState?.openCloneProjectDialog
                ? () => sidebarHeaderState.openCloneProjectDialog?.(true)
                : undefined}
              onDeleteProject={sidebarHeaderState?.openDeleteProjectDialog}
              playerControlProps={isDocumentsWorkspaceActive(activeTabId) && activeThread ? {
                sentences: activeSource === 'pdf' ? pdfSentences : chatSentences,
                sourceKey: activeSource === 'pdf' ? `pdf:${fileHash || 'none'}` : chatPlaybackSourceKey,
                currentId: activeSource === 'pdf' ? currentPdfId : currentChatId,
                onCurrentChange: (id) => {
                  if (activeSource === 'pdf') setCurrentPdfId(id);
                  else setCurrentChatId(id);
                  setPlayRequestId(null);
                },
                playRequestId,
                autoScroll,
                onAutoScrollChange: setAutoScroll,
                highlightEnabled,
                onHighlightEnabledChange: setHighlightEnabled,
              } : null}
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
              onBackToProjects={handleOpenHome}
              onHeaderStateChange={setSidebarHeaderState}
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
                  setChatPlaybackSourceKey={setChatPlaybackSourceKey}
                  currentChatId={currentChatId}
                  activeSource={activeSource}
                  onJump={(id) => { setActiveSource('chat'); setCurrentChatId(id); setPlayRequestId(id); }}
                  onResetChatId={() => { setCurrentChatId(null); setPlayRequestId(null); }}
                  onThreadForked={handleThreadForked}
                  onThreadUpdate={handleThreadUpdated}
                  onOpenThread={handleOpenThreadInChat}
                  onOpenTrace={handleOpenTrace}
                  onOpenCanvas={handleOpenCanvas}
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
        {chunkInspectorTab && chunkInspectorTarget ? (
          <DocumentChunkInspectorDialog
            open
            onClose={() => setChunkInspectorTab(null)}
            fileHash={chunkInspectorTab.fileHash}
            fileName={chunkInspectorTab.fileName}
            scope={chunkInspectorTarget.scope}
            scopeId={chunkInspectorTarget.id}
          />
        ) : null}
      </Box>
      </ThreadChatSettingsProvider>
    </ThemeProvider>
  );
}
