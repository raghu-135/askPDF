import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPluginRegistration } from "@embedpdf/core";
import { EmbedPDF, type PDFContextState } from "@embedpdf/core/react";
import { PdfAnnotationObject } from "@embedpdf/models";
import { usePdfiumEngine } from "@embedpdf/engines/react";

type AnnotationEvent = {
  type: string;
  committed?: boolean;
  annotation?: PdfAnnotationObject;
};
import {
  DocumentContent,
  DocumentManagerPluginPackage,
} from "@embedpdf/plugin-document-manager/react";
import {
  Viewport,
  ViewportPluginPackage,
} from "@embedpdf/plugin-viewport/react";
import {
  Scroller,
  ScrollPluginPackage,
  ScrollStrategy,
  useScroll,
} from "@embedpdf/plugin-scroll/react";
import {
  RenderLayer,
  RenderPluginPackage,
} from "@embedpdf/plugin-render/react";
import {
  ZoomPluginPackage,
  ZoomMode,
  useZoom,
} from "@embedpdf/plugin-zoom/react";
import {
  InteractionManagerPluginPackage,
  PagePointerProvider,
} from "@embedpdf/plugin-interaction-manager/react";
import {
  SelectionLayer,
  SelectionPluginPackage,
  type SelectionSelectionMenuProps,
} from "@embedpdf/plugin-selection/react";
import {
  AnnotationLayer,
  AnnotationPluginPackage,
  useAnnotation,
  useAnnotationCapability,
  LockModeType,
  type AnnotationSelectionMenuProps,
} from "@embedpdf/plugin-annotation/react";
import { HistoryPluginPackage } from "@embedpdf/plugin-history";
import { ThumbnailPluginPackage } from "@embedpdf/plugin-thumbnail/react";
import {
  useInteractionManager,
  useInteractionManagerCapability,
} from "@embedpdf/plugin-interaction-manager/react";
import { useSelectionCapability } from "@embedpdf/plugin-selection/react";
import {
  Box,
  IconButton,
  Stack,
  Tooltip,
  Typography,
  Divider,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { useHistoryCapability } from "@embedpdf/plugin-history/react";
import { PdfSidebar, type SidebarTab } from "./PdfSidebar";
import { usePersistAnnotations } from "../hooks/usePersistAnnotations";
import { AnnotationToolbar } from "./annotation/AnnotationToolbar";
import { AnnotationSelectionMenu } from "./annotation/AnnotationSelectionMenu";
import { STANDARD_COLORS } from "./annotation/constants";
import AddCommentIcon from "@mui/icons-material/AddComment";
import DeleteIcon from "@mui/icons-material/Delete";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";


type BBox = {
  page: number;
  x: number;
  y: number;
  width: number;
  height: number;
  page_height: number;
  page_width: number;
};

type Sentence = {
  id: number;
  text: string;
  bboxes: BBox[];
};

type Props = {
  pdfUrl: string;
  sentences: Sentence[];
  currentId: number | null;
  onJump: (id: number) => void;
  autoScroll: boolean;
  isResizing?: boolean;
  highlightEnabled?: boolean;
  darkMode?: boolean;
  threadId?: string | null;
  fileHash?: string | null;
};

// Memoize tool configuration outside buildPlugins to prevent recreation
const ANNOTATION_TOOLS = [
  {
    id: 'highlight',
    defaults: {
      strokeColor: '#ffeb3b',
      opacity: 0.3,
    },
  },
  {
    id: 'underline',
    defaults: {
      strokeColor: '#2196f3',
      opacity: 1,
    },
  },
  {
    id: 'strikeout',
    defaults: {
      strokeColor: '#f44336',
      opacity: 1,
    },
  },
  {
    id: 'squiggly',
    defaults: {
      strokeColor: '#ff9800',
      opacity: 1,
    },
  },
  {
    id: 'ink',
    defaults: {
      strokeColor: '#f44336',
      strokeWidth: 2,
      opacity: 1,
    },
  },
  {
    id: 'line',
    defaults: {
      strokeColor: '#f44336',
      strokeWidth: 2,
      opacity: 1,
    },
  },
  {
    id: 'square',
    defaults: {
      strokeColor: '#f44336',
      strokeWidth: 2,
      opacity: 1,
    },
  },
  {
    id: 'circle',
    defaults: {
      strokeColor: '#f44336',
      strokeWidth: 2,
      opacity: 1,
    },
  },
];

function buildPlugins(pdfUrl: string) {
  return [
    createPluginRegistration(DocumentManagerPluginPackage, {
      initialDocuments: [{ url: pdfUrl, autoActivate: true }],
    }),
    createPluginRegistration(ViewportPluginPackage, {
      viewportGap: 0,
    }),
    createPluginRegistration(ScrollPluginPackage, {
      defaultStrategy: ScrollStrategy.Vertical,
      defaultPageGap: 1,
    }),
    createPluginRegistration(RenderPluginPackage),
    createPluginRegistration(ZoomPluginPackage, {
      defaultZoomLevel: ZoomMode.FitWidth,
    }),
    createPluginRegistration(InteractionManagerPluginPackage),
    createPluginRegistration(SelectionPluginPackage),
    createPluginRegistration(HistoryPluginPackage),
    createPluginRegistration(AnnotationPluginPackage, {
      annotationAuthor: "AskPDF",
      colorPresets: [...STANDARD_COLORS],
      tools: ANNOTATION_TOOLS,
    }),
    createPluginRegistration(ThumbnailPluginPackage),
  ];
}

function sentencesByPageMap(sentences: Sentence[]) {
  const map: { [key: number]: (Sentence & { pageBBoxes: BBox[] })[] } = {};
  if (!sentences) return map;
  sentences.forEach((s) => {
    if (!s.bboxes) return;
    s.bboxes.forEach((b) => {
      if (!map[b.page]) map[b.page] = [];
      let entry = map[b.page].find((e) => e.id === s.id);
      if (!entry) {
        entry = { ...s, pageBBoxes: [] };
        map[b.page].push(entry);
      }
      entry.pageBBoxes.push(b);
    });
  });
  return map;
}

function SelectionCopyMenu({
  selected,
  menuWrapperProps,
  rect,
  documentId,
}: SelectionSelectionMenuProps & { documentId: string }) {
  const { provides: selectionCapability } = useSelectionCapability();

  if (!selected) return null;

  return (
    <div {...menuWrapperProps} style={{ ...menuWrapperProps?.style, zIndex: 100 }}>
      <Box
        sx={{
          position: "absolute",
          top: rect.size.height + 8,
          left: 0,
          pointerEvents: "auto",
          minWidth: "max-content",
          display: "flex",
          gap: 0.5,
        }}
      >
        <Tooltip title="Copy selected text">
          <IconButton
            size="small"
            onClick={() => selectionCapability?.copyToClipboard(documentId)}
            sx={{
              bgcolor: "background.paper",
              boxShadow: 1,
              "&:hover": { bgcolor: "background.default" },
            }}
          >
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </div>
  );
}

function DocumentLoadedSync({
  isLoaded,
  setPdfLoaded,
}: {
  isLoaded: boolean;
  setPdfLoaded: (v: boolean) => void;
}) {
  useEffect(() => {
    if (!isLoaded) {
      setPdfLoaded(false);
      return;
    }
    const t = window.setTimeout(() => setPdfLoaded(true), 150);
    return () => window.clearTimeout(t);
  }, [isLoaded, setPdfLoaded]);
  return null;
}

function EmbedPdfDocumentBody({
  documentId,
  pdfUrl: _pdfUrl,
  sentences,
  currentId,
  onJump,
  autoScroll,
  isResizing,
  highlightEnabled,
  darkMode,
  threadId,
  fileHash,
  pdfLoaded,
  setPdfLoaded,
  isHistoryProcessingRef,
}: Props & {
  documentId: string;
  pdfLoaded: boolean;
  setPdfLoaded: (v: boolean) => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
}) {
  const theme = useTheme();
  const [showSidebar, setShowSidebar] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("thumbnails");
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const [commentComposerRequest, setCommentComposerRequest] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const sentenceRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  const { provides: zoomScope } = useZoom(documentId);
  const zoomRef = useRef(zoomScope);
  zoomRef.current = zoomScope;
  const isResizingRef = useRef(isResizing);
  isResizingRef.current = isResizing;

  const byPage = useMemo(() => sentencesByPageMap(sentences), [sentences]);

  const { provides: annotationApi } = useAnnotation(documentId);
  const { provides: selectionCapability } = useSelectionCapability();
  const { provides: scrollApi } = useScroll(documentId);
  const scrollRef = useRef(scrollApi);
  scrollRef.current = scrollApi;
  const pendingScrollIdRef = useRef<number | null>(null);
  const { provides: historyApi } = useHistoryCapability();
  const historyRef = useRef({ provides: historyApi });
  historyRef.current = { provides: historyApi };
  const {
    hydrateAnnotationsRef,
    loadPersistedAnnotations,
    schedulePersistAnnotations,
  } = usePersistAnnotations({
    annotationApi,
    threadId,
    fileHash,
  });

  useEffect(() => {
    void loadPersistedAnnotations();
  }, [loadPersistedAnnotations]);

  
  useEffect(() => {
    const savedWidth = window.localStorage.getItem("askpdf.pdfSidebarWidth");
    const parsedWidth = savedWidth ? Number(savedWidth) : NaN;
    if (Number.isFinite(parsedWidth) && parsedWidth > 0) {
      setSidebarWidth(Math.min(520, Math.max(220, parsedWidth)));
    }

    const savedTab = window.localStorage.getItem("askpdf.pdfSidebarTab");
    if (savedTab === "comments" || savedTab === "thumbnails") {
      setSidebarTab(savedTab);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem("askpdf.pdfSidebarWidth", String(sidebarWidth));
  }, [sidebarWidth]);

  useEffect(() => {
    window.localStorage.setItem("askpdf.pdfSidebarTab", sidebarTab);
  }, [sidebarTab]);

  const openCommentsPane = useCallback(() => {
    setShowSidebar(true);
    setSidebarTab("comments");
    setCommentComposerRequest((value) => value + 1);
  }, []);

  useEffect(() => {
    // Priority: Don't let TTS scroll fight with a manual history undo/redo
    if (isHistoryProcessingRef.current) return;

    if (!autoScroll || currentId === null) {
      pendingScrollIdRef.current = null;
      return;
    }
    const s = sentences[currentId];
    if (!s?.bboxes?.length) return;

    const el = sentenceRefs.current[currentId];
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      scrollRef.current?.scrollToPage({
        pageNumber: s.bboxes[0].page,
        behavior: "smooth",
      });
      pendingScrollIdRef.current = currentId;
    }
  }, [currentId, autoScroll, sentences, isHistoryProcessingRef]);

  const scrollToAnnotation = useCallback(
    (anno: PdfAnnotationObject) => {
      if (!scrollApi) return;
      scrollApi.scrollToPage({
        pageNumber: anno.pageIndex + 1,
        pageCoordinates: {
          x: anno.rect.origin.x,
          y: anno.rect.origin.y,
        },
        alignY: 50,
        behavior: "smooth",
      });
    },
    [scrollApi]
  );

  const isCommittedMutationEvent = useCallback((event: AnnotationEvent) => {
    return (
      ["create", "update", "delete"].includes(event.type) &&
      Boolean(event.committed)
    );
  }, []);

  // Persistence / DB Sync & Auto-scroll on all committed changes (Undo/Redo/etc)
  useEffect(() => {
    if (!annotationApi) return;
    const sub = annotationApi.onAnnotationEvent((event) => {
      if (event.type === "loaded") return;

      // 1. Persistence logging
      if (isCommittedMutationEvent(event)) {
        if (!hydrateAnnotationsRef.current) {
          schedulePersistAnnotations();
        }
      }

      // 2. Auto-scroll on Undo/Redo
      // Only scroll during history actions to avoid locking the scroll during normal interaction.
      if (isCommittedMutationEvent(event) && isHistoryProcessingRef.current) {
        if (event.annotation) {
          scrollToAnnotation(event.annotation);
        }
      }
    });
    return () => sub();
  }, [
    annotationApi,
    documentId,
    isCommittedMutationEvent,
    schedulePersistAnnotations,
    scrollToAnnotation,
  ]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "c") {
        const target = e.target as HTMLElement;
        if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) {
          return;
        }

        const selectionState = selectionCapability?.getState(documentId);
        if (selectionState?.selection) {
          e.preventDefault();
          selectionCapability?.copyToClipboard(documentId);
        }
        return;
      }

      // Handle Delete/Backspace
      if (e.key === "Delete" || e.key === "Backspace") {
        const target = e.target as HTMLElement;
        if (target && ["INPUT", "TEXTAREA"].includes(target.tagName)) {
          return;
        }
        const selection = annotationApi?.getSelectedAnnotations() || [];
        if (selection.length > 0) {
          e.preventDefault();
          annotationApi?.deleteAnnotations(
            selection.map((s) => ({
              pageIndex: s.object.pageIndex,
              id: s.object.id,
            }))
          );
        }
        return;
      }

      // Handle Undo/Redo shortcuts (Cmd+Z / Cmd+Shift+Z)
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        const { provides: history } = historyRef.current;
        if (!history) return;

        e.preventDefault();
        isHistoryProcessingRef.current = true;
        
        if (e.shiftKey) {
          if (history.canRedo()) history.redo();
        } else {
          if (history.canUndo()) history.undo();
        }

        setTimeout(() => {
          isHistoryProcessingRef.current = false;
        }, 150);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [annotationApi, documentId, selectionCapability]);

  const handlePageContextMenuCapture = useCallback(
    (e: React.MouseEvent) => {
      const selectionState = selectionCapability?.getState(documentId);
      if (!selectionState?.selection) return;

      e.preventDefault();
      selectionCapability?.copyToClipboard(documentId);
    },
    [documentId, selectionCapability]
  );

  /** Panel split drag ended — refit once. Do not depend on `zoomScope` identity (it changes every render and would refit every frame → scroll flicker). */
  useEffect(() => {
    if (isResizing) return;
    zoomRef.current?.requestZoom(ZoomMode.FitWidth);
  }, [isResizing]);

  /** Refit when the viewer container width changes (window/panel resize), not on every scroll. */
  useEffect(() => {
    const el = containerRef.current;
    if (typeof ResizeObserver === "undefined" || !el) return;

    let lastWidth = Math.round(el.getBoundingClientRect().width);
    let rafId: number | null = null;

    const scheduleFit = () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = null;
        zoomRef.current?.requestZoom(ZoomMode.FitWidth);
      });
    };

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const w = Math.round(entry.contentRect.width);
      if (Math.abs(w - lastWidth) < 2) return;
      lastWidth = w;
      scheduleFit();
    });

    ro.observe(el);
    return () => {
      ro.disconnect();
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [documentId]);



  const handlePageDoubleClickCapture = useCallback(
    (e: React.MouseEvent, pageNumber: number) => {
      if (e.detail !== 2) return;
      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;

      const pageData = byPage[pageNumber] || [];
      for (const sentence of pageData) {
        for (const bbox of sentence.pageBBoxes) {
          const bLeft = bbox.x / bbox.page_width;
          const bTop =
            (bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height;
          const bWidth = bbox.width / bbox.page_width;
          const bHeight = bbox.height / bbox.page_height;

          if (
            x >= bLeft &&
            x <= bLeft + bWidth &&
            y >= bTop &&
            y <= bTop + bHeight
          ) {
            e.preventDefault();
            e.stopPropagation();
            onJump(sentence.id);
            return;
          }
        }
      }
    },
    [byPage, onJump],
  );

  const handlePageClick = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement;
      
      // Don't deselect if clicking on obvious UI elements
      if (
        target.tagName === 'BUTTON' ||
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'A' ||
        target.closest('button, input, textarea, a, [role="button"]')
      ) {
        return;
      }

      // If clicking on canvas, svg, or img (the actual PDF content), deselect annotation
      // This is the most reliable way to detect clicks on the page background
      if (
        target.tagName === 'CANVAS' ||
        target.tagName === 'SVG' ||
        target.tagName === 'IMG' ||
        target.closest('canvas, svg, img')
      ) {
        annotationApi?.deselectAnnotation();
      }
    },
    [annotationApi]
  );

  const renderPage = useCallback(
    (props: {
      pageIndex: number;
      width: number;
      height: number;
    }) => {
      const { pageIndex, width, height } = props;
      const zoomState = zoomRef.current?.getState();
      const scale = zoomState?.currentZoomLevel || 1;
      const pageNumber = pageIndex + 1;

      if (!highlightEnabled) {
        return (
          <div
            style={{
              position: "relative",
              width,
              height,
              marginBottom: 0,
            }}
            onDoubleClickCapture={(e) =>
              handlePageDoubleClickCapture(e, pageNumber)
            }
            onContextMenuCapture={handlePageContextMenuCapture}
            onClick={handlePageClick}
          >
            <PagePointerProvider
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
            >
              <RenderLayer
                documentId={documentId}
                pageIndex={pageIndex}
                scale={scale}
              />
              <SelectionLayer
                documentId={documentId}
                pageIndex={pageIndex}
                scale={scale}
                selectionMenu={(props) => (
                  <SelectionCopyMenu {...props} documentId={documentId} />
                )}
              />
              <AnnotationLayer
                documentId={documentId}
                pageIndex={pageIndex}
                scale={scale}
                selectionMenu={(props) => (
                  <AnnotationSelectionMenu
                    {...props}
                    documentId={documentId}
                    onOpenComments={openCommentsPane}
                  />
                )}
              />
            </PagePointerProvider>
          </div>
        );
      }

      const pageData = byPage[pageNumber] || [];
      const activeSentence = pageData.find((s) => s.id === currentId);

      return (
        <div
          style={{
            position: "relative",
            width,
            height,
            marginBottom: 0,
          }}
          onDoubleClickCapture={(e) =>
            handlePageDoubleClickCapture(e, pageNumber)
          }
          onContextMenuCapture={handlePageContextMenuCapture}
          onClick={handlePageClick}
        >
          <PagePointerProvider
            documentId={documentId}
            pageIndex={pageIndex}
            scale={scale}
          >
            <RenderLayer
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
            />
            <SelectionLayer
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
              selectionMenu={(props) => (
                <SelectionCopyMenu {...props} documentId={documentId} />
              )}
            />
            <AnnotationLayer
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
              selectionMenu={(props) => (
                <AnnotationSelectionMenu
                  {...props}
                  documentId={documentId}
                  onOpenComments={openCommentsPane}
                />
              )}
            />
            {activeSentence ? (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  pointerEvents: "none",
                  zIndex: 25,
                }}
              >
                {activeSentence.pageBBoxes.map((bbox, idx) => (
                  <div
                    key={idx}
                    ref={(el) => {
                      if (idx === 0) {
                        sentenceRefs.current[activeSentence.id] = el;
                        
                        if (el && pendingScrollIdRef.current === activeSentence.id) {
                          pendingScrollIdRef.current = null;
                          requestAnimationFrame(() => {
                            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                          });
                        }
                      }
                    }}
                    style={{
                      position: "absolute",
                      left: `${(bbox.x / bbox.page_width) * 100}%`,
                      top: `${((bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height) * 100}%`,
                      width: `${(bbox.width / bbox.page_width) * 100}%`,
                      height: `${(bbox.height / bbox.page_height) * 100}%`,
                      backgroundColor: "rgba(0, 180, 255, 0.28)",
                    }}
                  />
                ))}
              </div>
            ) : null}
          </PagePointerProvider>
        </div>
      );
    },
    [
      documentId,
      byPage,
      currentId,
      highlightEnabled,
      handlePageContextMenuCapture,
      handlePageDoubleClickCapture,
      handlePageClick,
      openCommentsPane,
    ],
  );

  const viewportBg = darkMode ? theme.palette.background.default : "#f1f3f5";

  return (
    <DocumentContent documentId={documentId}>
      {({ isLoaded }) => {
        if (!isLoaded) {
          return (
            <Box sx={{ p: 2 }}>
              <Typography>Loading PDF...</Typography>
            </Box>
          );
        }
        return (
          <>
            <DocumentLoadedSync
              isLoaded={isLoaded}
              setPdfLoaded={setPdfLoaded}
            />

            <AnnotationToolbar
              documentId={documentId}
              showSidebar={showSidebar}
              onToggleSidebar={() => setShowSidebar((value) => !value)}
              isHistoryProcessingRef={isHistoryProcessingRef}
            />

            <Box
              sx={{
                display: "flex",
                flex: 1,
                minHeight: 0,
                width: "100%",
                overflow: "hidden",
              }}
            >
              {showSidebar && (
                <PdfSidebar
                  documentId={documentId}
                  width={sidebarWidth}
                  activeTab={sidebarTab}
                  onTabChange={setSidebarTab}
                  onWidthChange={setSidebarWidth}
                  onToggleSidebar={() => setShowSidebar(false)}
                  commentComposerRequest={commentComposerRequest}
                />
              )}
              <Box
                ref={containerRef}
                onDragStart={(e) => e.preventDefault()}
                sx={{
                  position: "relative",
                  flex: 1,
                  minHeight: 0,
                  minWidth: 0,
                  display: "flex",
                  flexDirection: "column",
                  "& img, & svg, & canvas": {
                    WebkitUserDrag: "none",
                    userDrag: "none",
                  },
                }}
              >
                {darkMode && !pdfLoaded && (
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      zIndex: 100,
                      bgcolor: theme.palette.background.default,
                      color: theme.palette.text.primary,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexDirection: "column",
                    }}
                  >
                    <Typography variant="h6" sx={{ mb: 2 }}>
                      Rendering PDF...
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Preparing dark mode view
                    </Typography>
                  </Box>
                )}
                <Box
                  sx={{
                    flex: 1,
                    minHeight: 0,
                    position: "relative",
                    filter:
                      darkMode && pdfLoaded
                        ? "invert(0.92) hue-rotate(180deg) brightness(0.85)"
                        : undefined,
                    /** Own compositor layer to reduce full-layer repaints while scrolling with CSS filter. */
                    transform:
                      darkMode && pdfLoaded ? "translateZ(0)" : undefined,
                    visibility: darkMode && !pdfLoaded ? "hidden" : "visible",
                  }}
                >
                  <Viewport
                    documentId={documentId}
                    style={{
                      height: "100%",
                      width: "100%",
                      backgroundColor: viewportBg,
                    }}
                  >
                    <Scroller documentId={documentId} renderPage={renderPage} />
                  </Viewport>
                </Box>
              </Box>
            </Box>
          </>
        );
      }}
    </DocumentContent>
  );
}

const PdfViewer = React.memo(function PdfViewer({
  pdfUrl,
  sentences,
  currentId,
  onJump,
  autoScroll,
  isResizing,
  highlightEnabled = true,
  darkMode = false,
  threadId = null,
  fileHash = null,
}: Props) {
  const theme = useTheme();
  const { engine, isLoading, error } = usePdfiumEngine();
  const plugins = useMemo(() => buildPlugins(pdfUrl), [pdfUrl]);
  const [pdfLoaded, setPdfLoaded] = useState(false);
  const isHistoryProcessingRef = useRef(false);

  useEffect(() => {
    setPdfLoaded(false);
  }, [pdfUrl]);

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">Failed to initialize PDF engine.</Typography>
      </Box>
    );
  }

  if (isLoading || !engine) {
    return (
      <Box
        sx={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: darkMode ? theme.palette.background.default : "transparent",
        }}
      >
        <Typography sx={{ p: 2 }}>Loading PDF engine...</Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        height: "100%",
        width: "100%",
        display: "flex",
        flexDirection: "column",
        alignItems: "stretch",
        bgcolor: darkMode ? theme.palette.background.default : "transparent",
        p: 0,
        m: 0,
        boxSizing: "border-box",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <EmbedPDF key={pdfUrl} engine={engine} plugins={plugins}>
        {(ctx: PDFContextState) =>
          ctx.activeDocumentId ? (
            <EmbedPdfDocumentBody
              documentId={ctx.activeDocumentId}
              pdfUrl={pdfUrl}
              sentences={sentences}
              currentId={currentId}
              onJump={onJump}
              autoScroll={autoScroll}
              isResizing={isResizing}
              highlightEnabled={highlightEnabled}
              darkMode={darkMode}
              threadId={threadId}
              fileHash={fileHash}
              pdfLoaded={pdfLoaded}
              setPdfLoaded={setPdfLoaded}
              isHistoryProcessingRef={isHistoryProcessingRef}
            />
          ) : null
        }
      </EmbedPDF>
    </Box>
  );
});

export default PdfViewer;
