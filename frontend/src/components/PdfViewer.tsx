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
import BorderColorIcon from "@mui/icons-material/BorderColor";
import DrawIcon from "@mui/icons-material/Draw";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import ViewSidebarIcon from "@mui/icons-material/ViewSidebar";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import GestureIcon from "@mui/icons-material/Gesture";
import DeleteIcon from "@mui/icons-material/Delete";
import UndoIcon from "@mui/icons-material/Undo";
import RedoIcon from "@mui/icons-material/Redo";
import CircleIcon from "@mui/icons-material/Circle";
import AddCommentIcon from "@mui/icons-material/AddComment";
import PanToolAltIcon from "@mui/icons-material/PanToolAlt";
import EditNoteIcon from "@mui/icons-material/EditNote";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { Slider } from "@mui/material";
import { useHistoryCapability } from "@embedpdf/plugin-history/react";
import { PdfSidebar, type SidebarTab } from "./PdfSidebar";
import { usePersistAnnotations } from "../hooks/usePersistAnnotations";

const MARKUP_TOOLS = ["highlight", "underline", "strikeout", "squiggly"];
const SHAPE_TOOLS = ["ink", "line", "square", "circle"];
const VIEW_ONLY_MODE_ID = "view-only";

const STANDARD_COLORS = [
  "#ffeb3b", // Yellow
  "#4caf50", // Green
  "#2196f3", // Blue
  "#f44336", // Red
  "#e91e63", // Pink
  "#ff9800", // Orange
  "#00bcd4", // Cyan
  "#9c27b0", // Purple
];

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



function AnnotationToolStrip({
  documentId,
  showSidebar,
  onToggleSidebar,
  onOpenComments,
  isHistoryProcessingRef,
}: {
  documentId: string;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  onOpenComments: () => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
}) {
  const { provides } = useAnnotation(documentId);
  const { provides: annotationCapability } = useAnnotationCapability();
  const { provides: interactionCapability } = useInteractionManagerCapability();
  const { provides: selectionCapability } = useSelectionCapability();
  const { provides: interactionScope } = useInteractionManager(documentId);
  const { provides: history } = useHistoryCapability();
  const active = provides?.getActiveTool();
  const activeId = active && "id" in active ? (active as { id: string }).id : null;
  const defaultModeId = interactionCapability?.getDefaultMode() ?? "default";
  const activeModeId =
    interactionScope?.getActiveInteractionMode()?.id ?? defaultModeId;
  const isViewOnly = activeModeId === VIEW_ONLY_MODE_ID;

  // Per-category settings state
  const [markupSettings, setMarkupSettings] = useState({
    strokeColor: "#ffeb3b",
    opacity: 0.3,
  });
  const [shapeSettings, setShapeSettings] = useState({
    strokeColor: "#f44336",
    strokeWidth: 2,
    opacity: 1,
  });

  // Sync settings with tool defaults whenever they change or tool is activated
  useEffect(() => {
    if (!annotationCapability || !activeId) return;

    if (MARKUP_TOOLS.includes(activeId)) {
      annotationCapability.setToolDefaults(activeId, markupSettings);
    } else if (SHAPE_TOOLS.includes(activeId)) {
      annotationCapability.setToolDefaults(activeId, shapeSettings);
    }
  }, [annotationCapability, activeId, markupSettings, shapeSettings]);

  useEffect(() => {
    if (!interactionCapability) return;
    interactionCapability.registerMode({
      id: VIEW_ONLY_MODE_ID,
      scope: "page",
      exclusive: false,
      cursor: "default",
    });
  }, [interactionCapability]);

  useEffect(() => {
    if (!selectionCapability) return;

    selectionCapability.enableForMode(
      defaultModeId,
      {
        enableSelection: true,
        showSelectionRects: true,
        enableMarquee: false,
      },
      documentId
    );

    selectionCapability.enableForMode(
      VIEW_ONLY_MODE_ID,
      {
        enableSelection: false,
        showSelectionRects: false,
        enableMarquee: false,
      },
      documentId
    );
  }, [defaultModeId, documentId, selectionCapability]);

  const clearCurrentSelection = useCallback(() => {
    selectionCapability?.clear(documentId);
    provides?.deselectAnnotation();
  }, [documentId, provides, selectionCapability]);

  const activateSelectMode = useCallback(() => {
    interactionScope?.activateDefaultMode();
    clearCurrentSelection();
  }, [clearCurrentSelection, interactionScope]);

  const activateViewOnlyMode = useCallback(() => {
    interactionScope?.activate(VIEW_ONLY_MODE_ID);
    clearCurrentSelection();
  }, [clearCurrentSelection, interactionScope]);

  useEffect(() => {
    if (!annotationCapability) return;
    annotationCapability.setLocked({
      type: isViewOnly ? LockModeType.All : LockModeType.None,
    });
  }, [annotationCapability, isViewOnly]);

  useEffect(() => {
    if (!activeId || !isViewOnly) return;
    activateSelectMode();
  }, [activeId, activateSelectMode, isViewOnly]);

  useEffect(() => {
    if (activeId || isViewOnly) return;
    interactionScope?.activateDefaultMode();
  }, [activeId, interactionScope, isViewOnly]);

  const tool = (id: string | null, icon: React.ReactNode, title: string) => (
    <Tooltip title={title} key={title + (id || "select")}>
      <IconButton
        size="small"
        onClick={() => provides?.setActiveTool(id)}
        color={activeId === id ? "primary" : "default"}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );

  const isMarkup = activeId && MARKUP_TOOLS.includes(activeId);
  const isShape = activeId && SHAPE_TOOLS.includes(activeId);
  const currentSettings = isMarkup ? markupSettings : isShape ? shapeSettings : null;
  const rawSetSettings = isMarkup ? setMarkupSettings : isShape ? setShapeSettings : null;

  const handleSelectToggle = useCallback(() => {
    if (activeId) {
      provides?.setActiveTool(null);
      activateSelectMode();
      return;
    }

    if (isViewOnly) {
      activateSelectMode();
      return;
    }

    activateViewOnlyMode();
  }, [activeId, activateSelectMode, activateViewOnlyMode, isViewOnly, provides]);

  const updateSelection = useCallback(
    (patch: Partial<PdfAnnotationObject>) => {
      const selectedIds = provides?.getSelectedAnnotationIds() || [];
      if (selectedIds.length === 0 || !annotationCapability) return;

      const selectedAnnotations = provides?.getSelectedAnnotations() || [];
      const patches = selectedAnnotations
        .map((ta) => {
          return {
            pageIndex: ta.object.pageIndex,
            id: ta.object.id,
            patch,
          };
        });

      if (patches.length > 0) {
        annotationCapability.updateAnnotations(patches);
      }
    },
    [provides, annotationCapability]
  );

  const setSettings = useCallback(
    (updater: (prev: any) => any) => {
      if (!rawSetSettings) return;
      rawSetSettings((prev: any) => {
        const next = updater(prev);
        // Compare keys to find what changed
        const patch: any = {};
        for (const key in next) {
          if (next[key] !== prev[key]) {
            patch[key] = next[key];
          }
        }
        if (Object.keys(patch).length > 0) {
          updateSelection(patch);
        }
        return next;
      });
    },
    [rawSetSettings, updateSelection]
  );

  const handleUndo = useCallback(() => {
    if (isHistoryProcessingRef) isHistoryProcessingRef.current = true;
    history?.undo();
    setTimeout(() => {
      if (isHistoryProcessingRef) isHistoryProcessingRef.current = false;
    }, 100);
  }, [history, isHistoryProcessingRef]);

  const handleRedo = useCallback(() => {
    if (isHistoryProcessingRef) isHistoryProcessingRef.current = true;
    history?.redo();
    setTimeout(() => {
      if (isHistoryProcessingRef) isHistoryProcessingRef.current = false;
    }, 100);
  }, [history, isHistoryProcessingRef]);

  return (
    <Stack direction="column" sx={{ width: "100%" }}>
      <Stack
        direction="row"
        spacing={0.5}
        alignItems="center"
        sx={{
          flexShrink: 0,
          px: 1,
          py: 0.5,
          borderBottom: 1,
          borderColor: "divider",
          bgcolor: "background.paper",
          width: "100%",
          justifyContent: "space-between",
        }}
      >
        <Stack direction="row" spacing={0.5} alignItems="center">
          <Tooltip title="Toggle Sidebar">
            <IconButton onClick={onToggleSidebar} size="small" sx={{ mr: 0.5 }}>
              <ViewSidebarIcon
                fontSize="small"
                color={showSidebar ? "primary" : "inherit"}
              />
            </IconButton>
          </Tooltip>
          <Tooltip title="Comments">
            <IconButton onClick={onOpenComments} size="small" sx={{ mr: 0.5 }}>
              <AddCommentIcon fontSize="small" color="inherit" />
            </IconButton>
          </Tooltip>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          <Tooltip title={isViewOnly ? "Select" : "View only"}>
            <IconButton
              size="small"
              onClick={handleSelectToggle}
              color={isViewOnly ? "primary" : "default"}
            >
              {isViewOnly ? <EditNoteIcon fontSize="small" /> : <PanToolAltIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          {/* Markup Tools */}
          {tool("highlight", <BorderColorIcon fontSize="small" />, "Highlight")}
          {tool("underline", <FormatUnderlinedIcon fontSize="small" />, "Underline")}
          {tool(
            "strikeout",
            <StrikethroughSIcon fontSize="small" />,
            "Strikeout"
          )}
          {tool(
            "squiggly",
            <GestureIcon
              fontSize="small"
              sx={{ transform: "rotate(90deg)" }}
            />,
            "Squiggly"
          )}

          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          {/* Shape Tools */}
          {tool("ink", <DrawIcon fontSize="small" />, "Draw")}
          {tool("line", <GestureIcon fontSize="small" />, "Line")}
          {tool("square", <CropSquareIcon fontSize="small" />, "Rectangle")}
          {tool(
            "circle",
            <RadioButtonUncheckedIcon fontSize="small" />,
            "Ellipse"
          )}
        </Stack>

        <Stack direction="row" spacing={0.5} alignItems="center">
          <Tooltip title="Undo">
            <span>
              <IconButton
                size="small"
                onClick={handleUndo}
                disabled={!history?.canUndo()}
              >
                <UndoIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Redo">
            <span>
              <IconButton
                size="small"
                onClick={handleRedo}
                disabled={!history?.canRedo()}
              >
                <RedoIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
        </Stack>
      </Stack>

      {(isMarkup || isShape) && currentSettings && setSettings && (
        <Stack
          direction="row"
          spacing={2}
          alignItems="center"
          sx={{
            px: 2,
            py: 0.5,
            borderBottom: 1,
            borderColor: "divider",
            bgcolor: "background.default",
            height: 40,
          }}
        >
          <Stack direction="row" spacing={0.5}>
            {STANDARD_COLORS.map((color) => (
              <IconButton
                key={color}
                size="small"
                onClick={() =>
                  setSettings((prev: any) => ({ ...prev, strokeColor: color }))
                }
                sx={{
                  p: 0.25,
                  border: "2px solid",
                  borderColor:
                    currentSettings.strokeColor === color
                      ? "primary.main"
                      : "transparent",
                }}
              >
                <CircleIcon sx={{ color, fontSize: 18 }} />
              </IconButton>
            ))}
          </Stack>

          {isShape && (
            <Stack direction="row" spacing={1} alignItems="center" sx={{ width: 150 }}>
              <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
                Width: {(currentSettings as any).strokeWidth}px
              </Typography>
              <Slider
                size="small"
                value={(currentSettings as any).strokeWidth}
                min={1}
                max={12}
                onChange={(_, val) =>
                  setSettings((prev: any) => ({ ...prev, strokeWidth: val as number }))
                }
              />
            </Stack>
          )}

          <Stack direction="row" spacing={1} alignItems="center" sx={{ width: 120 }}>
            <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
              Opacity: {Math.round(currentSettings.opacity * 100)}%
            </Typography>
            <Slider
              size="small"
              value={currentSettings.opacity}
              min={0.1}
              max={1}
              step={0.1}
              onChange={(_, val) =>
                setSettings((prev: any) => ({ ...prev, opacity: val as number }))
              }
            />
          </Stack>
        </Stack>
      )}
    </Stack>
  );
}

function AnnotationSelectionMenu({
  selected,
  context,
  menuWrapperProps,
  rect,
  documentId,
  onOpenComments,
}: AnnotationSelectionMenuProps & { documentId: string; onOpenComments: () => void }) {
  const { provides: annotationCapability } = useAnnotationCapability();

  const handleDelete = () => {
    if (!context?.annotation?.object) return;
    const { pageIndex, id } = context.annotation.object;
    annotationCapability?.forDocument(documentId).deleteAnnotation(pageIndex, id);
  };

  if (!selected || !context?.annotation) return null;

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
        <Tooltip title="Comment">
          <IconButton
            size="small"
            onClick={onOpenComments}
            sx={{
              bgcolor: "background.paper",
              boxShadow: 1,
              "&:hover": { bgcolor: "background.default" },
            }}
          >
            <AddCommentIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Delete">
          <IconButton
            size="small"
            color="error"
            onClick={handleDelete}
            sx={{
              bgcolor: "background.paper",
              boxShadow: 1,
              "&:hover": { bgcolor: "background.default" },
            }}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </div>
  );
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
  const [showSidebar, setShowSidebar] = useState(true);
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

  const isCommittedMutationEvent = useCallback((event: any) => {
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
        scrollToAnnotation((event as any).annotation);
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

            <AnnotationToolStrip
              documentId={documentId}
              showSidebar={showSidebar}
              onToggleSidebar={() => setShowSidebar((value) => !value)}
              onOpenComments={openCommentsPane}
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
