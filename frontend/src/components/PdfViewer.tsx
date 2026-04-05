import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPluginRegistration } from "@embedpdf/core";
import { EmbedPDF, type PDFContextState } from "@embedpdf/core/react";
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
} from "@embedpdf/plugin-selection/react";
import {
  AnnotationLayer,
  AnnotationPluginPackage,
  useAnnotation,
} from "@embedpdf/plugin-annotation/react";
import { HistoryPluginPackage } from "@embedpdf/plugin-history";
import { Box, IconButton, Stack, Tooltip, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import BorderColorIcon from "@mui/icons-material/BorderColor";
import DrawIcon from "@mui/icons-material/Draw";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import NearMeIcon from "@mui/icons-material/NearMe";

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

function TtsScrollSync({
  documentId,
  currentId,
  sentences,
  autoScroll,
}: {
  documentId: string;
  currentId: number | null;
  sentences: Sentence[];
  autoScroll: boolean;
}) {
  const { provides: scroll } = useScroll(documentId);
  const scrollRef = useRef(scroll);
  scrollRef.current = scroll;

  useEffect(() => {
    if (!autoScroll || currentId === null) return;
    const s = sentences[currentId];
    if (!s?.bboxes?.length) return;
    const page = s.bboxes[0].page;
    scrollRef.current?.scrollToPage({
      pageNumber: page,
      behavior: "smooth",
      alignY: 50,
    });
    /** `scroll` from useScroll is not referentially stable; omitting it avoids re-scrolling every render (flicker). */
  }, [documentId, currentId, autoScroll, sentences]);

  return null;
}

function AnnotationToolStrip({ documentId }: { documentId: string }) {
  const { provides } = useAnnotation(documentId);
  const active = provides?.getActiveTool();
  const activeId =
    active && "id" in active ? (active as { id: string }).id : null;

  const tool = (id: string | null, icon: React.ReactNode, title: string) => (
    <Tooltip title={title} key={title}>
      <IconButton
        size="small"
        onClick={() => provides?.setActiveTool(id)}
        color={activeId === id ? "primary" : "default"}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );

  return (
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
      }}
    >
      {tool(null, <NearMeIcon fontSize="small" />, "Select")}
      {tool("highlight", <BorderColorIcon fontSize="small" />, "Highlight")}
      {tool("ink", <DrawIcon fontSize="small" />, "Draw")}
      {tool("square", <CropSquareIcon fontSize="small" />, "Rectangle")}
      {tool("circle", <RadioButtonUncheckedIcon fontSize="small" />, "Ellipse")}
    </Stack>
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
  pdfLoaded,
  setPdfLoaded,
}: Props & {
  documentId: string;
  pdfLoaded: boolean;
  setPdfLoaded: (v: boolean) => void;
}) {
  const theme = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const sentenceRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

  const { provides: zoomScope } = useZoom(documentId);
  const zoomRef = useRef(zoomScope);
  zoomRef.current = zoomScope;
  const isResizingRef = useRef(isResizing);
  isResizingRef.current = isResizing;

  const byPage = useMemo(() => sentencesByPageMap(sentences), [sentences]);

  /** Panel split drag ended — refit once. Do not depend on `zoomScope` identity (it changes every render and would refit every frame → scroll flicker). */
  useEffect(() => {
    if (isResizing) return;
    zoomRef.current?.requestZoom(ZoomMode.FitWidth);
  }, [isResizing]);

  /** Refit when the viewer container width changes (window/panel resize), not on every scroll. Debounced; ignores sub-pixel noise. */
  useEffect(() => {
    const el = containerRef.current;
    if (typeof ResizeObserver === "undefined" || !el) return;

    let lastWidth = Math.round(el.getBoundingClientRect().width);
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;

    const scheduleFit = () => {
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        debounceTimer = null;
        zoomRef.current?.requestZoom(ZoomMode.FitWidth);
      }, 120);
    };

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const w = Math.round(entry.contentRect.width);
      if (Math.abs(w - lastWidth) < 2) return;
      lastWidth = w;
      if (isResizingRef.current) return;
      scheduleFit();
    });

    ro.observe(el);
    return () => {
      ro.disconnect();
      if (debounceTimer !== null) clearTimeout(debounceTimer);
    };
  }, [documentId]);

  useEffect(() => {
    if (autoScroll && currentId !== null && sentenceRefs.current[currentId]) {
      sentenceRefs.current[currentId]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [currentId, autoScroll]);

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
      scale: number;
    }) => {
      const { pageIndex, width, height, scale } = props;
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
              <AnnotationLayer
                documentId={documentId}
                pageIndex={pageIndex}
                scale={scale}
              />
              <SelectionLayer
                documentId={documentId}
                pageIndex={pageIndex}
                scale={scale}
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
            <AnnotationLayer
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
            />
            <SelectionLayer
              documentId={documentId}
              pageIndex={pageIndex}
              scale={scale}
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
      handlePageDoubleClickCapture,
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
            <TtsScrollSync
              documentId={documentId}
              currentId={currentId}
              sentences={sentences}
              autoScroll={autoScroll}
            />
            <AnnotationToolStrip documentId={documentId} />
            <Box
              ref={containerRef}
              sx={{
                position: "relative",
                flex: 1,
                minHeight: 0,
                width: "100%",
                display: "flex",
                flexDirection: "column",
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
}: Props) {
  const theme = useTheme();
  const { engine, isLoading, error } = usePdfiumEngine();
  const plugins = useMemo(() => buildPlugins(pdfUrl), [pdfUrl]);
  const [pdfLoaded, setPdfLoaded] = useState(false);

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
              pdfLoaded={pdfLoaded}
              setPdfLoaded={setPdfLoaded}
            />
          ) : null
        }
      </EmbedPDF>
    </Box>
  );
});

export default PdfViewer;
