import React, { useCallback, useState, useMemo } from "react";
import {
  IconButton,
  Stack,
  Tooltip,
  Divider,
} from "@mui/material";
import {
  useAnnotation,
  useAnnotationCapability,
  LockModeType,
} from "@embedpdf/plugin-annotation/react";
import { useHistoryCapability } from "@embedpdf/plugin-history/react";
import BorderColorIcon from "@mui/icons-material/BorderColor";
import DrawIcon from "@mui/icons-material/Draw";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import ViewSidebarIcon from "@mui/icons-material/ViewSidebar";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import GestureIcon from "@mui/icons-material/Gesture";
import ArrowRightAltIcon from "@mui/icons-material/ArrowRightAlt";
import UndoIcon from "@mui/icons-material/Undo";
import RedoIcon from "@mui/icons-material/Redo";
import NearMeIcon from "@mui/icons-material/NearMe";
import NearMeDisabledIcon from "@mui/icons-material/NearMeDisabled";

export interface AnnotationToolbarProps {
  documentId: string;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
  searchControls?: React.ReactNode;
}

/**
 * Simplified annotation toolbar using EmbedPDF native patterns.
 * Properties moved to contextual selection menu.
 */
export const AnnotationToolbar: React.FC<AnnotationToolbarProps> = React.memo(function AnnotationToolbar({
  documentId,
  showSidebar,
  onToggleSidebar,
  isHistoryProcessingRef,
  searchControls,
}) {
  // EmbedPDF hooks
  const { provides: annotationApi, state } = useAnnotation(documentId);
  const { provides: annotationCapability } = useAnnotationCapability();
  const { provides: history } = useHistoryCapability();

  // View-only mode state (simple boolean, no custom interaction mode)
  const [isViewOnly, setIsViewOnly] = useState(false);

  // Memoize icon JSX elements to prevent recreation on every render
  const icons = useMemo(() => ({
    highlight: <BorderColorIcon fontSize="small" />,
    underline: <FormatUnderlinedIcon fontSize="small" />,
    strikeout: <StrikethroughSIcon fontSize="small" />,
    squiggly: <GestureIcon fontSize="small" sx={{ transform: "rotate(90deg)" }} />,
    ink: <DrawIcon fontSize="small" />,
    line: <ArrowRightAltIcon fontSize="small" />,
    square: <CropSquareIcon fontSize="small" />,
    circle: <RadioButtonUncheckedIcon fontSize="small" />,
  }), []);

  // Toggle view-only mode
  const toggleViewOnly = useCallback(() => {
    const next = !isViewOnly;
    setIsViewOnly(next);
    annotationCapability?.setLocked({
      type: next ? LockModeType.All : LockModeType.None,
    });
    // Clear selection and deactivate active tool when switching to view-only mode
    if (next) {
      annotationApi?.deselectAnnotation();
      annotationApi?.setActiveTool(null);
    }
  }, [isViewOnly, annotationCapability, annotationApi]);

  // Tool button component
  const toolButton = useCallback((id: string | null, icon: React.ReactNode, title: string) => {
    const isActive = state.activeToolId === id;
    return (
      <Tooltip title={title} key={title + (id || "select")}>
        <IconButton
          size="small"
          onClick={() => annotationApi?.setActiveTool(isActive ? null : id)}
          disabled={isViewOnly}
          color={isActive ? "primary" : "default"}
        >
          {icon}
        </IconButton>
      </Tooltip>
    );
  }, [state.activeToolId, annotationApi, isViewOnly]);

  // History handlers
  // Note: isHistoryProcessingRef is used to prevent auto-scroll conflicts with TTS and comments
  // When history actions (undo/redo) occur, we set this flag to temporarily disable auto-scroll
  // This prevents the scroll system from fighting with the history navigation
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
      {/* Main toolbar - tools only */}
      <Stack
        direction="row"
        spacing={0.5}
        sx={{
          alignItems: "center",
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
        {/* Left section - Tools and controls */}
        <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
          <Tooltip title="Toggle Sidebar">
            <IconButton onClick={onToggleSidebar} size="small" sx={{ mr: 0.5 }}>
              <ViewSidebarIcon
                fontSize="small"
                color={showSidebar ? "primary" : "inherit"}
                sx={{ transform: "scaleX(-1)" }}
              />
            </IconButton>
          </Tooltip>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          <Tooltip title={isViewOnly ? "Edit PDF" : "View only"}>
            <IconButton
              size="small"
              onClick={toggleViewOnly}
              color={isViewOnly ? "primary" : "default"}
            >
              {isViewOnly ? <NearMeIcon fontSize="small" /> : <NearMeDisabledIcon fontSize="small" />}
            </IconButton>
          </Tooltip>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          {/* Markup Tools */}
          {toolButton("highlight", icons.highlight, "Highlight")}
          {toolButton("underline", icons.underline, "Underline")}
          {toolButton("strikeout", icons.strikeout, "Strikeout")}
          {toolButton("squiggly", icons.squiggly, "Squiggly")}

          <Divider orientation="vertical" flexItem sx={{ mx: 0.5, my: 0.5 }} />

          {/* Shape Tools */}
          {toolButton("ink", icons.ink, "Draw")}
          {toolButton("line", icons.line, "Line")}
          {toolButton("square", icons.square, "Rectangle")}
          {toolButton("circle", icons.circle, "Ellipse")}
        </Stack>

        {/* Right section - Search and history controls */}
        <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
          {searchControls}
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
    </Stack>
  );
});
