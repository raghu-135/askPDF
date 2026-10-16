import React, { useCallback, useMemo, useState } from "react";
import { IconButton, Stack, Tooltip } from "@mui/material";
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
import PanToolIcon from "@mui/icons-material/PanTool";

const activeButtonSx = {
  bgcolor: "primary.main",
  color: "primary.contrastText",
  "&:hover": {
    bgcolor: "primary.dark",
  },
  "&.Mui-disabled": {
    bgcolor: "action.disabledBackground",
    color: "action.disabled",
  },
};

export const AnnotationChromeItems = React.memo(function AnnotationChromeItems({
  documentId,
  showSidebar,
  onToggleSidebar,
  isHistoryProcessingRef,
}: {
  documentId: string;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
}) {
  const { provides: annotationApi, state } = useAnnotation(documentId);
  const { provides: annotationCapability } = useAnnotationCapability();
  const { provides: history } = useHistoryCapability();
  const [isViewOnly, setIsViewOnly] = useState(false);

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

  const toggleViewOnly = useCallback(() => {
    const next = !isViewOnly;
    setIsViewOnly(next);
    annotationCapability?.setLocked({
      type: next ? LockModeType.All : LockModeType.None,
    });
    if (next) {
      annotationApi?.deselectAnnotation();
      annotationApi?.setActiveTool(null);
    }
  }, [annotationApi, annotationCapability, isViewOnly]);

  const toolButton = useCallback((id: string | null, icon: React.ReactNode, title: string) => {
    const isActive = state.activeToolId === id;
    return (
      <Tooltip title={title} key={title + (id || "select")}>
        <IconButton
          size="small"
          onClick={() => annotationApi?.setActiveTool(isActive ? null : id)}
          disabled={isViewOnly}
          color={isActive ? "primary" : "default"}
          sx={isActive ? activeButtonSx : undefined}
        >
          {icon}
        </IconButton>
      </Tooltip>
    );
  }, [annotationApi, isViewOnly, state.activeToolId]);

  const handleUndo = useCallback(() => {
    if (isHistoryProcessingRef) isHistoryProcessingRef.current = true;
    history?.undo();
    window.setTimeout(() => {
      if (isHistoryProcessingRef) isHistoryProcessingRef.current = false;
    }, 100);
  }, [history, isHistoryProcessingRef]);

  const handleRedo = useCallback(() => {
    if (isHistoryProcessingRef) isHistoryProcessingRef.current = true;
    history?.redo();
    window.setTimeout(() => {
      if (isHistoryProcessingRef) isHistoryProcessingRef.current = false;
    }, 100);
  }, [history, isHistoryProcessingRef]);

  return (
    <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
      <Tooltip title="Toggle Sidebar">
        <IconButton
          onClick={onToggleSidebar}
          size="small"
          sx={showSidebar ? activeButtonSx : undefined}
        >
          <ViewSidebarIcon fontSize="small" sx={{ transform: "scaleX(-1)" }} />
        </IconButton>
      </Tooltip>

      <Tooltip title={isViewOnly ? "Edit PDF" : "View only"}>
        <IconButton
          size="small"
          onClick={toggleViewOnly}
          color={isViewOnly ? "primary" : "default"}
          sx={isViewOnly ? activeButtonSx : undefined}
        >
          <PanToolIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      {toolButton("highlight", icons.highlight, "Highlight")}
      {toolButton("underline", icons.underline, "Underline")}
      {toolButton("strikeout", icons.strikeout, "Strikeout")}
      {toolButton("squiggly", icons.squiggly, "Squiggly")}
      {toolButton("ink", icons.ink, "Draw")}
      {toolButton("line", icons.line, "Line")}
      {toolButton("square", icons.square, "Rectangle")}
      {toolButton("circle", icons.circle, "Ellipse")}

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
  );
});
