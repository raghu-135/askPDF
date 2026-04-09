import React, { useCallback, useEffect, useMemo } from "react";
import {
  Box,
  IconButton,
  Stack,
  Tooltip,
  Typography,
  Divider,
  Slider,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import {
  useAnnotation,
  useAnnotationCapability,
  LockModeType,
} from "@embedpdf/plugin-annotation/react";
import {
  useInteractionManager,
  useInteractionManagerCapability,
} from "@embedpdf/plugin-interaction-manager/react";
import { useSelectionCapability } from "@embedpdf/plugin-selection/react";
import { useHistoryCapability } from "@embedpdf/plugin-history/react";
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
import { PdfAnnotationObject } from "@embedpdf/models";

import {
  MARKUP_TOOLS,
  SHAPE_TOOLS,
  VIEW_ONLY_MODE_ID,
  STANDARD_COLORS,
} from "./constants";
import { useAnnotationSelection } from "./useAnnotationSelection";
import { useAnnotationSettings } from "./useAnnotationSettings";

export interface AnnotationToolbarProps {
  documentId: string;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  onOpenComments: () => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
}

/**
 * Single unified annotation toolbar that eliminates rerendering issues.
 * Always visible with conditional sections for tools and properties.
 */
export const AnnotationToolbar: React.FC<AnnotationToolbarProps> = React.memo(function AnnotationToolbar({
  documentId,
  showSidebar,
  onToggleSidebar,
  onOpenComments,
  isHistoryProcessingRef,
}) {
  const theme = useTheme();
  
  // Core annotation hooks
  const { provides: annotationApi } = useAnnotation(documentId);
  const { provides: annotationCapability } = useAnnotationCapability();
  const { provides: interactionCapability } = useInteractionManagerCapability();
  const { provides: selectionCapability } = useSelectionCapability();
  const { provides: interactionScope } = useInteractionManager(documentId);
  const { provides: history } = useHistoryCapability();

  // Custom hooks for optimized state management
  const { hasSelection, annotationProperties } = useAnnotationSelection(documentId);
  const {
    markupSettings,
    shapeSettings,
    currentSettings,
    updateColor,
    updateWidth,
    updateOpacity,
    updateSettingsFromSelection,
    updateSelectedAnnotationDirectly,
  } = useAnnotationSettings(documentId);

  // Active tool and mode state
  const active = annotationApi?.getActiveTool();
  const activeId = active && "id" in active ? (active as { id: string }).id : null;
  const defaultModeId = interactionCapability?.getDefaultMode() ?? "default";
  const activeModeId =
    interactionScope?.getActiveInteractionMode()?.id ?? defaultModeId;
  const isViewOnly = activeModeId === VIEW_ONLY_MODE_ID;

  // Tool type detection
  const isMarkup = activeId && MARKUP_TOOLS.includes(activeId as any);
  const isShape = activeId && SHAPE_TOOLS.includes(activeId as any);
  const hasActiveTool = Boolean(activeId);

  // Update settings when annotation is selected
  useEffect(() => {
    if (hasSelection && annotationProperties) {
      // Create a mock annotation object for settings update
      const mockAnnotation = {
        type: "unknown",
        ...annotationProperties,
      } as unknown as PdfAnnotationObject;
      updateSettingsFromSelection(mockAnnotation);
    }
  }, [hasSelection, annotationProperties, updateSettingsFromSelection]);

  // Register view-only mode
  useEffect(() => {
    if (!interactionCapability) return;
    interactionCapability.registerMode({
      id: VIEW_ONLY_MODE_ID,
      scope: "page",
      exclusive: false,
      cursor: "default",
    });
  }, [interactionCapability]);

  // Configure selection for modes
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

  // Lock annotations in view-only mode
  useEffect(() => {
    if (!annotationCapability) return;
    annotationCapability.setLocked({
      type: isViewOnly ? LockModeType.All : LockModeType.None,
    });
  }, [annotationCapability, isViewOnly]);

  // Clear selection functions
  const clearCurrentSelection = useCallback(() => {
    selectionCapability?.clear(documentId);
    annotationApi?.deselectAnnotation();
  }, [documentId, annotationApi, selectionCapability]);

  // Mode activation functions
  const activateSelectMode = useCallback(() => {
    interactionScope?.activateDefaultMode();
    clearCurrentSelection();
  }, [clearCurrentSelection, interactionScope]);

  const activateViewOnlyMode = useCallback(() => {
    interactionScope?.activate(VIEW_ONLY_MODE_ID);
    clearCurrentSelection();
  }, [clearCurrentSelection, interactionScope]);

  // Auto-switch to select mode when tool is active in view-only mode
  useEffect(() => {
    if (!activeId || !isViewOnly) return;
    activateSelectMode();
  }, [activeId, activateSelectMode, isViewOnly]);

  // Auto-switch to default mode when no tool is active and not in view-only
  useEffect(() => {
    if (activeId || isViewOnly) return;
    interactionScope?.activateDefaultMode();
  }, [activeId, interactionScope, isViewOnly]);

  // Tool toggle handler
  const handleSelectToggle = useCallback(() => {
    if (activeId) {
      annotationApi?.setActiveTool(null);
      activateSelectMode();
      return;
    }

    if (isViewOnly) {
      activateSelectMode();
      return;
    }

    activateViewOnlyMode();
  }, [activeId, activateSelectMode, activateViewOnlyMode, isViewOnly, annotationApi]);

  // Tool button component
  const tool = useMemo(() => (id: string | null, icon: React.ReactNode, title: string) => (
    <Tooltip title={title} key={title + (id || "select")}>
      <IconButton
        size="small"
        onClick={() => annotationApi?.setActiveTool(id)}
        color={activeId === id ? "primary" : "default"}
      >
        {icon}
      </IconButton>
    </Tooltip>
  ), [activeId, annotationApi]);

  // History handlers
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

  // Get current display values for properties
  const displayValues = useMemo(() => {
    if (hasSelection && annotationProperties) {
      return {
        strokeColor: annotationProperties.strokeColor || currentSettings?.strokeColor || "#ffeb3b",
        strokeWidth: annotationProperties.strokeWidth || currentSettings?.strokeWidth || 2,
        opacity: annotationProperties.opacity || currentSettings?.opacity || 1,
      };
    }
    return {
      strokeColor: currentSettings?.strokeColor || "#ffeb3b",
      strokeWidth: currentSettings?.strokeWidth || 2,
      opacity: currentSettings?.opacity || 1,
    };
  }, [hasSelection, annotationProperties, currentSettings, markupSettings, shapeSettings]);

  // Determine if width slider should be shown
  const showWidthSlider = isShape || (hasSelection && annotationProperties && 'strokeWidth' in annotationProperties && !isMarkup);

  return (
    <Stack direction="column" sx={{ width: "100%" }}>
      {/* Main toolbar - always visible */}
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
          {tool("strikeout", <StrikethroughSIcon fontSize="small" />, "Strikeout")}
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

        {/* Right section - History controls */}
        <Stack direction="row" spacing={0.5} sx={{ alignItems: "center" }}>
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

      {/* Properties section - always visible but values change */}
      <Stack
        direction="row"
        spacing={2}
        sx={{
          alignItems: "center",
          px: 2,
          py: 0.5,
          borderBottom: 1,
          borderColor: "divider",
          bgcolor: "background.default",
          height: 40,
        }}
      >
        {/* Color picker */}
        <Stack direction="row" spacing={0.5}>
          {STANDARD_COLORS.map((color) => (
            <IconButton
              key={color}
              size="small"
              onClick={() => {
                updateColor(color);
                // Also update selected annotation directly if there's a selection
                if (hasSelection) {
                  updateSelectedAnnotationDirectly({ strokeColor: color });
                }
              }}
              sx={{
                p: 0.25,
                border: "2px solid",
                borderColor: displayValues.strokeColor === color ? "primary.main" : "transparent",
              }}
            >
              <CircleIcon sx={{ color, fontSize: 18 }} />
            </IconButton>
          ))}
        </Stack>

        {/* Width slider - shown conditionally but container remains */}
        {showWidthSlider && (
          <Stack direction="row" spacing={1} sx={{ alignItems: "center", width: 150 }}>
            <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
              Width: {displayValues.strokeWidth}px
            </Typography>
            <Slider
              size="small"
              value={displayValues.strokeWidth}
              min={1}
              max={12}
              onChange={(_, val) => {
                const width = val as number;
                updateWidth(width);
                // Also update selected annotation directly if there's a selection
                if (hasSelection) {
                  updateSelectedAnnotationDirectly({ strokeWidth: width });
                }
              }}
            />
          </Stack>
        )}

        {/* Opacity slider - always visible */}
        <Stack direction="row" spacing={1} sx={{ alignItems: "center", width: 120 }}>
          <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
            Opacity: {Math.round(displayValues.opacity * 100)}%
          </Typography>
          <Slider
            size="small"
            value={displayValues.opacity}
            min={0.1}
            max={1}
            step={0.1}
            onChange={(_, val) => {
              const opacity = val as number;
              updateOpacity(opacity);
              // Also update selected annotation directly if there's a selection
              if (hasSelection) {
                updateSelectedAnnotationDirectly({ opacity });
              }
            }}
          />
        </Stack>
      </Stack>
    </Stack>
  );
});
