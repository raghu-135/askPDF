import React, { useState, useCallback, useEffect, useMemo } from "react";
import {
  useAnnotationCapability,
  type AnnotationSelectionMenuProps,
} from "@embedpdf/plugin-annotation/react";
import { PdfAnnotationObject, PdfAnnotationSubtypeName } from "@embedpdf/models";
import { Box, IconButton, Stack, Slider, Tooltip } from "@mui/material";
import AddCommentIcon from "@mui/icons-material/AddComment";
import DeleteIcon from "@mui/icons-material/Delete";
import CircleIcon from "@mui/icons-material/Circle";
import { STANDARD_COLORS } from "./constants";

const MENU_SIZES = {
  colorSwatch: 26,
  colorIcon: 16,
  slider: 128,
  sliderThumb: 12,
  actionButton: 28,
};

const DEFAULT_STROKE_WIDTH = 2;
const DEFAULT_OPACITY = 1;

// Memoized color button component to prevent recreation on every render
const ColorButton = React.memo(
  ({
    color,
    localColor,
    onClick,
  }: {
    color: string;
    localColor: string | undefined;
    onClick: () => void;
  }) => {
    const selected = localColor === color;

    return (
      <IconButton
        size="small"
        onClick={onClick}
        sx={{
          width: MENU_SIZES.colorSwatch,
          height: MENU_SIZES.colorSwatch,
          p: 0,
          border: 1,
          borderColor: selected ? "primary.main" : "divider",
          bgcolor: selected ? "action.selected" : "background.paper",
          boxShadow: selected ? "0 0 0 2px rgba(25, 118, 210, 0.16)" : "none",
          transition:
            "background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease",
          "&:hover": {
            bgcolor: "action.hover",
            borderColor: selected ? "primary.main" : "text.secondary",
          },
        }}
      >
        <CircleIcon
          sx={{
            color,
            fontSize: MENU_SIZES.colorIcon,
            filter: "drop-shadow(0 0 0.5px rgba(0,0,0,0.45))",
          }}
        />
      </IconButton>
    );
  },
);

ColorButton.displayName = "ColorButton";

export const AnnotationSelectionMenu: React.FC<
  AnnotationSelectionMenuProps & { documentId: string; onOpenComments: () => void }
> = ({ selected, context, menuWrapperProps, rect, documentId, onOpenComments }) => {
  const { provides: annotationCapability } = useAnnotationCapability();
  const annotationScope = annotationCapability?.forDocument(documentId);
  const annotation = context?.annotation?.object;

  // Local state for property inputs (with type guards)
  const [localColor, setLocalColor] = useState(
    annotation && "strokeColor" in annotation ? annotation.strokeColor : undefined
  );
  const [localWidth, setLocalWidth] = useState(
    annotation && "strokeWidth" in annotation ? annotation.strokeWidth : DEFAULT_STROKE_WIDTH
  );
  const [localOpacity, setLocalOpacity] = useState(
    annotation && "opacity" in annotation ? annotation.opacity : DEFAULT_OPACITY
  );

  // Sync local state when annotation changes externally (e.g., undo/redo)
  useEffect(() => {
    if (!annotation) return;
    setLocalColor(annotation && "strokeColor" in annotation ? annotation.strokeColor : undefined);
    setLocalWidth(annotation && "strokeWidth" in annotation ? annotation.strokeWidth : DEFAULT_STROKE_WIDTH);
    setLocalOpacity(annotation && "opacity" in annotation ? annotation.opacity : DEFAULT_OPACITY);
  }, [annotation]);

  // Update annotation properties
  const handlePropertyChange = useCallback(
    (patch: Partial<PdfAnnotationObject>) => {
      if (!annotation || !annotationScope) return;
      annotationScope.updateAnnotation(annotation.pageIndex, annotation.id, patch);
    },
    [annotation, annotationScope]
  );

  // Update tool defaults when properties change (bidirectional sync)
  const handlePropertyChangeWithDefaults = useCallback(
    (patch: Partial<PdfAnnotationObject>) => {
      handlePropertyChange(patch);
      // Update tool defaults for future annotations
      const toolId = annotation?.type ? PdfAnnotationSubtypeName[annotation.type] : undefined;
      if (toolId) {
        annotationCapability?.setToolDefaults(toolId, patch);
      }
    },
    [handlePropertyChange, annotation, annotationCapability]
  );

  const handleDelete = useCallback(() => {
    if (!annotation || !annotationScope) return;
    annotationScope.deleteAnnotation(annotation.pageIndex, annotation.id);
  }, [annotation, annotationScope]);

  // Memoize action icons to prevent recreation on every render
  const actionIcons = useMemo(() => ({
    addComment: <AddCommentIcon fontSize="small" />,
    delete: <DeleteIcon fontSize="small" />,
  }), []);

  if (!selected || !annotation) return null;

  const hasWidth = "strokeWidth" in annotation;

  return (
    <div {...menuWrapperProps}>
      <Box
        sx={{
          position: "absolute",
          top: rect.size.height + 8,
          left: 0,
          pointerEvents: "auto",
          minWidth: "max-content",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 0.75,
          bgcolor: "background.paper",
          p: 1,
          border: 1,
          borderColor: "divider",
          borderRadius: 1.5,
          boxShadow: "0 8px 24px rgba(15, 23, 42, 0.16)",
          zIndex: 100,
          "&::before": {
            content: '""',
            position: "absolute",
            top: -6,
            left: 14,
            width: 10,
            height: 10,
            bgcolor: "background.paper",
            borderLeft: 1,
            borderTop: 1,
            borderColor: "divider",
            transform: "rotate(45deg)",
          },
        }}
      >
        {/* Color picker - two rows for compact layout */}
        <Stack spacing={0.5} justifyContent="center" sx={{ px: 0.25 }}>
          <Stack direction="row" spacing={0.5} justifyContent="center">
            {STANDARD_COLORS.slice(0, 4).map((color) => (
              <ColorButton
                key={color}
                color={color}
                localColor={localColor}
                onClick={() => handlePropertyChangeWithDefaults({ strokeColor: color })}
              />
            ))}
          </Stack>
          <Stack direction="row" spacing={0.5} justifyContent="center">
            {STANDARD_COLORS.slice(4).map((color) => (
              <ColorButton
                key={color}
                color={color}
                localColor={localColor}
                onClick={() => handlePropertyChangeWithDefaults({ strokeColor: color })}
              />
            ))}
          </Stack>
        </Stack>

        {/* Width slider (only for shapes) */}
        {hasWidth && (
          <Tooltip title={`Width: ${localWidth}px`} placement="top">
            <Box sx={{ width: MENU_SIZES.slider, px: 0.75, py: 0.25 }}>
              <Slider
                value={localWidth || DEFAULT_STROKE_WIDTH}
                min={1}
                max={12}
                step={1}
                size="small"
                onChange={(_, val) => setLocalWidth(val as number)}
                onChangeCommitted={() =>
                  handlePropertyChangeWithDefaults({ strokeWidth: localWidth })
                }
                valueLabelDisplay="off"
                sx={{
                  py: 0.75,
                  "& .MuiSlider-thumb": {
                    width: MENU_SIZES.sliderThumb,
                    height: MENU_SIZES.sliderThumb,
                  },
                }}
              />
            </Box>
          </Tooltip>
        )}

        {/* Opacity slider */}
        <Tooltip title={`Opacity: ${Math.round((localOpacity || DEFAULT_OPACITY) * 100)}%`} placement="top">
          <Box sx={{ width: MENU_SIZES.slider, px: 0.75, py: 0.25 }}>
            <Slider
              value={localOpacity || DEFAULT_OPACITY}
              min={0.1}
              max={1}
              step={0.1}
              size="small"
              onChange={(_, val) => setLocalOpacity(val as number)}
              onChangeCommitted={() =>
                handlePropertyChangeWithDefaults({ opacity: localOpacity })
              }
              valueLabelDisplay="off"
              sx={{
                py: 0.75,
                "& .MuiSlider-thumb": {
                  width: MENU_SIZES.sliderThumb,
                  height: MENU_SIZES.sliderThumb,
                },
              }}
            />
          </Box>
        </Tooltip>

        {/* Action buttons - comment and delete */}
        <Stack
          direction="row"
          spacing={0.5}
          sx={{
            pt: 0.5,
            mt: 0.25,
            borderTop: 1,
            borderColor: "divider",
            width: "100%",
            justifyContent: "center",
          }}
        >
          <Tooltip title="Add Comment">
            <IconButton
              size="small"
              onClick={onOpenComments}
              sx={{
                width: MENU_SIZES.actionButton,
                height: MENU_SIZES.actionButton,
                color: "text.secondary",
                "&:hover": {
                  color: "primary.main",
                  bgcolor: "action.hover",
                },
              }}
            >
              {actionIcons.addComment}
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton
              size="small"
              onClick={handleDelete}
              sx={{
                width: MENU_SIZES.actionButton,
                height: MENU_SIZES.actionButton,
                color: "text.secondary",
                "&:hover": {
                  color: "error.main",
                  bgcolor: "rgba(211, 47, 47, 0.08)",
                },
              }}
            >
              {actionIcons.delete}
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>
    </div>
  );
};
