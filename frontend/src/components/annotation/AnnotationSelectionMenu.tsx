import React, { useState, useCallback, useEffect } from "react";
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
    annotation && "strokeWidth" in annotation ? annotation.strokeWidth : 2
  );
  const [localOpacity, setLocalOpacity] = useState(
    annotation && "opacity" in annotation ? annotation.opacity : 1
  );

  // Sync local state when annotation changes externally (e.g., undo/redo)
  useEffect(() => {
    if (!annotation) return;
    setLocalColor(annotation && "strokeColor" in annotation ? annotation.strokeColor : undefined);
    setLocalWidth(annotation && "strokeWidth" in annotation ? annotation.strokeWidth : 2);
    setLocalOpacity(annotation && "opacity" in annotation ? annotation.opacity : 1);
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
          gap: 1,
          bgcolor: "background.paper",
          p: 1.5,
          borderRadius: 1,
          boxShadow: 3,
          zIndex: 100,
        }}
      >
        {/* Color picker - two rows for compact layout */}
        <Stack spacing={0.5} justifyContent="center">
          <Stack direction="row" spacing={0.5} justifyContent="center">
            {STANDARD_COLORS.slice(0, 4).map((color) => (
              <IconButton
                key={color}
                size="small"
                onClick={() => handlePropertyChangeWithDefaults({ strokeColor: color })}
                sx={{
                  border: localColor === color ? 2 : 1,
                  borderColor: localColor === color ? "primary.main" : "grey.300",
                  p: 0.25,
                }}
              >
                <CircleIcon sx={{ color, fontSize: 16 }} />
              </IconButton>
            ))}
          </Stack>
          <Stack direction="row" spacing={0.5} justifyContent="center">
            {STANDARD_COLORS.slice(4).map((color) => (
              <IconButton
                key={color}
                size="small"
                onClick={() => handlePropertyChangeWithDefaults({ strokeColor: color })}
                sx={{
                  border: localColor === color ? 2 : 1,
                  borderColor: localColor === color ? "primary.main" : "grey.300",
                  p: 0.25,
                }}
              >
                <CircleIcon sx={{ color, fontSize: 16 }} />
              </IconButton>
            ))}
          </Stack>
        </Stack>

        {/* Width slider (only for shapes) */}
        {hasWidth && (
          <Tooltip title={`Width: ${localWidth}px`} placement="top">
            <Box sx={{ width: 120, px: 1 }}>
              <Slider
                value={localWidth || 2}
                min={1}
                max={12}
                step={1}
                size="small"
                onChange={(_, val) => setLocalWidth(val as number)}
                onChangeCommitted={() =>
                  handlePropertyChangeWithDefaults({ strokeWidth: localWidth })
                }
                valueLabelDisplay="auto"
                valueLabelFormat={(val) => `${val}px`}
              />
            </Box>
          </Tooltip>
        )}

        {/* Opacity slider */}
        <Tooltip title={`Opacity: ${Math.round((localOpacity || 1) * 100)}%`} placement="top">
          <Box sx={{ width: 120, px: 1 }}>
            <Slider
              value={localOpacity || 1}
              min={0.1}
              max={1}
              step={0.1}
              size="small"
              onChange={(_, val) => setLocalOpacity(val as number)}
              onChangeCommitted={() =>
                handlePropertyChangeWithDefaults({ opacity: localOpacity })
              }
              valueLabelDisplay="auto"
              valueLabelFormat={(val) => `${Math.round(val * 100)}%`}
            />
          </Box>
        </Tooltip>

        {/* Action buttons - comment and delete */}
        <Stack direction="row" spacing={0.5}>
          <Tooltip title="Add Comment">
            <IconButton size="small" onClick={onOpenComments}>
              <AddCommentIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Delete">
            <IconButton size="small" onClick={handleDelete} color="error">
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </Box>
    </div>
  );
};
