import { useCallback, useEffect, useState } from "react";
import { PdfAnnotationObject } from "@embedpdf/models";
import { useAnnotation, useAnnotationCapability } from "@embedpdf/plugin-annotation/react";
import {
  MarkupSettings,
  ShapeSettings,
  CurrentSettings,
  DEFAULT_MARKUP_SETTINGS,
  DEFAULT_SHAPE_SETTINGS,
  MARKUP_TOOLS,
  SHAPE_TOOLS,
  AnnotationTool,
} from "./constants";

export interface UseAnnotationSettingsReturn {
  markupSettings: MarkupSettings;
  shapeSettings: ShapeSettings;
  currentSettings: CurrentSettings | null;
  updateColor: (color: string) => void;
  updateWidth: (width: number) => void;
  updateOpacity: (opacity: number) => void;
  updateSettingsFromSelection: (annotation: PdfAnnotationObject) => void;
  updateSelectedAnnotationDirectly: (patch: Partial<PdfAnnotationObject>) => void;
}

/**
 * Centralized hook for managing annotation tool settings.
 * Handles markup and shape settings, tool defaults synchronization, and selection-based updates.
 */
export function useAnnotationSettings(documentId: string): UseAnnotationSettingsReturn {
  const { provides: annotationApi } = useAnnotation(documentId);
  const { provides: annotationCapability } = useAnnotationCapability();
  
  const [markupSettings, setMarkupSettings] = useState<MarkupSettings>(DEFAULT_MARKUP_SETTINGS);
  const [shapeSettings, setShapeSettings] = useState<ShapeSettings>(DEFAULT_SHAPE_SETTINGS);
  const [activeTool, setActiveTool] = useState<string | null>(null);

  // Get current active tool
  useEffect(() => {
    if (!annotationApi) return;
    
    const active = annotationApi.getActiveTool();
    const toolId = active && "id" in active ? (active as { id: string }).id : null;
    setActiveTool(toolId);
  }, [annotationApi]);

  // Sync settings with tool defaults whenever they change or tool is activated
  useEffect(() => {
    if (!annotationCapability || !activeTool) return;

    if (MARKUP_TOOLS.includes(activeTool as any)) {
      annotationCapability.setToolDefaults(activeTool, markupSettings as any);
    } else if (SHAPE_TOOLS.includes(activeTool as any)) {
      annotationCapability.setToolDefaults(activeTool, shapeSettings as any);
    }
  }, [annotationCapability, activeTool, markupSettings, shapeSettings]);

  // Determine current settings based on active tool
  const isMarkup = activeTool && MARKUP_TOOLS.includes(activeTool as any);
  const isShape = activeTool && SHAPE_TOOLS.includes(activeTool as any);
  const currentSettings = isMarkup ? markupSettings : isShape ? shapeSettings : null;

  // Update settings when annotation is selected
  const updateSettingsFromSelection = useCallback((annotation: PdfAnnotationObject) => {
    const annotationType = annotation.type as unknown as string;
    
    if (MARKUP_TOOLS.includes(annotationType as any)) {
      setMarkupSettings({
        strokeColor: (annotation as any).strokeColor || DEFAULT_MARKUP_SETTINGS.strokeColor,
        opacity: (annotation as any).opacity || DEFAULT_MARKUP_SETTINGS.opacity,
      });
    } else if (SHAPE_TOOLS.includes(annotationType as any)) {
      setShapeSettings({
        strokeColor: (annotation as any).strokeColor || DEFAULT_SHAPE_SETTINGS.strokeColor,
        strokeWidth: (annotation as any).strokeWidth || DEFAULT_SHAPE_SETTINGS.strokeWidth,
        opacity: (annotation as any).opacity || DEFAULT_SHAPE_SETTINGS.opacity,
      });
    }
  }, []);

  // Update selected annotations with new settings
  const updateSelectedAnnotations = useCallback((patch: Partial<PdfAnnotationObject>) => {
    if (!annotationApi || !annotationCapability) return;

    const selectedIds = annotationApi.getSelectedAnnotationIds() || [];
    if (selectedIds.length === 0) return;

    const selectedAnnotations = annotationApi.getSelectedAnnotations() || [];
    const patches = selectedAnnotations.map((ta) => ({
      pageIndex: ta.object.pageIndex,
      id: ta.object.id,
      patch,
    }));

    if (patches.length > 0) {
      annotationCapability.updateAnnotations(patches);
      
      // Also update local state to keep sliders in sync
      const firstAnnotation = selectedAnnotations[0]?.object;
      if (firstAnnotation) {
        updateSettingsFromSelection(firstAnnotation);
      }
    }
  }, [annotationApi, annotationCapability, updateSettingsFromSelection]);

  // Generic settings updater
  const updateSettings = useCallback((
    updater: (prev: MarkupSettings | ShapeSettings) => MarkupSettings | ShapeSettings,
    isMarkupUpdate: boolean
  ) => {
    const setSettings = isMarkupUpdate ? setMarkupSettings : setShapeSettings;

    setSettings((prev) => {
      const next = updater(prev);
      
      // Find what changed and update selected annotations
      const patch: any = {};
      for (const key in next) {
        if (next[key] !== prev[key]) {
          patch[key] = next[key];
        }
      }
      
      // Always update selected annotations if there are any
      updateSelectedAnnotations(patch);
      
      return next;
    });
  }, [updateSelectedAnnotations]);

  // Specific update functions
  const updateColor = useCallback((color: string) => {
    if (isMarkup) {
      updateSettings((prev) => ({ ...prev, strokeColor: color }), true);
    } else if (isShape) {
      updateSettings((prev) => ({ ...prev, strokeColor: color }), false);
    } else {
      // If no active tool, try to update based on what might be selected
      // This handles the case where an annotation is selected but no tool is active
      updateSettings((prev) => ({ ...prev, strokeColor: color }), true);
      updateSettings((prev) => ({ ...prev, strokeColor: color }), false);
    }
  }, [isMarkup, isShape, updateSettings]);

  const updateWidth = useCallback((width: number) => {
    if (isShape) {
      updateSettings((prev) => ({ ...prev, strokeWidth: width }), false);
    } else {
      // Update shape settings even if no active tool, for selected annotations
      updateSettings((prev) => ({ ...prev, strokeWidth: width }), false);
    }
  }, [isShape, updateSettings]);

  const updateOpacity = useCallback((opacity: number) => {
    if (isMarkup) {
      updateSettings((prev) => ({ ...prev, opacity }), true);
    } else if (isShape) {
      updateSettings((prev) => ({ ...prev, opacity }), false);
    } else {
      // Update both if no active tool, for selected annotations
      updateSettings((prev) => ({ ...prev, opacity }), true);
      updateSettings((prev) => ({ ...prev, opacity }), false);
    }
  }, [isMarkup, isShape, updateSettings]);

  return {
    markupSettings,
    shapeSettings,
    currentSettings,
    updateColor,
    updateWidth,
    updateOpacity,
    updateSettingsFromSelection,
    updateSelectedAnnotationDirectly: updateSelectedAnnotations,
  };
}
