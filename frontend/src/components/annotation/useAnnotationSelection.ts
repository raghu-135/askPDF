import { useCallback, useEffect, useRef, useState } from "react";
import { PdfAnnotationObject } from "@embedpdf/models";
import { useAnnotation } from "@embedpdf/plugin-annotation/react";

export interface UseAnnotationSelectionReturn {
  selectedAnnotation: PdfAnnotationObject | null;
  hasSelection: boolean;
  annotationProperties: {
    strokeColor?: string;
    strokeWidth?: number;
    opacity?: number;
  } | null;
}

/**
 * Optimized hook for managing annotation selection state without causing rerenders.
 * Uses refs and debounced polling to minimize unnecessary updates.
 */
export function useAnnotationSelection(documentId: string): UseAnnotationSelectionReturn {
  const { provides: annotationApi } = useAnnotation(documentId);
  const [selectedAnnotation, setSelectedAnnotation] = useState<PdfAnnotationObject | null>(null);
  const [hasSelection, setHasSelection] = useState(false);
  const [annotationProperties, setAnnotationProperties] = useState<{
    strokeColor?: string;
    strokeWidth?: number;
    opacity?: number;
  } | null>(null);
  
  const lastSelectionRef = useRef<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const checkSelection = useCallback(() => {
    if (!annotationApi) return;

    const selectedAnnotations = annotationApi.getSelectedAnnotations() || [];
    const currentSelection = selectedAnnotations.length > 0 ? selectedAnnotations[0] : null;
    const currentSelectionId = currentSelection?.object?.id || null;

    // Always update selection state to ensure displayValues recalculation
    if (currentSelectionId !== lastSelectionRef.current) {
      lastSelectionRef.current = currentSelectionId;
      
      if (currentSelection) {
        const annotation = currentSelection.object;
        setSelectedAnnotation(annotation);
        setHasSelection(true);
        
        // Extract properties that can be edited
        const properties: {
          strokeColor?: string;
          strokeWidth?: number;
          opacity?: number;
        } = {};
        
        if ('strokeColor' in annotation) {
          properties.strokeColor = (annotation as any).strokeColor;
        }
        if ('strokeWidth' in annotation) {
          properties.strokeWidth = (annotation as any).strokeWidth;
        }
        if ('opacity' in annotation) {
          properties.opacity = (annotation as any).opacity;
        }
        
        setAnnotationProperties(properties);
      } else {
        setSelectedAnnotation(null);
        setHasSelection(false);
        setAnnotationProperties(null);
      }
    } else if (currentSelection && currentSelection.object) {
      // Even if same ID, update properties in case they changed
      const annotation = currentSelection.object;
      const properties: {
        strokeColor?: string;
        strokeWidth?: number;
        opacity?: number;
      } = {};
      
      if ('strokeColor' in annotation) {
        properties.strokeColor = (annotation as any).strokeColor;
      }
      if ('strokeWidth' in annotation) {
        properties.strokeWidth = (annotation as any).strokeWidth;
      }
      if ('opacity' in annotation) {
        properties.opacity = (annotation as any).opacity;
      }
      
      setAnnotationProperties(properties);
    }
  }, [annotationApi]);

  // Debounced selection checking to avoid excessive polling
  const debouncedCheckSelection = useCallback(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(checkSelection, 50);
  }, [checkSelection]);

  useEffect(() => {
    if (!annotationApi) return;

    // Initial check
    checkSelection();

    // Set up optimized polling
    intervalRef.current = setInterval(debouncedCheckSelection, 200);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
    };
  }, [annotationApi, checkSelection, debouncedCheckSelection]);

  return {
    selectedAnnotation,
    hasSelection,
    annotationProperties,
  };
}
