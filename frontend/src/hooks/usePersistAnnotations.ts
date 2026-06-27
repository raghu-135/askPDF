import { useCallback, useEffect, useRef } from "react";
import { PdfAnnotationSubtype } from "@embedpdf/models";
import { getThreadFileAnnotations, updateThreadFileAnnotations } from "../lib/api";
import { serializeAnnotationItems } from "../lib/annotation-utils";

type AnnotationApi = {
  importAnnotations: (annotations: any[]) => void;
  exportAnnotations: () => { toPromise: () => Promise<any[]> };
  deleteAnnotations: (selection: { pageIndex: number; id: string }[]) => void;
};

type UsePersistAnnotationsParams = {
  annotationApi: AnnotationApi | null | undefined;
  threadId?: string | null;
  fileHash?: string | null;
  pdfLoaded?: boolean;
};

type SnapshotData = {
  annotations: any[];
  serialized: string;
  threadId: string;
  fileHash: string;
};

function getAnnotationPayload(item: any) {
  return item?.annotation || item;
}

function getAnnotationSelection(item: any): { pageIndex: number; id: string } | null {
  const annotation = getAnnotationPayload(item);
  if (typeof annotation?.pageIndex !== "number" || typeof annotation?.id !== "string") {
    return null;
  }
  return { pageIndex: annotation.pageIndex, id: annotation.id };
}

function filterPersistableAnnotations(items: any[]): any[] {
  return items.filter((item) => {
    const annotation = getAnnotationPayload(item);
    return annotation?.type !== PdfAnnotationSubtype.LINK;
  });
}

/**
 * Hydrate, debounce-save, and flush EmbedPDF annotations for a single thread/file pair.
 *
 * The hook owns the persistence lifecycle so the viewer can stay focused on rendering
 * and event wiring.
 */
export function usePersistAnnotations({
  annotationApi,
  threadId,
  fileHash,
  pdfLoaded = true,
}: UsePersistAnnotationsParams) {
  const hydrateAnnotationsRef = useRef(false);
  const hydratedAnnotationKeyRef = useRef<string | null>(null);
  const persistTimerRef = useRef<number | null>(null);
  const lastPersistedSnapshotRef = useRef<string | null>(null);
  // Stores the latest annotation snapshot for unmount persistence (Bug 1 fix)
  const latestSnapshotRef = useRef<SnapshotData | null>(null);

  /** Clear any pending debounce timer for annotation persistence. */
  const clearPersistTimer = useCallback(() => {
    if (persistTimerRef.current) {
      window.clearTimeout(persistTimerRef.current);
      persistTimerRef.current = null;
    }
  }, []);

  /**
   * Reset effect - only runs when threadId or fileHash changes.
   * This ensures state is reset for a new context, but NOT on every effect run
   * (which would happen when annotationApi or pdfLoaded change).
   */
  useEffect(() => {
    hydrateAnnotationsRef.current = false;
    hydratedAnnotationKeyRef.current = null;
    lastPersistedSnapshotRef.current = null;
    // Don't clear latestSnapshotRef here - unmount cleanup needs it!
    clearPersistTimer();
  }, [threadId, fileHash, clearPersistTimer]);

  /**
   * Load effect - runs when annotationApi or pdfLoaded become available.
   * Only loads if not already hydrated for the current context.
   */
  useEffect(() => {
    // Only attempt to load when we have all required params and PDF is ready
    if (!annotationApi || !threadId || !fileHash || !pdfLoaded) {
      return;
    }

    const key = `${threadId}:${fileHash}`;
    if (hydratedAnnotationKeyRef.current === key) {
      return;
    }
    hydratedAnnotationKeyRef.current = key;

    const loadAnnotations = async () => {
      try {
        const payload = await getThreadFileAnnotations(threadId, fileHash);
        const annotations = payload.annotations || [];

        // Clear existing annotations first to prevent duplicate React keys
        const existing = await annotationApi.exportAnnotations().toPromise();
        const existingSelections = (existing as any[])
          .map(getAnnotationSelection)
          .filter((selection): selection is { pageIndex: number; id: string } => Boolean(selection));
        if (existingSelections.length > 0) {
          annotationApi.deleteAnnotations(existingSelections);
        }

        const persistedAnnotations = filterPersistableAnnotations(annotations);

        if (persistedAnnotations.length === 0) {
          return;
        }

        // Note: We skip deduplication by ID since EmbedPDF annotations may not have unique IDs
        const serializedSnapshot = JSON.stringify(serializeAnnotationItems(persistedAnnotations));
        lastPersistedSnapshotRef.current = serializedSnapshot;
        hydrateAnnotationsRef.current = true;
        annotationApi.importAnnotations(persistedAnnotations);
      } catch (error: any) {
        const message = String(error?.message || error || "");
        if (!message.includes("404")) {
          console.warn("Failed to load persisted annotations:", error);
        }
      } finally {
        hydrateAnnotationsRef.current = false;
      }
    };

    void loadAnnotations();
  }, [annotationApi, fileHash, threadId, pdfLoaded]);

  /**
   * Capture a snapshot of the current annotations without persisting.
   * This is called eagerly after each committed mutation to ensure the unmount
   * cleanup has access to the latest data even if annotationApi is destroyed.
   */
  const captureSnapshot = useCallback(async () => {
    if (!annotationApi || !threadId || !fileHash) return;

    try {
      const exported = await annotationApi.exportAnnotations().toPromise();
      const persistable = filterPersistableAnnotations(exported as any[]);
      const serialized = JSON.stringify(serializeAnnotationItems(persistable));
      // Store context in snapshot so unmount cleanup knows which file this belongs to
      latestSnapshotRef.current = { annotations: persistable, serialized, threadId, fileHash };
    } catch (error: any) {
      // "Document not found" is expected during cleanup - silently ignore
      const message = String(error?.message || error || "");
      if (!message.includes("Document not found")) {
        console.warn("Failed to capture annotation snapshot:", error);
      }
    }
  }, [annotationApi, fileHash, threadId]);

  /**
   * Export the current annotation graph from EmbedPDF and persist it if it changed.
   * Also updates the snapshot ref so unmount cleanup has the latest data.
   */
  const persistAnnotations = useCallback(async () => {
    if (!annotationApi || !threadId || !fileHash) return;

    try {
      const exported = await annotationApi.exportAnnotations().toPromise();
      // Note: We don't deduplicate by ID because EmbedPDF may not assign unique IDs
      const uniqueExported = filterPersistableAnnotations(exported as any[]);
      const snapshot = JSON.stringify(serializeAnnotationItems(uniqueExported));
      if (snapshot === lastPersistedSnapshotRef.current) {
        return;
      }

      await updateThreadFileAnnotations(threadId, fileHash, uniqueExported as any);
      lastPersistedSnapshotRef.current = snapshot;
      // Update snapshot ref so unmount has the latest (with context)
      latestSnapshotRef.current = { annotations: uniqueExported, serialized: snapshot, threadId, fileHash };
    } catch (error: any) {
      // "Document not found" is expected during cleanup/unmount - don't spam console
      const message = String(error?.message || error || "");
      if (!message.includes("Document not found")) {
        console.warn("Failed to persist annotations:", error);
      }
    }
  }, [annotationApi, fileHash, threadId]);

  /**
   * Schedule a persistence write after a short debounce window.
   *
   * This reduces backend chatter during rapid annotation edits and undo/redo bursts.
   */
  const schedulePersistAnnotations = useCallback(() => {
    clearPersistTimer();
    persistTimerRef.current = window.setTimeout(() => {
      persistTimerRef.current = null;
      void persistAnnotations();
    }, 400);
  }, [clearPersistTimer, persistAnnotations]);

  /**
   * Flush any pending save when the hook unmounts.
   * Uses the captured snapshot to persist even if annotationApi is already destroyed.
   * Note: We use the threadId/fileHash stored IN the snapshot, not from closure,
   * because the effect cleanup runs with new values when dependencies change.
   */
  useEffect(() => {
    return () => {
      clearPersistTimer();
      // Use the captured snapshot for unmount persistence (Bug 1 fix)
      // This avoids calling exportAnnotations on a potentially-destroyed API
      const snapshot = latestSnapshotRef.current;
      if (snapshot) {
        const { annotations, serialized, threadId: snapThreadId, fileHash: snapFileHash } = snapshot;
        if (serialized !== lastPersistedSnapshotRef.current) {
          void updateThreadFileAnnotations(snapThreadId, snapFileHash, annotations as any)
            .then(() => {
              lastPersistedSnapshotRef.current = serialized;
              // Clear snapshot after successful persistence
              latestSnapshotRef.current = null;
            })
            .catch((error: any) => {
              const message = String(error?.message || error || "");
              if (!message.includes("Document not found")) {
                console.warn("Failed to persist annotations on unmount:", error);
              }
            });
        }
      }
    };
  }, [clearPersistTimer]);

  return {
    hydrateAnnotationsRef,
    schedulePersistAnnotations,
    captureSnapshot,
  };
}
