import { useCallback, useEffect, useRef } from "react";
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
};

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
}: UsePersistAnnotationsParams) {
  const hydrateAnnotationsRef = useRef(false);
  const hydratedAnnotationKeyRef = useRef<string | null>(null);
  const persistTimerRef = useRef<number | null>(null);
  const lastPersistedSnapshotRef = useRef<string | null>(null);

  /** Clear any pending debounce timer for annotation persistence. */
  const clearPersistTimer = useCallback(() => {
    if (persistTimerRef.current) {
      window.clearTimeout(persistTimerRef.current);
      persistTimerRef.current = null;
    }
  }, []);

  /**
   * Reset transient persistence state when the active thread or file changes.
   *
   * This clears hydration and dedupe state so the next PDF context starts cleanly.
   */
  const resetForContextChange = useCallback(() => {
    hydrateAnnotationsRef.current = false;
    hydratedAnnotationKeyRef.current = null;
    lastPersistedSnapshotRef.current = null;
    clearPersistTimer();
  }, [clearPersistTimer]);

  /** Clear cached persistence state whenever the active thread/file identity changes. */
  useEffect(() => {
    resetForContextChange();
  }, [resetForContextChange, threadId, fileHash]);

  /**
   * Export the current annotation graph from EmbedPDF and persist it if it changed.
   */
  const persistAnnotations = useCallback(async () => {
    if (!annotationApi || !threadId || !fileHash) return;

    try {
      const exported = await annotationApi.exportAnnotations().toPromise();
      // Deduplicate before saving to clean up stale duplicate data
      const uniqueExported = Array.from(
        new Map((exported as any[]).map((a: any) => [a.id, a])).values()
      );
      const snapshot = JSON.stringify(serializeAnnotationItems(uniqueExported));
      if (snapshot === lastPersistedSnapshotRef.current) {
        return;
      }

      await updateThreadFileAnnotations(threadId, fileHash, uniqueExported as any);
      lastPersistedSnapshotRef.current = snapshot;
    } catch (error: any) {
      // "Document not found" is expected during cleanup/unmount - don't spam console
      const message = String(error?.message || error || "");
      if (!message.includes("Document not found")) {
        console.warn("Failed to persist annotations:", error);
      }
    }
  }, [annotationApi, fileHash, threadId]);

  /**
   * Load persisted annotations for the active thread/file pair and import them into EmbedPDF.
   *
   * The key guard prevents repeated hydration for the same document identity.
   */
  const loadPersistedAnnotations = useCallback(async () => {
    if (!annotationApi || !threadId || !fileHash) return;

    const key = `${threadId}:${fileHash}`;
    if (hydratedAnnotationKeyRef.current === key) return;
    hydratedAnnotationKeyRef.current = key;

    try {
      const payload = await getThreadFileAnnotations(threadId, fileHash);

      const annotations = payload.annotations || [];

      // Clear existing annotations first to prevent duplicate React keys
      const existing = await annotationApi.exportAnnotations().toPromise();
      if ((existing as any[]).length > 0) {
        annotationApi.deleteAnnotations(
          (existing as any[]).map((a: any) => ({ pageIndex: a.pageIndex, id: a.id }))
        );
      }

      if (annotations.length === 0) {
        return;
      }

      // Deduplicate by annotation ID to prevent React key errors in AnnotationLayer
      const uniqueAnnotations = Array.from(
        new Map(annotations.map((a: any) => [a.id, a])).values()
      );

      const serializedSnapshot = JSON.stringify(serializeAnnotationItems(uniqueAnnotations));
      lastPersistedSnapshotRef.current = serializedSnapshot;
      hydrateAnnotationsRef.current = true;
      annotationApi.importAnnotations(uniqueAnnotations);
    } catch (error: any) {
      const message = String(error?.message || error || "");
      if (!message.includes("404")) {
        console.warn("Failed to load persisted annotations:", error);
      }
    } finally {
      hydrateAnnotationsRef.current = false;
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

  /** Flush any pending save when the hook unmounts or the active PDF context changes. */
  useEffect(() => {
    return () => {
      clearPersistTimer();
      void persistAnnotations();
    };
  }, [clearPersistTimer, persistAnnotations, threadId, fileHash]);

  return {
    hydrateAnnotationsRef,
    loadPersistedAnnotations,
    schedulePersistAnnotations,
  };
}
