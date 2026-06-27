import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCoreState } from "@embedpdf/core/react";
import {
  MatchFlag,
  type PdfEngine,
  type SearchResult,
} from "@embedpdf/models";
import { getAdjacentCyclicIndex } from "../lib/pdf-utils";

export type PdfSearchState = {
  query: string;
  setQuery: (query: string) => void;
  isOpen: boolean;
  setIsOpen: (isOpen: boolean) => void;
  isLoading: boolean;
  error: string | null;
  results: SearchResult[];
  activeIndex: number;
  activeResult: SearchResult | null;
  navigationVersion: number;
  caseSensitive: boolean;
  setCaseSensitive: (enabled: boolean) => void;
  wholeWord: boolean;
  setWholeWord: (enabled: boolean) => void;
  progressPage: number | null;
  goNext: () => void;
  goPrevious: () => void;
  clear: () => void;
};

type UsePdfSearchOptions = {
  documentId: string;
  engine: PdfEngine<Blob> | null;
  resetKey?: string | null;
};

const SEARCH_DEBOUNCE_MS = 250;

export function usePdfSearch({
  documentId,
  engine,
  resetKey,
}: UsePdfSearchOptions): PdfSearchState {
  const coreState = useCoreState();
  const document = coreState?.documents[documentId]?.document ?? null;
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [navigationVersion, setNavigationVersion] = useState(0);
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [progressPage, setProgressPage] = useState<number | null>(null);
  const requestIdRef = useRef(0);

  const trimmedQuery = query.trim();

  const flags = useMemo(() => {
    const nextFlags: MatchFlag[] = [];
    if (caseSensitive) nextFlags.push(MatchFlag.MatchCase);
    if (wholeWord) nextFlags.push(MatchFlag.MatchWholeWord);
    return nextFlags;
  }, [caseSensitive, wholeWord]);

  const clear = useCallback(() => {
    requestIdRef.current += 1;
    setQuery("");
    setResults([]);
    setActiveIndex(-1);
    setNavigationVersion(0);
    setIsLoading(false);
    setError(null);
    setProgressPage(null);
  }, []);

  useEffect(() => {
    clear();
  }, [clear, resetKey]);

  useEffect(() => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    if (!engine || !document || !trimmedQuery) {
      setResults([]);
      setActiveIndex(-1);
      setIsLoading(false);
      setError(null);
      setProgressPage(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    setProgressPage(null);

    const timer = window.setTimeout(() => {
      const task = engine.searchAllPages(document, trimmedQuery, { flags });
      task.onProgress((progress) => {
        if (requestIdRef.current !== requestId) return;
        setProgressPage(progress.page + 1);
      });
      task
        .toPromise()
        .then((value) => {
          if (requestIdRef.current !== requestId) return;
          setResults(value.results);
          setActiveIndex(value.results.length > 0 ? 0 : -1);
          setIsLoading(false);
          setProgressPage(null);
        })
        .catch(() => {
          if (requestIdRef.current !== requestId) return;
          setResults([]);
          setActiveIndex(-1);
          setIsLoading(false);
          setProgressPage(null);
          setError("Search failed");
        });
    }, SEARCH_DEBOUNCE_MS);

    return () => window.clearTimeout(timer);
  }, [document, engine, flags, trimmedQuery]);

  const goNext = useCallback(() => {
    if (results.length > 0) {
      setNavigationVersion((version) => version + 1);
    }
    setActiveIndex((index) => {
      return getAdjacentCyclicIndex(index, results.length, "next");
    });
  }, [results.length]);

  const goPrevious = useCallback(() => {
    if (results.length > 0) {
      setNavigationVersion((version) => version + 1);
    }
    setActiveIndex((index) => {
      return getAdjacentCyclicIndex(index, results.length, "previous");
    });
  }, [results.length]);

  return {
    query,
    setQuery,
    isOpen,
    setIsOpen,
    isLoading,
    error,
    results,
    activeIndex,
    activeResult: activeIndex >= 0 ? results[activeIndex] ?? null : null,
    navigationVersion,
    caseSensitive,
    setCaseSensitive,
    wholeWord,
    setWholeWord,
    progressPage,
    goNext,
    goPrevious,
    clear,
  };
}
