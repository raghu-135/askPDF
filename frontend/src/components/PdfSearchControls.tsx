import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCoreState } from "@embedpdf/core/react";
import {
  MatchFlag,
  type PdfEngine,
  type SearchResult,
} from "@embedpdf/models";
import {
  Box,
  CircularProgress,
  IconButton,
  InputAdornment,
  Stack,
  TextField,
  ToggleButton,
  Tooltip,
  Typography,
} from "@mui/material";
import AbcIcon from "@mui/icons-material/Abc";
import CloseIcon from "@mui/icons-material/Close";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp";
import SearchIcon from "@mui/icons-material/Search";
import TextFieldsIcon from "@mui/icons-material/TextFields";
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
    setActiveIndex((index) => {
      return getAdjacentCyclicIndex(index, results.length, "next");
    });
  }, [results.length]);

  const goPrevious = useCallback(() => {
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

type PdfSearchControlsProps = {
  search: PdfSearchState;
};

export const PdfSearchControls = React.memo(function PdfSearchControls({
  search,
}: PdfSearchControlsProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const resultText = search.error
    ? search.error
    : search.query.trim() && !search.isLoading && search.results.length === 0
      ? "No results"
      : search.results.length > 0
        ? `${search.activeIndex + 1} / ${search.results.length}`
        : search.isLoading && search.progressPage
          ? `Page ${search.progressPage}`
          : "";

  useEffect(() => {
    if (!search.isOpen) return;
    const t = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(t);
  }, [search.isOpen]);

  const handleToggleOpen = useCallback(() => {
    search.setIsOpen(!search.isOpen);
  }, [search]);

  const handleClose = useCallback(() => {
    search.clear();
    search.setIsOpen(false);
  }, [search]);

  if (!search.isOpen) {
    return (
      <Tooltip title="Search document">
        <IconButton size="small" onClick={handleToggleOpen}>
          <SearchIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    );
  }

  return (
    <Stack direction="row" spacing={0.5} sx={{ alignItems: "center", minWidth: 0 }}>
      <TextField
        size="small"
        value={search.query}
        inputRef={inputRef}
        placeholder="Search"
        onChange={(event) => search.setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter") return;
          event.preventDefault();
          if (event.shiftKey) {
            search.goPrevious();
          } else {
            search.goNext();
          }
        }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon fontSize="small" />
            </InputAdornment>
          ),
          endAdornment: search.isLoading ? (
            <InputAdornment position="end">
              <CircularProgress size={14} />
            </InputAdornment>
          ) : null,
        }}
        sx={{
          width: { xs: 140, sm: 200, md: 240 },
          "& .MuiInputBase-root": {
            height: 32,
          },
          "& input": {
            py: 0.5,
          },
        }}
      />

      <Box sx={{ minWidth: 68, textAlign: "center" }}>
        <Typography
          variant="caption"
          color={search.error ? "error" : "text.secondary"}
          sx={{ whiteSpace: "nowrap" }}
        >
          {resultText}
        </Typography>
      </Box>

      <Tooltip title="Previous result">
        <span>
          <IconButton
            size="small"
            onClick={search.goPrevious}
            disabled={search.results.length === 0}
          >
            <KeyboardArrowUpIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>

      <Tooltip title="Next result">
        <span>
          <IconButton
            size="small"
            onClick={search.goNext}
            disabled={search.results.length === 0}
          >
            <KeyboardArrowDownIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>

      <Tooltip title="Match case">
        <ToggleButton
          size="small"
          value="case"
          selected={search.caseSensitive}
          onChange={() => search.setCaseSensitive(!search.caseSensitive)}
          sx={{ width: 30, height: 30, p: 0 }}
        >
          <TextFieldsIcon fontSize="small" />
        </ToggleButton>
      </Tooltip>

      <Tooltip title="Whole word">
        <ToggleButton
          size="small"
          value="word"
          selected={search.wholeWord}
          onChange={() => search.setWholeWord(!search.wholeWord)}
          sx={{ width: 30, height: 30, p: 0 }}
        >
          <AbcIcon fontSize="small" />
        </ToggleButton>
      </Tooltip>

      <Tooltip title="Close search">
        <IconButton size="small" onClick={handleClose}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Tooltip>
    </Stack>
  );
});
