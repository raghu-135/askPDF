import React, { useCallback, useEffect, useRef } from "react";
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
import type { PdfSearchState } from "../hooks/usePdfSearch";

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
