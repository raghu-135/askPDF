import React, { useState } from "react";
import {
  Box,
  Button,
  CircularProgress,
  IconButton,
  InputAdornment,
  TextField,
  Tooltip,
} from "@mui/material";
import LanguageIcon from "@mui/icons-material/Language";
import ClearIcon from "@mui/icons-material/Clear";
import { addWebSourceToThread } from "../lib/api";

type IndexedData = {
  fileHash: string;
  url: string;
  title?: string;
  status: "accepted" | "error";
  message?: string;
};

type Props = {
  /** Active thread ID — indexing is disabled when null. */
  threadId: string | null;
  /** Called when indexing has been accepted by the server. */
  onIndexed: (data: IndexedData) => void;
  disabled?: boolean;
  tooltipText?: string;
};

export default function WebUploader({ threadId, onIndexed, disabled, tooltipText }: Props) {
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isDisabled = disabled || isLoading || !threadId;

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!threadId || !url.trim()) return;

    const trimmed = url.trim();
    if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
      setError("URL must start with http:// or https://");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await addWebSourceToThread(threadId, trimmed);
      setUrl("");
      onIndexed({
        fileHash: result.file_hash,
        url: trimmed,
        status: "accepted",
      });
    } catch (err: any) {
      const msg = err?.message || "Failed to index webpage";
      setError(msg);
      onIndexed({ fileHash: "", url: trimmed, status: "error", message: msg });
    } finally {
      setIsLoading(false);
    }
  };

  const inner = (
    <Box
      component="form"
      onSubmit={handleSubmit}
      sx={{ display: "flex", alignItems: "center", gap: 1 }}
    >
      <TextField
        size="small"
        placeholder="https://example.com"
        value={url}
        onChange={(e) => {
          setUrl(e.target.value);
          setError(null);
        }}
        disabled={isDisabled}
        error={!!error}
        helperText={error || undefined}
        sx={{ minWidth: 260 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <LanguageIcon fontSize="small" color={isDisabled ? "disabled" : "action"} />
            </InputAdornment>
          ),
          endAdornment: url ? (
            <InputAdornment position="end">
              <IconButton size="small" onClick={() => { setUrl(""); setError(null); }} tabIndex={-1}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </InputAdornment>
          ) : undefined,
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            handleSubmit();
          }
        }}
      />
      <Button
        type="submit"
        variant="contained"
        disabled={isDisabled || !url.trim()}
        startIcon={isLoading ? <CircularProgress size={14} color="inherit" /> : <LanguageIcon />}
        sx={{ whiteSpace: "nowrap" }}
      >
        {isLoading ? "Indexing…" : "Add Webpage"}
      </Button>
    </Box>
  );

  if (tooltipText && (!threadId || disabled)) {
    return <Tooltip title={tooltipText}><span>{inner}</span></Tooltip>;
  }

  return inner;
}
