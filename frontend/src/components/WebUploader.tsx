import React, { useState } from "react";
import {
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

type AddWebSourceResult = {
  status: string;
  file_hash: string;
  url: string;
  title?: string;
  indexing: string;
};

type Props = {
  /** Active thread ID — indexing is disabled when null. */
  threadId: string | null;
  /** Called when indexing has been accepted by the server. */
  onIndexed: (data: IndexedData) => void;
  disabled?: boolean;
  tooltipText?: string;
  /** External control: current URL value */
  value?: string;
  /** External control: callback when URL changes */
  onChange?: (url: string) => void;
  /** External control: callback to clear URL */
  onClear?: () => void;
  /** External control: callback when user presses Enter */
  onSubmit?: () => void;
  /** External control: loading state during submission */
  isLoading?: boolean;
};

const WebUploader = React.memo(function WebUploader({ 
  threadId, 
  onIndexed, 
  disabled, 
  tooltipText,
  value,
  onChange,
  onClear,
  onSubmit,
  isLoading: externalLoading
}: Props) {
  const [internalUrl, setInternalUrl] = useState("");
  const [internalLoading, setInternalLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Use controlled value if provided, otherwise internal state
  const url = value !== undefined ? value : internalUrl;
  const setUrl = (newUrl: string) => {
    if (onChange) {
      onChange(newUrl);
    } else {
      setInternalUrl(newUrl);
    }
  };

  // Use external loading state if provided, otherwise internal
  const isLoading = externalLoading !== undefined ? externalLoading : internalLoading;
  const setIsLoading = (loading: boolean) => {
    if (externalLoading === undefined) {
      setInternalLoading(loading);
    }
  };

  const isDisabled = disabled || isLoading || !threadId;

  const validateUrl = (urlToValidate: string): string | null => {
    const trimmed = urlToValidate.trim();
    if (!trimmed) return null;
    if (!trimmed.startsWith("http://") && !trimmed.startsWith("https://")) {
      return "URL must start with http:// or https://";
    }
    return null;
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!threadId || !url.trim()) return;

    const trimmed = url.trim();
    const validationError = validateUrl(url);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await addWebSourceToThread(threadId, trimmed);
      setUrl("");
      onClear?.();
      onIndexed({
        fileHash: result.file_hash,
        url: trimmed,
        title: result.title,
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

  // Expose submit handler for external control
  React.useImperativeHandle(
    React.useRef<{ submit: () => Promise<void> }>({ submit: handleSubmit }),
    () => ({ submit: handleSubmit })
  );

  const handleClear = () => {
    setUrl("");
    setError(null);
    onClear?.();
  };

  const inner = (
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
      sx={{ minWidth: 300, flex: 1 }}
      InputProps={{
        startAdornment: (
          <InputAdornment position="start">
            <LanguageIcon fontSize="small" color="action" />
          </InputAdornment>
        ),
        endAdornment: url ? (
          <InputAdornment position="end">
            <IconButton
              size="small"
              onClick={handleClear}
              tabIndex={-1}
            >
              <ClearIcon fontSize="small" />
            </IconButton>
          </InputAdornment>
        ) : null,
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" && url.trim()) {
          e.preventDefault();
          onSubmit?.();
        }
      }}
    />
  );

  if (tooltipText && (!threadId || disabled)) {
    return <Tooltip title={tooltipText}><span>{inner}</span></Tooltip>;
  }

  return inner;
});

export default WebUploader;
