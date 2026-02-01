import { Button, Tooltip, CircularProgress, Box, Typography } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import React from "react";
import { getFileIndexStatus, IndexingStatus as IndexStatus } from "../lib/api";

type Props = {
  embedModel: string;
  onUploaded: (data: { sentences: any[]; pdfUrl: string; fileHash: string; fileName?: string }) => void;
  onIndexingComplete?: (fileHash: string) => void;
  disabled?: boolean;
  tooltipText?: string;
};

export default function PdfUploader({ embedModel, onUploaded, onIndexingComplete, disabled, tooltipText }: Props) {
  const inputId = "pdf-upload-input";
  const [isUploading, setIsUploading] = React.useState(false);
  const [indexingState, setIndexingState] = React.useState<{
    fileHash: string;
    status: IndexStatus;
    progress: number;
    error?: string;
  } | null>(null);

  const isDisabled = disabled || !embedModel || isUploading;

  // Poll for indexing status
  React.useEffect(() => {
    if (!indexingState || indexingState.status === 'ready' || indexingState.status === 'failed') {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const status = await getFileIndexStatus(indexingState.fileHash);
        setIndexingState({
          fileHash: status.file_hash,
          status: status.status,
          progress: status.progress || 0,
          error: status.error
        });

        if (status.status === 'ready') {
          clearInterval(pollInterval);
          onIndexingComplete?.(status.file_hash);
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error("Failed to check indexing status", error);
      }
    }, 2000);

    return () => clearInterval(pollInterval);
  }, [indexingState?.fileHash, indexingState?.status, onIndexingComplete]);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setIndexingState(null);
    
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("embedding_model", embedModel);
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      
      // Set initial indexing state
      setIndexingState({
        fileHash: data.fileHash,
        status: data.indexingStatus || 'pending',
        progress: 0
      });
      
      onUploaded({ ...data, fileName: file.name });
    } catch (error) {
      console.error("Upload failed", error);
    } finally {
      setIsUploading(false);
      e.target.value = ""; // reset
    }
  };

  const renderIndexingStatus = () => {
    if (!indexingState) return null;

    const { status, progress, error } = indexingState;

    if (status === 'ready') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 2 }}>
          <CheckCircleIcon color="success" fontSize="small" />
          <Typography variant="body2" color="success.main">
            Indexed
          </Typography>
        </Box>
      );
    }

    if (status === 'failed') {
      return (
        <Tooltip title={error || 'Indexing failed'}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 2 }}>
            <ErrorIcon color="error" fontSize="small" />
            <Typography variant="body2" color="error">
              Failed
            </Typography>
          </Box>
        </Tooltip>
      );
    }

    if (status === 'pending' || status === 'indexing') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, ml: 2 }}>
          <CircularProgress size={16} />
          <Typography variant="body2" color="text.secondary">
            Indexing{progress > 0 ? ` (${progress}%)` : '...'}
          </Typography>
        </Box>
      );
    }

    return null;
  };

  const button = (
    <>
      <input
        id={inputId}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        style={{ display: "none" }}
        disabled={isDisabled}
      />
      <label htmlFor={inputId}>
        <Button
          variant="contained"
          component="span"
          disabled={isDisabled}
        >
          {isUploading ? "Uploading..." : "Upload PDF"}
        </Button>
      </label>
    </>
  );

  const content = (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      {button}
      {renderIndexingStatus()}
    </Box>
  );

  if (tooltipText && isDisabled) {
    return <Tooltip title={tooltipText}><span>{content}</span></Tooltip>;
  }

  return content;
}
