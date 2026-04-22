declare const process: {
  env: Record<string, string | undefined>;
};
import { Button, Tooltip, CircularProgress, Box, Typography } from "@mui/material";
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import React from "react";
import { getFileStatus, getParsedSentences, FileStatus, ProcessStatusHelper } from "../lib/api";

type Props = {
  threadId?: string | null;
  embeddingModel?: string | null;
  onUploaded: (data: { sentences: any[] | null; pdfUrl: string; fileHash: string; fileName?: string }) => void;
  onIndexingComplete?: (fileHash: string) => void;
  onParsingComplete?: (fileHash: string, sentences: any[]) => void;
  disabled?: boolean;
  tooltipText?: string;
  /** Current web URL value - when provided, button switches to web mode */
  webUrl?: string;
  /** Handler for web URL submission when in web mode */
  onWebSubmit?: () => void;
  /** Loading state for web submission */
  isWebLoading?: boolean;
};

const PdfUploader = React.memo(function PdfUploader({
  threadId,
  embeddingModel,
  onUploaded,
  onIndexingComplete,
  onParsingComplete,
  disabled,
  tooltipText,
  webUrl,
  onWebSubmit,
  isWebLoading = false
}: Props) {
  const inputId = "pdf-upload-input";
  const [isUploading, setIsUploading] = React.useState(false);
  const [fileStatus, setFileStatus] = React.useState<{
    fileHash: string;
    status: FileStatus;
  } | null>(null);

  const isWebMode = !!webUrl?.trim();
  const isDisabled = disabled || isUploading || isWebLoading;

  // Poll for file status (parsing and indexing)
  React.useEffect(() => {
    if (!fileStatus) {
      return;
    }

    // Check if it's a full FileStatus object
    if ('parsing' in fileStatus.status && 'indexing' in fileStatus.status) {
      const { parsing, indexing } = fileStatus.status;
      // Stop polling if both parsing and indexing are completed or failed
      if (ProcessStatusHelper.isTerminal(parsing.status) && ProcessStatusHelper.isTerminal(indexing.status)) {
        if (ProcessStatusHelper.isCompleted(parsing.status) && ProcessStatusHelper.isCompleted(indexing.status)) {
          onIndexingComplete?.(fileStatus.fileHash);
        }
        return;
      }
    }

    const pollInterval = setInterval(async () => {
      try {
        const status = await getFileStatus(fileStatus.fileHash, {
          embeddingModel: embeddingModel || undefined,
          threadId: threadId || undefined,
        });
        // Ensure we have a full FileStatus object
        const fullStatus: FileStatus = 'parsing' in status && 'indexing' in status
          ? status as FileStatus
          : {
              parsing: { status: 'unknown' },
              indexing: { status: 'unknown' },
              indexing_status: { summary: { status: 'unknown' }, models: {} },
              updated_at: new Date().toISOString(),
            };

        setFileStatus({
          fileHash: fileStatus.fileHash,
          status: fullStatus
        });

        // Check if parsing just completed
        if (ProcessStatusHelper.isCompleted(fullStatus.parsing.status) && onParsingComplete) {
          try {
            const parsedData = await getParsedSentences(fileStatus.fileHash);
            onParsingComplete(fileStatus.fileHash, parsedData.sentences);
          } catch (error) {
            console.error("Failed to fetch parsed sentences", error);
          }
        }

        // Check if both parsing and indexing are completed
        if (ProcessStatusHelper.isCompleted(fullStatus.parsing.status) && ProcessStatusHelper.isCompleted(fullStatus.indexing.status)) {
          clearInterval(pollInterval);
          onIndexingComplete?.(fileStatus.fileHash);
        } else if (ProcessStatusHelper.isFailed(fullStatus.parsing.status) || ProcessStatusHelper.isFailed(fullStatus.indexing.status)) {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error("Failed to check file status", error);
      }
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [embeddingModel, fileStatus?.fileHash, fileStatus?.status, onIndexingComplete, onParsingComplete, threadId]);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setFileStatus(null);
    
    try {
      if (!threadId) {
        throw new Error("A thread must be selected before uploading.");
      }
      const form = new FormData();
      form.append("file", file);
      form.append("thread_id", threadId);
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      
      // Set initial file status - will be updated by polling
      setFileStatus({
        fileHash: data.fileHash,
        status: {
          parsing: { status: 'pending' },
          indexing: { status: 'pending' },
          indexing_status: { summary: { status: 'pending' }, models: {} },
          updated_at: new Date().toISOString()
        }
      });
      
      onUploaded({ ...data, fileName: file.name });
    } catch (error) {
      console.error("Upload failed", error);
    } finally {
      setIsUploading(false);
      e.target.value = ""; // reset
    }
  };


  const handleButtonClick = () => {
    if (isWebMode && onWebSubmit) {
      onWebSubmit();
    }
    // If not in web mode, the label click will trigger file input
  };

  const buttonLabel = isWebLoading 
    ? "Uploading..." 
    : isWebMode 
      ? "Upload webpage" 
      : isUploading 
        ? "Uploading..." 
        : "Upload PDF";

  const button = (
    <>
      <input
        id={inputId}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        style={{ display: "none" }}
        disabled={isDisabled || isWebMode}
      />
      <label htmlFor={inputId} style={{ cursor: isWebMode ? 'default' : 'pointer' }}>
        <Button
          variant="contained"
          component="span"
          disabled={isDisabled}
          onClick={isWebMode ? handleButtonClick : undefined}
        >
          {buttonLabel}
        </Button>
      </label>
    </>
  );

  const content = button;

  if (tooltipText && isDisabled) {
    return <Tooltip title={tooltipText}><span>{content}</span></Tooltip>;
  }

  return content;
});

export default PdfUploader;
