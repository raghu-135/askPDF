import { Button, Tooltip } from "@mui/material";
import React from "react";
import { getFileStatus, getParsedSentences, FileStatus, ProcessStatusHelper, uploadPdf as apiUploadPdf } from "../lib/api";
import { isRetryableError, isNotFoundError } from "../lib/error-utils";

type Props = {
  threadId?: string | null;
  embeddingModel?: string | null;
  onUploaded: (data: { sentences: any[] | null; pdfUrl: string; fileHash: string; fileName?: string }) => void;
  onIndexingComplete?: (fileHash: string) => void;
  onParsingComplete?: (fileHash: string, sentences: any[]) => void;
  disabled?: boolean;
  tooltipText?: string;
};

const PdfUploader = React.memo(function PdfUploader({
  threadId,
  embeddingModel,
  onUploaded,
  onIndexingComplete,
  onParsingComplete,
  disabled,
  tooltipText,
}: Props) {
  const inputId = "pdf-upload-input";
  const [isUploading, setIsUploading] = React.useState(false);
  const [fileStatus, setFileStatus] = React.useState<{
    fileHash: string;
    status: FileStatus;
  } | null>(null);

  const isDisabled = disabled || isUploading;

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
        if (!threadId) {
          console.error("Thread ID is required for file status polling");
          return;
        }
        const status = await getFileStatus(fileStatus.fileHash, threadId, {
          embeddingModel: embeddingModel || undefined,
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
            if (!threadId) {
              console.error("Thread ID is required for fetching parsed sentences");
              return;
            }
            const parsedData = await getParsedSentences(fileStatus.fileHash, threadId);
            onParsingComplete(fileStatus.fileHash, parsedData.sentences);
          } catch (error) {
            console.error("Failed to fetch parsed sentences", error);
            
            // Check if error should stop retrying
            if (!isRetryableError(error)) {
              if (isNotFoundError(error)) {
                console.log('File or thread no longer exists, stopping sentences fetch');
              } else {
                console.log('Permanent error in sentences fetch:', error?.message);
              }
              // Don't retry for permanent errors
            }
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
        
        // Check if error should stop polling
        if (!isRetryableError(error)) {
          if (isNotFoundError(error)) {
            console.log('File or thread no longer exists, stopping polling');
          } else {
            console.log('Permanent error in file status polling:', error?.message);
          }
          clearInterval(pollInterval);
        }
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
      const data = await apiUploadPdf(file, threadId);

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
      
      // Provide better error feedback
      if (isNotFoundError(error)) {
        console.error('Thread not found for upload');
      } else if (!isRetryableError(error)) {
        console.error('Permanent upload error:', error?.message);
      }
    } finally {
      setIsUploading(false);
      e.target.value = ""; // reset
    }
  };


  const buttonLabel = isUploading ? "Uploading..." : "Upload PDF";

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
