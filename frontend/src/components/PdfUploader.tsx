import { Button, Tooltip } from "@mui/material";
import React from "react";
import { getTargetFileStatus, getParsedSentencesForTarget, FileStatus, ProcessStatusHelper, uploadPdfToTarget, type KnowledgeTarget } from "../lib/api";
import { ProcessStatus } from "../lib/enums";
import { isRetryableError, isNotFoundError } from "../lib/error-utils";

type Props = {
  target?: KnowledgeTarget | null;
  onUploaded: (data: { sentences: any[] | null; downloadUrl: string; fileHash: string; fileName?: string }) => void;
  onIndexingComplete?: (fileHash: string) => void;
  onParsingComplete?: (fileHash: string, sentences: any[]) => void;
  disabled?: boolean;
  tooltipText?: string;
};

const PdfUploader = React.memo(function PdfUploader({
  target,
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
    target: KnowledgeTarget;
  } | null>(null);
  const indexingNotifiedRef = React.useRef<string | null>(null);
  const parsingNotifiedRef = React.useRef<string | null>(null);

  const isDisabled = disabled || isUploading;

  // Poll for file status (parsing and indexing)
  React.useEffect(() => {
    if (!fileStatus) {
      return;
    }

    // Check if it's a full FileStatus object
    if ('parsing' in fileStatus.status && 'indexing' in fileStatus.status) {
      const { parsing, indexing } = fileStatus.status;
      if (
        ProcessStatusHelper.isCompleted(indexing.status) &&
        indexingNotifiedRef.current !== fileStatus.fileHash
      ) {
        indexingNotifiedRef.current = fileStatus.fileHash;
        onIndexingComplete?.(fileStatus.fileHash);
      }

      // Stop polling if both parsing and indexing are completed or failed
      if (ProcessStatusHelper.isTerminal(parsing.status) && ProcessStatusHelper.isTerminal(indexing.status)) {
        return;
      }
    }

    const pollInterval = setInterval(async () => {
      try {
        const status = await getTargetFileStatus(fileStatus.fileHash, fileStatus.target);
        // Ensure we have a full FileStatus object
        const fullStatus: FileStatus = 'parsing' in status && 'indexing' in status
          ? status as FileStatus
          : {
              parsing: { status: ProcessStatus.Unknown },
              indexing: { status: ProcessStatus.Unknown },
              indexing_status: { summary: { status: ProcessStatus.Unknown }, models: {} },
              updated_at: new Date().toISOString(),
            };

        setFileStatus({
          fileHash: fileStatus.fileHash,
          status: fullStatus,
          target: fileStatus.target,
        });

        if (
          ProcessStatusHelper.isCompleted(fullStatus.indexing.status) &&
          indexingNotifiedRef.current !== fileStatus.fileHash
        ) {
          indexingNotifiedRef.current = fileStatus.fileHash;
          onIndexingComplete?.(fileStatus.fileHash);
        }

        // Check if parsing just completed
        if (
          ProcessStatusHelper.isCompleted(fullStatus.parsing.status) &&
          parsingNotifiedRef.current !== fileStatus.fileHash &&
          onParsingComplete
        ) {
          try {
            if (!target) {
              return;
            }
            const parsedData = await getParsedSentencesForTarget(fileStatus.fileHash, target);
            parsingNotifiedRef.current = fileStatus.fileHash;
            onParsingComplete(fileStatus.fileHash, parsedData.sentences);
          } catch (error) {
            console.error("Failed to fetch parsed sentences", error);
            
            // Check if error should stop retrying
            if (!isRetryableError(error)) {
              // Don't retry for permanent errors.
            }
          }
        }

        // Check if both parsing and indexing are completed
        if (ProcessStatusHelper.isCompleted(fullStatus.parsing.status) && ProcessStatusHelper.isCompleted(fullStatus.indexing.status)) {
          clearInterval(pollInterval);
        } else if (ProcessStatusHelper.isFailed(fullStatus.parsing.status) || ProcessStatusHelper.isFailed(fullStatus.indexing.status)) {
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error("Failed to check file status", error);
        
        // Check if error should stop polling
        if (!isRetryableError(error)) {
          clearInterval(pollInterval);
        }
      }
    }, 5000);

    return () => clearInterval(pollInterval);
  }, [fileStatus?.fileHash, fileStatus?.status, fileStatus?.target, onIndexingComplete, onParsingComplete]);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setFileStatus(null);
    indexingNotifiedRef.current = null;
    parsingNotifiedRef.current = null;

    try {
      if (!target) {
        throw new Error("A thread or project must be selected before uploading.");
      }
      const data = await uploadPdfToTarget(file, target);

      // Set initial file status - will be updated by polling
      setFileStatus({
        fileHash: data.fileHash,
        target,
        status: {
          parsing: { status: ProcessStatus.Pending },
          indexing: { status: ProcessStatus.Pending },
          indexing_status: { summary: { status: ProcessStatus.Pending }, models: {} },
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
