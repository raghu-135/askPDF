/**
 * useFileProcessing.ts - Hook for robust file processing status polling.
 *
 * This hook handles the race condition between file status updates and sentence availability
 * by continuing to poll when status is completed but sentences are not yet available.
 *
 * Phase 7: Frontend Polling Resilience - PostgreSQL Migration
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getFileStatus,
  getParsedSentences,
  type FileStatus,
  type ProcessStatus,
  ProcessStatusHelper,
} from '@/lib/api';

interface UseFileProcessingReturn {
  status: FileStatus | null;
  sentences: any[] | null;
  error: string | null;
  isPolling: boolean;
  retryCount: number;
}

interface UseFileProcessingOptions {
  pollInterval?: number;      // milliseconds, default 5000
  maxAttempts?: number;         // default 60 (5 minutes at 5s intervals)
  sentenceRetryDelay?: number;  // milliseconds to wait after completed status before retrying sentences
  maxSentenceRetries?: number; // max retries for sentences after completed status
}

/**
 * Hook for polling file processing status with resilience against race conditions.
 * 
 * Race condition handling:
 * - Backend updates status and sentences in a single atomic transaction (Phase 6)
 * - However, due to replication lag or other delays, we still implement defensive polling
 * - When status shows "completed" but sentences fetch fails, we continue polling
 * 
 * @param fileHash - The file hash to poll for
 * @param threadId - The thread ID context
 * @param options - Polling configuration options
 */
export function useFileProcessing(
  fileHash: string | null,
  threadId: string | null,
  options: UseFileProcessingOptions = {}
): UseFileProcessingReturn {
  const {
    pollInterval = 5000,
    maxAttempts = 60,
    sentenceRetryDelay = 1000,
    maxSentenceRetries = 5,
  } = options;

  const [status, setStatus] = useState<FileStatus | null>(null);
  const [sentences, setSentences] = useState<any[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [retryCount, setRetryCount] = useState(0);

  // Reset state when file or thread changes
  useEffect(() => {
    setStatus(null);
    setSentences(null);
    setError(null);
    setRetryCount(0);
  }, [fileHash, threadId]);

  const fetchSentencesWithRetry = useCallback(async (
    targetFileHash: string,
    targetThreadId: string
  ): Promise<{ sentences: any[] | null; success: boolean }> => {
    let sentenceAttempts = 0;
    
    while (sentenceAttempts < maxSentenceRetries) {
      try {
        const data = await getParsedSentences(targetFileHash, targetThreadId);
        if (data.sentences && data.sentences.length > 0) {
          return { sentences: data.sentences, success: true };
        }
        // Empty sentences array - might be valid or might be race condition
        return { sentences: data.sentences || [], success: true };
      } catch (e) {
        sentenceAttempts++;
        console.log(`Sentences fetch attempt ${sentenceAttempts} failed, retrying...`);
        
        if (sentenceAttempts < maxSentenceRetries) {
          await new Promise(resolve => setTimeout(resolve, sentenceRetryDelay));
        }
      }
    }
    
    return { sentences: null, success: false };
  }, [maxSentenceRetries, sentenceRetryDelay]);

  useEffect(() => {
    // Don't poll if missing required params
    if (!fileHash || !threadId) {
      return;
    }

    setIsPolling(true);
    let attempts = 0;
    let completedSeen = false;
    let pollIntervalId: NodeJS.Timeout | null = null;

    const poll = async () => {
      attempts++;
      setRetryCount(attempts);

      if (attempts > maxAttempts) {
        setError(`Timeout waiting for file processing after ${maxAttempts} attempts`);
        setIsPolling(false);
        if (pollIntervalId) {
          clearInterval(pollIntervalId);
        }
        return;
      }

      try {
        const fileStatus = await getFileStatus(fileHash, threadId) as FileStatus;
        setStatus(fileStatus);

        const parsingStatus = fileStatus.parsing?.status as ProcessStatus;

        // Handle completed status
        if (ProcessStatusHelper.isCompleted(parsingStatus)) {
          completedSeen = true;

          // Try to fetch sentences with retry logic
          const { sentences: fetchedSentences, success } = await fetchSentencesWithRetry(
            fileHash,
            threadId
          );

          if (success) {
            setSentences(fetchedSentences);
            setIsPolling(false);
            if (pollIntervalId) {
              clearInterval(pollIntervalId);
            }
            return;
          }

          // Sentences fetch failed after retries
          // Continue polling if we haven't exceeded max attempts
          if (attempts >= maxAttempts) {
            setError('Sentences not available after processing completed');
            setIsPolling(false);
            if (pollIntervalId) {
              clearInterval(pollIntervalId);
            }
          }
          return;
        }

        // Handle failed status
        if (ProcessStatusHelper.isFailed(parsingStatus)) {
          setError(fileStatus.parsing?.error || 'File processing failed');
          setIsPolling(false);
          if (pollIntervalId) {
            clearInterval(pollIntervalId);
          }
          return;
        }

        // Handle terminal indexing status (completed or failed)
        const indexingStatus = fileStatus.indexing?.status as ProcessStatus;
        if (ProcessStatusHelper.isTerminal(indexingStatus) && !completedSeen) {
          // Indexing done but parsing not marked - unusual but handle it
          console.warn('Indexing terminal but parsing not completed');
        }

        // Continue polling for pending/running states
      } catch (e) {
        console.error('Polling error:', e);
        // Don't stop polling on transient errors
      }
    };

    // Start polling immediately
    poll();

    // Set up interval for continued polling
    pollIntervalId = setInterval(poll, pollInterval);

    // Cleanup on unmount or when dependencies change
    return () => {
      if (pollIntervalId) {
        clearInterval(pollIntervalId);
      }
    };
  }, [fileHash, threadId, pollInterval, maxAttempts, fetchSentencesWithRetry]);

  return {
    status,
    sentences,
    error,
    isPolling,
    retryCount,
  };
}

export default useFileProcessing;
