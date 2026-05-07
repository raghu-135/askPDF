/**
 * Retry utilities for API calls with exponential backoff and smart error handling
 */

import { classifyError, shouldStopPolling, isRetryableError, isNotFoundError, getRateLimitDelay } from './error-utils';

export interface RetryOptions {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  retryableErrors?: (error: any) => boolean;
}

export interface RetryResult<T> {
  success: boolean;
  data?: T;
  error?: any;
  attempts: number;
  stopped?: boolean; // For polling operations
  resourceNotFound?: boolean; // For deleted resources
}

/**
 * Generic retry function with exponential backoff
 */
export async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<RetryResult<T>> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 10000,
    retryableErrors = isRetryableError
  } = options;

  let lastError: any;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const data = await operation();
      return {
        success: true,
        data,
        attempts: attempt
      };
    } catch (error) {
      lastError = error;
      
      // Check if error is retryable
      if (!retryableErrors(error) || attempt === maxRetries) {
        return {
          success: false,
          error: lastError,
          attempts: attempt,
          resourceNotFound: isNotFoundError(lastError)
        };
      }
      
      // Calculate delay with exponential backoff
      const delay = Math.min(baseDelay * Math.pow(2, attempt - 1), maxDelay);
      // Use rate limit delay if applicable
      const actualDelay = getRateLimitDelay(lastError, delay);
      console.warn(`Retry attempt ${attempt}/${maxRetries} failed, retrying in ${actualDelay}ms:`, error);
      
      await new Promise(resolve => setTimeout(resolve, actualDelay));
    }
  }
  
  return {
    success: false,
    error: lastError,
    attempts: maxRetries,
    resourceNotFound: isNotFoundError(lastError)
  };
}

/**
 * Retry function specifically for polling operations
 */
export async function withPollingRetry<T>(
  operation: () => Promise<T>,
  options: {
    maxRetries?: number;
    interval?: number;
    shouldStop?: (result: T) => boolean;
    retryableErrors?: (error: any) => boolean;
  } = {}
): Promise<{ success: boolean; data?: T; error?: any; stopped: boolean; resourceNotFound?: boolean }> {
  const {
    maxRetries = 3,
    interval = 5000,
    shouldStop = () => false,
    retryableErrors = isRetryableError
  } = options;

  let consecutiveFailures = 0;
  
  while (consecutiveFailures < maxRetries) {
    try {
      const result = await operation();
      
      // Check if we should stop based on result
      if (shouldStop(result)) {
        return {
          success: true,
          data: result,
          stopped: true,
          resourceNotFound: false
        };
      }
      
      // Success - reset failure counter
      consecutiveFailures = 0;
      
    } catch (error) {
      consecutiveFailures++;
      const errorClassified = classifyError(error);
      
      console.error(`Polling attempt ${consecutiveFailures}/${maxRetries} failed (${errorClassified.type}):`, error);
      
      // Check if error should stop polling
      if (shouldStopPolling(error) || !retryableErrors(error)) {
        return {
          success: false,
          error,
          stopped: false,
          resourceNotFound: isNotFoundError(error)
        };
      }
    }
    
    // Wait before next attempt
    if (consecutiveFailures < maxRetries) {
      // Use rate limit delay if applicable
      const lastError = consecutiveFailures > 0 ? new Error('Previous error') : null;
      const actualDelay = lastError ? getRateLimitDelay(lastError, interval) : interval;
      await new Promise(resolve => setTimeout(resolve, actualDelay));
    }
  }
  
  return {
    success: false,
    error: consecutiveFailures > 0 ? new Error(`Max retries (${maxRetries}) exceeded`) : undefined,
    stopped: false,
    resourceNotFound: false
  };
}
