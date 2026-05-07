/**
 * Error classification utilities for handling different types of HTTP errors
 */

export interface ClassifiedError {
  type: 'permanent' | 'transient' | 'rate_limit' | 'unknown';
  status?: number;
  message: string;
  isNotFound: boolean;
}

/**
 * Extract HTTP status code from error message or response
 */
function extractStatusCode(error: any): number | undefined {
  // Try to get status from error object
  if (error?.status) return error.status;
  if (error?.statusCode) return error.statusCode;
  
  // Try to parse status from error message
  const message = error?.message || error?.toString() || '';
  const statusMatch = message.match(/status\s*[:=]\s*(\d+)/i);
  if (statusMatch) {
    return parseInt(statusMatch[1], 10);
  }
  
  return undefined;
}

/**
 * Check if error represents a not found (404) error
 */
function isNotFoundStatus(status: number | undefined): boolean {
  return status === 404;
}

/**
 * Check if error message contains not found indicators
 */
function hasNotFoundMessage(message: string): boolean {
  const notFoundIndicators = [
    'not found',
    'does not exist',
    'resource not found',
    'thread not found',
    'file not found',
    'message not found'
  ];
  
  const lowerMessage = message.toLowerCase();
  return notFoundIndicators.some(indicator => lowerMessage.includes(indicator));
}

/**
 * Classify error type based on HTTP status and message content
 */
export function classifyError(error: any): ClassifiedError {
  const message = error?.message || error?.toString() || 'Unknown error';
  const status = extractStatusCode(error);
  
  // Check for not found specifically
  const isNotFound = isNotFoundStatus(status) || hasNotFoundMessage(message);
  
  // Determine error type based on status code
  let type: ClassifiedError['type'];
  
  if (status) {
    if (status === 429) {
      type = 'rate_limit';
    } else if (status >= 400 && status < 500) {
      // Client errors (4xx) are typically permanent
      type = 'permanent';
    } else if (status >= 500) {
      // Server errors (5xx) are typically transient
      type = 'transient';
    } else {
      type = 'unknown';
    }
  } else {
    // Network errors or other issues without status codes
    if (message.toLowerCase().includes('network') || 
        message.toLowerCase().includes('fetch') ||
        message.toLowerCase().includes('connection')) {
      type = 'transient';
    } else {
      type = 'unknown';
    }
  }
  
  return {
    type,
    status,
    message,
    isNotFound
  };
}

/**
 * Check if error should stop polling (permanent error)
 */
export function shouldStopPolling(error: any): boolean {
  const classified = classifyError(error);
  return classified.type === 'permanent' || classified.type === 'unknown';
}

/**
 * Check if error is retryable (transient or rate limit)
 */
export function isRetryableError(error: any): boolean {
  const classified = classifyError(error);
  return classified.type === 'transient' || classified.type === 'rate_limit';
}

/**
 * Check if error specifically indicates resource not found
 */
export function isNotFoundError(error: any): boolean {
  const classified = classifyError(error);
  return classified.isNotFound;
}

/**
 * Get appropriate retry delay for rate limiting
 */
export function getRateLimitDelay(error: any, defaultDelay: number = 1000): number {
  const classified = classifyError(error);
  if (classified.type !== 'rate_limit') return defaultDelay;
  
  // Try to extract retry-after header or from message
  const message = classified.message.toLowerCase();
  const retryMatch = message.match(/retry\s*after\s*[:=]\s*(\d+)/i);
  if (retryMatch) {
    return parseInt(retryMatch[1], 10) * 1000; // Convert to milliseconds
  }
  
  // Default exponential backoff for rate limiting
  return defaultDelay * 2;
}
