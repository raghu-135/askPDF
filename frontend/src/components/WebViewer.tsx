/**
 * WebViewer
 *
 * Renders a captured webpage inside a sandboxed <iframe>.
 *
 * How it works (and why there are no X-Frame-Options / CSP errors):
 *  1. The backend fetches the URL server-side, inlines all CSS and images as
 *     base64 data URIs, and saves a self-contained HTML file.
 *  2. This component fetches that HTML text from our own backend.
 *  3. It wraps the HTML in a Blob URL (`blob:…`) and sets that as the iframe
 *     `src`.  Because it is a local Blob — not a request to the original site —
 *     the browser never sees the original `X-Frame-Options` or CSP headers.
 *
 * The iframe uses `sandbox="allow-scripts"`.  External <script src> tags are
 * stripped by the backend so only inline scripts remain; CSS/images are already
 * inlined, so no cross-origin fetches are needed inside the frame.
 */

import React, { useEffect, useRef, useState } from 'react';
import { Box, CircularProgress, Typography, Button, Tooltip, IconButton } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { captureWebPage, fetchWebPageHtml } from '../lib/api';

type Props = {
  /** The original URL of the webpage. */
  url: string;
  /** MD5 hash of the URL — used to retrieve the saved HTML. */
  fileHash: string;
  darkMode?: boolean;
  isResizing?: boolean;
};

type Status = 'idle' | 'capturing' | 'loading' | 'ready' | 'error';

const WebViewer = React.memo(function WebViewer({ url, fileHash, darkMode = false, isResizing = false }: Props) {
  const [status, setStatus] = useState<Status>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pageTitle, setPageTitle] = useState<string>('');
  const blobUrlRef = useRef<string | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const load = async (force = false) => {
    setStatus('capturing');
    setErrorMsg(null);

    try {
      // Step 1 – ask backend to capture (caches after first call)
      const capture = await captureWebPage(url, force);
      setPageTitle(capture.title);
    } catch (err: any) {
      setStatus('error');
      setErrorMsg(`Capture failed: ${err.message}`);
      return;
    }

    setStatus('loading');

    try {
      // Step 2 – fetch the saved HTML as text
      const html = await fetchWebPageHtml(fileHash);

      // Step 3 – revoke any previous blob URL to avoid memory leaks
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
      }

      // Step 4 – create a local Blob URL and point the iframe at it
      const blob = new Blob([html], { type: 'text/html' });
      blobUrlRef.current = URL.createObjectURL(blob);

      if (iframeRef.current) {
        iframeRef.current.src = blobUrlRef.current;
      }

      setStatus('ready');
    } catch (err: any) {
      setStatus('error');
      setErrorMsg(`Failed to load page: ${err.message}`);
    }
  };

  // Trigger initial load once on mount
  useEffect(() => {
    load();
    return () => {
      // Clean up blob URL on unmount
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileHash, url]);

  const isLoading = status === 'capturing' || status === 'loading';

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        bgcolor: darkMode ? '#1a1a1a' : '#f5f5f5',
        overflow: 'hidden',
      }}
    >
      {/* Toolbar */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1.5,
          py: 0.75,
          borderBottom: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
          bgcolor: darkMode ? '#242424' : '#fff',
          minHeight: 40,
          flexShrink: 0,
        }}
      >
        <Typography
          variant="caption"
          noWrap
          sx={{
            flex: 1,
            color: darkMode ? '#aaa' : 'text.secondary',
            fontFamily: 'monospace',
            fontSize: '0.7rem',
          }}
        >
          {pageTitle || url}
        </Typography>

        <Tooltip title="Refresh capture">
          <span>
            <IconButton
              size="small"
              onClick={() => load(true)}
              disabled={isLoading}
              sx={{ color: darkMode ? '#aaa' : 'inherit' }}
            >
              <RefreshIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Open original in browser">
          <IconButton
            size="small"
            onClick={() => window.open(url, '_blank', 'noopener,noreferrer')}
            sx={{ color: darkMode ? '#aaa' : 'inherit' }}
          >
            <OpenInNewIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Content area */}
      <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {/* Loading overlay */}
        {isLoading && (
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: darkMode ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.8)',
              zIndex: 10,
              gap: 2,
            }}
          >
            <CircularProgress size={36} />
            <Typography variant="body2" color={darkMode ? 'grey.300' : 'text.secondary'}>
              {status === 'capturing' ? 'Fetching and saving page…' : 'Loading saved page…'}
            </Typography>
          </Box>
        )}

        {/* Error state */}
        {status === 'error' && (
          <Box
            sx={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
              p: 4,
            }}
          >
            <Typography color="error" variant="body2" align="center">
              {errorMsg}
            </Typography>
            <Button variant="outlined" size="small" onClick={() => load(false)}>
              Retry
            </Button>
            <Button
              variant="text"
              size="small"
              onClick={() => window.open(url, '_blank', 'noopener,noreferrer')}
            >
              Open original ↗
            </Button>
          </Box>
        )}

        {/* The iframe — always mounted so the blob URL assignment works */}
        <iframe
          ref={iframeRef}
          title={pageTitle || url}
          sandbox="allow-scripts"
          style={{
            width: '100%',
            height: '100%',
            border: 'none',
            display: 'block',
            // Hide during resize to avoid iframe swallowing mouse events
            pointerEvents: isResizing ? 'none' : 'auto',
            // Apply dark inversion matching PdfViewer dark mode
            filter:
              darkMode && status === 'ready'
                ? 'invert(1) hue-rotate(180deg)'
                : 'none',
          }}
        />
      </Box>
    </Box>
  );
});

export default WebViewer;
