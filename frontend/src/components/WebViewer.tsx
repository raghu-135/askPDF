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
import {
  Box, CircularProgress, Typography, Button, Tooltip, IconButton,
  Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions,
  Snackbar, Alert,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import { captureWebPage, fetchWebPageHtml, refreshWebSource } from '../lib/api';

type Props = {
  /** The original URL of the webpage. */
  url: string;
  /** MD5 hash of the URL — used to retrieve the saved HTML. */
  fileHash: string;
  /** Thread ID — if provided, refresh also re-indexes the page in the RAG service. */
  threadId?: string;
  darkMode?: boolean;
  isResizing?: boolean;
};

type Status = 'idle' | 'capturing' | 'loading' | 'ready' | 'error';

const WebViewer = React.memo(function WebViewer({ url, fileHash, threadId, darkMode = false, isResizing = false }: Props) {
  const [status, setStatus] = useState<Status>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [pageTitle, setPageTitle] = useState<string>('');
  const blobUrlRef = useRef<string | null>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  // Confirmation dialog state (content changed, warn before re-indexing)
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingContentHash, setPendingContentHash] = useState<string | null>(null);

  // Snackbar for "no changes detected" feedback
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'info' }>({
    open: false, message: '', severity: 'info',
  });

  const loadHtml = async () => {
    setStatus('loading');
    try {
      const html = await fetchWebPageHtml(fileHash);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      const blob = new Blob([html], { type: 'text/html' });
      blobUrlRef.current = URL.createObjectURL(blob);
      if (iframeRef.current) iframeRef.current.src = blobUrlRef.current;
      setStatus('ready');
    } catch (err: any) {
      setStatus('error');
      setErrorMsg(`Failed to load page: ${err.message}`);
    }
  };

  const load = async (force = false) => {
    setStatus('capturing');
    setErrorMsg(null);

    try {
      // Step 1 – capture (or use cache) and always get the content_hash
      const capture = await captureWebPage(url, force);
      setPageTitle(capture.title);

      // Step 2 – if forced AND in a thread context, run the two-phase refresh check
      if (force && threadId) {
        try {
          const result = await refreshWebSource(threadId, fileHash, capture.content_hash, false);

          if (result.status === 'unchanged') {
            setSnackbar({ open: true, message: 'Page content has not changed. Index is already up to date.', severity: 'info' });
            // Still reload the visual view with fresh HTML
            await loadHtml();
            return;
          }

          if (result.status === 'confirmation_required') {
            // Store hash, show dialog — loadHtml will run after confirmation
            setPendingContentHash(result.new_content_hash ?? capture.content_hash);
            setConfirmOpen(true);
            // Reload visual without re-indexing for now
            await loadHtml();
            return;
          }
          // 'accepted' (no content_hash sent / forced straight through): fall through to loadHtml
        } catch (reindexErr: any) {
          // Non-fatal: visual refresh still continues; log for debugging
          console.warn('Web source refresh check failed:', reindexErr.message);
        }
      }
    } catch (err: any) {
      setStatus('error');
      setErrorMsg(`Capture failed: ${err.message}`);
      return;
    }

    await loadHtml();
  };

  /** Called when the user confirms the "replace index" dialog. */
  const handleConfirmReindex = async () => {
    setConfirmOpen(false);
    if (!threadId) return;
    try {
      await refreshWebSource(threadId, fileHash, pendingContentHash, true);
      setSnackbar({ open: true, message: 'Re-indexing started. New content will be available shortly.', severity: 'success' });
    } catch (err: any) {
      console.warn('Re-index (after confirmation) failed:', err.message);
    }
    setPendingContentHash(null);
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

      {/* Confirmation dialog — shown when page content has changed */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Update page index?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            The content of this page has changed since it was last indexed.
            Re-indexing will <strong>remove the existing indexed data</strong> for this page
            and replace it with the new content. The updated knowledge will be available
            to the AI after indexing completes.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setConfirmOpen(false); setPendingContentHash(null); }}>
            Keep existing index
          </Button>
          <Button onClick={handleConfirmReindex} variant="contained" color="primary">
            Re-index new content
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for non-blocking feedback */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar(s => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar(s => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
});

export default WebViewer;
