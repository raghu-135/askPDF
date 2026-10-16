import React from 'react';
import {
  Box,
  CircularProgress,
  IconButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import DataObjectIcon from '@mui/icons-material/DataObject';
import DeleteIcon from '@mui/icons-material/Delete';
import ErrorIcon from '@mui/icons-material/Error';
import FolderIcon from '@mui/icons-material/Folder';
import LanguageIcon from '@mui/icons-material/Language';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import OpenInBrowserIcon from '@mui/icons-material/OpenInBrowser';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import ReplayIcon from '@mui/icons-material/Replay';
import { truncateFileName } from '../../lib/pdf-utils';
import { ProcessStatus, ThreadFileSourceType } from '../../lib/enums';
import type { DocumentWorkspaceTab } from './WorkspaceTabs';

const stopTabAction = (event: React.SyntheticEvent) => {
  event.preventDefault();
  event.stopPropagation();
};

const handleCopySourceUrl = async (sourceUrl: string) => {
  try {
    await navigator.clipboard?.writeText(sourceUrl);
  } catch {
    window.prompt('Copy source URL', sourceUrl);
  }
};

export default function DocumentInstanceTabLabel({
  tab,
  documentContext = 'thread',
  onClose,
  onDocumentRemove,
  onDocumentPromote,
  onDocumentRetry,
  onInspectChunks,
}: {
  tab: DocumentWorkspaceTab;
  documentContext?: 'thread' | 'project';
  onClose?: (tabId: string) => void;
  onDocumentRemove?: (tabId: string) => void;
  onDocumentPromote?: (tabId: string) => void;
  onDocumentRetry?: (tabId: string) => void;
  onInspectChunks?: (tab: DocumentWorkspaceTab) => void;
}) {
  const [menuAnchor, setMenuAnchor] = React.useState<HTMLElement | null>(null);
  const isBrowserDocument = tab.sourceType === ThreadFileSourceType.Browser;
  let label = truncateFileName(tab.fileName);
  if (isBrowserDocument && tab.sourceUrl) {
    try {
      label = new URL(tab.sourceUrl).hostname;
    } catch {
      label = tab.fileName;
    }
  }
  const fullTitle = isBrowserDocument ? (tab.sourceUrl || tab.fileName) : tab.fileName;

  const closeMenu = () => setMenuAnchor(null);

  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 0, maxWidth: 200 }}>
        {isBrowserDocument ? (
          <Tooltip title="Open source webpage">
            <span>
              <Box
                component="span"
                role={tab.sourceUrl ? 'button' : undefined}
                tabIndex={tab.sourceUrl ? 0 : -1}
                onClick={(event) => {
                  stopTabAction(event);
                  if (tab.sourceUrl) {
                    window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                  }
                }}
                onKeyDown={(event) => {
                  if (tab.sourceUrl && (event.key === 'Enter' || event.key === ' ')) {
                    stopTabAction(event);
                    window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
                  }
                }}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flex: '0 0 auto',
                  cursor: tab.sourceUrl ? 'pointer' : 'default',
                  p: '2px',
                  borderRadius: 1,
                  color: 'primary.main',
                }}
              >
                <LanguageIcon fontSize="small" />
              </Box>
            </span>
          </Tooltip>
        ) : (
          <PictureAsPdfIcon fontSize="small" sx={{ color: 'error.main', opacity: 0.7, flex: '0 0 auto' }} />
        )}
        <Tooltip title={fullTitle} placement="bottom">
          <Typography
            component="span"
            variant="body2"
            noWrap
            sx={{ minWidth: 0, flex: '1 1 auto', textAlign: 'left' }}
          >
            {label}
          </Typography>
        </Tooltip>
        {tab.parsingStatus === ProcessStatus.Pending && <CircularProgress size={13} sx={{ flex: '0 0 auto' }} />}
        {tab.parsingStatus === ProcessStatus.Failed && (
          <Tooltip title={tab.processingError || 'Processing failed'}>
            <ErrorIcon color="error" sx={{ fontSize: 15, flex: '0 0 auto' }} />
          </Tooltip>
        )}
        {tab.associationScope === 'project' && (
          <Tooltip title="Project knowledge">
            <FolderIcon sx={{ fontSize: 14, color: 'primary.main', flex: '0 0 auto' }} />
          </Tooltip>
        )}
        <Tooltip title="Document actions">
          <span>
            <Box
              component="span"
              role="button"
              tabIndex={0}
              aria-label={`Document actions for ${fullTitle}`}
              onClick={(event) => {
                stopTabAction(event);
                setMenuAnchor(event.currentTarget);
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  stopTabAction(event);
                  setMenuAnchor(event.currentTarget);
                }
              }}
              sx={{ p: 0.2, display: 'inline-flex', borderRadius: 1, opacity: 0.72, flex: '0 0 auto' }}
            >
              <MoreVertIcon sx={{ fontSize: 16 }} />
            </Box>
          </span>
        </Tooltip>
        {onClose && (
          <Tooltip title="Close tab">
            <IconButton
              size="small"
              onClick={(event) => {
                stopTabAction(event);
                onClose(tab.id);
              }}
              sx={{ p: 0.2 }}
            >
              <CloseIcon sx={{ fontSize: 15 }} />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={closeMenu}
        onClick={(event) => event.stopPropagation()}
      >
        {onInspectChunks && (
          <MenuItem
            onClick={() => {
              onInspectChunks(tab);
              closeMenu();
            }}
          >
            <ListItemIcon><DataObjectIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Inspect vector chunks</ListItemText>
          </MenuItem>
        )}
        {isBrowserDocument && tab.sourceUrl && (
          <MenuItem
            onClick={() => {
              window.open(tab.sourceUrl, '_blank', 'noopener,noreferrer');
              closeMenu();
            }}
          >
            <ListItemIcon><OpenInBrowserIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Open source webpage</ListItemText>
          </MenuItem>
        )}
        {isBrowserDocument && tab.sourceUrl && (
          <MenuItem
            onClick={() => {
              void handleCopySourceUrl(tab.sourceUrl!);
              closeMenu();
            }}
          >
            <ListItemIcon><ContentCopyIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Copy source URL</ListItemText>
          </MenuItem>
        )}
        {onDocumentPromote && tab.associationScope === 'thread' && !tab.isProjectKnowledge && (
          <MenuItem
            onClick={() => {
              onDocumentPromote(tab.id);
              closeMenu();
            }}
          >
            <ListItemIcon><CreateNewFolderIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Add to project knowledge</ListItemText>
          </MenuItem>
        )}
        {tab.parsingStatus === ProcessStatus.Failed && onDocumentRetry && (
          <MenuItem
            onClick={() => {
              onDocumentRetry(tab.id);
              closeMenu();
            }}
          >
            <ListItemIcon><ReplayIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Retry processing</ListItemText>
          </MenuItem>
        )}
        {onDocumentRemove && (documentContext === 'project' || tab.associationScope !== 'project') && (
          <MenuItem
            onClick={() => {
              onDocumentRemove(tab.id);
              closeMenu();
            }}
          >
            <ListItemIcon><DeleteIcon color="error" fontSize="small" /></ListItemIcon>
            <ListItemText>{documentContext === 'project' ? 'Remove from project' : 'Delete from thread'}</ListItemText>
          </MenuItem>
        )}
        {onClose && (
          <MenuItem
            onClick={() => {
              onClose(tab.id);
              closeMenu();
            }}
          >
            <ListItemIcon><CloseIcon fontSize="small" /></ListItemIcon>
            <ListItemText>Close tab</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </>
  );
}
