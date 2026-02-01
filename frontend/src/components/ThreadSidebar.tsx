import React, { useState, useEffect } from 'react';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Typography,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Divider,
  Chip,
  Paper,
  Collapse,
  CircularProgress,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import ChatIcon from '@mui/icons-material/Chat';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DescriptionIcon from '@mui/icons-material/Description';
import LockIcon from '@mui/icons-material/Lock';
import {
  Thread,
  createThread,
  listThreads,
  deleteThread,
  updateThread,
} from '../lib/api';

interface ThreadSidebarProps {
  activeThreadId: string | null;
  onThreadSelect: (thread: Thread | null) => void;
  availableEmbedModels: string[];
  onEmbedModelChange?: (model: string) => void;
}

const ThreadSidebar: React.FC<ThreadSidebarProps> = ({
  activeThreadId,
  onThreadSelect,
  availableEmbedModels,
  onEmbedModelChange,
}) => {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newThreadName, setNewThreadName] = useState('');
  const [newThreadEmbedModel, setNewThreadEmbedModel] = useState('');
  const [creating, setCreating] = useState(false);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState('');
  const [expanded, setExpanded] = useState(true);

  // Load threads on mount
  useEffect(() => {
    loadThreads();
  }, []);

  const loadThreads = async () => {
    try {
      setLoading(true);
      const response = await listThreads();
      setThreads(response.threads);
    } catch (error) {
      console.error('Failed to load threads:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateThread = async () => {
    if (!newThreadName.trim() || !newThreadEmbedModel) return;
    
    try {
      setCreating(true);
      const thread = await createThread(newThreadName.trim(), newThreadEmbedModel);
      setThreads(prev => [thread, ...prev]);
      onThreadSelect(thread);
      if (onEmbedModelChange) {
        onEmbedModelChange(newThreadEmbedModel);
      }
      setCreateDialogOpen(false);
      setNewThreadName('');
      setNewThreadEmbedModel('');
    } catch (error) {
      console.error('Failed to create thread:', error);
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteThread = async (threadId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (!confirm('Delete this thread and all its messages?')) return;
    
    try {
      await deleteThread(threadId);
      setThreads(prev => prev.filter(t => t.id !== threadId));
      if (activeThreadId === threadId) {
        onThreadSelect(null);
      }
    } catch (error) {
      console.error('Failed to delete thread:', error);
    }
  };

  const handleEditThread = async (threadId: string) => {
    if (!editingName.trim()) return;
    
    try {
      const updated = await updateThread(threadId, editingName.trim());
      setThreads(prev => prev.map(t => t.id === threadId ? { ...t, name: updated.name } : t));
      setEditingThreadId(null);
      setEditingName('');
    } catch (error) {
      console.error('Failed to update thread:', error);
    }
  };

  const startEditing = (thread: Thread, event: React.MouseEvent) => {
    event.stopPropagation();
    setEditingThreadId(thread.id);
    setEditingName(thread.name);
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) return 'Today';
    if (days === 1) return 'Yesterday';
    if (days < 7) return `${days} days ago`;
    return date.toLocaleDateString();
  };

  return (
    <Paper 
      elevation={0} 
      sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        bgcolor: 'grey.50'
      }}
    >
      {/* Header */}
      <Box sx={{ 
        p: 1.5, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between',
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
          <Typography variant="subtitle2" fontWeight="bold">
            Threads
          </Typography>
          <Chip label={threads.length} size="small" />
        </Box>
        <Tooltip title="Create new thread">
          <IconButton 
            size="small" 
            color="primary" 
            onClick={() => setCreateDialogOpen(true)}
          >
            <AddIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Thread List */}
      <Collapse in={expanded} sx={{ flex: 1, overflow: 'auto' }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : threads.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No threads yet
            </Typography>
            <Button 
              size="small" 
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
              sx={{ mt: 1 }}
            >
              Create Thread
            </Button>
          </Box>
        ) : (
          <List dense sx={{ p: 0 }}>
            {threads.map((thread) => (
              <ListItem
                key={thread.id}
                disablePadding
                sx={{
                  bgcolor: activeThreadId === thread.id ? 'primary.light' : 'transparent',
                  '&:hover': { bgcolor: activeThreadId === thread.id ? 'primary.light' : 'grey.100' }
                }}
              >
                <ListItemButton 
                  onClick={() => onThreadSelect(thread)}
                  selected={activeThreadId === thread.id}
                  sx={{ py: 1 }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    <ChatIcon fontSize="small" color={activeThreadId === thread.id ? 'primary' : 'action'} />
                  </ListItemIcon>
                  
                  {editingThreadId === thread.id ? (
                    <TextField
                      size="small"
                      value={editingName}
                      onChange={(e) => setEditingName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleEditThread(thread.id);
                        if (e.key === 'Escape') setEditingThreadId(null);
                      }}
                      onBlur={() => handleEditThread(thread.id)}
                      autoFocus
                      fullWidth
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <ListItemText
                      primary={
                        <Typography 
                          variant="body2" 
                          fontWeight={activeThreadId === thread.id ? 'bold' : 'normal'}
                          noWrap
                        >
                          {thread.name}
                        </Typography>
                      }
                      secondary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                          <Typography variant="caption" color="text.secondary">
                            {formatDate(thread.created_at)}
                          </Typography>
                          {thread.message_count !== undefined && thread.message_count > 0 && (
                            <Chip 
                              label={`${thread.message_count} msgs`} 
                              size="small" 
                              sx={{ height: 16, fontSize: '0.65rem' }}
                            />
                          )}
                          {thread.file_count !== undefined && thread.file_count > 0 && (
                            <Chip 
                              icon={<DescriptionIcon sx={{ fontSize: '0.7rem !important' }} />}
                              label={thread.file_count} 
                              size="small" 
                              sx={{ height: 16, fontSize: '0.65rem' }}
                            />
                          )}
                        </Box>
                      }
                    />
                  )}
                </ListItemButton>
                
                <ListItemSecondaryAction>
                  <IconButton 
                    size="small" 
                    onClick={(e) => startEditing(thread, e)}
                    sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                  >
                    <EditIcon fontSize="small" />
                  </IconButton>
                  <IconButton 
                    size="small" 
                    onClick={(e) => handleDeleteThread(thread.id, e)}
                    sx={{ opacity: 0.6, '&:hover': { opacity: 1, color: 'error.main' } }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        )}
      </Collapse>

      {/* Create Thread Dialog */}
      <Dialog 
        open={createDialogOpen} 
        onClose={() => setCreateDialogOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Create New Thread</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Thread Name"
            fullWidth
            value={newThreadName}
            onChange={(e) => setNewThreadName(e.target.value)}
            placeholder="e.g., Research Paper Analysis"
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={newThreadEmbedModel}
              label="Embedding Model"
              onChange={(e) => setNewThreadEmbedModel(e.target.value)}
            >
              {availableEmbedModels.map((model) => (
                <MenuItem key={model} value={model}>{model}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box sx={{ mt: 2, p: 1.5, bgcolor: 'warning.light', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <LockIcon fontSize="small" />
              <Typography variant="caption" color="text.secondary">
                The embedding model is locked once a thread is created. 
                To use a different model, create a new thread.
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateThread}
            variant="contained"
            disabled={!newThreadName.trim() || !newThreadEmbedModel || creating}
          >
            {creating ? <CircularProgress size={20} /> : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default ThreadSidebar;
