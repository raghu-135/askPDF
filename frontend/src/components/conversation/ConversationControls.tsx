import React, { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  CircularProgress,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  TextField,
  Tooltip,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined';

export const ConversationPanelShell = React.forwardRef<HTMLDivElement, {
  children: React.ReactNode;
  sx?: Record<string, any>;
}>(function ConversationPanelShell({ children, sx }, ref) {
  return (
    <Paper
      ref={ref}
      elevation={0}
      sx={{
        height: '100%',
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default',
        color: 'text.primary',
        ...sx,
      }}
    >
      {children}
    </Paper>
  );
});

export function ConversationModelControls({
  models,
  model,
  contextWindow,
  disabled = false,
  onModelChange,
  onContextWindowChange,
}: {
  models: string[];
  model: string;
  contextWindow: number;
  disabled?: boolean;
  onModelChange: (model: string) => void;
  onContextWindowChange: (contextWindow: number) => void;
}) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, minWidth: 0 }}>
      <TextField
        size="small"
        label="Ctx size"
        type="number"
        value={contextWindow || ''}
        disabled={disabled}
        onChange={(event) => onContextWindowChange(Math.max(0, Number.parseInt(event.target.value, 10) || 0))}
        sx={{ width: 100, flex: '0 0 auto' }}
        slotProps={{ htmlInput: { min: 1, step: 1, style: { textAlign: 'right' } } }}
      />
      <FormControl fullWidth size="small" disabled={disabled}>
        <InputLabel id="conversation-llm-label">Select LLM</InputLabel>
        <Select
          labelId="conversation-llm-label"
          value={model}
          label="Select LLM"
          onChange={(event) => onModelChange(String(event.target.value))}
        >
          {models.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
        </Select>
      </FormControl>
    </Box>
  );
}

export function ConversationComposer({
  inputRef,
  seedText = '',
  seedVersion = 0,
  placeholder,
  disabled = false,
  busy = false,
  stopping = false,
  minRows = 3,
  clearOnSubmit = true,
  auxiliaryActions,
  onDraftChange,
  onSubmit,
  onStop,
}: {
  inputRef?: React.Ref<HTMLInputElement | HTMLTextAreaElement>;
  seedText?: string;
  seedVersion?: number;
  placeholder: string;
  disabled?: boolean;
  busy?: boolean;
  stopping?: boolean;
  minRows?: number;
  clearOnSubmit?: boolean;
  auxiliaryActions?: React.ReactNode;
  onDraftChange?: (text: string) => void;
  onSubmit: (text: string) => void;
  onStop?: () => void;
}) {
  const [draft, setDraft] = useState(seedText);

  useEffect(() => {
    setDraft(seedText);
  }, [seedText, seedVersion]);

  const submit = () => {
    const value = draft.trim();
    if (!value || disabled || busy) return;
    onSubmit(value);
    if (clearOnSubmit) {
      setDraft('');
      onDraftChange?.('');
    }
  };

  return (
    <Box sx={{ display: 'flex', gap: 1, alignItems: 'stretch', px: 1 }}>
      <TextField
        inputRef={inputRef}
        fullWidth
        variant="outlined"
        multiline
        minRows={minRows}
        maxRows={10}
        placeholder={placeholder}
        value={draft}
        onChange={(event) => {
          setDraft(event.target.value);
          onDraftChange?.(event.target.value);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        disabled={disabled}
      />
      <Box sx={{ width: 40, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', alignItems: 'center' }}>
        {busy && onStop ? (
          <Tooltip title={stopping ? 'Stopping' : 'Stop'}>
            <span>
              <IconButton color="error" onClick={onStop} disabled={stopping} aria-label="Stop">
                {stopping ? <CircularProgress size="1em" color="inherit" /> : <StopCircleOutlinedIcon />}
              </IconButton>
            </span>
          </Tooltip>
        ) : (
          <IconButton color="primary" onClick={submit} disabled={disabled || busy || !draft.trim()} aria-label="Send">
            {busy ? <CircularProgress size="1em" /> : <SendIcon />}
          </IconButton>
        )}
        {auxiliaryActions}
      </Box>
    </Box>
  );
}

export function ConversationDecisionPanel({
  variant,
  children,
}: {
  variant: 'clarification' | 'conflict' | 'approval';
  children: React.ReactNode;
}) {
  return (
    <Alert
      severity={variant === 'conflict' ? 'warning' : variant === 'approval' ? 'info' : 'success'}
      sx={{
        mx: 1,
        mt: 1,
        maxHeight: '42%',
        minHeight: 80,
        overflow: 'auto',
        resize: 'vertical',
        alignItems: 'flex-start',
      }}
    >
      {children}
    </Alert>
  );
}
