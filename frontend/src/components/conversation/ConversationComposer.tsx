import React, { useEffect, useState } from 'react';
import {
  Box,
  CircularProgress,
  IconButton,
  TextField,
  Tooltip,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined';
import {
  getConversationComposerButtonState,
} from '../../lib/conversation-ui-state';

export function ConversationComposer({
  inputRef,
  seedText = '',
  seedVersion = 0,
  placeholder,
  disabled = false,
  busy = false,
  showStop = false,
  canStop = false,
  stopping = false,
  stopTooltip = 'Stop',
  stopAriaLabel = 'Stop',
  minRows = 3,
  clearOnSubmit = true,
  disableWhenEmpty = true,
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
  showStop?: boolean;
  canStop?: boolean;
  stopping?: boolean;
  stopTooltip?: React.ReactNode;
  stopAriaLabel?: string;
  minRows?: number;
  clearOnSubmit?: boolean;
  disableWhenEmpty?: boolean;
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
  const buttonState = getConversationComposerButtonState({
    disabled,
    busy,
    showStop,
    canStop,
    stopping,
    hasDraft: Boolean(draft.trim()),
    disableWhenEmpty,
  });

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
        sx={{
          '& .MuiOutlinedInput-root': {
            bgcolor: 'background.paper',
            color: 'text.primary',
            '& fieldset': {
              borderColor: 'primary.light',
              borderWidth: '1px',
            },
            '&:hover fieldset': {
              borderColor: 'primary.main',
            },
          },
        }}
      />
      <Box sx={{ flex: '0 0 auto', width: 40, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', alignItems: 'center' }}>
        {buttonState.mode === 'stop' ? (
          <Tooltip title={stopTooltip}>
            <span>
              <IconButton
                size="medium"
                color="error"
                onClick={onStop}
                disabled={buttonState.disabled}
                aria-label={stopAriaLabel}
              >
                {buttonState.spinning
                  ? <CircularProgress size="1em" color="inherit" />
                  : <StopCircleOutlinedIcon fontSize="medium" />}
              </IconButton>
            </span>
          </Tooltip>
        ) : (
          <IconButton
            size="medium"
            color="primary"
            onClick={submit}
            disabled={buttonState.disabled}
            aria-label="Send"
          >
            {buttonState.spinning ? <CircularProgress size="1em" /> : <SendIcon fontSize="medium" />}
          </IconButton>
        )}
        {auxiliaryActions}
      </Box>
    </Box>
  );
}
