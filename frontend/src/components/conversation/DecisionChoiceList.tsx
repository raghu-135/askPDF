import React, { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  IconButton,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddCommentOutlinedIcon from '@mui/icons-material/AddCommentOutlined';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';

export type DecisionChoiceItem = {
  id: string;
  label: string;
  description?: string;
  text?: string;
};

export function DecisionChoiceList({
  choices,
  presentation = 'buttons',
  disabled = false,
  onSelect,
  onChoiceTextChange,
  onCustomSubmit,
  customLabel = 'Something else',
  customPlaceholder = 'Share a different response',
}: {
  choices: DecisionChoiceItem[];
  presentation?: 'buttons' | 'editable';
  disabled?: boolean;
  onSelect: (choice: DecisionChoiceItem, text: string) => void;
  onChoiceTextChange?: (choice: DecisionChoiceItem, text: string) => void;
  onCustomSubmit: (text: string) => void;
  customLabel?: string;
  customPlaceholder?: string;
}) {
  const [customOpen, setCustomOpen] = useState(false);
  const [customText, setCustomText] = useState('');
  const choiceKey = useMemo(() => choices.map((choice) => choice.id).join('|'), [choices]);

  useEffect(() => {
    setCustomOpen(false);
    setCustomText('');
  }, [choiceKey]);

  const submitCustom = () => {
    const text = customText.trim();
    if (!text || disabled) return;
    onCustomSubmit(text);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, width: '100%', minWidth: 0 }}>
      {choices.map((choice) => {
        const value = choice.text ?? choice.label;
        if (presentation === 'editable') {
          return (
            <Box
              key={choice.id}
              sx={{
                display: 'grid',
                gridTemplateColumns: 'minmax(0, 1fr) 2.5rem',
                gap: 1,
                alignItems: 'flex-start',
                width: '100%',
                minWidth: 0,
              }}
            >
              <TextField
                fullWidth
                size="small"
                multiline
                label={choice.label}
                value={value}
                disabled={disabled}
                sx={{ '& .MuiOutlinedInput-root': { bgcolor: 'action.hover' } }}
                onChange={(event) => onChoiceTextChange?.(choice, event.target.value)}
              />
              <Tooltip title="Send this response">
                <Box component="span" sx={{ width: '2.5rem', display: 'flex', justifyContent: 'center' }}>
                  <IconButton
                    color="primary"
                    size="medium"
                    disabled={!value.trim() || disabled}
                    onClick={() => onSelect(choice, value.trim())}
                    sx={{ mt: 0.25 }}
                    aria-label={`Send ${choice.label}`}
                  >
                    <SendIcon fontSize="medium" />
                  </IconButton>
                </Box>
              </Tooltip>
            </Box>
          );
        }
        return (
          <Button
            key={choice.id}
            variant="outlined"
            size="small"
            onClick={() => onSelect(choice, value)}
            disabled={disabled}
            sx={{ justifyContent: 'flex-start', textAlign: 'left' }}
          >
            <Box sx={{ minWidth: 0 }}>
              <Typography variant="body2">{choice.label}</Typography>
              {choice.description && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  {choice.description}
                </Typography>
              )}
            </Box>
          </Button>
        );
      })}

      {!customOpen ? (
        <Button
          variant="text"
          size="small"
          startIcon={<AddCommentOutlinedIcon fontSize="small" />}
          onClick={() => setCustomOpen(true)}
          disabled={disabled}
          sx={{ alignSelf: 'flex-start' }}
        >
          {customLabel}
        </Button>
      ) : (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: 'minmax(0, 1fr) 2.5rem 2.5rem',
            gap: 0.5,
            alignItems: 'flex-start',
            width: '100%',
            minWidth: 0,
          }}
        >
          <TextField
            autoFocus
            fullWidth
            size="small"
            multiline
            minRows={2}
            maxRows={6}
            label="Your response"
            placeholder={customPlaceholder}
            value={customText}
            disabled={disabled}
            onChange={(event) => setCustomText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                submitCustom();
              }
            }}
          />
          <Tooltip title="Send your response">
            <Box component="span" sx={{ width: '2.5rem', display: 'flex', justifyContent: 'center' }}>
              <IconButton
                color="primary"
                size="medium"
                disabled={!customText.trim() || disabled}
                onClick={submitCustom}
                sx={{ mt: 0.25 }}
                aria-label="Send custom response"
              >
                <SendIcon fontSize="medium" />
              </IconButton>
            </Box>
          </Tooltip>
          <Tooltip title="Cancel custom response">
            <Box component="span" sx={{ width: '2.5rem', display: 'flex', justifyContent: 'center' }}>
              <IconButton
                size="medium"
                disabled={disabled}
                onClick={() => {
                  setCustomOpen(false);
                  setCustomText('');
                }}
                sx={{ mt: 0.25 }}
                aria-label="Cancel custom response"
              >
                <CloseIcon fontSize="medium" />
              </IconButton>
            </Box>
          </Tooltip>
        </Box>
      )}
    </Box>
  );
}
