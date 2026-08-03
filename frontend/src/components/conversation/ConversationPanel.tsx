import React, { useId, useState } from 'react';
import {
  Box,
  IconButton,
  FormControl,
  InputLabel,
  List,
  MenuItem,
  Paper,
  Select,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import type { SxProps, Theme } from '@mui/material/styles';

export const ConversationPanelTemplate = React.forwardRef<HTMLDivElement, {
  header: React.ReactNode;
  transcript: React.ReactNode;
  status?: React.ReactNode;
  decision?: React.ReactNode;
  composer: React.ReactNode;
  footer?: React.ReactNode;
  sx?: SxProps<Theme>;
}>(function ConversationPanelTemplate({
  header,
  transcript,
  status,
  decision,
  composer,
  footer,
  sx,
}, ref) {
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
      {header}
      {status}
      {transcript}
      {decision}
      {composer}
      {footer}
    </Paper>
  );
});

export function ConversationHeader({
  models,
  model,
  contextWindow,
  disabled = false,
  leading,
  beforeModelControls,
  trailingActions,
  onModelChange,
  onContextWindowChange,
}: {
  models: string[];
  model: string;
  contextWindow: number;
  disabled?: boolean;
  leading?: React.ReactNode;
  beforeModelControls?: React.ReactNode;
  trailingActions?: React.ReactNode;
  onModelChange: (model: string) => void;
  onContextWindowChange: (contextWindow: number) => void;
}) {
  const labelId = useId();
  const [showContextHighlight, setShowContextHighlight] = useState(false);
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const selectOutlineSx = {
    '& fieldset': {
      borderColor: 'transparent',
      borderWidth: '1px',
    },
    '&:hover fieldset': {
      borderColor: 'primary.main',
    },
    '&.Mui-focused fieldset': {
      borderColor: 'primary.main',
    },
  };

  const handleModelChange = (nextModel: string) => {
    setShowContextHighlight(true);
    setTooltipOpen(true);
    onModelChange(nextModel);
  };

  const dismissContextHelp = () => {
    setShowContextHighlight(false);
    setTooltipOpen(false);
  };

  return (
    <Box sx={{
      mb: 0.5,
      pt: 0.5,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 2,
      flexShrink: 0,
      minWidth: 0,
    }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
        {leading}
      </Box>
      <Box sx={{
        display: 'flex',
        alignItems: 'center',
        flexGrow: 1,
        maxWidth: 350,
        minWidth: 0,
        gap: 1,
      }}>
        {beforeModelControls}
        <Tooltip
          title={
            <Box sx={{ p: 0.5 }}>
              <Typography variant="caption" sx={{ display: 'block' }}>
                Set context window size for the LLM.
              </Typography>
              <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                Find the model context length at{' '}
                <a
                  href="https://llm-explorer.com/list/"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#90caf9', textDecoration: 'underline' }}
                >
                  llm-explorer.com
                </a>
                . Enter the numeric Context Len value, such as 8000 or 128000.
                Larger windows allow more context but can increase latency and cost.
              </Typography>
            </Box>
          }
          placement="top"
          open={tooltipOpen}
          onOpen={() => setTooltipOpen(true)}
          onClose={() => {
            if (!showContextHighlight) setTooltipOpen(false);
          }}
        >
          <TextField
            size="small"
            label="Ctx size"
            type="number"
            value={contextWindow}
            disabled={disabled}
            onChange={(event) => onContextWindowChange(Number.parseInt(event.target.value, 10) || 0)}
            onClick={dismissContextHelp}
            onFocus={dismissContextHelp}
            sx={{
              width: 100,
              flex: '0 0 auto',
              '& .MuiOutlinedInput-root': {
                transition: 'all 0.3s ease',
                backgroundColor: showContextHighlight ? 'rgba(255, 235, 59, 0.1)' : 'transparent',
                ...selectOutlineSx,
                '& fieldset': {
                  borderColor: showContextHighlight ? 'primary.main' : 'transparent',
                  borderWidth: showContextHighlight ? '2px' : '1px',
                },
              },
            }}
            slotProps={{ htmlInput: { min: 1, step: 1, style: { textAlign: 'right' } } }}
          />
        </Tooltip>
        <FormControl fullWidth size="small" disabled={disabled}>
          <InputLabel id={labelId}>Select LLM</InputLabel>
          <Select
            labelId={labelId}
            value={model}
            label="Select LLM"
            onChange={(event) => handleModelChange(String(event.target.value))}
            sx={selectOutlineSx}
          >
            {models.map((item) => <MenuItem key={item} value={item}>{item}</MenuItem>)}
          </Select>
        </FormControl>
        {trailingActions}
      </Box>
    </Box>
  );
}

export function WorkspaceContextHeader({
  title,
  subtitle,
  icon,
  onBack,
  backLabel = 'Back',
  actions,
}: {
  title: string;
  subtitle?: React.ReactNode;
  icon?: React.ReactNode;
  onBack?: () => void;
  backLabel?: string;
  actions?: React.ReactNode;
}) {
  return (
    <Box
      sx={{
        minHeight: 49,
        px: 1,
        borderBottom: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        display: 'flex',
        alignItems: 'center',
        gap: 1,
        flexShrink: 0,
        minWidth: 0,
      }}
    >
      {onBack && (
        <Tooltip title={backLabel}>
          <IconButton size="small" onClick={onBack} aria-label={backLabel}>
            <ArrowBackIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )}
      {icon}
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="subtitle2" fontWeight={800} noWrap>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary" component="div" noWrap>
            {subtitle}
          </Typography>
        )}
      </Box>
      {actions}
    </Box>
  );
}

export const ConversationTranscriptFrame = React.forwardRef<HTMLDivElement, {
  children: React.ReactNode;
  sx?: SxProps<Theme>;
}>(function ConversationTranscriptFrame({ children, sx }, ref) {
  return (
    <List
      component="div"
      ref={ref}
      sx={{
        flexGrow: 1,
        minHeight: 0,
        overflow: 'auto',
        borderRadius: 1,
        mb: 1,
        p: 1,
        ...sx,
      }}
    >
      {children}
    </List>
  );
});
