import React, { useCallback, useEffect, useMemo, useState } from 'react';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
  CircularProgress,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { listThreads, type Thread } from '../../lib/api';
import { checkLlmModelReady, fetchAvailableLlmModelsStrict } from '../../lib/models-api';

const LAST_TEST_THREAD_KEY = 'askpdf.agentWorkflowBuilder.lastTestThread';
const LAST_LLM_MODEL_KEY = 'last_llm_model';

const localValue = (key: string) => {
  if (typeof window === 'undefined') return '';
  return window.localStorage.getItem(key) || '';
};

const remember = (key: string, value: string) => {
  if (typeof window === 'undefined') return;
  if (value) window.localStorage.setItem(key, value);
  else window.localStorage.removeItem(key);
};

export interface BuilderModelHealth {
  checking: boolean;
  ready: boolean | null;
  supportsTools: boolean | null;
}

export function BuilderThreadPicker({
  value,
  onChange,
  disabled,
  lockedThreadId,
}: {
  value: string;
  onChange: (threadId: string) => void;
  disabled?: boolean;
  lockedThreadId?: string | null;
}) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectionIssue, setSelectionIssue] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listThreads();
      setThreads(response.threads);
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load threads.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    if (!loaded) return;
    if (lockedThreadId) {
      if (threads.some((thread) => thread.id === lockedThreadId)) {
        if (value !== lockedThreadId) onChange(lockedThreadId);
        setSelectionIssue(null);
      } else {
        setSelectionIssue('The thread required by the interrupted run is no longer available.');
      }
      return;
    }
    if (value && !threads.some((thread) => thread.id === value)) {
      onChange('');
      remember(LAST_TEST_THREAD_KEY, '');
      setSelectionIssue('The previously selected thread is no longer available. Choose another thread.');
      return;
    }
    if (!value) {
      const remembered = localValue(LAST_TEST_THREAD_KEY);
      if (remembered && threads.some((thread) => thread.id === remembered)) onChange(remembered);
      else if (remembered) remember(LAST_TEST_THREAD_KEY, '');
    }
  }, [loaded, lockedThreadId, onChange, threads, value]);

  useEffect(() => {
    if (value && threads.some((thread) => thread.id === value)) remember(LAST_TEST_THREAD_KEY, value);
  }, [threads, value]);

  const selected = threads.find((thread) => thread.id === value) || null;

  return (
    <Stack spacing={0.75}>
      <Autocomplete
        options={threads}
        value={selected}
        loading={loading}
        disabled={disabled || Boolean(lockedThreadId)}
        getOptionLabel={(thread) => thread.name || thread.id}
        isOptionEqualToValue={(option, candidate) => option.id === candidate.id}
        onChange={(_, thread) => {
          setSelectionIssue(null);
          onChange(thread?.id || '');
        }}
        noOptionsText={loading ? 'Loading threads…' : 'No threads available'}
        renderOption={(props, thread) => (
          <Box component="li" {...props} key={thread.id} sx={{ display: 'block !important' }}>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>{thread.name || 'Untitled thread'}</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{thread.id}</Typography>
            <Typography variant="caption" color="text.secondary">
              {thread.embeddingModel || 'No embedding model'} · {thread.file_count || 0} files · {thread.message_count || 0} messages
            </Typography>
          </Box>
        )}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Thread"
            required
            helperText={lockedThreadId ? 'Locked to the thread used by the interrupted test.' : 'Select the thread whose context this test should read.'}
            slotProps={{
              input: {
                ...params.InputProps,
                endAdornment: (
                  <>
                    {loading ? <CircularProgress size={16} /> : null}
                    <Tooltip title="Refresh threads">
                      <span><IconButton size="small" disabled={disabled || loading || Boolean(lockedThreadId)} onClick={() => void load()}><RefreshIcon fontSize="small" /></IconButton></span>
                    </Tooltip>
                    {params.InputProps.endAdornment}
                  </>
                ),
              },
            }}
          />
        )}
      />
      {error ? <Alert severity="error" action={<Button size="small" onClick={() => void load()}>Retry</Button>}>{error}</Alert> : null}
      {selectionIssue ? <Alert severity="warning">{selectionIssue}</Alert> : null}
    </Stack>
  );
}

export function BuilderLlmModelPicker({
  value,
  onChange,
  onHealthChange,
  disabled,
}: {
  value: string;
  onChange: (model: string) => void;
  onHealthChange: (health: BuilderModelHealth) => void;
  disabled?: boolean;
}) {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectionIssue, setSelectionIssue] = useState<string | null>(null);
  const [health, setHealth] = useState<BuilderModelHealth>({ checking: false, ready: null, supportsTools: null });

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setModels(await fetchAvailableLlmModelsStrict());
      setLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load models.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    if (!loaded) return;
    if (value && models.includes(value)) return;
    if (value) setSelectionIssue('The previously selected model is no longer available.');
    const remembered = localValue(LAST_LLM_MODEL_KEY);
    const next = remembered && models.includes(remembered) ? remembered : models[0] || '';
    onChange(next);
    if (next) remember(LAST_LLM_MODEL_KEY, next);
  }, [loaded, models, onChange, value]);

  useEffect(() => {
    let cancelled = false;
    if (!value || !models.includes(value)) {
      const next = { checking: false, ready: null, supportsTools: null };
      setHealth(next);
      onHealthChange(next);
      return;
    }
    const checking = { checking: true, ready: null, supportsTools: null };
    setHealth(checking);
    onHealthChange(checking);
    void checkLlmModelReady(value).then((result) => {
      if (cancelled) return;
      const next = { checking: false, ready: result.ready, supportsTools: result.supportsTools };
      setHealth(next);
      onHealthChange(next);
    });
    return () => { cancelled = true; };
  }, [models, onHealthChange, value]);

  const status = useMemo(() => {
    if (!value) return null;
    if (health.checking) return <Chip size="small" label="Checking" />;
    if (health.ready) return <Chip size="small" color="success" label="Ready" />;
    if (health.ready === false) return <Chip size="small" color="error" label="Offline" />;
    return null;
  }, [health, value]);

  return (
    <Stack spacing={0.75}>
      <Autocomplete
        options={models}
        value={models.includes(value) ? value : null}
        loading={loading}
        disabled={disabled}
        disableClearable
        onChange={(_, model) => {
          const next = model || '';
          setSelectionIssue(null);
          onChange(next);
          remember(LAST_LLM_MODEL_KEY, next);
        }}
        noOptionsText={loading ? 'Loading models…' : 'No models available'}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Model"
            required
            slotProps={{
              input: {
                ...params.InputProps,
                endAdornment: (
                  <>
                    {loading ? <CircularProgress size={16} /> : status}
                    <Tooltip title="Refresh models">
                      <span><IconButton size="small" disabled={disabled || loading} onClick={() => void load()}><RefreshIcon fontSize="small" /></IconButton></span>
                    </Tooltip>
                    {params.InputProps.endAdornment}
                  </>
                ),
              },
            }}
          />
        )}
      />
      {error ? <Alert severity="error" action={<Button size="small" onClick={() => void load()}>Retry</Button>}>{error}</Alert> : null}
      {selectionIssue ? <Alert severity="warning">{selectionIssue}</Alert> : null}
      {health.ready === false ? <Alert severity="error">The selected model is not available for chat completion.</Alert> : null}
      {health.ready && health.supportsTools === false ? <Alert severity="warning">This model does not report native tool-calling support. Graph-managed tools can still run.</Alert> : null}
    </Stack>
  );
}
