import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  IconButton,
  List,
  ListItem,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import MemoryIcon from '@mui/icons-material/Memory';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  applyMemoryCuratorChanges,
  respondToMemoryCurator,
  type MemoryCuratorMessage,
  type MemoryCuratorOperation,
  type MemoryCuratorResponse,
} from '../lib/api';
import { checkLlmModelReady, fetchAvailableLlmModels } from '../lib/models-api';
import {
  buildCuratorContext,
  curatorTitle,
  type MemoryCuratorIntent,
} from '../lib/memory-curator';
import {
  ConversationComposer,
  ConversationDecisionPanel,
  ConversationModelControls,
  ConversationPanelShell,
} from './conversation/ConversationControls';

const defaultContextWindow = 8192;

const initialPrompt = (intent: MemoryCuratorIntent) => {
  if (intent.mode === 'conversation_review') {
    return 'Review the next eligible completed conversation turns and suggest only durable memory changes.';
  }
  if (intent.mode === 'edit') {
    return `I want to review and edit this memory:\n\n${intent.memory?.content || ''}`;
  }
  return 'Help me add a durable memory. Ask what should be remembered and help me choose the right scope.';
};

const initialAssistantMessage = (intent: MemoryCuratorIntent): MemoryCuratorMessage | null => {
  if (intent.mode === 'conversation_review') return null;
  if (intent.mode === 'edit') {
    return {
      role: 'assistant',
      content: `What would you like to change about this memory?\n\n> ${intent.memory?.content || ''}`,
    };
  }
  return {
    role: 'assistant',
    content: 'What should be remembered? I will check for related or conflicting memory before proposing a change.',
  };
};

const operationLabel = (operation: MemoryCuratorOperation) => {
  const scope = operation.scope_type === 'user' ? 'Global' : operation.scope_type === 'project' ? 'Project' : 'Thread';
  return `${operation.action.toUpperCase()} ${scope || ''}`.trim();
};

export default function MemoryCuratorPanel({
  intent,
  onClose,
  onDirtyChange,
  onApplied,
}: {
  intent: MemoryCuratorIntent;
  onClose: () => void;
  onDirtyChange: (dirty: boolean) => void;
  onApplied: () => void;
}) {
  const [models, setModels] = useState<string[]>([]);
  const [llmModel, setLlmModel] = useState('');
  const [contextWindow, setContextWindow] = useState(defaultContextWindow);
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [messages, setMessages] = useState<MemoryCuratorMessage[]>(() => {
    const initial = initialAssistantMessage(intent);
    return initial ? [initial] : [];
  });
  const [decision, setDecision] = useState<MemoryCuratorResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);
  const context = useMemo(() => buildCuratorContext(intent), [intent]);
  const hasUnconfirmedDecision = Boolean(
    decision?.state === 'proposal'
    || (decision?.state === 'no_changes' && decision.review?.cursor),
  );

  useEffect(() => {
    onDirtyChange(hasUnconfirmedDecision);
  }, [hasUnconfirmedDecision, onDirtyChange]);

  useEffect(() => {
    let cancelled = false;
    const savedModel = window.localStorage.getItem('last_llm_model') || '';
    const savedContext = Number.parseInt(window.localStorage.getItem('last_context_window') || '', 10);
    if (savedContext > 0) setContextWindow(savedContext);
    fetchAvailableLlmModels().then((items) => {
      if (cancelled) return;
      setModels(items);
      setLlmModel(savedModel && items.includes(savedModel) ? savedModel : items[0] || savedModel);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!llmModel) {
      setModelReady(null);
      return;
    }
    let cancelled = false;
    setModelReady(null);
    checkLlmModelReady(llmModel).then((result) => {
      if (!cancelled) setModelReady(result.ready);
    });
    window.localStorage.setItem('last_llm_model', llmModel);
    return () => {
      cancelled = true;
    };
  }, [llmModel]);

  const respond = useCallback(async (nextMessages: MemoryCuratorMessage[]) => {
    if (!llmModel || modelReady !== true) return;
    const boundedMessages = nextMessages.slice(-24);
    setBusy(true);
    setError(null);
    setApplied(false);
    try {
      const response = await respondToMemoryCurator({
        mode: intent.mode,
        context,
        memory_id: intent.memory?.id,
        messages: boundedMessages,
        llm_model: llmModel,
        context_window: contextWindow,
      });
      setMessages([...boundedMessages, { role: 'assistant', content: response.message }]);
      setDecision(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The memory curator could not respond.');
    } finally {
      setBusy(false);
    }
  }, [context, contextWindow, intent.memory?.id, intent.mode, llmModel, modelReady]);

  useEffect(() => {
    if (intent.mode !== 'conversation_review' || !llmModel || modelReady !== true || messages.length) return;
    const firstMessage: MemoryCuratorMessage = { role: 'user', content: initialPrompt(intent) };
    setMessages([firstMessage]);
    void respond([firstMessage]);
  }, [intent, llmModel, messages.length, modelReady, respond]);

  const submitMessage = (text: string) => {
    const next = [...messages, { role: 'user' as const, content: text }];
    setDecision(null);
    setMessages(next);
    void respond(next);
  };

  const apply = async () => {
    if (!decision) return;
    setBusy(true);
    setError(null);
    try {
      const result = await applyMemoryCuratorChanges({
        context,
        operations: decision.operations,
        review_cursor: decision.review?.cursor || undefined,
        actor_id: 'ui',
      });
      setApplied(true);
      setDecision(null);
      onDirtyChange(false);
      onApplied();
      const summary = result.warnings.length
        ? `Changes were saved with ${result.warnings.length} indexing warning(s). Failed records remain available for Retry.`
        : result.review_cursor_advanced && result.changed_memories.length === 0
          ? 'Review completed. No memory changes were saved.'
          : 'The confirmed memory changes were saved.';
      setMessages((current) => [...current, { role: 'assistant', content: summary }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The memory changes could not be applied.');
    } finally {
      setBusy(false);
    }
  };

  const modelChecking = Boolean(llmModel) && modelReady === null;
  const canCompose = Boolean(llmModel && contextWindow >= 256 && modelReady === true && !busy);

  return (
    <ConversationPanelShell sx={{ display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr) auto', bgcolor: 'background.paper' }}>
      <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <MemoryIcon color="primary" fontSize="small" />
          <Typography variant="subtitle2" sx={{ fontWeight: 700, flex: 1 }}>{curatorTitle(intent)}</Typography>
          <Tooltip title="Close memory curator">
            <IconButton size="small" onClick={onClose} aria-label="Close memory curator"><CloseIcon fontSize="small" /></IconButton>
          </Tooltip>
        </Stack>
        <Box sx={{ mt: 1 }}>
          <ConversationModelControls
            models={models}
            model={llmModel}
            contextWindow={contextWindow}
            disabled={busy}
            onModelChange={setLlmModel}
            onContextWindowChange={(value) => {
              setContextWindow(value);
              if (value > 0) window.localStorage.setItem('last_context_window', String(value));
            }}
          />
        </Box>
      </Box>

      <Box sx={{ minHeight: 0, overflow: 'auto' }}>
        {error && <Alert severity="error" onClose={() => setError(null)} sx={{ m: 1 }}>{error}</Alert>}
        {modelReady === false && <Alert severity="warning" sx={{ m: 1 }}>Select a ready chat model to continue.</Alert>}
        {decision?.review && (
          <Alert severity="info" sx={{ m: 1 }}>
            Reviewed {decision.review.reviewed_count} turn(s). {decision.review.remaining_count} remain after this batch.
          </Alert>
        )}
        {Boolean(decision?.embedding_readiness.length) && (
          <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ px: 1, pb: 0.5 }}>
            {decision?.embedding_readiness.map((item) => (
              <Chip
                key={item.embedding_model}
                size="small"
                variant="outlined"
                color={item.ready ? 'success' : 'warning'}
                label={`${item.embedding_model}: ${item.ready ? 'ready' : 'fallback'}`}
              />
            ))}
          </Stack>
        )}
        {decision?.consent?.effective_user_recall === false && (
          <Alert severity="info" sx={{ m: 1 }}>
            Global recall is disabled for this thread. You can still inspect and manage stored memory here.
          </Alert>
        )}
        <List disablePadding>
          {messages.map((message, index) => (
            <ListItem key={`${message.role}-${index}`} alignItems="flex-start" sx={{ px: 1.5, py: 1, bgcolor: message.role === 'user' ? 'action.hover' : 'transparent' }}>
              <Box sx={{ minWidth: 0, width: '100%', '& p': { my: 0.5 }, overflowWrap: 'anywhere' }}>
                <Typography variant="caption" color="text.secondary">{message.role === 'user' ? 'You' : 'Memory curator'}</Typography>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
              </Box>
            </ListItem>
          ))}
        </List>
        {busy && <Box sx={{ display: 'grid', placeItems: 'center', py: 2 }}><CircularProgress size={24} /></Box>}
        {decision && ['clarification', 'conflict'].includes(decision.state) && (
          <ConversationDecisionPanel variant={decision.state === 'conflict' ? 'conflict' : 'clarification'}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">{decision.state === 'conflict' ? 'Choose how to resolve this conflict' : 'Choose an option'}</Typography>
              {decision.choices.map((choice) => (
                <Button
                  key={choice.id}
                  variant="outlined"
                  size="small"
                  onClick={() => submitMessage(choice.user_message)}
                  disabled={busy}
                  sx={{ justifyContent: 'flex-start', textAlign: 'left' }}
                >
                  <Box>
                    <Typography variant="body2">{choice.label}</Typography>
                    {choice.description && <Typography variant="caption" color="text.secondary">{choice.description}</Typography>}
                  </Box>
                </Button>
              ))}
            </Stack>
          </ConversationDecisionPanel>
        )}
        {decision?.state === 'proposal' && (
          <ConversationDecisionPanel variant="approval">
            <Stack spacing={1}>
              <Typography variant="subtitle2">Confirm memory changes</Typography>
              {decision.operations.filter((operation) => operation.action !== 'noop').map((operation, index) => (
                <Box key={`${operation.action}-${operation.memory_id || index}`}>
                  {index > 0 && <Divider sx={{ mb: 1 }} />}
                  <Chip size="small" label={operationLabel(operation)} />
                  {operation.content && <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap' }}>{operation.content}</Typography>}
                </Box>
              ))}
              <Stack direction="row" spacing={1}>
                <Button variant="contained" size="small" onClick={() => void apply()} disabled={busy}>Confirm</Button>
                <Button size="small" onClick={() => setDecision(null)} disabled={busy}>Reject</Button>
              </Stack>
            </Stack>
          </ConversationDecisionPanel>
        )}
        {decision?.state === 'no_changes' && decision.review?.cursor && (
          <ConversationDecisionPanel variant="approval">
            <Stack spacing={1}>
              <Typography variant="body2">Confirm that this review found no durable memory changes.</Typography>
              <Stack direction="row" spacing={1}>
                <Button variant="contained" size="small" onClick={() => void apply()} disabled={busy}>Complete review</Button>
                <Button size="small" onClick={() => setDecision(null)} disabled={busy}>Cancel</Button>
              </Stack>
            </Stack>
          </ConversationDecisionPanel>
        )}
        {applied && <Alert severity="success" sx={{ m: 1 }}>Memory workspace refreshed.</Alert>}
        {decision?.embedding_readiness.some((item) => item.degraded) && (
          <Alert severity="warning" sx={{ m: 1 }}>Related-memory search was degraded; recent memory was used as fallback.</Alert>
        )}
      </Box>

      <Box sx={{ py: 1, borderTop: 1, borderColor: 'divider' }}>
        <ConversationComposer
          placeholder={intent.mode === 'edit' ? 'Describe the correction...' : 'Tell the curator what to remember...'}
          disabled={!canCompose}
          busy={busy || modelChecking}
          onSubmit={submitMessage}
        />
      </Box>
    </ConversationPanelShell>
  );
}
