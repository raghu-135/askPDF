import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  IconButton,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import MemoryIcon from '@mui/icons-material/Memory';
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
  ConversationHeader,
  ConversationMessageBubble,
  ConversationPanelTemplate,
  ConversationTranscriptFrame,
  ResizableDecisionPanel,
} from './conversation';

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

type CuratorUiMessage = MemoryCuratorMessage & { id: string };

let curatorMessageSequence = 0;
const curatorMessage = (
  role: MemoryCuratorMessage['role'],
  content: string,
): CuratorUiMessage => ({
  id: `curator-message-${Date.now()}-${curatorMessageSequence++}`,
  role,
  content,
});

const initialAssistantMessage = (intent: MemoryCuratorIntent): CuratorUiMessage | null => {
  if (intent.mode === 'conversation_review') return null;
  if (intent.mode === 'edit') {
    return curatorMessage(
      'assistant',
      `What would you like to change about this memory?\n\n> ${intent.memory?.content || ''}`,
    );
  }
  return curatorMessage(
    'assistant',
    'What should be remembered? I will check for related or conflicting memory before proposing a change.',
  );
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
  const [messages, setMessages] = useState<CuratorUiMessage[]>(() => {
    const initial = initialAssistantMessage(intent);
    return initial ? [initial] : [];
  });
  const [decision, setDecision] = useState<MemoryCuratorResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
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

  const respond = useCallback(async (nextMessages: CuratorUiMessage[]) => {
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
        messages: boundedMessages.map(({ role, content }) => ({ role, content })),
        llm_model: llmModel,
        context_window: contextWindow,
      });
      setMessages([...boundedMessages, curatorMessage('assistant', response.message)]);
      setDecision(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The memory curator could not respond.');
    } finally {
      setBusy(false);
    }
  }, [context, contextWindow, intent.memory?.id, intent.mode, llmModel, modelReady]);

  useEffect(() => {
    if (intent.mode !== 'conversation_review' || !llmModel || modelReady !== true || messages.length) return;
    const firstMessage = curatorMessage('user', initialPrompt(intent));
    setMessages([firstMessage]);
    void respond([firstMessage]);
  }, [intent, llmModel, messages.length, modelReady, respond]);

  const submitMessage = (text: string) => {
    const next = [...messages, curatorMessage('user', text)];
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
      setMessages((current) => [...current, curatorMessage('assistant', summary)]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The memory changes could not be applied.');
    } finally {
      setBusy(false);
    }
  };

  const modelChecking = Boolean(llmModel) && modelReady === null;
  const canCompose = Boolean(llmModel && contextWindow >= 256 && modelReady === true && !busy);
  const copyMessage = async (message: CuratorUiMessage) => {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId((current) => current === message.id ? null : current), 2000);
  };

  let decisionPanel: React.ReactNode;
  if (decision && ['clarification', 'conflict'].includes(decision.state)) {
    decisionPanel = (
      <ResizableDecisionPanel
        title={decision.state === 'conflict' ? 'Choose how to resolve this conflict' : 'Choose an option'}
        variant={decision.state === 'conflict' ? 'conflict' : 'clarification'}
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
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
      </ResizableDecisionPanel>
    );
  } else if (decision?.state === 'proposal') {
    decisionPanel = (
      <ResizableDecisionPanel
        title="Confirm memory changes"
        variant="approval"
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
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
      </ResizableDecisionPanel>
    );
  } else if (decision?.state === 'no_changes' && decision.review?.cursor) {
    decisionPanel = (
      <ResizableDecisionPanel
        title="Complete memory review"
        variant="approval"
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
        <Typography variant="body2">Confirm that this review found no durable memory changes.</Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="contained" size="small" onClick={() => void apply()} disabled={busy}>Complete review</Button>
          <Button size="small" onClick={() => setDecision(null)} disabled={busy}>Cancel</Button>
        </Stack>
      </ResizableDecisionPanel>
    );
  }

  return (
    <ConversationPanelTemplate
      ref={panelRef}
      sx={{ p: 1, cursor: 'default' }}
      header={(
        <ConversationHeader
          models={models}
          model={llmModel}
          contextWindow={contextWindow}
          disabled={busy}
          onModelChange={setLlmModel}
          onContextWindowChange={(value) => {
            setContextWindow(value);
            if (value > 0) window.localStorage.setItem('last_context_window', String(value));
          }}
          leading={(
            <>
              <MemoryIcon color="primary" fontSize="small" />
              <Typography
                variant="subtitle2"
                noWrap
                sx={{ fontWeight: 700, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}
              >
                {curatorTitle(intent)}
              </Typography>
            </>
          )}
          trailingActions={(
            <Tooltip title="Close memory curator">
              <IconButton size="small" onClick={onClose} aria-label="Close memory curator">
                <CloseIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        />
      )}
      status={(
        <Box sx={{ maxHeight: '35%', overflowY: 'auto', flexShrink: 0 }}>
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
          {applied && <Alert severity="success" sx={{ m: 1 }}>Memory workspace refreshed.</Alert>}
          {decision?.embedding_readiness.some((item) => item.degraded) && (
            <Alert severity="warning" sx={{ m: 1 }}>Related-memory search was degraded; recent memory was used as fallback.</Alert>
          )}
        </Box>
      )}
      transcript={(
        <ConversationTranscriptFrame>
          {messages.map((message) => (
            <ConversationMessageBubble
              key={message.id}
              role={message.role}
              content={message.content}
              actions={(
                <Tooltip title={copiedId === message.id ? 'Copied!' : 'Copy message'}>
                  <IconButton
                    size="small"
                    onClick={() => void copyMessage(message)}
                    aria-label="Copy message"
                    sx={{ color: 'inherit', p: 0.5 }}
                  >
                    {copiedId === message.id ? <CheckIcon fontSize="small" /> : <ContentCopyIcon fontSize="small" />}
                  </IconButton>
                </Tooltip>
              )}
            />
          ))}
        </ConversationTranscriptFrame>
      )}
      decision={decisionPanel}
      composer={(
        <Box sx={{ py: 1 }}>
          <ConversationComposer
            placeholder={intent.mode === 'edit' ? 'Describe the correction...' : 'Tell the curator what to remember...'}
            disabled={!canCompose}
            busy={busy || modelChecking}
            disableWhenEmpty
            onSubmit={submitMessage}
          />
        </Box>
      )}
    />
  );
}
