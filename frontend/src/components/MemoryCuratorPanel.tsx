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
  getMemoryWorkspaceStatus,
  prepareMemoryWorkspace,
  respondToMemoryCurator,
  type MemoryCuratorMessage,
  type MemoryCuratorOperation,
  type MemoryCuratorResponse,
  type MemoryChangeReceipt,
  type MemoryCuratorWebSource,
  type MemoryConsistencyReviewCursor,
  type MemoryWorkspaceReadiness,
} from '../lib/api';
import { checkLlmModelReady, fetchAvailableLlmModels } from '../lib/models-api';
import {
  buildCuratorContext,
  curatorTitle,
  toMemoryConsistencyReviewCursor,
  type MemoryCuratorIntent,
} from '../lib/memory-curator';
import {
  ConversationComposer,
  ConversationHeader,
  ConversationMessageBubble,
  ConversationPanelTemplate,
  ConversationTranscriptFrame,
  DecisionChoiceList,
  ResizableDecisionPanel,
  WebSearchModeControl,
  WebSourceList,
} from './conversation';
import { useWebSearchMode } from '../hooks/useWebSearchMode';
import { getChatComposerState } from '../lib/chat-composer-state';
import { ChatComposerIndexingStatus, ChatComposerStatus } from '../lib/enums';
import EmbeddingModelReadinessIndicator from './EmbeddingModelReadinessIndicator';

const defaultContextWindow = 8192;

const initialPrompt = (intent: MemoryCuratorIntent) => {
  if (intent.mode === 'memory_review') {
    return 'Review related memories for duplicates, conflicts, superseded statements, and stale override relationships.';
  }
  if (intent.mode === 'conversation_review') {
    return 'Review the next eligible completed conversation turns and suggest only durable memories for this thread.';
  }
  if (intent.mode === 'edit') {
    return `I want to review and edit this memory:\n\n${intent.memory?.content || ''}`;
  }
  return 'Help me add a durable memory. Ask what should be remembered and help me choose the right scope.';
};

type CuratorUiMessage = MemoryCuratorMessage & { id: string; web_sources?: MemoryCuratorWebSource[] };

let curatorMessageSequence = 0;
const curatorMessage = (
  role: MemoryCuratorMessage['role'],
  content: string,
  choiceId?: string,
): CuratorUiMessage => ({
  id: `curator-message-${Date.now()}-${curatorMessageSequence++}`,
  role,
  content,
  ...(choiceId ? { choice_id: choiceId } : {}),
});

const initialAssistantMessage = (intent: MemoryCuratorIntent): CuratorUiMessage | null => {
  if (intent.mode === 'conversation_review' || intent.mode === 'memory_review') return null;
  if (intent.mode === 'edit') {
    return curatorMessage(
      'assistant',
      `What would you like to change about this memory?\n\n> ${intent.memory?.content || ''}`,
    );
  }
  return curatorMessage(
    'assistant',
    'What should be remembered?',
  );
};

const operationLabel = (operation: MemoryCuratorOperation) => {
  const scope = operation.scope_type === 'user' ? 'Global' : operation.scope_type === 'project' ? 'Project' : 'Thread';
  return `${operation.action.toUpperCase()} ${scope || ''}`.trim();
};

const scopeLabel = (scope?: { scope_type: string; scope_id: string }) => {
  if (!scope) return 'Stored';
  if (scope.scope_type === 'user') return 'Global';
  if (scope.scope_type === 'project') return 'Project';
  return 'Thread';
};

const receiptMessage = (receipt: MemoryChangeReceipt) => {
  const result = receipt.result_memory_id ? ` (${receipt.result_memory_id})` : '';
  if (receipt.action === 'move') {
    return `Moved the memory from ${scopeLabel(receipt.source_scope)} to ${scopeLabel(receipt.destination_scope)}${result}.`;
  }
  if (receipt.action === 'delete') return `Deleted the ${scopeLabel(receipt.source_scope)} memory.`;
  if (receipt.action === 'set_overrides') return `Updated the memory's override relationships${result}.`;
  if (receipt.action === 'update') return `Updated the ${scopeLabel(receipt.destination_scope)} memory${result}.`;
  return `Created the ${scopeLabel(receipt.destination_scope)} memory${result}.`;
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
  const [workspaceReadiness, setWorkspaceReadiness] = useState<MemoryWorkspaceReadiness | null>(null);
  const [workspaceReadinessFailed, setWorkspaceReadinessFailed] = useState(false);
  const [workspaceReadinessVersion, setWorkspaceReadinessVersion] = useState(0);
  const { mode: webSearchMode, setMode: setWebSearchMode } = useWebSearchMode();
  const [messages, setMessages] = useState<CuratorUiMessage[]>(() => {
    const initial = initialAssistantMessage(intent);
    return initial ? [initial] : [];
  });
  const [decision, setDecision] = useState<MemoryCuratorResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [reviewCursor, setReviewCursor] = useState<MemoryConsistencyReviewCursor | null>(null);
  const [reviewCanContinue, setReviewCanContinue] = useState(false);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const context = useMemo(() => buildCuratorContext(intent), [intent]);
  const curatorEmbeddingModel = intent.embeddingModel
    || workspaceReadiness?.embedding_model
    || decision?.embedding_readiness[0]?.embedding_model
    || '';
  const hasUnconfirmedDecision = Boolean(
    decision?.state === 'proposal'
    || decision?.state === 'web_search_approval'
    || (decision?.state === 'no_changes' && (decision.review?.cursor || decision.memory_review)),
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

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async (prepare: boolean) => {
      try {
        const status = prepare
          ? await prepareMemoryWorkspace({ threadId: intent.threadId, projectId: intent.projectId })
          : await getMemoryWorkspaceStatus({ threadId: intent.threadId, projectId: intent.projectId });
        if (cancelled) return;
        setWorkspaceReadinessFailed(false);
        setWorkspaceReadiness(status);
        if (status.status === 'indexing') {
          timer = window.setTimeout(() => void poll(false), 1500);
        }
      } catch (requestError) {
        if (!cancelled) {
          setWorkspaceReadinessFailed(true);
          setError(requestError instanceof Error
            ? requestError.message
            : 'Memory workspace readiness could not be checked.');
        }
      }
    };
    setWorkspaceReadiness(null);
    setWorkspaceReadinessFailed(false);
    void poll(true);
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [intent.projectId, intent.threadId, workspaceReadinessVersion]);

  const respond = useCallback(async (
    nextMessages: CuratorUiMessage[],
    webSearchDecision?: { query: string; approved: boolean },
  ) => {
    if (!llmModel || modelReady !== true) return;
    setBusy(true);
    setError(null);
    setApplied(false);
    try {
      const response = await respondToMemoryCurator({
        mode: intent.mode,
        context,
        memory_id: intent.memory?.id,
        messages: nextMessages.map(({ role, content, choice_id }) => ({
          role,
          content,
          ...(choice_id ? { choice_id } : {}),
        })),
        llm_model: llmModel,
        context_window: contextWindow,
        web_search_mode: webSearchMode,
        ...(webSearchDecision ? { web_search_decision: webSearchDecision } : {}),
        ...(intent.mode === 'memory_review' && reviewCursor ? { memory_review_cursor: reviewCursor } : {}),
      });
      setMessages([...nextMessages, {
        ...curatorMessage('assistant', response.message),
        web_sources: response.web_sources || [],
      }]);
      setDecision(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'The memory curator could not respond.');
    } finally {
      setBusy(false);
    }
  }, [context, contextWindow, intent.memory?.id, intent.mode, llmModel, modelReady, reviewCursor, webSearchMode]);

  useEffect(() => {
    if (!['conversation_review', 'memory_review'].includes(intent.mode) || !llmModel || modelReady !== true || messages.length) return;
    const firstMessage = curatorMessage('user', initialPrompt(intent));
    setMessages([firstMessage]);
    void respond([firstMessage]);
  }, [intent, llmModel, messages.length, modelReady, respond]);

  const submitMessage = (text: string, choiceId?: string) => {
    const next = [...messages, curatorMessage('user', text, choiceId)];
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
        memory_review_cursor: decision.memory_review
          ? toMemoryConsistencyReviewCursor(decision.memory_review)
          : undefined,
        actor_id: 'ui',
      });
      setApplied(true);
      setDecision(null);
      setReviewCanContinue(Boolean(
        decision.memory_review
        && decision.memory_review.remaining_anchor_count > 0
        && !result.memory_review_completed
      ));
      if (decision.memory_review) {
        setReviewCursor(toMemoryConsistencyReviewCursor(decision.memory_review));
      }
      onDirtyChange(false);
      onApplied();
      setWorkspaceReadinessVersion((version) => version + 1);
      const warningSuffix = result.warnings.length
        ? `\n${result.warnings.length} indexing warning(s) occurred. Failed records remain available for Retry.`
        : '';
      const summary = result.receipts?.length
        ? `${result.receipts.map(receiptMessage).join('\n')}${warningSuffix}`
        : result.warnings.length
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

  const workspaceIndexingStatus = workspaceReadinessFailed
    ? ChatComposerIndexingStatus.Error
    : workspaceReadiness === null
    ? ChatComposerIndexingStatus.Checking
    : workspaceReadiness.status === 'ready'
      ? ChatComposerIndexingStatus.Ready
      : workspaceReadiness.status === 'blocked'
        ? ChatComposerIndexingStatus.Blocked
        : workspaceReadiness.status === 'error'
          ? ChatComposerIndexingStatus.Error
          : ChatComposerIndexingStatus.Indexing;
  const composerState = useMemo(() => getChatComposerState({
    loading: busy,
    llmModel,
    isLlmModelValid: modelReady,
    isLlmToolsSupported: true,
    isEmbeddingModelValid: workspaceReadinessFailed
      ? true
      : workspaceReadiness?.embedding_model_ready ?? null,
    indexingStatus: workspaceIndexingStatus,
    hasInput: false,
  }), [busy, llmModel, modelReady, workspaceIndexingStatus, workspaceReadiness?.embedding_model_ready, workspaceReadinessFailed]);
  const readyPlaceholder = intent.mode === 'memory_review'
    ? 'Add context for this review...'
    : intent.mode === 'edit'
      ? 'Describe the correction...'
      : 'Tell the curator what to remember...';
  const copyMessage = async (message: CuratorUiMessage) => {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId((current) => current === message.id ? null : current), 2000);
  };

  let decisionPanel: React.ReactNode;
  if (decision?.state === 'web_search_approval' && decision.pending_web_search) {
    const pending = decision.pending_web_search;
    decisionPanel = (
      <ResizableDecisionPanel
        title="Allow internet search?"
        variant="approval"
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
        <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>{pending.reason}</Typography>
        <Typography variant="caption" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>
          Query: {pending.query}
        </Typography>
        <DecisionChoiceList
          choices={[
            { id: 'approve-web-search', label: 'Approve search', description: 'Run this exact query once.', text: 'Approve this internet search.' },
            { id: 'deny-web-search', label: 'Continue without search', description: 'Continue without running this query.', text: 'Continue without this internet search.' },
          ]}
          disabled={busy}
          onSelect={(choice, text) => {
            const next = [...messages, curatorMessage('user', text, choice.id)];
            setDecision(null);
            setMessages(next);
            void respond(next, { query: pending.query, approved: choice.id === 'approve-web-search' });
          }}
          onCustomSubmit={(text) => submitMessage(text)}
          customPlaceholder="Tell the curator how to proceed"
        />
      </ResizableDecisionPanel>
    );
  } else if (decision && ['clarification', 'conflict'].includes(decision.state)) {
    decisionPanel = (
      <ResizableDecisionPanel
        title={decision.state === 'conflict' ? 'Choose how to resolve this conflict' : 'Choose an option'}
        variant={decision.state === 'conflict' ? 'conflict' : 'clarification'}
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
        <DecisionChoiceList
          choices={decision.choices.map((choice) => ({
            id: choice.id,
            label: choice.label,
            description: choice.description,
            text: choice.user_message,
          }))}
          disabled={busy}
          onSelect={(choice, text) => submitMessage(text, choice.id)}
          onCustomSubmit={(text) => submitMessage(text)}
          customPlaceholder="Tell the curator what outcome you prefer"
        />
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
        {(decision.operation_summaries?.length
          ? decision.operation_summaries
          : decision.operations.filter((operation) => operation.action !== 'noop').map((operation) => ({
              operation_group_id: operation.operation_group_id || `${operation.action}-${operation.memory_id || ''}`,
              action: operation.semantic_action || operation.action,
              label: operationLabel(operation),
              content: operation.content,
              override_target_ids: operation.override_targets?.map((target) => target.memory_id) || [],
              removed_incoming_override_count: 0,
            }))).map((operation, index) => (
          <Box key={operation.operation_group_id || `${operation.action}-${index}`}>
            {index > 0 && <Divider sx={{ mb: 1 }} />}
            <Chip size="small" label={operation.label} />
            {operation.content && <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap' }}>{operation.content}</Typography>}
            {Boolean(operation.override_target_ids?.length) && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Overrides {operation.override_target_ids?.length} broader {operation.override_target_ids?.length === 1 ? 'memory' : 'memories'}.
              </Typography>
            )}
            {Boolean(operation.removed_incoming_override_count) && (
              <Typography variant="caption" color="warning.main" sx={{ display: 'block', mt: 0.5 }}>
                Removes {operation.removed_incoming_override_count} incoming override relationship(s).
              </Typography>
            )}
          </Box>
        ))}
        <Stack direction="row" spacing={1}>
          <Button variant="contained" size="small" onClick={() => void apply()} disabled={busy}>Confirm</Button>
          <Button size="small" onClick={() => setDecision(null)} disabled={busy}>Revise</Button>
        </Stack>
      </ResizableDecisionPanel>
    );
  } else if (decision?.state === 'no_changes' && (decision.review?.cursor || decision.memory_review)) {
    decisionPanel = (
      <ResizableDecisionPanel
        title={decision.memory_review?.remaining_anchor_count ? 'Continue memory review' : 'Complete memory review'}
        variant="approval"
        rootRef={panelRef}
        horizontalInset={1}
        minHeight={80}
      >
        <Typography variant="body2">Confirm that this group needs no memory changes.</Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="contained" size="small" onClick={() => void apply()} disabled={busy}>
            {decision.memory_review?.remaining_anchor_count ? 'Accept and continue' : 'Complete review'}
          </Button>
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
          beforeModelControls={(
            <WebSearchModeControl mode={webSearchMode} disabled={busy} onChange={setWebSearchMode} />
          )}
          leading={(
            <>
              {curatorEmbeddingModel && (
                <EmbeddingModelReadinessIndicator
                  model={curatorEmbeddingModel}
                  ready={workspaceReadiness?.embedding_model_ready ?? null}
                  size={18}
                />
              )}
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
          {decision?.memory_review && (
            <Alert severity={decision.memory_review.representation_pending ? 'warning' : 'info'} sx={{ m: 1 }}>
              Reviewed {decision.memory_review.reviewed_anchor_count} anchor(s). {decision.memory_review.remaining_anchor_count} remain.
              {decision.memory_review.representation_pending
                ? ` ${decision.memory_review.missing_representation_count} Global representation(s) are warming for ${decision.memory_review.embedding_model} and were omitted from this review.`
                : ''}
            </Alert>
          )}
          {reviewCanContinue && reviewCursor && (
            <Box sx={{ px: 1, pb: 1 }}>
              <Button
                size="small"
                variant="outlined"
                startIcon={<MemoryIcon />}
                disabled={busy || modelReady !== true}
                onClick={() => {
                  setReviewCanContinue(false);
                  void respond(messages);
                }}
              >
                Review next group
              </Button>
            </Box>
          )}
          {decision?.consent?.effective_user_recall === false && (
            <Alert severity="info" sx={{ m: 1 }}>
              Global recall is disabled for this thread. You can still inspect and manage stored memory here.
            </Alert>
          )}
          {applied && <Alert severity="success" sx={{ m: 1 }}>Memory workspace refreshed.</Alert>}
        </Box>
      )}
      transcript={(
        <ConversationTranscriptFrame>
          {messages.map((message) => (
            <ConversationMessageBubble
              key={message.id}
              role={message.role}
              content={message.content}
              afterContent={<WebSourceList sources={message.web_sources || []} />}
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
      composer={hasUnconfirmedDecision ? null : (
        <Box sx={{ py: 1 }}>
          <ConversationComposer
            placeholder={composerState.status === ChatComposerStatus.Ready ? readyPlaceholder : composerState.placeholder}
            disabled={composerState.disabled || contextWindow < 256}
            busy={composerState.busy}
            disableWhenEmpty
            onSubmit={submitMessage}
          />
        </Box>
      )}
    />
  );
}
