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
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DriveFileMoveRtlIcon from '@mui/icons-material/DriveFileMoveRtl';
import PsychologyIcon from '@mui/icons-material/Psychology';
import PsychologyAltIcon from '@mui/icons-material/PsychologyAlt';
import {
  applyMemoryManagerPlan,
  getMemoryWorkspaceStatus,
  planMemoryManager,
  prepareMemoryWorkspace,
  type MemoryCuratorMessage,
  type MemoryCuratorOperation,
  type MemoryCuratorResponse,
  type MemoryCuratorWebSource,
  type MemoryManagerOperation,
  type MemoryManagerPlan,
  type MemoryConsistencyReviewCursor,
  type MemoryWorkspaceReadiness,
} from '../lib/api';
import { checkLlmModelReady, fetchAvailableLlmModels } from '../lib/models-api';
import {
  buildCuratorContext,
  toMemoryConsistencyReviewCursor,
  type MemoryManagerIntent,
} from '../lib/memory-manager';
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
  WorkspaceContextHeader,
} from './conversation';
import { useWebSearchMode } from '../hooks/useWebSearchMode';
import { getChatComposerState } from '../lib/chat-composer-state';
import { ChatComposerIndexingStatus, ChatComposerStatus } from '../lib/enums';
import EmbeddingModelReadinessIndicator from './EmbeddingModelReadinessIndicator';
import {
  memoryOperationLabel,
  memoryReceiptMessage,
  memoryWorkspaceTitle,
} from '../lib/memory-ui';

const defaultContextWindow = 8192;

const initialPrompt = (intent: MemoryManagerIntent) => {
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

const conversationReviewComposerText = 'Click Send to review this conversation and extract durable memories for this thread.';

const managerMode = (mode: MemoryManagerIntent['mode']): 'direct_edit' | 'conversation_extract' | 'consistency_review' => (
  mode === 'conversation_review'
    ? 'conversation_extract'
    : mode === 'memory_review'
      ? 'consistency_review'
      : 'direct_edit'
);

const managerOperationToCuratorOperation = (operation: MemoryManagerOperation): MemoryCuratorOperation => ({
  action: operation.type === 'memory_create' ? 'create' : operation.type === 'memory_delete' ? 'delete' : 'update',
  scope_type: operation.scope_type,
  scope_id: operation.scope_id,
  memory_id: operation.memory_id,
  expected_updated_at: operation.expected_updated_at,
  content: operation.content,
  attributes: operation.attributes,
  override_targets: operation.override_target_ids.map((memoryId) => ({
    memory_id: memoryId,
    expected_updated_at: operation.override_target_versions?.[memoryId] || '',
  })),
  semantic_action: operation.type === 'relationship_replace' ? 'set_overrides' : undefined,
  operation_group_id: operation.operation_group_id,
  move_source_memory_id: operation.source_memory_id,
  move_destination_memory_id: operation.destination_memory_id,
});

const managerPlanToCuratorResponse = (plan: MemoryManagerPlan): MemoryCuratorResponse => ({
  message: plan.message,
  state: plan.state === 'blocked' ? 'clarification' : plan.state,
  choices: plan.choices,
  operations: plan.operations.map(managerOperationToCuratorOperation),
  operation_summaries: plan.analysis,
  review: plan.review,
  memory_review: plan.memory_review,
  embedding_readiness: plan.embedding_readiness,
  pending_web_search: plan.pending_web_search,
  web_sources: plan.web_sources,
  consent: plan.consent,
});

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

const initialAssistantMessage = (intent: MemoryManagerIntent): CuratorUiMessage | null => {
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

export default function MemoryManagerPanel({
  intent,
  onBack,
  backLabel = 'Back',
  contextSubtitle,
  onDirtyChange,
  onApplied,
}: {
  intent: MemoryManagerIntent;
  onBack: () => void;
  backLabel?: string;
  contextSubtitle?: React.ReactNode;
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
  const [managerPlan, setManagerPlan] = useState<MemoryManagerPlan | null>(null);
  const [reviewId, setReviewId] = useState<string | null>(null);
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
      const plan = await planMemoryManager({
        mode: managerMode(intent.mode),
        context,
        memory_id: intent.memory?.id,
        messages: nextMessages.map(({ role, content, choice_id }) => ({
          role,
          content,
          ...(choice_id ? { choice_id } : {}),
        })),
        llm_model: llmModel,
        context_window: contextWindow,
        review_round: reviewCursor ? 2 : 1,
        review_id: reviewId,
        web_search_mode: webSearchMode,
        ...(webSearchDecision ? { web_search_decision: webSearchDecision } : {}),
        ...(intent.mode === 'memory_review' && reviewCursor ? { memory_review_cursor: reviewCursor } : {}),
      });
      setManagerPlan(plan);
      if (plan.review_id) setReviewId(plan.review_id);
      const response = managerPlanToCuratorResponse(plan);
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
  }, [context, contextWindow, intent.memory?.id, intent.mode, llmModel, modelReady, reviewCursor, reviewId, webSearchMode]);

  useEffect(() => {
    if (intent.mode !== 'memory_review' || !llmModel || modelReady !== true || messages.length) return;
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
    if (!decision || !managerPlan) return;
    setBusy(true);
    setError(null);
    try {
      const result = await applyMemoryManagerPlan({
        plan: managerPlan,
        idempotency_key: `memory-manager:${managerPlan.plan_id}`,
        actor_id: 'ui',
      });
      setApplied(true);
      setDecision(null);
      setManagerPlan(null);
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
        ? `${result.receipts.map(memoryReceiptMessage).join('\n')}${warningSuffix}`
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
    ? 'Add context for this existing-memory review...'
    : intent.mode === 'conversation_review'
      ? 'Add guidance for reusable thread memories...'
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
        customPlaceholder="Describe how you want to proceed"
        />
      </ResizableDecisionPanel>
    );
  } else if (decision && ['clarification', 'conflict'].includes(decision.state)) {
    decisionPanel = (
      <ResizableDecisionPanel
        title={decision.state === 'conflict' ? 'Choose how to resolve this conflict' : 'Provide clarification'}
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
          customPlaceholder="Describe the intended outcome"
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
              label: memoryOperationLabel(operation),
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
      sx={{ cursor: 'default' }}
      header={(
        <>
          <WorkspaceContextHeader
            title={memoryWorkspaceTitle(intent.scopeType)}
            icon={
              intent.mode === 'conversation_review'
                ? <PsychologyAltIcon color="primary" fontSize="small" />
                : <PsychologyIcon color="primary" fontSize="small" />
            }
            onBack={onBack}
            backIcon={backLabel === 'Back to Project' ? <DriveFileMoveRtlIcon fontSize="small" sx={{ flex: '0 0 auto' }} /> : undefined}
            backLabel={backLabel}
            backContextLabel={contextSubtitle || backLabel}
          />
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
            sx={{
              minHeight: 72,
              px: 1.5,
              py: 1,
              mb: 0,
            }}
            beforeModelControls={(
              <WebSearchModeControl mode={webSearchMode} disabled={busy} onChange={setWebSearchMode} />
            )}
            leading={(
              <Box sx={{ minWidth: 0, overflow: 'hidden' }}>
                {curatorEmbeddingModel && (
                  <EmbeddingModelReadinessIndicator
                    model={curatorEmbeddingModel}
                    ready={workspaceReadiness?.embedding_model_ready ?? null}
                    size={18}
                  />
                )}
              </Box>
            )}
          />
        </>
      )}
      status={(
        <Box sx={{ maxHeight: '35%', overflowY: 'auto', flexShrink: 0 }}>
          {error && <Alert severity="error" onClose={() => setError(null)} sx={{ m: 1 }}>{error}</Alert>}
          {modelReady === false && <Alert severity="warning" sx={{ m: 1 }}>Select a ready chat model to continue.</Alert>}
          {decision?.review && (
            <Alert severity="info" sx={{ m: 1 }}>
              Reviewed {decision.review.reviewed_count} completed turn(s) for reusable Thread memories. {decision.review.remaining_count} remain after this batch.
            </Alert>
          )}
          {decision?.memory_review && (
            <Alert severity={decision.memory_review.representation_pending ? 'warning' : 'info'} sx={{ m: 1 }}>
              Reviewed {decision.memory_review.reviewed_anchor_count} existing memory anchor(s) for conflicts or duplicates. {decision.memory_review.remaining_anchor_count} remain.
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
                startIcon={intent.mode === 'conversation_review' ? <PsychologyAltIcon /> : <PsychologyIcon />}
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
        <Box sx={{ px: 1, py: 1 }}>
          <ConversationComposer
            seedText={intent.mode === 'conversation_review' ? conversationReviewComposerText : ''}
            seedVersion={intent.mode === 'conversation_review' ? 1 : 0}
            placeholder={composerState.status === ChatComposerStatus.Ready ? readyPlaceholder : composerState.placeholder}
            disabled={composerState.disabled || contextWindow < 256}
            busy={composerState.busy}
            disableWhenEmpty
            onSubmit={(text) => submitMessage(intent.mode === 'conversation_review' ? initialPrompt(intent) : text)}
          />
        </Box>
      )}
    />
  );
}
