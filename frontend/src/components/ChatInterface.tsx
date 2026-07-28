import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useTheme } from '@mui/material/styles';
import {
    Box,
    TextField,
    Button,
    List,
    ListItem,
    Typography,
    Select,
    MenuItem,
    Paper,
    FormControl,
    InputLabel,
    IconButton,
    Tooltip,
    Chip,
    CircularProgress,
} from '@mui/material';
import WifiTwoToneIcon from '@mui/icons-material/WifiTwoTone';
import WifiOffTwoToneIcon from '@mui/icons-material/WifiOffTwoTone';
import WifiPasswordIcon from '@mui/icons-material/WifiPassword';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import EditIcon from '@mui/icons-material/Edit';
import MemoryIcon from '@mui/icons-material/Memory';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import CallSplitIcon from '@mui/icons-material/CallSplit';
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined';
import RouteIcon from '@mui/icons-material/Route';
import dynamic from 'next/dynamic';
import remarkGfm from 'remark-gfm';
import { useVirtualizer } from '@tanstack/react-virtual';
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });
import { deriveChatSentences, type ChatSentenceCache } from '../lib/chat-sentence-cache';
import { getChatComposerState } from '../lib/chat-composer-state';
import { canRequestChatCancellation, recoverCanceledChat } from '../lib/chat-run-cancellation';
import {
    Thread,
    Message,
    WebSource,
    PromptToolDefinition,
    streamThreadChat,
    getThreadMessages,
    deleteMessage,
    forkThread,
    listProjects,
    listThreads,
    getThreadIndexStatus,
    getThreadSettings,
    getProject,
    updateThreadSettings,
    getPromptTools,
    getPromptPreview,
    listMemoryCandidates,
    listAgentWorkflows,
    getAgentRun,
    listThreadAgentRuns,
    AgentRunDetails,
    AgentTraceRefs,
    AgentRunPendingInterrupt,
    AgentRunResumeAction,
    AgentWorkflow,
    streamResumeAgentRun,
    cancelChatAgentRun,
    cancelAgentWorkflowBuilderTest,
    getLatestAgentWorkflowBuilderTest,
    resumeAgentWorkflowBuilderTest,
    streamAgentWorkflowBuilderTest,
    resolveMemoryCandidate,
    type MemoryCandidate,
    type Project,
    type ThreadChatResponse,
} from '../lib/api';
import type { AgentExecutionStreamEnvelope } from '../lib/agent-execution-stream';
import { withPollingRetry, withRetry } from '../lib/retry-utils';
import { isRetryableError } from '../lib/error-utils';
import { fetchAvailableLlmModels, checkLlmModelReady, checkEmbeddingModelReady } from '../lib/models-api';
import {
    AgentRunResumeAction as AgentRunResumeActionValue,
    ChatComposerIndexingStatus,
    EmbeddingReadinessStatus,
    InterruptStatus,
    MessageRole,
    ReasoningFormat,
    type ChatComposerIndexingStatus as ChatComposerIndexingStatusValue,
    type ReasoningFormat as ReasoningFormatValue,
} from '../lib/enums';
import ChatSettingsDialog from './ChatSettingsDialog';
import ThreadLineageTooltipContent from './ThreadLineageTooltipContent';
import ThreadForkDialog, { MemoryCopyMode } from './ThreadForkDialog';
import { buildLiveTraceView, buildRunTraceView } from './agent-debug/agent-trace-projection';
import type { TraceRunView } from './agent-debug/agent-trace-projection';
import useBatchedExecutionEvents from './agent-graph/useBatchedExecutionEvents';
import {
    transientMessagesForRequest,
    workflowSpecFingerprint,
    type BuilderTestSession,
} from '../lib/builder-test-session';
import {
    isLiveTraceTerminalEvent,
    liveTraceStatusFromEvent,
    LiveTraceStreamController,
} from '../lib/live-trace-stream';

interface ChatMessage extends Message {
    isRecollected?: boolean;
    reasoning?: string;
    reasoning_available?: boolean;
    reasoning_format?: ReasoningFormatValue;
    rewritten_query?: string;
    web_sources?: WebSource[];
    agent_run_id?: string;
    agent_run_turn_kind?: string;
    agent_run_sequence?: number | null;
    agent_trace_refs?: AgentTraceRefs | null;
    agent_workflow_id?: string;
    agent_route?: string;
    agent_route_reason?: string;
    pending_human_review?: boolean;
}

type WebSearchMode = 'on' | 'ask' | 'off';

type LiveChatExecution = {
    messageId: string;
    runId?: string;
    running: boolean;
    canceling?: boolean;
    error?: string;
};

type ClarificationChoice = {
    text: string;
    isOriginal: boolean;
};

const clarificationChoiceText = (choice: unknown): string => {
    if (typeof choice === 'string') return choice;
    if (choice && typeof choice === 'object') {
        const candidate = choice as { text?: unknown; label?: unknown; title?: unknown; question?: unknown };
        for (const value of [candidate.text, candidate.label, candidate.title, candidate.question]) {
            if (typeof value === 'string') return value;
        }
    }
    return '';
};

type PendingHumanReview = {
    runId: string;
    interrupt: AgentRunPendingInterrupt;
    localUserMessageId: string;
    localAssistantMessageId: string;
};

type NormalThreadRuntimeState = {
    kind: 'normal-thread';
    persistent: true;
    historyReadOnly: false;
};

type BuilderTestRuntimeState = {
    kind: 'builder-test';
    persistent: false;
    historyReadOnly: true;
    sessionIdRef: React.MutableRefObject<string>;
    updateMessage: (
        messageId: string,
        patch: Partial<BuilderTestSession['messages'][number]>,
    ) => void;
};

type RuntimeState = NormalThreadRuntimeState | BuilderTestRuntimeState;

const createBuilderTestSessionId = () => (
    globalThis.crypto?.randomUUID?.() || `builder-${Date.now()}`
);

const builderTestMessageToChatMessage = (
    message: BuilderTestSession['messages'][number],
    baseWorkflowId: string,
): ChatMessage => ({
    id: message.id,
    role: message.role === 'user' ? MessageRole.User : MessageRole.Assistant,
    content: message.content,
    created_at: message.createdAt,
    agent_run_id: message.runId,
    agent_workflow_id: baseWorkflowId,
    pending_human_review: message.status === 'review',
});

const useNormalThreadChatRuntime = (): NormalThreadRuntimeState => useMemo(() => ({
    kind: 'normal-thread',
    persistent: true,
    historyReadOnly: false,
}), []);

const useBuilderTestChatRuntime = (
    testRuntime: BuilderTestConversationRuntime | undefined,
    setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
): BuilderTestRuntimeState | null => {
    const sessionIdRef = useRef(createBuilderTestSessionId());

    useEffect(() => {
        if (!testRuntime) return;
        const temporary = testRuntime.session.messages.map((message) => (
            builderTestMessageToChatMessage(message, testRuntime.baseWorkflowId)
        ));
        setMessages((current) => [
            ...current.filter((message) => !message.id.startsWith('test-')),
            ...temporary,
        ]);
    }, [setMessages, testRuntime?.baseWorkflowId, testRuntime?.session.messages]);

    const updateMessage = useCallback((
        messageId: string,
        patch: Partial<BuilderTestSession['messages'][number]>,
    ) => {
        if (!testRuntime) return;
        testRuntime.onSessionChange((current) => ({
            ...current,
            messages: current.messages.map((message) => (
                message.id === messageId ? { ...message, ...patch } : message
            )),
        }));
    }, [testRuntime]);

    if (!testRuntime) return null;
    return {
        kind: 'builder-test',
        persistent: false,
        historyReadOnly: true,
        sessionIdRef,
        updateMessage,
    };
};

const ChatComposer = React.memo(function ChatComposer({
    inputRef,
    seedText,
    seedVersion,
    loading,
    llmModel,
    isLlmModelValid,
    isLlmToolsSupported,
    isEmbeddingModelValid,
    indexingStatus,
    liveExecution,
    isTestRuntime,
    onSubmit,
    onStop,
    onOpenSettings,
}: {
    inputRef: React.Ref<HTMLInputElement | HTMLTextAreaElement>;
    seedText: string;
    seedVersion: number;
    loading: boolean;
    llmModel: string;
    isLlmModelValid: boolean | null;
    isLlmToolsSupported: boolean | null;
    isEmbeddingModelValid: boolean | null;
    indexingStatus: ChatComposerIndexingStatusValue;
    liveExecution: LiveChatExecution | null;
    isTestRuntime: boolean;
    onSubmit: (text: string) => void;
    onStop: () => void;
    onOpenSettings: () => void;
}) {
    const theme = useTheme();
    const [draft, setDraft] = useState(seedText);

    useEffect(() => {
        setDraft(seedText);
    }, [seedText, seedVersion]);

    const composerState = useMemo(() => getChatComposerState({
        loading,
        llmModel,
        isLlmModelValid,
        isLlmToolsSupported,
        isEmbeddingModelValid,
        indexingStatus,
        hasInput: Boolean(draft),
    }), [
        draft,
        indexingStatus,
        isEmbeddingModelValid,
        isLlmModelValid,
        isLlmToolsSupported,
        llmModel,
        loading,
    ]);

    return (
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'stretch', px: 1 }}>
            <TextField
                inputRef={inputRef}
                fullWidth
                variant="outlined"
                multiline
                minRows={3}
                maxRows={10}
                placeholder={composerState.placeholder}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        onSubmit(draft);
                    }
                }}
                disabled={composerState.disabled}
                sx={{
                    '& .MuiOutlinedInput-root': {
                        bgcolor: theme.palette.background.paper,
                        color: theme.palette.text.primary,
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
            <Box
                sx={{
                    flex: '0 0 auto',
                    width: '2.5rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                }}
            >
                {loading && liveExecution ? (
                    <Tooltip title={
                        liveExecution.canceling
                            ? 'Stopping after the current LLM or tool call finishes'
                            : liveExecution.runId
                                ? 'Stop after the current step'
                                : 'Preparing the chat run'
                    }>
                        <span>
                            <IconButton
                                size="medium"
                                color="error"
                                aria-label={liveExecution.canceling ? 'Stopping chat run' : 'Stop chat run'}
                                onClick={onStop}
                                disabled={!liveExecution.runId || liveExecution.canceling || !liveExecution.running}
                            >
                                {liveExecution.canceling
                                    ? <CircularProgress size="1em" color="inherit" />
                                    : <StopCircleOutlinedIcon fontSize="medium" />}
                            </IconButton>
                        </span>
                    </Tooltip>
                ) : (
                    <IconButton
                        size="medium"
                        color="primary"
                        onClick={() => onSubmit(draft)}
                        disabled={composerState.disabled}
                    >
                        {composerState.busy ? <CircularProgress size="1em" /> : <SendIcon fontSize="medium" />}
                    </IconButton>
                )}
                <Tooltip
                    title={isTestRuntime
                        ? 'AI prompt settings for this test session'
                        : 'AI prompt settings for this thread'}
                    placement="top"
                >
                    <IconButton
                        size="medium"
                        onClick={onOpenSettings}
                        sx={{
                            color: 'text.secondary',
                            '&:hover': {
                                bgcolor: 'action.hover',
                                color: 'text.primary',
                            },
                        }}
                    >
                        <SettingsIcon fontSize="medium" />
                    </IconButton>
                </Tooltip>
            </Box>
        </Box>
    );
});

const MemoizedMarkdown = React.memo(function MemoizedMarkdown({ content }: { content: string }) {
    return (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content}
        </ReactMarkdown>
    );
});

type ChatMessageItemProps = {
    msg: ChatMessage;
    index: number;
    isRecollected: boolean;
    isActive: boolean;
    isEditing: boolean;
    isOlderQuestion: boolean;
    copied: boolean;
    liveForMessage: LiveChatExecution | null;
    showAgentRunDebug: boolean;
    forking: boolean;
    loading: boolean;
    isTestRuntime: boolean;
    onCopy: (text: string, messageId: string) => void;
    onReadAloud: (messageIdx: number) => void;
    onForkFromMessage: (messageId: string, event: React.MouseEvent) => void;
    onEditQuestion: (msg: ChatMessage, event: React.MouseEvent) => void;
    onDeleteMessage: (messageId: string, event: React.MouseEvent) => void;
    onOpenAgentRun: (msg: ChatMessage) => void;
    formatAgentWorkflowLabel: (msg: ChatMessage) => string;
};

const ChatMessageItem = React.memo(function ChatMessageItem({
    msg,
    index,
    isRecollected,
    isActive,
    isEditing,
    isOlderQuestion,
    copied,
    liveForMessage,
    showAgentRunDebug,
    forking,
    loading,
    isTestRuntime,
    onCopy,
    onReadAloud,
    onForkFromMessage,
    onEditQuestion,
    onDeleteMessage,
    onOpenAgentRun,
    formatAgentWorkflowLabel,
}: ChatMessageItemProps) {
    const theme = useTheme();
    const isUser = msg.role === MessageRole.User;
    const content = typeof msg.content === 'string' ? msg.content : String(msg.content ?? '');
    const editTooltip = isEditing
        ? "Editing this question"
        : isOlderQuestion
            ? "Edit and ask again at the end"
            : "Edit question";

    return (
        <ListItem
            alignItems="flex-start"
            sx={{
                flexDirection: 'column',
                alignItems: isUser ? 'flex-end' : 'flex-start',
                px: 0,
                py: 0.5
            }}
        >
            <Paper
                sx={{
                    p: 1.5,
                    bgcolor: isUser
                        ? theme.palette.mode === 'dark'
                            ? theme.palette.primary.dark
                            : theme.palette.primary.main
                        : theme.palette.mode === 'dark'
                            ? theme.palette.background.paper
                            : theme.palette.grey[100],
                    color: isUser
                        ? theme.palette.getContrastText(theme.palette.primary.main)
                        : theme.palette.text.primary,
                    width: showAgentRunDebug ? `calc(100% - ${theme.spacing(6)})` : 'fit-content',
                    maxWidth: isUser ? '90%' : `calc(100% - ${theme.spacing(6)})`,
                    minWidth: 0,
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
                    boxShadow: isActive
                        ? '0 0 10px rgba(255, 255, 0, 0.4)'
                        : isRecollected
                            ? '0 0 10px rgba(156, 39, 176, 0.5)'
                            : 'none',
                    border: (isRecollected || isEditing) ? '2px solid' : 'none',
                    borderColor: isEditing
                        ? 'warning.main'
                        : isRecollected
                            ? 'secondary.main'
                            : 'transparent',
                    borderRadius: '12px',
                    transition: 'all 0.2s ease',
                    cursor: 'default',
                    position: 'relative',
                    contain: 'layout paint style',
                    '&:hover .message-actions': {
                        opacity: 1
                    }
                }}
            >
                {isRecollected && (
                    <Chip
                        icon={<MemoryIcon fontSize="small" />}
                        label="Used as context"
                        size="small"
                        color="secondary"
                        sx={{
                            position: 'absolute',
                            top: -10,
                            left: 10,
                            height: 20,
                            fontSize: '0.65rem'
                        }}
                    />
                )}

                <Box
                    className="message-actions"
                    sx={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        display: 'flex',
                        gap: 0.25,
                        opacity: 0,
                        transition: 'opacity 0.2s ease',
                        bgcolor: isUser
                            ? theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.2)'
                            : theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
                        backdropFilter: 'blur(4px)',
                        borderRadius: '20px',
                        p: 0.4,
                        boxShadow: 1,
                        zIndex: 10,
                        '&:hover': { opacity: 1 }
                    }}
                >
                    <Tooltip title={copied ? "Copied!" : "Copy message"}>
                        <IconButton
                            size="small"
                            onClick={() => onCopy(content, msg.id)}
                            sx={{
                                color: 'inherit',
                                p: 0.5,
                                '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                            }}
                        >
                            {copied ? <CheckIcon fontSize="small" /> : <ContentCopyIcon fontSize="small" />}
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Read aloud">
                        <IconButton
                            size="small"
                            onClick={() => onReadAloud(index)}
                            sx={{
                                color: isUser ? 'inherit' : (isActive ? 'primary.main' : 'inherit'),
                                p: 0.5,
                                '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                            }}
                        >
                            <VolumeUpIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                    {!isTestRuntime && !isUser && (
                        <Tooltip title="Fork from here">
                            <span>
                                <IconButton
                                    size="small"
                                    onClick={(e) => onForkFromMessage(msg.id, e)}
                                    disabled={forking || loading}
                                    sx={{
                                        color: 'inherit',
                                        p: 0.5,
                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                    }}
                                >
                                    {forking ? <CircularProgress size={14} /> : <CallSplitIcon fontSize="small" />}
                                </IconButton>
                            </span>
                        </Tooltip>
                    )}
                    {!isTestRuntime && isUser && (
                        <Tooltip title={editTooltip}>
                            <span>
                                <IconButton
                                    size="small"
                                    onClick={(e) => onEditQuestion(msg, e)}
                                    disabled={loading}
                                    sx={{
                                        color: isEditing ? 'warning.light' : 'inherit',
                                        p: 0.5,
                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                    }}
                                >
                                    <EditIcon fontSize="small" />
                                </IconButton>
                            </span>
                        </Tooltip>
                    )}
                    {!isTestRuntime && (
                        <Tooltip title="Delete message">
                            <span>
                                <IconButton
                                    size="small"
                                    onClick={(e) => onDeleteMessage(msg.id, e)}
                                    disabled={loading || isEditing}
                                    sx={{
                                        color: 'inherit',
                                        p: 0.5,
                                        '&:hover': { color: 'error.main' },
                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                    }}
                                >
                                    <DeleteIcon fontSize="small" />
                                </IconButton>
                            </span>
                        </Tooltip>
                    )}
                </Box>

                {showAgentRunDebug && (
                    <Box sx={{ mb: 1 }}>
                        <Button
                            size="small"
                            variant="text"
                            startIcon={<RouteIcon fontSize="small" />}
                            onClick={() => void onOpenAgentRun(msg)}
                            sx={{ minHeight: 26, px: 0.5, textTransform: 'none' }}
                        >
                            {liveForMessage?.canceling
                                ? 'Stopping after current step…'
                                : liveForMessage?.running
                                    ? 'Open live trace'
                                    : `Open trace · ${formatAgentWorkflowLabel(msg)}${msg.agent_route ? ` · ${msg.agent_route}` : ''}`}
                        </Button>
                    </Box>
                )}
                {msg.role === MessageRole.Assistant && msg.reasoning_available && msg.reasoning && (
                    <Box sx={{ mb: 1 }}>
                        <details>
                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>Reasoning</summary>
                            <Typography
                                variant="caption"
                                component="pre"
                                sx={{ mt: 0.75, mb: 0, p: 1, borderRadius: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word', bgcolor: 'rgba(0,0,0,0.04)' }}
                            >
                                {msg.reasoning}
                            </Typography>
                        </details>
                    </Box>
                )}

                <Typography variant="body2" component="div" sx={{
                    cursor: 'text',
                    pr: 2,
                    minWidth: 0,
                    maxWidth: '100%',
                    overflowWrap: 'anywhere',
                    wordBreak: 'break-word',
                    '& p': { m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& p:last-child': { mb: 0 },
                    '& ul, & ol': { pl: 2, m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& li': { mb: 0.5, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& blockquote': { m: 0, pl: 1.5, borderLeft: '3px solid', borderColor: 'divider', overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& a': { overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& code': { bgcolor: msg.role === MessageRole.User ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace', overflowWrap: 'anywhere', wordBreak: 'break-word' },
                    '& pre': { maxWidth: '100%', bgcolor: msg.role === MessageRole.User ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 },
                    '& pre code': { overflowWrap: 'normal', wordBreak: 'normal' },
                    '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto', borderCollapse: 'collapse', mb: 1 },
                    '& th, & td': { border: '1px solid', borderColor: 'divider', px: 0.75, py: 0.5 }
                }}>
                    <MemoizedMarkdown content={content} />
                </Typography>
                {msg.role === MessageRole.User && msg.rewritten_query && (
                    <Box sx={{ mt: 1 }}>
                        <details>
                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                <Tooltip
                                    title="This question was rewritten with thread context for retrieval."
                                    arrow
                                >
                                    <span>Rewritten for context</span>
                                </Tooltip>
                            </summary>
                            <Typography
                                variant="caption"
                                sx={{
                                    mt: 1,
                                    display: 'block',
                                    fontStyle: 'italic',
                                    opacity: 0.9,
                                    p: 1,
                                    borderRadius: 1,
                                    bgcolor: 'rgba(255,255,255,0.1)',
                                    minWidth: 0,
                                    maxWidth: '100%',
                                    overflowWrap: 'anywhere',
                                    wordBreak: 'break-word'
                                }}
                            >
                                {msg.rewritten_query}
                            </Typography>
                        </details>
                    </Box>
                )}
                {msg.role === MessageRole.Assistant && msg.web_sources && msg.web_sources.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                        <details>
                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                🌐 Web sources used ({msg.web_sources.length})
                            </summary>
                            <Box sx={{ mt: 0.75, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
                                {msg.web_sources.map((source, i) => (
                                    <Box
                                        key={i}
                                        sx={{
                                            p: 1,
                                            borderRadius: 1,
                                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)',
                                            borderLeft: '3px solid',
                                            borderColor: 'primary.light',
                                        }}
                                    >
                                        {source.url ? (
                                            <Typography
                                                variant="caption"
                                                component="a"
                                                href={source.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                sx={{
                                                    color: 'primary.main',
                                                    display: 'block',
                                                    fontWeight: 600,
                                                    textDecoration: 'none',
                                                    mb: 0.25,
                                                    '&:hover': { textDecoration: 'underline' },
                                                }}
                                            >
                                                {source.title || source.url}
                                            </Typography>
                                        ) : (
                                            <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.25 }}>
                                                {source.title || 'Web result'}
                                            </Typography>
                                        )}
                                        {source.url && (
                                            <Typography
                                                variant="caption"
                                                sx={{ color: 'text.secondary', display: 'block', wordBreak: 'break-all', mb: 0.25 }}
                                            >
                                                {source.url}
                                            </Typography>
                                        )}
                                        <Typography
                                            variant="caption"
                                            sx={{ display: 'block', opacity: 0.85, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                                        >
                                            {source.text}
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        </details>
                    </Box>
                )}
            </Paper>
        </ListItem>
    );
});

const normalizeAgentWorkflowForUi = (workflowId?: string | null) => (
    workflowId ? String(workflowId) : ''
);

const isBuiltinAgentWorkflow = (workflows: AgentWorkflow[], workflowId?: string | null) => (
    Boolean(workflows.find((workflow) => workflow.id === workflowId)?.is_builtin)
);

const workflowSupportsReplans = (workflows: AgentWorkflow[], workflowId?: string | null) => (
    Boolean(workflows.find((workflow) => workflow.id === workflowId)?.supports_replans)
);

export interface ConversationRuntime {
    kind: 'normal-thread' | 'builder-test';
    persistent: boolean;
    historyReadOnly: boolean;
}

export interface NormalThreadConversationRuntime extends ConversationRuntime {
    kind: 'normal-thread';
    persistent: true;
    historyReadOnly: false;
}

export interface BuilderTestConversationRuntime extends ConversationRuntime {
    kind: 'builder-test';
    persistent: false;
    historyReadOnly: true;
    spec: Record<string, any>;
    baseWorkflowId: string;
    session: BuilderTestSession;
    onSessionChange: React.Dispatch<React.SetStateAction<BuilderTestSession>>;
}

export interface ChatInterfaceProps {
    ragApiUrl?: string;
    activeThread: Thread | null;
    chatSentences: any[];
    setChatSentences: (sentences: any[]) => void;
    currentChatId: number | null;
    activeSource: 'pdf' | 'chat';
    onJump: (id: number) => void;
    onResetChatId?: () => void;
    onThreadUpdate?: () => void;
    onThreadForked?: (thread: Thread) => void;
    onOpenThread?: (thread: Thread) => void;
    hideInlineLineage?: boolean;
    darkMode?: boolean;
    autoScroll?: boolean;
    isPanelResizing?: boolean;
    onOpenTrace?: (trace: ChatTraceDescriptor) => void;
    testRuntime?: BuilderTestConversationRuntime;
}

export type ChatTraceDescriptor = {
    id: string;
    messageId: string;
    label: string;
    status?: string;
    routeReason?: string;
    traceRefs?: AgentTraceRefs | null;
    runDetails?: AgentRunDetails;
    liveTraceView?: TraceRunView;
    loading?: boolean;
    error?: string;
    running?: boolean;
};

const PersistentChatInterface: React.FC<ChatInterfaceProps> = ({
    activeThread,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump,
    onThreadUpdate,
    onThreadForked,
    onOpenThread,
    hideInlineLineage = false,
    onResetChatId,
    darkMode = false,
    autoScroll = true,
    isPanelResizing = false,
    onOpenTrace,
    testRuntime,
}) => {
    const ragApiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!ragApiUrl) {
        console.error("ERROR: NEXT_PUBLIC_API_URL environment variable is not set. Please configure it in docker-compose.yml");
    }
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const theme = useTheme();
    const [composerSeed, setComposerSeed] = useState('');
    const [composerSeedVersion, setComposerSeedVersion] = useState(0);
    const [loading, setLoading] = useState(false);
    const normalRuntime = useNormalThreadChatRuntime();
    const builderRuntime = useBuilderTestChatRuntime(testRuntime, setMessages);
    const conversationRuntime: RuntimeState = builderRuntime || normalRuntime;
    const isTestRuntime = conversationRuntime.kind === 'builder-test';

    const [indexingStatus, setIndexingStatus] = useState<ChatComposerIndexingStatusValue>(ChatComposerIndexingStatus.Checking);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [contextWindow, setContextWindow] = useState<number>(0);
    const [replans, setReplans] = useState(1);
    const [replansLimit, setReplansLimit] = useState<number | null>(null);
    const [defaultSystemRole, setDefaultSystemRole] = useState('');
    const [defaultCustomInstructions, setDefaultCustomInstructions] = useState('');
    const [systemRole, setSystemRole] = useState('');
    const [toolCatalog, setToolCatalog] = useState<PromptToolDefinition[]>([]);
    const [toolInstructions, setToolInstructions] = useState<Record<string, string>>({});
    const [customInstructions, setCustomInstructions] = useState('');
    const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
    const [savingSettings, setSavingSettings] = useState(false);
    const [promptPreview, setPromptPreview] = useState('');
    const [showContextHighlight, setShowContextHighlight] = useState(false);
    const [tooltipOpen, setTooltipOpen] = useState(false);
    const [recollectedIds, setRecollectedIds] = useState<Set<string>>(new Set());
    const [clarificationOptions, setClarificationOptions] = useState<ClarificationChoice[] | null>(null);
    const [clarificationPanelRatio, setClarificationPanelRatio] = useState(0.3);
    const [isClarificationResizing, setIsClarificationResizing] = useState(false);
    const [hitlWebApproval, setHitlWebApproval] = useState(false);
    const [savingWebSearchMode, setSavingWebSearchMode] = useState(false);
    const [defaultHitlWebApproval, setDefaultHitlWebApproval] = useState(false);
    const [useReranker, setUseReranker] = useState(false);
    const [defaultUseReranker, setDefaultUseReranker] = useState(false);
    const [useProjectMemory, setUseProjectMemory] = useState(true);
    const [useGlobalMemory, setUseGlobalMemory] = useState(false);
    const [projectAllowsGlobalMemory, setProjectAllowsGlobalMemory] = useState(false);
    const [agentWorkflowId, setAgentWorkflowId] = useState('');
    const [agentWorkflows, setAgentWorkflows] = useState<AgentWorkflow[]>([]);

    // Model selection
    const [llmModel, setLlmModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    const [isLlmModelValid, setIsLlmModelValid] = useState<boolean | null>(true);
    const [isLlmToolsSupported, setIsLlmToolsSupported] = useState<boolean | null>(null);
    const [isEmbeddingModelValid, setIsEmbeddingModelValid] = useState<boolean | null>(null);
    const [copiedId, setCopiedId] = useState<string | null>(null);
    const [forkingMessageId, setForkingMessageId] = useState<string | null>(null);
    const [forkDialogMessageId, setForkDialogMessageId] = useState<string | null>(null);
    const [forkProjects, setForkProjects] = useState<Project[]>([]);
    const [lineageThreads, setLineageThreads] = useState<Thread[]>([]);
    const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
    const [editingOriginalText, setEditingOriginalText] = useState('');
    const [agentRunDetails, setAgentRunDetails] = useState<Record<string, AgentRunDetails>>({});
    const [agentRunLoading, setAgentRunLoading] = useState<Record<string, boolean>>({});
    const [agentRunErrors, setAgentRunErrors] = useState<Record<string, string>>({});
    const [openAgentRunIds, setOpenAgentRunIds] = useState<Set<string>>(new Set());
    const [workspaceTraceMessageId, setWorkspaceTraceMessageId] = useState<string | null>(null);
    const workspaceTraceMessageIdRef = useRef<string | null>(null);
    const [liveExecution, setLiveExecution] = useState<LiveChatExecution | null>(null);
    const { events: liveExecutionEvents, append: appendLiveExecutionEvent, reset: resetLiveExecutionEvents } = useBatchedExecutionEvents();
    const liveTraceView = useMemo(() => buildLiveTraceView(liveExecutionEvents), [liveExecutionEvents]);
    const [pendingHumanReview, setPendingHumanReview] = useState<PendingHumanReview | null>(null);
    const [humanReviewSubmitting, setHumanReviewSubmitting] = useState<AgentRunResumeAction | null>(null);
    const [humanReviewError, setHumanReviewError] = useState<string | null>(null);
    const [humanReviewEditText, setHumanReviewEditText] = useState('');
    const [pendingMemoryCandidates, setPendingMemoryCandidates] = useState<MemoryCandidate[]>([]);
    const [memoryCandidateActionId, setMemoryCandidateActionId] = useState<string | null>(null);
    const [memoryCandidateError, setMemoryCandidateError] = useState<string | null>(null);

    const messageListRef = useRef<HTMLDivElement | null>(null);
    const messageRefs = useRef<{ [key: number]: HTMLLIElement | null }>({});
    const sentenceCacheRef = useRef<ChatSentenceCache>(new Map());
    const chatRootRef = useRef<HTMLDivElement | null>(null);
    const composerInputRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null);
    const humanReviewSubmissionKeyRef = useRef<string | null>(null);
    const manuallyToggledAgentRunsRef = useRef(new Set<string>());
    const activeThreadIdRef = useRef<string | null>(activeThread?.id ?? null);
    activeThreadIdRef.current = activeThread?.id ?? null;
    const clarificationResizeRef = useRef({
        startY: 0,
        startRatio: 0.3,
    });
    const messageVirtualizer = useVirtualizer({
        count: messages.length,
        getScrollElement: () => messageListRef.current,
        estimateSize: (index) => messages[index]?.role === MessageRole.User ? 76 : 180,
        overscan: 8,
        getItemKey: (index) => messages[index]?.id ?? index,
    });

    const setComposerText = useCallback((text: string) => {
        setComposerSeed(text);
        setComposerSeedVersion((version) => version + 1);
    }, []);

    const applyThreadSettingsToState = useCallback((settings?: Thread['settings']) => {
        setReplans(settings?.replans ?? 1);
        setSystemRole(settings?.system_role ?? defaultSystemRole);
        setToolInstructions(settings?.tool_instructions ?? {});
        setCustomInstructions(settings?.custom_instructions ?? defaultCustomInstructions);
        setHitlWebApproval(settings?.hitl_web_approval ?? defaultHitlWebApproval);
        setUseReranker(settings?.use_reranker ?? defaultUseReranker);
        setUseProjectMemory(settings?.memory?.thread_reads_project_memory ?? true);
        setUseGlobalMemory(settings?.memory?.thread_reads_user_memory ?? false);
        setAgentWorkflowId(normalizeAgentWorkflowForUi(settings?.agent_workflow?.workflow_id));
    }, [
        defaultCustomInstructions,
        defaultSystemRole,
        defaultHitlWebApproval,
        defaultUseReranker,
    ]);

    const loadProjectMemorySettings = useCallback(async () => {
        const projectId = activeThread?.project_id;
        if (!projectId || isTestRuntime) {
            setProjectAllowsGlobalMemory(false);
            return;
        }
        try {
            const project = await getProject(projectId);
            if (activeThreadIdRef.current !== activeThread?.id) return;
            setProjectAllowsGlobalMemory(
                project.settings_json?.memory?.project_reads_user_memory === true
            );
        } catch (error) {
            console.error('Failed to load project memory settings:', error);
            setProjectAllowsGlobalMemory(false);
        }
    }, [activeThread?.id, activeThread?.project_id, isTestRuntime]);

    const clampClarificationRatio = (ratio: number) => Math.max(0.16, Math.min(0.58, ratio));

    const handleClarificationResizeStart = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        event.preventDefault();
        event.stopPropagation();
        clarificationResizeRef.current = {
            startY: event.clientY,
            startRatio: clarificationPanelRatio,
        };
        setIsClarificationResizing(true);
        event.currentTarget.setPointerCapture(event.pointerId);
    }, [clarificationPanelRatio]);

    const handleClarificationResizeMove = useCallback((event: PointerEvent) => {
        const chatHeight = chatRootRef.current?.getBoundingClientRect().height || window.innerHeight;
        const deltaRatio = (clarificationResizeRef.current.startY - event.clientY) / chatHeight;
        setClarificationPanelRatio(clampClarificationRatio(
            clarificationResizeRef.current.startRatio + deltaRatio
        ));
    }, []);

    const handleClarificationResizeEnd = useCallback(() => {
        setIsClarificationResizing(false);
    }, []);

    const loadPendingMemoryCandidates = useCallback(async () => {
        if (!activeThread || isTestRuntime) {
            setPendingMemoryCandidates([]);
            return;
        }
        try {
            const response = await listMemoryCandidates({
                status: 'pending',
                sourceProjectId: activeThread.project_id,
                limit: 20,
            });
            const rows = response.memory_candidates || [];
            setPendingMemoryCandidates(rows.filter(candidate => (
                candidate.source_thread_id === activeThread.id
                || candidate.proposed_scope_id === activeThread.id
                || Boolean(activeThread.project_id && candidate.source_project_id === activeThread.project_id)
            )));
            setMemoryCandidateError(null);
        } catch (error: any) {
            setMemoryCandidateError(error?.message || 'Unable to load memory candidates.');
        }
    }, [activeThread?.id, activeThread?.project_id, isTestRuntime]);

    const handleResolveMemoryCandidate = useCallback(async (candidate: MemoryCandidate, status: 'approved' | 'rejected') => {
        if (!activeThread) return;
        try {
            setMemoryCandidateActionId(candidate.id);
            await resolveMemoryCandidate(candidate.id, status, {
                actorId: 'ui',
            });
            setPendingMemoryCandidates(prev => prev.filter(item => item.id !== candidate.id));
            setMemoryCandidateError(null);
        } catch (error: any) {
            setMemoryCandidateError(error?.message || `Unable to ${status === 'approved' ? 'approve' : 'reject'} memory candidate.`);
        } finally {
            setMemoryCandidateActionId(null);
        }
    }, [activeThread?.id]);

    useEffect(() => {
        if (!isClarificationResizing) return;

        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        document.addEventListener('pointermove', handleClarificationResizeMove);
        document.addEventListener('pointerup', handleClarificationResizeEnd);
        document.addEventListener('pointercancel', handleClarificationResizeEnd);

        return () => {
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('pointermove', handleClarificationResizeMove);
            document.removeEventListener('pointerup', handleClarificationResizeEnd);
            document.removeEventListener('pointercancel', handleClarificationResizeEnd);
        };
    }, [handleClarificationResizeEnd, handleClarificationResizeMove, isClarificationResizing]);

    // Load messages when thread changes
    useEffect(() => {
        if (activeThread) {
            applyThreadSettingsToState(activeThread.settings);
            loadMessages();
            if (!isTestRuntime) {
                loadPendingMemoryCandidates();
                recoverPendingHumanReview(activeThread.id);
            }
            checkIndexStatus();
            loadThreadSettings();
            loadProjectMemorySettings();
            checkEmbeddingModelStatus();
        } else {
            setMessages([]);
            setClarificationOptions(null);
            setIndexingStatus(ChatComposerIndexingStatus.Ready);
            applyThreadSettingsToState(undefined);
            setIsEmbeddingModelValid(null);
            setIsLlmToolsSupported(null);
            setPendingMemoryCandidates([]);
            setMemoryCandidateError(null);
            setProjectAllowsGlobalMemory(false);
        }
    }, [activeThread?.id, activeThread?.file_count, activeThread?.settings, applyThreadSettingsToState, isTestRuntime, loadPendingMemoryCandidates, loadProjectMemorySettings]);

    useEffect(() => {
        if (activeThread) {
            setClarificationOptions(null);
            setEditingMessageId(null);
            setEditingOriginalText('');
            setAgentRunDetails({});
            setAgentRunLoading({});
            setAgentRunErrors({});
            setOpenAgentRunIds(new Set());
            manuallyToggledAgentRunsRef.current.clear();
            setLiveExecution(null);
            resetLiveExecutionEvents();
            setPendingHumanReview(null);
            setHumanReviewSubmitting(null);
            setHumanReviewError(null);
            setHumanReviewEditText('');
            setPendingMemoryCandidates([]);
            setMemoryCandidateError(null);
            setMemoryCandidateActionId(null);
        }
    }, [activeThread?.id, resetLiveExecutionEvents]);

    useEffect(() => {
        const parentThreadId = activeThread?.thread_metadata?.fork?.parent_thread_id;
        const childThreadIds = Array.isArray(activeThread?.thread_metadata?.fork_children)
            ? activeThread.thread_metadata.fork_children.filter((id): id is string => typeof id === 'string' && id.length > 0)
            : [];
        const hasLineage = !hideInlineLineage && Boolean(parentThreadId || childThreadIds.length > 0);

        if (!hasLineage) {
            setLineageThreads([]);
            return;
        }

        let cancelled = false;
        listThreads()
            .then(response => {
                if (!cancelled) {
                    setLineageThreads(response.threads);
                }
            })
            .catch(() => {
                if (!cancelled) {
                    setLineageThreads([]);
                }
            });

        return () => {
            cancelled = true;
        };
    }, [activeThread?.id, activeThread?.thread_metadata?.fork?.parent_thread_id, activeThread?.thread_metadata?.fork_children, hideInlineLineage]);

    useEffect(() => {
        const loadTools = async () => {
            try {
                const [res, patterns] = await Promise.all([
                    getPromptTools(),
                    listAgentWorkflows().catch((error) => {
                        console.error('Failed to load agent workflows:', error);
                        return { agent_workflows: [] };
                    }),
                ]);
                setToolCatalog(res.tools || []);
                setAgentWorkflows(patterns.agent_workflows || []);
                if (res.defaults) {
                    setReplansLimit(res.defaults.replans_limit);
                    setDefaultSystemRole(res.defaults.system_role ?? '');
                    setDefaultCustomInstructions(res.defaults.custom_instructions ?? '');
                    setDefaultHitlWebApproval(res.defaults.hitl_web_approval ?? false);
                    setDefaultUseReranker(res.defaults.use_reranker ?? false);
                    if (res.defaults.context_window && !localStorage.getItem('last_context_window')) {
                        setContextWindow(res.defaults.context_window);
                    }
                    if (!activeThread) {
                        setReplans(Math.min(1, res.defaults.replans_limit));
                        setSystemRole(res.defaults.system_role ?? '');
                        setCustomInstructions(res.defaults.custom_instructions ?? '');
                        setHitlWebApproval(res.defaults.hitl_web_approval ?? false);
                        setUseReranker(res.defaults.use_reranker ?? false);
                        setAgentWorkflowId(normalizeAgentWorkflowForUi(res.defaults.agent_workflow?.workflow_id));
                    }
                }
            } catch (error) {
                console.error('Failed to load prompt tools:', error);
                setToolCatalog([]);
                setAgentWorkflows([]);
            }
        };
        loadTools();
    }, [activeThread]);

    const loadThreadSettings = async () => {
        const threadId = activeThread?.id;
        if (!threadId) return;
        try {
            const settings = await getThreadSettings(threadId);
            if (activeThreadIdRef.current !== threadId) return;
            applyThreadSettingsToState(settings);
        } catch (error) {
            if (activeThreadIdRef.current !== threadId) return;
            console.error('Failed to load thread settings:', error);
            applyThreadSettingsToState(undefined);
        }
    };

    const loadMessages = async () => {
        if (!activeThread) return;
        try {
            const response = await getThreadMessages(activeThread.id);
            const persisted = response.messages.map(m => ({
                ...m,
                content: typeof m.content === 'string' ? m.content : String(m.content ?? ''),
                isRecollected: false,
                rewritten_query: m.role === MessageRole.User ? m.context_compact : undefined,
                web_sources: m.role === MessageRole.Assistant ? (m.web_sources || []) : undefined,
                agent_run_id: m.agent_run_id,
                agent_run_turn_kind: m.agent_run_turn_kind,
                agent_run_sequence: m.agent_run_sequence,
                agent_trace_refs: m.agent_trace_refs,
                agent_workflow_id: m.agent_workflow_id ?? m.metadata?.agent_workflow_id,
                agent_route: m.agent_route ?? m.metadata?.agent_route,
                agent_route_reason: m.agent_route_reason ?? m.metadata?.agent_route_reason,
            }));
            const temporary = testRuntime?.session.messages.map((message) => (
                builderTestMessageToChatMessage(message, testRuntime.baseWorkflowId)
            )) || [];
            setMessages([...persisted, ...temporary]);
        } catch (error) {
            console.error('Failed to load messages:', error);
        }
    };

    const recoverPendingHumanReview = async (threadId: string) => {
        try {
            const response = await listThreadAgentRuns(threadId, { status: 'awaiting_human', limit: 1 });
            if (activeThreadIdRef.current !== threadId) return;
            const latest = response.agent_runs?.[0];
            if (!latest?.id || !latest.pending_interrupt) {
                setPendingHumanReview(null);
                setHumanReviewEditText('');
                return;
            }

            const run = await getAgentRun(latest.id, threadId);
            if (activeThreadIdRef.current !== threadId) return;
            const interrupt = run.pending_interrupt || latest.pending_interrupt;
            if (!interrupt || interrupt.status && interrupt.status !== InterruptStatus.Pending) {
                setPendingHumanReview(null);
                setHumanReviewEditText('');
                return;
            }

            const localUserMessageId = `recovered-review-user-${run.id}`;
            const localAssistantMessageId = `recovered-review-asst-${run.id}`;
            setPendingHumanReview({
                runId: run.id,
                interrupt,
                localUserMessageId,
                localAssistantMessageId,
            });
            setHumanReviewError(null);
            setHumanReviewEditText(interrupt.proposed_final_answer || '');
            setAgentRunDetails(prev => ({ ...prev, [run.id]: run }));

            const inputSummary = interrupt.input_summary;
            const recoveredQuestion = typeof inputSummary === 'object' && inputSummary !== null
                ? String(inputSummary.question || '')
                : '';
            setMessages(prev => {
                if (prev.some(msg => msg.agent_run_id === run.id || msg.id === localAssistantMessageId)) {
                    return prev;
                }
                const recovered: ChatMessage[] = [];
                if (recoveredQuestion.trim()) {
                    recovered.push({
                        id: localUserMessageId,
                        role: MessageRole.User,
                        content: recoveredQuestion.trim(),
                        created_at: run.started_at || new Date().toISOString(),
                        agent_run_id: run.id,
                        agent_run_turn_kind: 'user_prompt',
                    });
                }
                recovered.push({
                    id: localAssistantMessageId,
                    role: MessageRole.Assistant,
                    content: interrupt.title || 'Human review required before the agent can continue.',
                    created_at: interrupt.requested_at || run.started_at || new Date().toISOString(),
                    agent_run_id: run.id,
                    agent_run_turn_kind: 'assistant_pending_review',
                    pending_human_review: true,
                });
                return [...prev, ...recovered];
            });
        } catch (error) {
            if (activeThreadIdRef.current !== threadId) return;
            console.error('Failed to recover pending human review:', error);
        }
    };

    const checkIndexStatus = async () => {
        if (!activeThread) return;
        try {
            setIndexingStatus(ChatComposerIndexingStatus.Checking);
            const status = await getThreadIndexStatus(activeThread.id);
            // Map rag-service status ('ready' | 'blocked') to UI status
            if (status.status === EmbeddingReadinessStatus.Ready) {
                setIndexingStatus(ChatComposerIndexingStatus.Ready);
            } else if (status.status === EmbeddingReadinessStatus.Blocked) {
                setIndexingStatus(ChatComposerIndexingStatus.Blocked);
            } else {
                // 'not_ready' means still indexing
                setIndexingStatus(ChatComposerIndexingStatus.Indexing);
            }
            // Update embedding model status from the same endpoint
            if (status.embeddingModelReady !== undefined) {
                setIsEmbeddingModelValid(status.embeddingModelReady);
            }
        } catch (error) {
            console.error('Failed to check index status:', error);
            // Set to error state instead of falsely claiming ready
            setIndexingStatus(ChatComposerIndexingStatus.Error);
            // Don't change embedding model status - keep previous state
        }
    };

    const checkEmbeddingModelStatus = async () => {
        if (!activeThread) return;
        
        setIsEmbeddingModelValid(null);
        
        const result = await withRetry(
            () => checkEmbeddingModelReady(activeThread.embeddingModel),
            {
                maxRetries: 2,
                baseDelay: 1000,
                retryableErrors: (error) => true // All errors are retryable for model checks
            }
        );
        
        if (result.success && result.data !== undefined) {
            setIsEmbeddingModelValid(result.data);
            if (!result.data) {
                setIndexingStatus(ChatComposerIndexingStatus.Blocked);
            }
        } else {
            console.error('Failed to check embedding model status after retries:', result.error);
            setIsEmbeddingModelValid(false);
            setIndexingStatus(ChatComposerIndexingStatus.Error);
        }
    };

    // Sync chatSentences with parent whenever messages change. Cache by message id/content
    // so streaming or appending a single message does not re-tokenize the whole thread.
    useEffect(() => {
        setChatSentences(deriveChatSentences(messages, sentenceCacheRef.current));
    }, [messages, setChatSentences]);

    const activeMessageIndex = useMemo(() => {
        if (activeSource !== 'chat' || currentChatId === null) return null;
        return chatSentences[currentChatId]?.messageIndex;
    }, [currentChatId, chatSentences, activeSource]);

    const effectiveToolInstructions = useMemo(() => {
        const merged: Record<string, string> = {};
        toolCatalog.forEach((toolDef) => {
            merged[toolDef.id] = toolInstructions[toolDef.id] || toolDef.default_prompt;
        });
        return merged;
    }, [toolCatalog, toolInstructions]);

    useEffect(() => {
        if (!settingsDialogOpen) return;
        let cancelled = false;
        const timeoutId = setTimeout(async () => {
            try {
                const res = await getPromptPreview({
                    context_window: contextWindow,
                    system_role: systemRole,
                    tool_instructions: effectiveToolInstructions,
                    custom_instructions: customInstructions,
                    use_web_search: useWebSearch,
                    agent_workflow_id: normalizeAgentWorkflowForUi(agentWorkflowId),
                });
                if (!cancelled) {
                    setPromptPreview(res.prompt || '');
                }
            } catch (error) {
                if (!cancelled) {
                    setPromptPreview('Unable to load prompt preview.');
                }
            }
        }, 200);
        return () => {
            cancelled = true;
            clearTimeout(timeoutId);
        };
    }, [settingsDialogOpen, contextWindow, systemRole, effectiveToolInstructions, customInstructions, useWebSearch, agentWorkflowId]);

    const resetAllSettingsToDefault = () => {
        const defaults: Record<string, string> = {};
        toolCatalog.forEach((toolDef) => {
            defaults[toolDef.id] = toolDef.default_prompt;
        });
        setReplans(Math.min(1, replansLimit ?? 1));
        setSystemRole(defaultSystemRole);
        setToolInstructions(defaults);
        setCustomInstructions(defaultCustomInstructions);
        setUseReranker(defaultUseReranker);
        setUseProjectMemory(true);
        setUseGlobalMemory(false);
        setAgentWorkflowId(agentWorkflows[0]?.id || '');
    };

    const resetToolInstructionToDefault = (toolId: string) => {
        const toolDef = toolCatalog.find((t) => t.id === toolId);
        if (!toolDef) return;
        setToolInstructions((prev) => ({
            ...prev,
            [toolId]: toolDef.default_prompt,
        }));
    };

    const resetSystemRoleToDefault = () => {
        setSystemRole(defaultSystemRole);
    };

    const resetCustomInstructionsToDefault = () => {
        setCustomInstructions('');
    };

    useEffect(() => {
        if (autoScroll && activeMessageIndex !== null) {
            messageVirtualizer.scrollToIndex(activeMessageIndex, {
                align: 'center',
                behavior: 'smooth',
            });
        }
    }, [activeMessageIndex, currentChatId, autoScroll, messageVirtualizer]);

    const scrollToBottom = () => {
        if (autoScroll && messages.length > 0) {
            messageVirtualizer.scrollToIndex(messages.length - 1, {
                align: 'end',
                behavior: 'smooth',
            });
        }
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages.length, autoScroll]);

    // Fetch available LLM models
    useEffect(() => {
        fetchAvailableLlmModels()
            .then(setAvailableModels)
            .catch(err => {
                console.error("Failed to fetch models", err);
                setAvailableModels([]);
            });
    }, []);

    const validateLlmModel = useCallback(async (model: string) => {
        if (!model) return;
        try {
            const result = await checkLlmModelReady(model);
            setIsLlmModelValid(result.ready);
            setIsLlmToolsSupported(result.ready ? result.supportsTools : null);
        } catch (err) {
            setIsLlmModelValid(false);
            setIsLlmToolsSupported(null);
        }
    }, []);

    // Validate LLM model when changed using chat-utils
    const handleLlmModelChange = async (model: string) => {
        setLlmModel(model);
        setIsLlmModelValid(null);
        setIsLlmToolsSupported(null);
        if (model) {
            setShowContextHighlight(true);
            setTooltipOpen(true);
            // Persist as last selected LLM in browser memory
            if (typeof window !== 'undefined') {
                localStorage.setItem('last_llm_model', model);
            }
        }
        if (!model) return;
        await validateLlmModel(model);
    };

    useEffect(() => {
        if (llmModel && isLlmModelValid === null) {
            validateLlmModel(llmModel);
        }
    }, [isLlmModelValid, llmModel, validateLlmModel]);

    const handleContextWindowChange = (val: number) => {
        setContextWindow(val);
        if (val > 0 && typeof window !== 'undefined') {
            localStorage.setItem('last_context_window', val.toString());
        }
    };

    const webSearchMode: WebSearchMode = !useWebSearch
        ? 'off'
        : hitlWebApproval
            ? 'ask'
            : 'on';

    const handleWebSearchModeChange = async () => {
        if (savingWebSearchMode) return;
        const nextMode: WebSearchMode = webSearchMode === 'off'
            ? 'ask'
            : webSearchMode === 'ask'
                ? 'on'
                : 'off';
        const previousUseWebSearch = useWebSearch;
        const previousHitlWebApproval = hitlWebApproval;
        const nextUseWebSearch = nextMode !== 'off';
        const nextHitlWebApproval = nextMode === 'ask';

        setUseWebSearch(nextUseWebSearch);
        setHitlWebApproval(nextHitlWebApproval);

        if (isTestRuntime || !activeThread) return;

        if (typeof window !== 'undefined') {
            localStorage.setItem('last_use_web_search', nextUseWebSearch ? '1' : '0');
        }
        setSavingWebSearchMode(true);
        try {
            await updateThreadSettings(activeThread.id, {
                hitl_web_approval: nextHitlWebApproval,
            });
        } catch (error) {
            console.error('Failed to update internet search mode:', error);
            setUseWebSearch(previousUseWebSearch);
            setHitlWebApproval(previousHitlWebApproval);
            if (typeof window !== 'undefined') {
                localStorage.setItem('last_use_web_search', previousUseWebSearch ? '1' : '0');
            }
        } finally {
            setSavingWebSearchMode(false);
        }
    };

    const webSearchModeLabel = webSearchMode === 'on'
        ? 'Internet Search On'
        : webSearchMode === 'ask'
            ? 'Ask me every time before internet search'
            : 'Internet Search Off';
    const nextWebSearchModeLabel = webSearchMode === 'off'
        ? 'Ask me every time'
        : webSearchMode === 'ask'
            ? 'Internet Search On'
            : 'Internet Search Off';

    // Polling for indexing and embedding model status
    useEffect(() => {
        if (!activeThread) return;
        if (indexingStatus === ChatComposerIndexingStatus.Blocked || isEmbeddingModelValid === false) return;
        // Keep polling if either indexing is in progress OR embedding model is not yet valid/checked
        if (indexingStatus !== ChatComposerIndexingStatus.Indexing && isEmbeddingModelValid === true) return;

        let intervalId: NodeJS.Timeout | null = null;

        const pollStatus = async () => {
            const result = await withPollingRetry(
                () => getThreadIndexStatus(activeThread.id),
                {
                    maxRetries: 3,
                    interval: 5000,
                    shouldStop: (status) => (
                        (status.status === EmbeddingReadinessStatus.Ready && status.embeddingModelReady === true) ||
                        status.status === EmbeddingReadinessStatus.Blocked ||
                        status.embeddingModelReady === false
                    ),
                    retryableErrors: (error) => isRetryableError(error) // Use smart error classification
                }
            );

            if (result.success && result.data) {
                // Update indexing status
                if (result.data.status === EmbeddingReadinessStatus.Ready) {
                    setIndexingStatus(ChatComposerIndexingStatus.Ready);
                } else if (result.data.status === EmbeddingReadinessStatus.Blocked) {
                    setIndexingStatus(ChatComposerIndexingStatus.Blocked);
                } else {
                    // 'not_ready' means still indexing
                    setIndexingStatus(ChatComposerIndexingStatus.Indexing);
                }

                // Update embedding model status
                if (result.data.embeddingModelReady !== undefined) {
                    setIsEmbeddingModelValid(result.data.embeddingModelReady);
                }
            }

            // Handle polling termination
            if (result.stopped || !result.success) {
                // Clear polling interval
                if (intervalId) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
                
                if (!result.success) {
                    // Handle different error types
                    if (result.resourceNotFound) {
                        // Thread was deleted - reset to initial state
                        setIndexingStatus(ChatComposerIndexingStatus.Checking);
                        setIsEmbeddingModelValid(false);
                    } else {
                        // Other error - set error state
                        setIndexingStatus(ChatComposerIndexingStatus.Error);
                    }
                }
            }
        };

        // Start polling
        pollStatus();

        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, [activeThread?.id, indexingStatus, isEmbeddingModelValid]);

    // Load browser memory settings (last selected LLM, context window, and web search) on mount
    useEffect(() => {
        if (typeof window === 'undefined') return;

        const savedLlm = localStorage.getItem('last_llm_model');
        if (savedLlm && !llmModel) {
            setLlmModel(savedLlm);
            setIsLlmModelValid(null);
            setIsLlmToolsSupported(null);
        }

        const savedCtx = localStorage.getItem('last_context_window');
        if (savedCtx) {
            const ctx = parseInt(savedCtx);
            if (!isNaN(ctx) && ctx > 0) {
                setContextWindow(ctx);
            }
        }

        const savedWebSearch = localStorage.getItem('last_use_web_search');
        if (savedWebSearch === '1' || savedWebSearch === '0') {
            setUseWebSearch(savedWebSearch === '1');
        }
    }, [llmModel]);

    const cancelQuestionEdit = useCallback(() => {
        setEditingMessageId(null);
        setEditingOriginalText('');
        setComposerText('');
    }, [setComposerText]);

    const handleEditQuestion = useCallback((msg: ChatMessage, event: React.MouseEvent) => {
        event.stopPropagation();
        if (loading || msg.role !== MessageRole.User) return;

        const content = typeof msg.content === 'string' ? msg.content : String(msg.content ?? '');
        setComposerText(content);
        setEditingMessageId(msg.id);
        setEditingOriginalText(content);
        setClarificationOptions(null);
    }, [loading, setComposerText]);

    const updateBuilderTestMessage = builderRuntime?.updateMessage || (() => undefined);

    const builderTestRequestContext = () => ({
        thread_id: activeThread!.id,
        llm_model: llmModel,
        use_web_search: useWebSearch,
        use_reranker: useReranker,
        context_window: contextWindow,
        replans: workflowSupportsReplans(agentWorkflows, testRuntime?.baseWorkflowId || '')
            ? replans
            : undefined,
        system_role_override: systemRole,
        tool_instructions_override: effectiveToolInstructions,
        custom_instructions_override: customInstructions,
        hitl_web_approval: hitlWebApproval,
        // In Builder Test, deliberately turning on the shared internet-search
        // control is the explicit confirmation required by the isolated runtime.
        allow_external_tools: useWebSearch,
        client_timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        client_locale: navigator.language,
        client_now_iso: new Date().toISOString(),
    });

    const handleBuilderTestSend = async (question: string) => {
        if (!testRuntime || !builderRuntime || !activeThread || loading) return;
        const now = new Date().toISOString();
        const stamp = Date.now();
        const userId = `test-user-${stamp}`;
        const assistantId = `test-assistant-${stamp}`;
        const fingerprint = workflowSpecFingerprint(testRuntime.spec);
        const priorMessages = testRuntime.session.messages;
        testRuntime.onSessionChange({
            threadId: activeThread.id,
            messages: [
                ...priorMessages,
                { id: userId, role: 'user', content: question, createdAt: now, status: 'completed', specFingerprint: fingerprint },
                { id: assistantId, role: 'assistant', content: '', createdAt: now, status: 'sending', specFingerprint: fingerprint },
            ],
        });
        setComposerText('');
        setLoading(true);
        resetLiveExecutionEvents();
        setPendingHumanReview(null);
        setHumanReviewError(null);
        let currentRunId = assistantId;
        let finalAnswer = '';
        let terminalError: string | undefined;
        const traceStream = new LiveTraceStreamController(assistantId);
        setLiveExecution({ messageId: assistantId, running: true });
        try {
            await streamAgentWorkflowBuilderTest({
                ...builderTestRequestContext(),
                builder_session_id: builderRuntime.sessionIdRef.current,
                base_workflow_id: testRuntime.baseWorkflowId,
                spec: testRuntime.spec,
                question,
                transient_messages: transientMessagesForRequest(priorMessages),
                workflow_spec_fingerprint: fingerprint,
            }, (event) => {
                if (event.event !== 'heartbeat') {
                    appendLiveExecutionEvent(event as AgentExecutionStreamEnvelope);
                }
                if (event.event === 'run.completed') {
                    finalAnswer = String(event.data?.answer || event.data?.final_output?.answer || '');
                }
                if (event.event === 'run.failed') {
                    terminalError = String(event.data?.error?.raw_message || event.data?.error || 'Workflow test failed.');
                }
                const snapshot = traceStream.append(event as AgentExecutionStreamEnvelope, terminalError);
                currentRunId = snapshot.runId;
                setLiveExecution({
                    messageId: assistantId,
                    runId: snapshot.runId,
                    running: snapshot.running,
                    error: terminalError,
                });
                onOpenTrace?.({
                    id: snapshot.runId,
                    messageId: assistantId,
                    label: `Test · ${fingerprint}`,
                    status: snapshot.status,
                    liveTraceView: buildLiveTraceView(snapshot.events),
                    running: snapshot.running,
                    error: terminalError,
                });
            });
            const runDetails = await getLatestAgentWorkflowBuilderTest(
                builderRuntime.sessionIdRef.current,
                testRuntime.baseWorkflowId,
            );
            const answer = finalAnswer
                || String(runDetails?.result_json?.answer || runDetails?.result_json?.final_answer || 'The workflow completed without a final answer.');
            updateBuilderTestMessage(assistantId, {
                content: answer,
                runId: runDetails?.id || currentRunId,
                status: runDetails?.pending_interrupt ? 'review' : 'completed',
            });
            if (runDetails) {
                setAgentRunDetails((current) => ({ ...current, [runDetails.id]: runDetails }));
                if (runDetails.pending_interrupt) {
                    setPendingHumanReview({
                        runId: runDetails.id,
                        interrupt: runDetails.pending_interrupt,
                        localUserMessageId: userId,
                        localAssistantMessageId: assistantId,
                    });
                    setHumanReviewEditText(runDetails.pending_interrupt.proposed_final_answer || '');
                }
                onOpenTrace?.({
                    id: runDetails.id,
                    messageId: assistantId,
                    label: `Test · ${fingerprint}`,
                    status: runDetails.status,
                    runDetails,
                    liveTraceView: buildRunTraceView(runDetails),
                    running: false,
                });
            }
        } catch (error: any) {
            const cancelled = error?.name === 'AbortError';
            const message = cancelled ? 'Workflow test cancelled.' : error?.message || terminalError || 'Workflow test failed.';
            updateBuilderTestMessage(assistantId, {
                content: message,
                runId: currentRunId,
                status: cancelled ? 'cancelled' : 'failed',
            });
        } finally {
            setLiveExecution(null);
            setLoading(false);
        }
    };

    const handleHumanReviewAction = async (action: AgentRunResumeAction, selectedOptionIds?: string[]): Promise<boolean> => {
        if (!pendingHumanReview || !activeThread) return false;
        const interrupt = pendingHumanReview.interrupt;
        if (!interrupt.interrupt_id) return false;
        if (testRuntime && builderRuntime) {
            setHumanReviewSubmitting(action);
            setHumanReviewError(null);
            resetLiveExecutionEvents();
            setLiveExecution({
                messageId: pendingHumanReview.localAssistantMessageId,
                runId: pendingHumanReview.runId,
                running: true,
            });
            try {
                const traceStream = new LiveTraceStreamController(pendingHumanReview.runId);
                await resumeAgentWorkflowBuilderTest(pendingHumanReview.runId, {
                    ...builderTestRequestContext(),
                    action,
                    interrupt_id: interrupt.interrupt_id,
                    resume_token: interrupt.resume_token || undefined,
                    resume_version: interrupt.resume_version || undefined,
                    selected_option_ids: selectedOptionIds,
                }, (event) => {
                    if (event.event !== 'heartbeat') {
                        appendLiveExecutionEvent(event as AgentExecutionStreamEnvelope);
                    }
                    const snapshot = traceStream.append(event as AgentExecutionStreamEnvelope);
                    onOpenTrace?.({
                        id: pendingHumanReview.runId,
                        messageId: pendingHumanReview.localAssistantMessageId,
                        label: `Test · ${workflowSpecFingerprint(testRuntime.spec)}`,
                        status: snapshot.status,
                        liveTraceView: buildLiveTraceView(snapshot.events),
                        running: snapshot.running,
                    });
                });
                const refreshed = await getLatestAgentWorkflowBuilderTest(
                    builderRuntime.sessionIdRef.current,
                    testRuntime.baseWorkflowId,
                );
                const answer = String(
                    refreshed?.result_json?.answer
                    || refreshed?.result_json?.final_answer
                    || 'The workflow test completed.',
                );
                updateBuilderTestMessage(pendingHumanReview.localAssistantMessageId, {
                    content: answer,
                    runId: refreshed?.id || pendingHumanReview.runId,
                    status: refreshed?.pending_interrupt ? 'review' : 'completed',
                });
                if (refreshed) {
                    setAgentRunDetails((current) => ({ ...current, [refreshed.id]: refreshed }));
                    onOpenTrace?.({
                        id: refreshed.id,
                        messageId: pendingHumanReview.localAssistantMessageId,
                        label: `Test · ${workflowSpecFingerprint(testRuntime.spec)}`,
                        status: refreshed.status,
                        runDetails: refreshed,
                        liveTraceView: buildRunTraceView(refreshed),
                        running: false,
                    });
                }
                if (refreshed?.pending_interrupt) {
                    setPendingHumanReview((current) => current ? {
                        ...current,
                        runId: refreshed.id,
                        interrupt: refreshed.pending_interrupt as AgentRunPendingInterrupt,
                    } : current);
                    setHumanReviewEditText(refreshed.pending_interrupt.proposed_final_answer || '');
                } else {
                    setPendingHumanReview(null);
                    setHumanReviewEditText('');
                }
                return true;
            } catch (error: any) {
                setHumanReviewError(error?.message || 'Unable to submit the test review decision.');
                return false;
            } finally {
                setLiveExecution(null);
                setHumanReviewSubmitting(null);
            }
        }
        const submissionKey = `${pendingHumanReview.runId}:${interrupt.interrupt_id}:${interrupt.resume_version ?? 1}:${action}`;
        if (humanReviewSubmissionKeyRef.current) return false;

        humanReviewSubmissionKeyRef.current = submissionKey;
        setHumanReviewSubmitting(action);
        setHumanReviewError(null);
        const liveMessageId = pendingHumanReview.localAssistantMessageId;
        resetLiveExecutionEvents();
        setLiveExecution({ messageId: liveMessageId, runId: pendingHumanReview.runId, running: true });
        setOpenAgentRunIds((current) => new Set(current).add(liveMessageId));
        try {
            let response: any;
            let terminalError: string | undefined;
            await streamResumeAgentRun(pendingHumanReview.runId, {
                action,
                interrupt_id: interrupt.interrupt_id,
                resume_token: interrupt.resume_token || undefined,
                resume_version: interrupt.resume_version || undefined,
                selected_option_ids: selectedOptionIds,
                thread_id: activeThread.id,
                edited_payload: action === AgentRunResumeActionValue.Edit ? { final_answer: humanReviewEditText } : undefined,
                client_metadata: { source: 'chat_pending_review_panel' },
            }, (event) => {
                if (event.event !== 'heartbeat') appendLiveExecutionEvent(event);
                if (['run.completed', 'run.failed', 'interrupt.created'].includes(event.event)) {
                    response = event.data?.response;
                    const rawError = event.data?.error;
                    terminalError = event.event === 'run.failed' && !response
                        ? String(rawError?.raw_message || rawError?.message || rawError || 'Unable to resume the agent run.')
                        : undefined;
                    setLiveExecution((current) => current?.messageId === liveMessageId
                        ? { ...current, running: false, error: terminalError }
                        : current);
                }
            });
            if (!response) throw new Error(terminalError || 'The resume stream ended before returning a result.');

            if (response.agent_run.status === 'awaiting_human' && response.agent_run.pending_interrupt) {
                setPendingHumanReview(prev => prev ? {
                    ...prev,
                    interrupt: response.agent_run.pending_interrupt as AgentRunPendingInterrupt,
                } : prev);
                setHumanReviewEditText(response.agent_run.pending_interrupt.proposed_final_answer || humanReviewEditText);
                return true;
            }

            setPendingHumanReview(null);
            setHumanReviewEditText('');
            await loadMessages();
            onThreadUpdate?.();
            return true;
        } catch (err: any) {
            setHumanReviewError(err?.message || 'Unable to submit review decision.');
            return false;
        } finally {
            if (humanReviewSubmissionKeyRef.current === submissionKey) {
                humanReviewSubmissionKeyRef.current = null;
            }
            setOpenAgentRunIds((current) => {
                const next = new Set(current);
                if (!manuallyToggledAgentRunsRef.current.has(liveMessageId)) next.delete(liveMessageId);
                return next;
            });
            setLiveExecution((current) => current?.messageId === liveMessageId ? null : current);
            setHumanReviewSubmitting(null);
        }
    };

    const handleSend = async (
        overrideInput?: string | React.SyntheticEvent,
        options?: { bypassClarification?: boolean },
    ) => {
        const rawTextToSend = typeof overrideInput === 'string' ? overrideInput : '';
        const textToSend = rawTextToSend.trim();
        const editMessageId = editingMessageId;
        const editOriginalText = editingOriginalText.trim();
        const isEditingQuestion = Boolean(editMessageId);

        if (!textToSend) {
            if (isEditingQuestion) {
                cancelQuestionEdit();
            }
            return;
        }
        if (!llmModel || !activeThread) return;
        if (testRuntime) {
            await handleBuilderTestSend(textToSend);
            return;
        }

        if (isEditingQuestion && textToSend === editOriginalText) {
            cancelQuestionEdit();
            return;
        }

        setComposerText('');
        setClarificationOptions(null);
        setLoading(true);

        const optimisticStamp = Date.now();
        const tempUserMsg: ChatMessage = {
            id: 'temp-user-' + optimisticStamp,
            role: MessageRole.User,
            content: textToSend,
            created_at: new Date().toISOString()
        };
        const tempAssistantMsg: ChatMessage = {
            id: 'temp-assistant-' + optimisticStamp,
            role: MessageRole.Assistant,
            content: '',
            created_at: new Date().toISOString(),
        };

        resetLiveExecutionEvents();
        setLiveExecution({ messageId: tempAssistantMsg.id, running: true });
        setOpenAgentRunIds((current) => new Set(current).add(tempAssistantMsg.id));

        let editDeletionCompleted = false;
        try {
            if (isEditingQuestion && editMessageId) {
                const { deleted_ids } = await deleteMessage(editMessageId);
                editDeletionCompleted = true;
                setMessages(prev => prev.filter(m => !deleted_ids.includes(m.id)));
                if (onResetChatId) {
                    onResetChatId();
                }
            }

            setMessages(prev => [...prev, tempUserMsg, tempAssistantMsg]);

            let response: ThreadChatResponse | undefined;
            let terminalStreamError: string | undefined;
            const traceStream = new LiveTraceStreamController(tempAssistantMsg.id);
            await streamThreadChat(
                activeThread.id,
                textToSend,
                llmModel,
                useWebSearch,
                useReranker,
                contextWindow,
                workflowSupportsReplans(agentWorkflows, agentWorkflowId) ? replans : undefined,
                systemRole,
                effectiveToolInstructions,
                customInstructions,
                Boolean(options?.bypassClarification),
                hitlWebApproval,
                (event: AgentExecutionStreamEnvelope) => {
                    if (event.event !== 'heartbeat') {
                        appendLiveExecutionEvent(event);
                    }
                    const snapshot = traceStream.append(event, terminalStreamError, response?.status);
                    if (event.event === 'run.started' && event.data?.run_id) {
                        setLiveExecution((current) => current?.messageId === tempAssistantMsg.id
                            ? { ...current, runId: snapshot.runId }
                            : current);
                    }
                    if (isLiveTraceTerminalEvent(event.event)) {
                        if (event.data?.response) response = event.data.response as ThreadChatResponse;
                        const rawError = event.data?.error;
                        terminalStreamError = event.event === 'run.failed' && !event.data?.response
                            ? String(rawError?.raw_message || rawError?.message || rawError || 'Chat workflow failed.')
                            : undefined;
                        const terminalSnapshot = traceStream.snapshot(event.event, terminalStreamError, response?.status);
                        setLiveExecution((current) => current?.messageId === tempAssistantMsg.id
                            ? { ...current, runId: terminalSnapshot.runId, running: false, error: terminalStreamError }
                            : current);
                        setOpenAgentRunIds((current) => {
                            const next = new Set(current);
                            if (!manuallyToggledAgentRunsRef.current.has(tempAssistantMsg.id)) next.delete(tempAssistantMsg.id);
                            return next;
                        });
                    }
                    if (workspaceTraceMessageIdRef.current === tempAssistantMsg.id && event.event !== 'heartbeat') {
                        const latestSnapshot = traceStream.snapshot(event.event, terminalStreamError, response?.status);
                        onOpenTrace?.({
                            id: latestSnapshot.runId,
                            messageId: tempAssistantMsg.id,
                            label: response?.agent_workflow_id || agentWorkflowId || 'agent',
                            status: latestSnapshot.status,
                            routeReason: response?.agent_route_reason,
                            traceRefs: response?.agent_trace_refs,
                            liveTraceView: buildLiveTraceView(latestSnapshot.events),
                            running: latestSnapshot.running,
                            error: terminalStreamError,
                        });
                    }
                },
            );
            if (!response) throw new Error(terminalStreamError || 'The chat stream ended before returning a response.');
            const traceWasOpenForTempMessage = workspaceTraceMessageIdRef.current === tempAssistantMsg.id;
            const refreshPersistedTrace = async (messageId: string) => {
                if (!traceWasOpenForTempMessage || !response?.agent_run_id || !activeThread) return;
                setWorkspaceTraceMessageId(messageId);
                workspaceTraceMessageIdRef.current = messageId;
                const label = `${response.agent_workflow_id || agentWorkflowId || 'agent'}${(response.agent_route || response.route) ? ` · ${response.agent_route || response.route}` : ''}`;
                try {
                    const run = await getAgentRun(response.agent_run_id, activeThread.id);
                    setAgentRunDetails(prev => ({ ...prev, [run.id]: run }));
                    onOpenTrace?.({
                        id: run.id,
                        messageId,
                        label,
                        status: run.status,
                        routeReason: response.agent_route_reason,
                        traceRefs: response.agent_trace_refs,
                        runDetails: run,
                        liveTraceView: buildRunTraceView(run),
                        running: false,
                    });
                } catch (error: any) {
                    const message = error?.message || 'Unable to load agent run.';
                    setAgentRunErrors(prev => ({
                        ...prev,
                        [response!.agent_run_id!]: message,
                    }));
                    onOpenTrace?.({
                        id: response.agent_run_id,
                        messageId,
                        label,
                        status: liveTraceStatusFromEvent('run.completed', terminalStreamError, response.status),
                        routeReason: response.agent_route_reason,
                        traceRefs: response.agent_trace_refs,
                        liveTraceView: buildLiveTraceView(traceStream.snapshot('run.completed', terminalStreamError, response.status).events),
                        running: false,
                        error: message,
                    });
                }
            };

            if (response.status === 'cancelled') {
                setMessages(prev => recoverCanceledChat(
                    prev,
                    tempUserMsg.id,
                    tempAssistantMsg.id,
                    textToSend,
                ).messages);
                setOpenAgentRunIds((current) => {
                    const next = new Set(current);
                    next.delete(tempAssistantMsg.id);
                    return next;
                });
                resetLiveExecutionEvents();
                setComposerText(textToSend);
                requestAnimationFrame(() => composerInputRef.current?.focus());
                if (traceWasOpenForTempMessage) {
                    onOpenTrace?.({
                        id: response.agent_run_id || traceStream.snapshot('run.cancelled').runId,
                        messageId: tempAssistantMsg.id,
                        label: response.agent_workflow_id || agentWorkflowId || 'agent',
                        status: 'cancelled',
                        routeReason: response.agent_route_reason,
                        traceRefs: response.agent_trace_refs,
                        liveTraceView: buildLiveTraceView(traceStream.snapshot('run.cancelled').events),
                        running: false,
                    });
                }
                return;
            }

            if (response.status === 'awaiting_human' && response.agent_run_id && response.pending_interrupt) {
                const localUserMessageId = response.user_message_id || ('review-user-' + Date.now());
                const localAssistantMessageId = response.assistant_message_id || ('review-asst-' + Date.now());
                const pendingReview: PendingHumanReview = {
                    runId: response.agent_run_id,
                    interrupt: response.pending_interrupt,
                    localUserMessageId,
                    localAssistantMessageId,
                };

                setPendingHumanReview(pendingReview);
                setHumanReviewError(null);
                setHumanReviewEditText(response.pending_interrupt.proposed_final_answer || response.answer || '');

                setMessages(prev => {
                    const updated = prev.filter(m => m.id !== tempUserMsg.id && m.id !== tempAssistantMsg.id);
                    return [
                        ...updated,
                        {
                            id: localUserMessageId,
                            role: MessageRole.User,
                            content: textToSend,
                            rewritten_query: response.rewritten_query && response.rewritten_query !== textToSend ? response.rewritten_query : undefined,
                            agent_run_id: response.agent_run_id,
                            agent_run_turn_kind: 'user_prompt',
                            agent_trace_refs: response.agent_trace_refs,
                            agent_workflow_id: response.agent_workflow_id,
                                                        agent_route: response.agent_route || response.route,
                            agent_route_reason: response.agent_route_reason,
                            created_at: new Date().toISOString()
                        },
                        {
                            id: localAssistantMessageId,
                            role: MessageRole.Assistant,
                            content: response.pending_interrupt.title || 'Human review required before the agent can continue.',
                            agent_run_id: response.agent_run_id,
                            agent_run_turn_kind: 'assistant_pending_review',
                            agent_trace_refs: response.agent_trace_refs,
                            agent_workflow_id: response.agent_workflow_id,
                                                        agent_route: response.agent_route || response.route,
                            agent_route_reason: response.agent_route_reason,
                            pending_human_review: true,
                            created_at: new Date().toISOString()
                        },
                    ];
                });

                try {
                    const run = await getAgentRun(response.agent_run_id, activeThread.id);
                    setAgentRunDetails(prev => ({ ...prev, [run.id]: run }));
                    if (traceWasOpenForTempMessage) {
                        setWorkspaceTraceMessageId(localAssistantMessageId);
                        workspaceTraceMessageIdRef.current = localAssistantMessageId;
                        onOpenTrace?.({
                            id: run.id,
                            messageId: localAssistantMessageId,
                            label: `${response.agent_workflow_id || agentWorkflowId || 'agent'}${(response.agent_route || response.route) ? ` · ${response.agent_route || response.route}` : ''}`,
                            status: run.status,
                            routeReason: response.agent_route_reason,
                            traceRefs: response.agent_trace_refs,
                            runDetails: run,
                            liveTraceView: buildRunTraceView(run),
                            running: false,
                        });
                    }
                } catch (error: any) {
                    setAgentRunErrors(prev => ({
                        ...prev,
                        [response.agent_run_id!]: error?.message || 'Unable to load agent run.',
                    }));
                    if (traceWasOpenForTempMessage) {
                        setWorkspaceTraceMessageId(localAssistantMessageId);
                        workspaceTraceMessageIdRef.current = localAssistantMessageId;
                        onOpenTrace?.({
                            id: response.agent_run_id,
                            messageId: localAssistantMessageId,
                            label: `${response.agent_workflow_id || agentWorkflowId || 'agent'}${(response.agent_route || response.route) ? ` · ${response.agent_route || response.route}` : ''}`,
                            status: 'review',
                            routeReason: response.agent_route_reason,
                            traceRefs: response.agent_trace_refs,
                            liveTraceView: buildLiveTraceView(traceStream.snapshot('interrupt.created').events),
                            running: false,
                            error: error?.message || 'Unable to load agent run.',
                        });
                    }
                }

                if (isEditingQuestion) {
                    setEditingMessageId(null);
                    setEditingOriginalText('');
                }
                return;
            }

            // Handle ambiguous query / clarification options
            if (response.clarification_options) {
                setClarificationOptions([
                    ...response.clarification_options
                        .map((choice) => clarificationChoiceText(choice).trim())
                        .filter(Boolean)
                        .map((text) => ({ text, isOriginal: false })),
                    { text: textToSend, isOriginal: true }
                ]);
                setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id && m.id !== tempAssistantMsg.id));
                if (isEditingQuestion) {
                    setEditingMessageId(null);
                    setEditingOriginalText('');
                }
            } else {
                // Normal flow: update messages with real IDs and add assistant response
                setMessages(prev => {
                    const updated = prev.filter(m => m.id !== tempUserMsg.id && m.id !== tempAssistantMsg.id);
                    const finalMessages = [...updated];

                    finalMessages.push({
                        id: response.user_message_id || ('final-user-' + Date.now()),
                        role: MessageRole.User,
                        content: textToSend, // Keep original input
                        rewritten_query: response.rewritten_query && response.rewritten_query !== textToSend ? response.rewritten_query : undefined,
                        agent_run_id: response.agent_run_id,
                        agent_run_turn_kind: response.agent_run_turn_kind,
                        agent_run_sequence: response.agent_run_sequence,
                        agent_trace_refs: response.agent_trace_refs,
                        agent_workflow_id: response.agent_workflow_id,
                                                agent_route: response.agent_route || response.route,
                        agent_route_reason: response.agent_route_reason,
                        created_at: new Date().toISOString()
                    });

                    if (response.assistant_message_id || response.answer) {
                        finalMessages.push({
                            id: response.assistant_message_id || ('assistant-' + Date.now()),
                            role: MessageRole.Assistant,
                            content: typeof response.answer === 'string' ? response.answer : String(response.answer ?? ''),
                            reasoning: response.reasoning || '',
                            reasoning_available: !!response.reasoning_available,
                            reasoning_format: response.reasoning_format || ReasoningFormat.None,
                            web_sources: response.web_sources || [],
                            agent_run_id: response.agent_run_id,
                            agent_run_turn_kind: response.agent_run_turn_kind,
                            agent_run_sequence: response.agent_run_sequence,
                            agent_trace_refs: response.agent_trace_refs,
                            agent_workflow_id: response.agent_workflow_id,
                                                        agent_route: response.agent_route || response.route,
                            agent_route_reason: response.agent_route_reason,
                            created_at: new Date().toISOString()
                        });
                    }

                    return finalMessages;
                });
                void refreshPersistedTrace(response.assistant_message_id || response.agent_run_id || traceStream.snapshot('run.completed').runId);

                // Mark recollected messages
                if (response.used_chat_ids && response.used_chat_ids.length > 0) {
                    setRecollectedIds(new Set(response.used_chat_ids));
                    // Clear recollection highlight after 10 seconds
                    setTimeout(() => setRecollectedIds(new Set()), 10000);
                }
                if (response.memory_candidate_ids && response.memory_candidate_ids.length > 0) {
                    loadPendingMemoryCandidates();
                }

                // Notify parent that thread was updated
                if (onThreadUpdate) {
                    onThreadUpdate();
                }
                if (isEditingQuestion) {
                    setEditingMessageId(null);
                    setEditingOriginalText('');
                }
            }

        } catch (err: any) {
            console.error(err);
            const errorMessage: ChatMessage = {
                id: 'error-' + Date.now(),
                role: MessageRole.Assistant,
                content: `Error: ${err.message || "Failed to get response."}`,
                created_at: new Date().toISOString()
            };
            if (editDeletionCompleted) {
                await loadMessages();
                setEditingMessageId(null);
                setEditingOriginalText('');
                setMessages(prev => [...prev, errorMessage]);
            } else {
                // Remove optimistic message and show error
                setMessages(prev => {
                    const updated = prev.filter(m => m.id !== tempUserMsg.id && m.id !== tempAssistantMsg.id);
                    return [
                        ...updated,
                        tempUserMsg,
                        errorMessage
                    ];
                });
            }
        } finally {
            setLiveExecution((current) => current?.messageId === tempAssistantMsg.id ? null : current);
            setLoading(false);
        }
    };

    const handleStopChat = async () => {
        const execution = liveExecution;
        if (!activeThread || !canRequestChatCancellation(execution)) return;
        setLiveExecution((current) => current?.runId === execution.runId
            ? { ...current, canceling: true, error: undefined }
            : current);
        try {
            if (testRuntime) {
                await cancelAgentWorkflowBuilderTest(execution.runId);
                return;
            }
            const result = await cancelChatAgentRun(execution.runId, activeThread.id);
            if (result.status === 'already_terminal') {
                setLiveExecution((current) => current?.runId === execution.runId
                    ? { ...current, canceling: false }
                    : current);
            }
        } catch (error: any) {
            setLiveExecution((current) => current?.runId === execution.runId
                ? {
                    ...current,
                    canceling: false,
                    error: error?.message || 'Unable to stop this chat run.',
                }
                : current);
        }
    };

    const handleDeleteMessage = useCallback(async (messageId: string, event: React.MouseEvent) => {
        event.stopPropagation();
        if (!confirm('Delete this message?')) return;

        // Frontend-only messages (error responses and their paired user messages) are never
        // persisted to the backend, so we remove them directly from local state.
        const isTempId = messageId.startsWith('error-') ||
            messageId.startsWith('temp-user-') ||
            messageId.startsWith('final-user-') ||
            messageId.startsWith('assistant-') ||
            messageId.startsWith('clarify-user-') ||
            messageId.startsWith('clarify-asst-') ||
            messageId.startsWith('review-user-') ||
            messageId.startsWith('review-asst-');
        if (isTempId) {
            setMessages(prev => {
                const idx = prev.findIndex(m => m.id === messageId);
                if (idx === -1) return prev;
                const msg = prev[idx];
                const isLocalUser = (id: string) =>
                    id.startsWith('temp-user-') ||
                    id.startsWith('final-user-') ||
                    id.startsWith('clarify-user-') ||
                    id.startsWith('review-user-');
                const isLocalAssistant = (id: string) =>
                    id.startsWith('error-') ||
                    id.startsWith('assistant-') ||
                    id.startsWith('clarify-asst-') ||
                    id.startsWith('review-asst-');

                // Deleting a local assistant message also removes its preceding local user message.
                if (msg.role === MessageRole.Assistant && idx > 0) {
                    const prevMsg = prev[idx - 1];
                    if (isLocalUser(prevMsg.id)) {
                        return prev.filter((_, i) => i !== idx && i !== idx - 1);
                    }
                }
                // Deleting a local user message also removes its following local assistant message.
                if (msg.role === MessageRole.User && idx < prev.length - 1) {
                    const nextMsg = prev[idx + 1];
                    if (isLocalAssistant(nextMsg.id)) {
                        return prev.filter((_, i) => i !== idx && i !== idx + 1);
                    }
                }
                return prev.filter(m => m.id !== messageId);
            });
            if (onResetChatId) onResetChatId();
            return;
        }

        try {
            const { deleted_ids } = await deleteMessage(messageId);
            setMessages(prev => prev.filter(m => !deleted_ids.includes(m.id)));
            if (editingMessageId && deleted_ids.includes(editingMessageId)) {
                setEditingMessageId(null);
                setEditingOriginalText('');
                setComposerText('');
            }

            // Critical: If the current active chat sentence belongs to a deleted message, 
            // reset the chat ID selection to prevent out-of-bounds access.
            if (onResetChatId) {
                onResetChatId();
            }

            if (onThreadUpdate) {
                onThreadUpdate();
            }
        } catch (error) {
            console.error('Failed to delete message:', error);
        }
    }, [editingMessageId, onResetChatId, onThreadUpdate, setComposerText]);

    const handleSaveThreadSettings = async () => {
        if (!activeThread) return;
        if (isTestRuntime) {
            // Builder Test settings are intentionally local to this page session.
            // State is already updated by the dialog controls, so applying only
            // closes the dialog and never mutates the selected real thread.
            setSettingsDialogOpen(false);
            return;
        }
        try {
            setSavingSettings(true);
            const nextSettings: Record<string, any> = {
                system_role: systemRole,
                tool_instructions: effectiveToolInstructions,
                custom_instructions: customInstructions,
                hitl_web_approval: hitlWebApproval,
                use_reranker: useReranker,
                memory: {
                    thread_reads_project_memory: useProjectMemory,
                    thread_reads_user_memory: useGlobalMemory,
                },
            };
            const normalizedWorkflowId = normalizeAgentWorkflowForUi(agentWorkflowId);
            if (normalizedWorkflowId) {
                nextSettings.agent_workflow = { workflow_id: normalizedWorkflowId };
            }
            if (workflowSupportsReplans(agentWorkflows, agentWorkflowId) && replansLimit !== null) {
                nextSettings.replans = Math.max(1, Math.min(replansLimit, replans));
            }
            const saved = await updateThreadSettings(activeThread.id, nextSettings);
            applyThreadSettingsToState(saved);
            setSettingsDialogOpen(false);
        } catch (error) {
            console.error('Failed to save thread settings:', error);
        } finally {
            setSavingSettings(false);
        }
    };

    const handleOpenThreadSettings = () => {
        if (activeThread?.settings) {
            applyThreadSettingsToState(activeThread.settings);
        }
        loadThreadSettings();
        loadProjectMemorySettings();
        setSettingsDialogOpen(true);
    };

    const refreshAgentWorkflows = useCallback(async () => {
        try {
            const patterns = await listAgentWorkflows();
            setAgentWorkflows(patterns.agent_workflows || []);
        } catch (error) {
            console.error('Failed to refresh agent workflows:', error);
        }
    }, []);

    const handleCloseThreadSettings = () => {
        setSettingsDialogOpen(false);
        loadThreadSettings();
    };

    const handleCopy = useCallback((text: string, messageId: string) => {
        navigator.clipboard.writeText(text);
        setCopiedId(messageId);
        setTimeout(() => setCopiedId(null), 2000);
    }, []);

    const handleReadAloud = useCallback((messageIdx: number) => {
        const firstSentence = chatSentences.find(s => s.messageIndex === messageIdx);
        if (firstSentence) onJump(firstSentence.id);
    }, [chatSentences, onJump]);

    const handleForkFromMessage = useCallback(async (messageId: string, event: React.MouseEvent) => {
        event.stopPropagation();
        if (!activeThread) return;
        setForkDialogMessageId(messageId);
        if (forkProjects.length === 0) {
            try {
                const response = await listProjects();
                setForkProjects(response.projects || []);
            } catch (error) {
                console.error('Failed to load projects for fork dialog:', error);
            }
        }
    }, [activeThread, forkProjects.length]);

    const submitMessageFork = async (options: { name?: string; targetProjectId?: string; memoryCopyMode?: MemoryCopyMode }) => {
        if (!activeThread || !forkDialogMessageId) return;
        try {
            setForkingMessageId(forkDialogMessageId);
            const forked = await forkThread(activeThread.id, { messageId: forkDialogMessageId, ...options });
            setForkDialogMessageId(null);
            onThreadForked?.(forked);
        } catch (error) {
            console.error('Failed to fork thread from message:', error);
            alert('Failed to fork thread from this message.');
        } finally {
            setForkingMessageId(null);
        }
    };

    const handleOpenAgentRun = useCallback(async (msg: ChatMessage) => {
        const runId = msg.agent_run_id;
        setWorkspaceTraceMessageId(msg.id);
        workspaceTraceMessageIdRef.current = msg.id;
        if (!runId || !activeThread || agentRunDetails[runId] || agentRunLoading[runId]) return;

        setAgentRunLoading(prev => ({ ...prev, [runId]: true }));
        setAgentRunErrors(prev => {
            const next = { ...prev };
            delete next[runId];
            return next;
        });
        try {
            const run = await getAgentRun(runId, activeThread.id);
            setAgentRunDetails(prev => ({ ...prev, [runId]: run }));
        } catch (error: any) {
            setAgentRunErrors(prev => ({
                ...prev,
                [runId]: error?.message || 'Unable to load agent run.',
            }));
        } finally {
            setAgentRunLoading(prev => ({ ...prev, [runId]: false }));
        }
    }, [activeThread, agentRunDetails, agentRunLoading]);

    useEffect(() => {
        if (!workspaceTraceMessageId || !onOpenTrace) return;
        const msg = messages.find((candidate) => candidate.id === workspaceTraceMessageId);
        if (!msg) {
            if (!workspaceTraceMessageId.startsWith('temp-assistant-')) {
                workspaceTraceMessageIdRef.current = null;
                setWorkspaceTraceMessageId(null);
            }
            return;
        }
        workspaceTraceMessageIdRef.current = workspaceTraceMessageId;
        const liveForMessage = liveExecution?.messageId === msg.id ? liveExecution : null;
        const runId = msg.agent_run_id || liveForMessage?.runId || msg.id;
        onOpenTrace({
            id: runId,
            messageId: msg.id,
            label: `${formatAgentWorkflowLabel(msg)}${msg.agent_route ? ` · ${msg.agent_route}` : ''}`,
            status: liveForMessage?.running ? 'running' : agentRunDetails[runId]?.status,
            routeReason: msg.agent_route_reason,
            traceRefs: msg.agent_trace_refs,
            runDetails: agentRunDetails[runId],
            liveTraceView: liveForMessage ? liveTraceView : undefined,
            loading: Boolean(agentRunLoading[runId]),
            error: liveForMessage?.error || agentRunErrors[runId],
            running: Boolean(liveForMessage?.running),
        });
    }, [
        agentRunDetails,
        agentRunErrors,
        agentRunLoading,
        liveExecution,
        liveTraceView,
        messages,
        onOpenTrace,
        workspaceTraceMessageId,
    ]);

    const handleAgentRunDetailsChange = useCallback((run: AgentRunDetails) => {
        setAgentRunDetails(prev => ({ ...prev, [run.id]: run }));
    }, []);

    const formatAgentWorkflowLabel = useCallback((msg: ChatMessage) => {
        return msg.agent_workflow_id || 'agent';
    }, []);

    if (!activeThread) {
        return (
            <Paper elevation={0} sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3, bgcolor: theme.palette.background.default, color: theme.palette.text.primary }}>
                <Box sx={{ textAlign: "center" }}>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                        No Thread Selected
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Create or select a thread from the sidebar to start chatting
                    </Typography>
                </Box>
            </Paper>
        );
    }

    const forkInfo = activeThread.thread_metadata?.fork;
    const childThreadIds = Array.isArray(activeThread.thread_metadata?.fork_children)
        ? activeThread.thread_metadata.fork_children.filter((id): id is string => typeof id === 'string' && id.length > 0)
        : [];
    const hasLineage = !isTestRuntime && !hideInlineLineage && Boolean(forkInfo || childThreadIds.length > 0);
    const lineageThreadsById = new Map(lineageThreads.map(thread => [thread.id, thread]));
    const headerSelectOutlineSx = {
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
    const latestUserMessageId = [...messages].reverse().find(m => m.role === MessageRole.User)?.id ?? null;
    const pendingReviewInterrupt = pendingHumanReview?.interrupt ?? null;
    const pendingReviewActions = Array.isArray(pendingReviewInterrupt?.allowed_actions)
        ? pendingReviewInterrupt.allowed_actions.map(String)
        : [];
    const pendingReviewTitle = pendingReviewInterrupt?.title
        || (pendingReviewInterrupt?.proposed_tool ? 'Approve tool use?' : 'Human review required');
    const pendingReviewProposedTool = pendingReviewInterrupt?.proposed_tool;
    const pendingReviewIsWebApproval = pendingReviewInterrupt?.target_node_id === 'web_worker'
        || pendingReviewInterrupt?.node_id === 'web_approval_gate'
        || (
            typeof pendingReviewProposedTool === 'object'
            && pendingReviewProposedTool !== null
            && pendingReviewProposedTool.name === 'search_web'
        );
    const approveLabel = pendingReviewIsWebApproval ? 'Approve web search' : 'Approve';
    const continueWithoutLabel = pendingReviewIsWebApproval ? 'Continue without web search' : 'Continue';
    const showDecisionPanel = Boolean(clarificationOptions || pendingHumanReview);

    return (
        <Paper ref={chatRootRef} elevation={0} sx={{ height: '100%', minHeight: 0, display: 'flex', flexDirection: 'column', p: 1, bgcolor: theme.palette.background.default, color: theme.palette.text.primary, cursor: 'default' }}>
            {/* Header */}
            <Box sx={{ mb: 0.5, pt: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, flexShrink: 0 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                    <Tooltip title={
                        isEmbeddingModelValid === null ? "Checking embedding model status..." :
                            isEmbeddingModelValid ? `Embedding model: ${activeThread.embeddingModel}` :
                                `Embedding model ${activeThread.embeddingModel} not found`
                    }>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            {isEmbeddingModelValid === true ? (
                                <CheckCircleIcon fontSize="medium" color="primary" />
                            ) : isEmbeddingModelValid === false ? (
                                <ErrorIcon fontSize="medium" color="error" />
                            ) : (
                                <CircularProgress size={20} />
                            )}
                            {isEmbeddingModelValid === null && (
                                <Typography variant="caption" color="warning.main" sx={{ ml: 0.5, fontWeight: 'bold' }}>CHECKING...</Typography>
                            )}
                            {isEmbeddingModelValid === false && <Typography variant="caption" color="error" sx={{ fontWeight: 'bold' }}>OFFLINE</Typography>}
                        </Box>
                    </Tooltip>
                    {hasLineage && (
                        <Tooltip
                            title={
                                <ThreadLineageTooltipContent
                                    thread={activeThread}
                                    threadsById={lineageThreadsById}
                                    onOpenThread={onOpenThread}
                                />
                            }
                            arrow
                            enterDelay={300}
                            leaveDelay={150}
                            disableInteractive={false}
                        >
                            <IconButton
                                size="small"
                                aria-label="Thread lineage"
                                sx={{ p: 0.5 }}
                            >
                                <CallSplitIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    )}
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, maxWidth: '350px', gap: 1 }}>
                    <Tooltip
                        title={`${webSearchModeLabel}. Click to switch to ${nextWebSearchModeLabel}.`}
                        placement="top"
                    >
                        <span>
                            <IconButton
                                aria-label={webSearchModeLabel}
                                color={webSearchMode === 'on'
                                    ? 'primary'
                                    : webSearchMode === 'ask'
                                        ? 'warning'
                                        : 'default'}
                                onClick={handleWebSearchModeChange}
                                disabled={savingWebSearchMode}
                                size="small"
                                sx={{ p: 0.5 }}
                            >
                                {webSearchMode === 'on'
                                    ? <WifiTwoToneIcon />
                                    : webSearchMode === 'ask'
                                        ? <WifiPasswordIcon />
                                        : <WifiOffTwoToneIcon />}
                            </IconButton>
                        </span>
                    </Tooltip>
                    <Tooltip
                        title={
                            <Box sx={{ p: 0.5 }}>
                                <Typography variant="caption" sx={{ display: "block" }}>
                                    Set context window size for the LLM.
                                </Typography>
                                <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                                    Search for your model here - <a
                                    href="https://llm-explorer.com/list/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    style={{ color: '#90caf9', marginLeft: '4px', textDecoration: 'underline' }}
                                >
                                    llm-explorer.com
                                </a>
                                    <br /> and plug in numbers only from column "Context Len" e.g. 8000 for 8k, 128000 for 128k: <br />
                                    <br /> the larger the context window, the more context the LLM can consider, but it may also increase latency and cost. Adjust according to your needs.
                                </Typography>
                            </Box>
                        }
                        placement="top"
                        open={tooltipOpen}
                        onOpen={() => setTooltipOpen(true)}
                        onClose={() => {
                            if (!showContextHighlight) {
                                setTooltipOpen(false);
                            }
                        }}
                    >
                        <TextField
                            size="small"
                            label="Ctx size"
                            type="number"
                            value={contextWindow}
                            onChange={(e) => handleContextWindowChange(parseInt(e.target.value) || 0)}
                            onClick={() => {
                                setShowContextHighlight(false);
                                setTooltipOpen(false);
                            }}
                            onFocus={() => {
                                setShowContextHighlight(false);
                                setTooltipOpen(false);
                            }}
                            sx={{
                                width: 'auto',
                                minWidth: 100,
                                maxWidth: 100,
                                '& .MuiOutlinedInput-root': {
                                    transition: 'all 0.3s ease',
                                    backgroundColor: showContextHighlight ? 'rgba(255, 235, 59, 0.1)' : 'transparent',
                                    ...headerSelectOutlineSx,
                                    '& fieldset': {
                                        borderColor: showContextHighlight ? 'primary.main' : 'transparent',
                                        borderWidth: showContextHighlight ? '2px' : '1px',
                                    },
                                },
                            }}
                            slotProps={{ htmlInput: { min: 1, step: 1, style: { textAlign: 'right' } } }}
                        />
                    </Tooltip>
                    <FormControl fullWidth size="small">
                        <InputLabel id="llm-label">Select LLM</InputLabel>
                        <Select
                            labelId="llm-label"
                            id="llm-select"
                            value={llmModel}
                            label="Select LLM"
                            onChange={(e) => handleLlmModelChange(e.target.value)}
                            sx={{
                                ...headerSelectOutlineSx,
                            }}
                        >
                            {availableModels.map(m => (
                                <MenuItem key={m} value={m}>{m}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Box>
            </Box>

            {/* Messages List */}
            <List
                component="div"
                ref={messageListRef}
                sx={{ flexGrow: 1, minHeight: 0, overflow: 'auto', borderRadius: 1, mb: 1, p: 1 }}
            >
                <Box
                    sx={{
                        height: `${messageVirtualizer.getTotalSize()}px`,
                        width: '100%',
                        position: 'relative',
                    }}
                >
                    {messageVirtualizer.getVirtualItems().map((virtualItem) => {
                        const msg = messages[virtualItem.index];
                        if (!msg) return null;
                        const liveForMessage = liveExecution?.messageId === msg.id ? liveExecution : null;
                        const showAgentRunDebug = msg.role === MessageRole.Assistant && Boolean(msg.agent_run_id || liveForMessage);
                        return (
                            <Box
                                key={virtualItem.key}
                                data-index={virtualItem.index}
                                ref={(node: HTMLDivElement | null) => {
                                    messageVirtualizer.measureElement(node);
                                    messageRefs.current[virtualItem.index] = node?.firstElementChild as HTMLLIElement | null;
                                }}
                                sx={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    transform: `translateY(${virtualItem.start}px)`,
                                }}
                            >
                                <ChatMessageItem
                                    msg={msg}
                                    index={virtualItem.index}
                                    isRecollected={recollectedIds.has(msg.id)}
                                    isActive={activeMessageIndex === virtualItem.index}
                                    isEditing={editingMessageId === msg.id}
                                    isOlderQuestion={msg.role === MessageRole.User && latestUserMessageId !== null && msg.id !== latestUserMessageId}
                                    copied={copiedId === msg.id}
                                    liveForMessage={liveForMessage}
                                    showAgentRunDebug={showAgentRunDebug}
                                    forking={forkingMessageId === msg.id}
                                    loading={loading}
                                    isTestRuntime={isTestRuntime}
                                    onCopy={handleCopy}
                                    onReadAloud={handleReadAloud}
                                    onForkFromMessage={handleForkFromMessage}
                                    onEditQuestion={handleEditQuestion}
                                    onDeleteMessage={handleDeleteMessage}
                                    onOpenAgentRun={handleOpenAgentRun}
                                    formatAgentWorkflowLabel={formatAgentWorkflowLabel}
                                />
                            </Box>
                        );
                    })}
                </Box>
                {false && messages.map((msg, idx) => {
                    const isRecollected = recollectedIds.has(msg.id);
                    const isUser = msg.role === MessageRole.User;
                    const liveForMessage = liveExecution?.messageId === msg.id ? liveExecution : null;
                    const showAgentRunDebug = msg.role === MessageRole.Assistant && Boolean(msg.agent_run_id || liveForMessage);
                    const isEditingThisMessage = editingMessageId === msg.id;
                    const isOlderQuestion = isUser && latestUserMessageId !== null && msg.id !== latestUserMessageId;
                    const editTooltip = isEditingThisMessage
                        ? "Editing this question"
                        : isOlderQuestion
                            ? "Edit and ask again at the end"
                            : "Edit question";
                    return (
                        <ListItem
                            key={msg.id}
                            ref={el => { messageRefs.current[idx] = el; }}
                            alignItems="flex-start"
                            sx={{
                                flexDirection: 'column',
                                alignItems: isUser ? 'flex-end' : 'flex-start',
                                px: 0,
                                py: 0.5
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 1.5,
                                    bgcolor: isUser
                                        ? theme.palette.mode === 'dark'
                                            ? theme.palette.primary.dark
                                            : theme.palette.primary.main
                                        : theme.palette.mode === 'dark'
                                            ? theme.palette.background.paper
                                            : theme.palette.grey[100],
                                    color: isUser
                                        ? theme.palette.getContrastText(theme.palette.primary.main)
                                        : theme.palette.text.primary,
                                    width: showAgentRunDebug ? `calc(100% - ${theme.spacing(6)})` : 'fit-content',
                                    maxWidth: isUser ? '90%' : `calc(100% - ${theme.spacing(6)})`,
                                    minWidth: 0,
                                    overflowWrap: 'anywhere',
                                    wordBreak: 'break-word',
                                    boxShadow: activeMessageIndex === idx
                                        ? '0 0 10px rgba(255, 255, 0, 0.4)'
                                        : isRecollected
                                            ? '0 0 10px rgba(156, 39, 176, 0.5)'
                                            : 'none',
                                    border: (isRecollected || isEditingThisMessage) ? '2px solid' : 'none',
                                    borderColor: isEditingThisMessage
                                        ? 'warning.main'
                                        : isRecollected
                                            ? 'secondary.main'
                                            : 'transparent',
                                    borderRadius: '12px',
                                    transition: 'all 0.2s ease',
                                    cursor: 'default',
                                    position: 'relative',
                                    '&:hover .message-actions': {
                                        opacity: 1
                                    }
                                }}
                            >
                                {/* Recollection indicator */}
                                {isRecollected && (
                                    <Chip
                                        icon={<MemoryIcon fontSize="small" />}
                                        label="Used as context"
                                        size="small"
                                        color="secondary"
                                        sx={{
                                            position: 'absolute',
                                            top: -10,
                                            left: 10,
                                            height: 20,
                                            fontSize: '0.65rem'
                                        }}
                                    />
                                )}

                                {/* Action buttons */}
                                <Box
                                    className="message-actions"
                                    sx={{
                                        position: 'absolute',
                                        top: 8,
                                        right: 8,
                                        display: 'flex',
                                        gap: 0.25,
                                        opacity: 0,
                                        transition: 'opacity 0.2s ease',
                                        bgcolor: isUser
                                            ? theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.2)'
                                            : theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)',
                                        backdropFilter: 'blur(4px)',
                                        borderRadius: '20px',
                                        p: 0.4,
                                        boxShadow: 1,
                                        zIndex: 10,
                                        '&:hover': { opacity: 1 }
                                    }}
                                >
                                    <Tooltip title={copiedId === msg.id ? "Copied!" : "Copy message"}>
                                        <IconButton
                                            size="small"
                                            onClick={() => handleCopy(typeof msg.content === 'string' ? msg.content : String(msg.content ?? ''), msg.id)}
                                            sx={{
                                                color: 'inherit',
                                                p: 0.5,
                                                '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                            }}
                                        >
                                            {copiedId === msg.id ? <CheckIcon fontSize="small" /> : <ContentCopyIcon fontSize="small" />}
                                        </IconButton>
                                    </Tooltip>
                                    <Tooltip title="Read aloud">
                                        <IconButton
                                            size="small"
                                            onClick={() => handleReadAloud(idx)}
                                            sx={{
                                                color: isUser ? 'inherit' : (activeMessageIndex === idx ? 'primary.main' : 'inherit'),
                                                p: 0.5,
                                                '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                            }}
                                        >
                                            <VolumeUpIcon fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                    {!isTestRuntime && !isUser && (
                                        <Tooltip title="Fork from here">
                                            <span>
                                                <IconButton
                                                    size="small"
                                                    onClick={(e) => handleForkFromMessage(msg.id, e)}
                                                    disabled={forkingMessageId === msg.id || loading}
                                                    sx={{
                                                        color: 'inherit',
                                                        p: 0.5,
                                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                                    }}
                                                >
                                                    {forkingMessageId === msg.id ? <CircularProgress size={14} /> : <CallSplitIcon fontSize="small" />}
                                                </IconButton>
                                            </span>
                                        </Tooltip>
                                    )}
                                    {!isTestRuntime && isUser && (
                                        <Tooltip title={editTooltip}>
                                            <span>
                                                <IconButton
                                                    size="small"
                                                    onClick={(e) => handleEditQuestion(msg, e)}
                                                    disabled={loading}
                                                    sx={{
                                                        color: isEditingThisMessage ? 'warning.light' : 'inherit',
                                                        p: 0.5,
                                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                                    }}
                                                >
                                                    <EditIcon fontSize="small" />
                                                </IconButton>
                                            </span>
                                        </Tooltip>
                                    )}
                                    {!isTestRuntime && (
                                        <Tooltip title="Delete message">
                                            <span>
                                                <IconButton
                                                    size="small"
                                                    onClick={(e) => handleDeleteMessage(msg.id, e)}
                                                    disabled={loading || isEditingThisMessage}
                                                    sx={{
                                                        color: 'inherit',
                                                        p: 0.5,
                                                        '&:hover': { color: 'error.main' },
                                                        '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                                    }}
                                                >
                                                    <DeleteIcon fontSize="small" />
                                                </IconButton>
                                            </span>
                                        </Tooltip>
                                    )}
                                </Box>

                                {showAgentRunDebug && (
                                    <Box sx={{ mb: 1 }}>
                                        <Button
                                            size="small"
                                            variant="text"
                                            startIcon={<RouteIcon fontSize="small" />}
                                            onClick={() => void handleOpenAgentRun(msg)}
                                            sx={{ minHeight: 26, px: 0.5, textTransform: 'none' }}
                                        >
                                            {liveForMessage?.canceling
                                                ? 'Stopping after current step…'
                                                : liveForMessage?.running
                                                    ? 'Open live trace'
                                                    : `Open trace · ${formatAgentWorkflowLabel(msg)}${msg.agent_route ? ` · ${msg.agent_route}` : ''}`}
                                        </Button>
                                    </Box>
                                )}
                                {msg.role === MessageRole.Assistant && msg.reasoning_available && msg.reasoning && (
                                    <Box sx={{ mb: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>Reasoning</summary>
                                            <Typography
                                                variant="caption"
                                                component="pre"
                                                sx={{ mt: 0.75, mb: 0, p: 1, borderRadius: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word', bgcolor: 'rgba(0,0,0,0.04)' }}
                                            >
                                                {msg.reasoning}
                                            </Typography>
                                        </details>
                                    </Box>
                                )}

                                <Typography variant="body2" component="div" sx={{
                                    cursor: 'text',
                                    pr: 2, // Add some padding to avoid immediate overlap with icons if possible
                                    minWidth: 0,
                                    maxWidth: '100%',
                                    overflowWrap: 'anywhere',
                                    wordBreak: 'break-word',
                                    '& p': { m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& p:last-child': { mb: 0 },
                                    '& ul, & ol': { pl: 2, m: 0, mb: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& li': { mb: 0.5, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1, overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& blockquote': { m: 0, pl: 1.5, borderLeft: '3px solid', borderColor: 'divider', overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& a': { overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& code': { bgcolor: msg.role === MessageRole.User ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace', overflowWrap: 'anywhere', wordBreak: 'break-word' },
                                    '& pre': { maxWidth: '100%', bgcolor: msg.role === MessageRole.User ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 },
                                    '& pre code': { overflowWrap: 'normal', wordBreak: 'normal' },
                                    '& table': { display: 'block', maxWidth: '100%', overflowX: 'auto', borderCollapse: 'collapse', mb: 1 },
                                    '& th, & td': { border: '1px solid', borderColor: 'divider', px: 0.75, py: 0.5 }
                                }}>
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {typeof msg.content === 'string' ? msg.content : String(msg.content ?? '')}
                                    </ReactMarkdown>
                                </Typography>
                                {msg.role === MessageRole.User && msg.rewritten_query && (
                                    <Box sx={{ mt: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                                <Tooltip
                                                    title="This question was rewritten with thread context for retrieval."
                                                    arrow
                                                >
                                                    <span>Rewritten for context</span>
                                                </Tooltip>
                                            </summary>
                                            <Typography
                                                variant="caption"
                                                sx={{
                                                    mt: 1,
                                                    display: 'block',
                                                    fontStyle: 'italic',
                                                    opacity: 0.9,
                                                    p: 1,
                                                    borderRadius: 1,
                                                    bgcolor: 'rgba(255,255,255,0.1)',
                                                    minWidth: 0,
                                                    maxWidth: '100%',
                                                    overflowWrap: 'anywhere',
                                                    wordBreak: 'break-word'
                                                }}
                                            >
                                                {msg.rewritten_query}
                                            </Typography>
                                        </details>
                                    </Box>
                                )}
                                {msg.role === MessageRole.Assistant && msg.web_sources && msg.web_sources.length > 0 && (
                                    <Box sx={{ mt: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                                🌐 Web sources used ({msg.web_sources.length})
                                            </summary>
                                            <Box sx={{ mt: 0.75, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
                                                {msg.web_sources.map((source, i) => (
                                                    <Box
                                                        key={i}
                                                        sx={{
                                                            p: 1,
                                                            borderRadius: 1,
                                                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)',
                                                            borderLeft: '3px solid',
                                                            borderColor: 'primary.light',
                                                        }}
                                                    >
                                                        {source.url ? (
                                                            <Typography
                                                                variant="caption"
                                                                component="a"
                                                                href={source.url}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                sx={{
                                                                    color: 'primary.main',
                                                                    display: 'block',
                                                                    fontWeight: 600,
                                                                    textDecoration: 'none',
                                                                    mb: 0.25,
                                                                    '&:hover': { textDecoration: 'underline' },
                                                                }}
                                                            >
                                                                {source.title || source.url}
                                                            </Typography>
                                                        ) : (
                                                            <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.25 }}>
                                                                {source.title || 'Web result'}
                                                            </Typography>
                                                        )}
                                                        {source.url && (
                                                            <Typography
                                                                variant="caption"
                                                                sx={{ color: 'text.secondary', display: 'block', wordBreak: 'break-all', mb: 0.25 }}
                                                            >
                                                                {source.url}
                                                            </Typography>
                                                        )}
                                                        <Typography
                                                            variant="caption"
                                                            sx={{ display: 'block', opacity: 0.85, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
                                                        >
                                                            {source.text}
                                                        </Typography>
                                                    </Box>
                                                ))}
                                            </Box>
                                        </details>
                                    </Box>
                                )}
                            </Paper>
                        </ListItem>
                    );
                })}
            </List>

            {/* Input Area */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, flexShrink: 0, minHeight: 0 }}>

                {(pendingMemoryCandidates.length > 0 || memoryCandidateError) && (
                    <Paper variant="outlined" sx={{ p: 1, borderRadius: 1, bgcolor: 'background.paper' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: pendingMemoryCandidates.length > 0 ? 1 : 0 }}>
                            <MemoryIcon fontSize="small" color="primary" />
                            <Typography variant="caption" sx={{ fontWeight: 700, flex: 1 }}>
                                Pending memory review
                            </Typography>
                            <Chip size="small" label={pendingMemoryCandidates.length} sx={{ height: 20 }} />
                        </Box>
                        {memoryCandidateError && (
                            <Typography variant="caption" color="error" sx={{ display: 'block', mb: pendingMemoryCandidates.length > 0 ? 1 : 0 }}>
                                {memoryCandidateError}
                            </Typography>
                        )}
                        {pendingMemoryCandidates.slice(0, 3).map(candidate => {
                            const busy = memoryCandidateActionId === candidate.id;
                            return (
                                <Box key={candidate.id} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 0.5, borderTop: 1, borderColor: 'divider', '&:first-of-type': { borderTop: 0 } }}>
                                    <Box sx={{ minWidth: 0, flex: 1 }}>
                                        <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>
                                            {candidate.content}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {candidate.proposed_scope_type} memory · {Math.round((candidate.confidence || 0) * 100)}%
                                        </Typography>
                                    </Box>
                                    <Tooltip title="Approve memory">
                                        <span>
                                            <IconButton size="small" color="primary" disabled={busy} onClick={() => handleResolveMemoryCandidate(candidate, 'approved')}>
                                                {busy ? <CircularProgress size={16} /> : <CheckIcon fontSize="small" />}
                                            </IconButton>
                                        </span>
                                    </Tooltip>
                                    <Tooltip title="Reject memory">
                                        <span>
                                            <IconButton size="small" color="default" disabled={busy} onClick={() => handleResolveMemoryCandidate(candidate, 'rejected')}>
                                                <CloseIcon fontSize="small" />
                                            </IconButton>
                                        </span>
                                    </Tooltip>
                                </Box>
                            );
                        })}
                    </Paper>
                )}

                {showDecisionPanel && (
                    <Box
                        sx={{
                            display: 'flex',
                            flexDirection: 'column',
                            mb: 1,
                            bgcolor: 'background.default',
                            borderRadius: 1,
                            maxHeight: `calc(100dvh * ${clarificationPanelRatio})`,
                            minHeight: 0,
                            overflow: 'hidden',
                            borderTop: '1px solid',
                            borderColor: 'divider',
                        }}
                    >
                        <Box
                            onPointerDown={handleClarificationResizeStart}
                            role="separator"
                            aria-orientation="horizontal"
                            aria-label="Resize decision panel"
                            sx={{
                                flex: '0 0 auto',
                                height: '1.5rem',
                                cursor: 'ns-resize',
                                touchAction: 'none',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'text.secondary',
                                '&::before': {
                                    content: '""',
                                    width: '18%',
                                    minWidth: '2rem',
                                    maxWidth: '5rem',
                                    height: '0.25rem',
                                    borderRadius: 999,
                                    bgcolor: isClarificationResizing ? 'primary.main' : 'divider',
                                },
                                '&:hover::before': {
                                    bgcolor: 'primary.main',
                                },
                            }}
                        />
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1, pb: 1, flexShrink: 0 }}>
                            <Typography variant="caption" sx={{ flex: 1, textAlign: 'center', color: 'text.secondary', fontWeight: 'bold' }}>
                                {clarificationOptions
                                    ? 'I need a bit more clarification. Did you mean one of these?'
                                    : pendingReviewTitle}
                            </Typography>
                            {clarificationOptions && (
                                <Tooltip title="Close clarification options">
                                    <IconButton
                                        size="small"
                                        onClick={() => {
                                            setClarificationOptions(null);
                                        }}
                                        sx={{ flex: '0 0 auto' }}
                                    >
                                        <CloseIcon fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, px: 1, pt: 1, pb: 1, overflowY: 'auto', minHeight: 0 }}>
                            {clarificationOptions && clarificationOptions.map((choice, i) => {
                                const choiceText = clarificationChoiceText(choice.text);
                                const trimmedChoiceText = choiceText.trim();
                                return (
                                    <Box
                                        key={i}
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
                                            label={choice.isOriginal ? 'Original question' : `Option ${i + 1}`}
                                            value={choiceText}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    bgcolor: 'action.hover',
                                                },
                                            }}
                                            onChange={(event) => {
                                                const nextText = event.target.value;
                                                setClarificationOptions(prev => prev?.map((item, index) => (
                                                    index === i ? { ...item, text: nextText } : item
                                                )) ?? null);
                                            }}
                                        />
                                        <Tooltip title="Send this question">
                                            <Box
                                                component="span"
                                                sx={{
                                                    width: '2.5rem',
                                                    display: 'flex',
                                                    justifyContent: 'center',
                                                    flex: '0 0 auto',
                                                }}
                                            >
                                                <IconButton
                                                    color="primary"
                                                    size="medium"
                                                    disabled={!trimmedChoiceText || loading}
                                                    onClick={() => handleSend(
                                                        trimmedChoiceText,
                                                        { bypassClarification: choice.isOriginal },
                                                    )}
                                                    sx={{ mt: 0.25 }}
                                                >
                                                    <SendIcon fontSize="medium" />
                                                </IconButton>
                                            </Box>
                                        </Tooltip>
                                    </Box>
                                );
                            })}
                            {pendingReviewInterrupt && (
                                <>
                                    {(pendingReviewInterrupt.prompt || pendingReviewInterrupt.body) && (
                                        <Typography variant="caption" sx={{ color: 'text.secondary', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                            {pendingReviewInterrupt.prompt || pendingReviewInterrupt.body}
                                        </Typography>
                                    )}
                                    {pendingReviewInterrupt.proposed_final_answer && (
                                        <TextField
                                            fullWidth
                                            size="small"
                                            multiline
                                            minRows={5}
                                            maxRows={12}
                                            label={pendingReviewActions.includes(AgentRunResumeActionValue.Edit) ? 'Final answer draft' : 'Proposed final answer'}
                                            value={humanReviewEditText}
                                            disabled={!pendingReviewActions.includes(AgentRunResumeActionValue.Edit) || Boolean(humanReviewSubmitting)}
                                            onChange={(event) => setHumanReviewEditText(event.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    bgcolor: 'action.hover',
                                                },
                                            }}
                                        />
                                    )}
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                                        {pendingReviewActions.includes(AgentRunResumeActionValue.Approve) && (
                                            <Button
                                                size="small"
                                                variant="contained"
                                                startIcon={<CheckIcon fontSize="inherit" />}
                                                disabled={Boolean(humanReviewSubmitting)}
                                                onClick={() => handleHumanReviewAction(AgentRunResumeActionValue.Approve)}
                                            >
                                                {humanReviewSubmitting === AgentRunResumeActionValue.Approve ? 'Approving...' : approveLabel}
                                            </Button>
                                        )}
                                        {pendingReviewActions.includes(AgentRunResumeActionValue.Edit) && (
                                            <Button
                                                size="small"
                                                variant="contained"
                                                color="secondary"
                                                startIcon={<EditIcon fontSize="inherit" />}
                                                disabled={Boolean(humanReviewSubmitting) || !humanReviewEditText.trim()}
                                                onClick={() => handleHumanReviewAction(AgentRunResumeActionValue.Edit)}
                                            >
                                                {humanReviewSubmitting === AgentRunResumeActionValue.Edit ? 'Saving...' : 'Save edit'}
                                            </Button>
                                        )}
                                        {pendingReviewActions.includes(AgentRunResumeActionValue.ContinueWithout) && (
                                            <Button
                                                size="small"
                                                variant="outlined"
                                                disabled={Boolean(humanReviewSubmitting)}
                                                onClick={() => handleHumanReviewAction(AgentRunResumeActionValue.ContinueWithout)}
                                            >
                                                {humanReviewSubmitting === AgentRunResumeActionValue.ContinueWithout ? 'Continuing...' : continueWithoutLabel}
                                            </Button>
                                        )}
                                        {pendingReviewActions.includes(AgentRunResumeActionValue.Reject) && (
                                            <Button
                                                size="small"
                                                variant="outlined"
                                                color="error"
                                                startIcon={<CloseIcon fontSize="inherit" />}
                                                disabled={Boolean(humanReviewSubmitting)}
                                                onClick={() => handleHumanReviewAction(AgentRunResumeActionValue.Reject)}
                                            >
                                                {humanReviewSubmitting === AgentRunResumeActionValue.Reject ? 'Rejecting...' : 'Reject'}
                                            </Button>
                                        )}
                                    </Box>
                                    {humanReviewError && (
                                        <Typography variant="caption" color="error">
                                            {humanReviewError}
                                        </Typography>
                                    )}
                                </>
                            )}
                        </Box>
                    </Box>
                )}

                {editingMessageId && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, px: 1 }}>
                        <Chip
                            icon={<EditIcon sx={{ fontSize: '0.9rem !important' }} />}
                            label="Editing question"
                            size="small"
                            color="warning"
                            sx={{ maxWidth: 'calc(100% - 3rem)', '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' } }}
                        />
                        <Tooltip title="Cancel edit">
                            <IconButton
                                size="small"
                                onClick={cancelQuestionEdit}
                                disabled={loading}
                                sx={{ flex: '0 0 auto' }}
                            >
                                <CloseIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                )}

                <ChatComposer
                    inputRef={composerInputRef}
                    seedText={composerSeed}
                    seedVersion={composerSeedVersion}
                    loading={loading}
                    llmModel={llmModel}
                    isLlmModelValid={isLlmModelValid}
                    isLlmToolsSupported={isLlmToolsSupported}
                    isEmbeddingModelValid={isEmbeddingModelValid}
                    indexingStatus={indexingStatus}
                    liveExecution={liveExecution}
                    isTestRuntime={isTestRuntime}
                    onSubmit={(text) => void handleSend(text)}
                    onStop={handleStopChat}
                    onOpenSettings={handleOpenThreadSettings}
                />
            </Box>

            <ChatSettingsDialog
                open={settingsDialogOpen}
                onClose={handleCloseThreadSettings}
                onSave={handleSaveThreadSettings}
                saving={savingSettings}
                description={isTestRuntime
                    ? 'Initialized from the selected thread. Changes apply only to this temporary Builder Test session and are not saved to the thread.'
                    : undefined}
                saveLabel={isTestRuntime ? 'Apply to test' : undefined}
                replans={replans}
                replansLimit={replansLimit}
                useReranker={useReranker}
                useProjectMemory={useProjectMemory}
                useGlobalMemory={useGlobalMemory}
                projectAllowsGlobalMemory={projectAllowsGlobalMemory}
                agentWorkflowId={agentWorkflowId}
                agentWorkflowIsCustom={!isBuiltinAgentWorkflow(agentWorkflows, agentWorkflowId)}
                agentWorkflows={agentWorkflows}
                systemRole={systemRole}
                toolInstructions={toolInstructions}
                customInstructions={customInstructions}
                toolCatalog={toolCatalog}
                effectiveToolInstructions={effectiveToolInstructions}
                promptPreview={promptPreview}
                onReplansChange={(value) => setReplans(value)}
                onRerankerChange={(checked) => setUseReranker(checked)}
                onProjectMemoryChange={(checked) => setUseProjectMemory(checked)}
                onGlobalMemoryChange={(checked) => setUseGlobalMemory(checked)}
                onAgentWorkflowChange={(value) => {
                    setAgentWorkflowId(value);
                }}
                onAgentWorkflowMenuOpen={refreshAgentWorkflows}
                onSystemRoleChange={(value) => setSystemRole(value)}
                onToolInstructionChange={(toolId, value) =>
                    setToolInstructions((prev) => ({
                        ...prev,
                        [toolId]: value,
                    }))
                }
                onCustomInstructionsChange={(value) => setCustomInstructions(value)}
                onResetAll={resetAllSettingsToDefault}
                onResetSystemRole={resetSystemRoleToDefault}
                onResetToolInstruction={resetToolInstructionToDefault}
                onResetCustomInstructions={resetCustomInstructionsToDefault}
            />
            <ThreadForkDialog
                open={Boolean(forkDialogMessageId)}
                sourceThread={activeThread}
                projects={forkProjects}
                fromMessageId={forkDialogMessageId}
                submitting={Boolean(forkingMessageId)}
                onClose={() => setForkDialogMessageId(null)}
                onSubmit={submitMessageFork}
            />
        </Paper>
    );
};

const ChatInterface: React.FC<ChatInterfaceProps> = (props) => {
    return <PersistentChatInterface {...props} />;
};

export default ChatInterface;
