import React, { useState, useEffect, useRef, useMemo } from 'react';
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
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import MemoryIcon from '@mui/icons-material/Memory';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import SettingsIcon from '@mui/icons-material/Settings';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import CheckIcon from '@mui/icons-material/Check';
import dynamic from 'next/dynamic';
import remarkGfm from 'remark-gfm';
const ReactMarkdown = dynamic(() => import('react-markdown'), { ssr: false });
import { splitIntoSentences, stripMarkdown } from '../lib/sentence-utils';
import {
    Thread,
    Message,
    WebSource,
    PromptToolDefinition,
    threadChat,
    getThreadMessages,
    deleteMessage,
    getThreadIndexStatus,
    getThreadSettings,
    updateThreadSettings,
    getPromptTools,
    getPromptPreview
} from '../lib/api';
import { withPollingRetry, withRetry } from '../lib/retry-utils';
import { fetchAvailableLlmModels, checkLlmModelReady, checkEmbedModelReady } from '../lib/models-api';
import ChatSettingsDialog from './ChatSettingsDialog';

interface ChatMessage extends Message {
    isRecollected?: boolean;
    reasoning?: string;
    reasoning_available?: boolean;
    reasoning_format?: 'structured' | 'tagged_text' | 'none';
    rewritten_query?: string;
    web_sources?: WebSource[];
}

interface ChatInterfaceProps {
    ragApiUrl?: string;
    activeThread: Thread | null;
    chatSentences: any[];
    setChatSentences: (sentences: any[]) => void;
    currentChatId: number | null;
    activeSource: 'pdf' | 'chat';
    onJump: (id: number) => void;
    onResetChatId?: () => void;
    onThreadUpdate?: () => void;
    darkMode?: boolean;
    autoScroll?: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    ragApiUrl: ragApiUrlProp,
    activeThread,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump,
    onThreadUpdate,
    onResetChatId,
    darkMode = false,
    autoScroll = true
}) => {
    const ragApiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!ragApiUrl) {
        console.error("ERROR: NEXT_PUBLIC_API_URL environment variable is not set. Please configure it in docker-compose.yml");
    }
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const theme = useTheme();
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    const [indexingStatus, setIndexingStatus] = useState<'checking' | 'indexing' | 'ready' | 'blocked' | 'error'>('checking');
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [contextWindow, setContextWindow] = useState<number>(0);
    const [maxIterations, setMaxIterations] = useState(0);
    const [defaultMaxIterations, setDefaultMaxIterations] = useState(0);
    const [minMaxIterations, setMinMaxIterations] = useState<number | null>(null);
    const [maxMaxIterations, setMaxMaxIterations] = useState<number | null>(null);
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
    const [clarificationOptions, setClarificationOptions] = useState<string[] | null>(null);
    const [useIntentAgent, setUseIntentAgent] = useState(true);
    const [intentAgentMaxIterations, setIntentAgentMaxIterations] = useState(1);
    const [defaultIntentAgentMaxIterations, setDefaultIntentAgentMaxIterations] = useState(1);
    const [reasoningMode, setReasoningMode] = useState(true);
    const [defaultReasoningMode, setDefaultReasoningMode] = useState(true);
    const [useReranker, setUseReranker] = useState(true);
    const [defaultUseReranker, setDefaultUseReranker] = useState(true);

    // Model selection
    const [llmModel, setLlmModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);

    const [isLlmModelValid, setIsLlmModelValid] = useState<boolean | null>(true);
    const [isLlmToolsSupported, setIsLlmToolsSupported] = useState<boolean | null>(null);
    const [isEmbedModelValid, setIsEmbedModelValid] = useState<boolean | null>(null);
    const [copiedId, setCopiedId] = useState<string | null>(null);

    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const messageRefs = useRef<{ [key: number]: HTMLLIElement | null }>({});
    const lastClarificationIdsRef = useRef<{ userId: string | null; assistantId: string | null } | null>(null);

    // Load messages when thread changes
    useEffect(() => {
        if (activeThread) {
            loadMessages();
            checkIndexStatus();
            loadThreadSettings();
            checkEmbedModelStatus();
        } else {
            setMessages([]);
            setClarificationOptions(null);
            lastClarificationIdsRef.current = null;
            setIndexingStatus('ready');
            setMaxIterations(defaultMaxIterations);
            setSystemRole(defaultSystemRole);
            setToolInstructions({});
            setCustomInstructions(defaultCustomInstructions);
            setUseIntentAgent(true);
            setIntentAgentMaxIterations(defaultIntentAgentMaxIterations);
            setReasoningMode(defaultReasoningMode);
            setIsEmbedModelValid(null);
            setIsLlmToolsSupported(null);
        }
    }, [activeThread?.id, activeThread?.file_count, defaultMaxIterations, defaultSystemRole, defaultCustomInstructions, defaultReasoningMode]);

    useEffect(() => {
        if (activeThread) {
            setClarificationOptions(null);
            lastClarificationIdsRef.current = null;
        }
    }, [activeThread?.id]);

    useEffect(() => {
        const loadTools = async () => {
            try {
                const res = await getPromptTools();
                setToolCatalog(res.tools || []);
                if (res.defaults) {
                    setDefaultMaxIterations(res.defaults.max_iterations);
                    setMinMaxIterations(res.defaults.min_max_iterations);
                    setMaxMaxIterations(res.defaults.max_max_iterations);
                    setDefaultSystemRole(res.defaults.system_role ?? '');
                    setDefaultCustomInstructions(res.defaults.custom_instructions ?? '');
                    setDefaultIntentAgentMaxIterations(res.defaults.intent_agent_max_iterations ?? 1);
                    setDefaultReasoningMode(res.defaults.reasoning_mode ?? true);
                    setDefaultUseReranker(res.defaults.use_reranker ?? true);
                    if (res.defaults.context_window && !localStorage.getItem('last_context_window')) {
                        setContextWindow(res.defaults.context_window);
                    }
                    if (!activeThread) {
                        setMaxIterations(res.defaults.max_iterations);
                        setSystemRole(res.defaults.system_role ?? '');
                        setCustomInstructions(res.defaults.custom_instructions ?? '');
                        setUseIntentAgent(res.defaults.use_intent_agent ?? true);
                        setIntentAgentMaxIterations(res.defaults.intent_agent_max_iterations ?? 1);
                        setReasoningMode(res.defaults.reasoning_mode ?? true);
                        setUseReranker(res.defaults.use_reranker ?? true);
                    }
                }
            } catch (error) {
                console.error('Failed to load prompt tools:', error);
                setToolCatalog([]);
            }
        };
        loadTools();
    }, [activeThread]);

    const loadThreadSettings = async () => {
        if (!activeThread) return;
        try {
            const settings = await getThreadSettings(activeThread.id);
            setMaxIterations(settings.max_iterations ?? defaultMaxIterations);
            setSystemRole(settings.system_role ?? defaultSystemRole);
            setToolInstructions(settings.tool_instructions ?? {});
            setCustomInstructions(settings.custom_instructions ?? defaultCustomInstructions);
            setUseIntentAgent(settings.use_intent_agent ?? true);
            setIntentAgentMaxIterations(settings.intent_agent_max_iterations ?? defaultIntentAgentMaxIterations);
            setReasoningMode(settings.reasoning_mode ?? defaultReasoningMode);
            setUseReranker(settings.use_reranker ?? defaultUseReranker);
        } catch (error) {
            console.error('Failed to load thread settings:', error);
            setMaxIterations(defaultMaxIterations);
            setSystemRole(defaultSystemRole);
            setToolInstructions({});
            setCustomInstructions(defaultCustomInstructions);
            setUseIntentAgent(true);
            setIntentAgentMaxIterations(defaultIntentAgentMaxIterations);
            setReasoningMode(defaultReasoningMode);
            setUseReranker(defaultUseReranker);
        }
    };

    const loadMessages = async () => {
        if (!activeThread) return;
        try {
            const response = await getThreadMessages(activeThread.id);
            setMessages(response.messages.map(m => ({
                ...m,
                content: typeof m.content === 'string' ? m.content : String(m.content ?? ''),
                isRecollected: false,
                rewritten_query: m.role === 'user' ? m.context_compact : undefined,
                web_sources: m.role === 'assistant' ? (m.web_sources || []) : undefined,
            })));
        } catch (error) {
            console.error('Failed to load messages:', error);
        }
    };

    const checkIndexStatus = async () => {
        if (!activeThread) return;
        try {
            setIndexingStatus('checking');
            const status = await getThreadIndexStatus(activeThread.id);
            // Map rag-service status ('ready' | 'blocked') to UI status
            if (status.status === 'ready') {
                setIndexingStatus('ready');
            } else if (status.status === 'blocked') {
                setIndexingStatus('blocked');
            } else {
                // 'not_ready' means still indexing
                setIndexingStatus('indexing');
            }
            // Update embedding model status from the same endpoint
            if (status.embed_model_ready !== undefined) {
                setIsEmbedModelValid(status.embed_model_ready);
            }
        } catch (error) {
            console.error('Failed to check index status:', error);
            // Set to error state instead of falsely claiming ready
            setIndexingStatus('error');
            // Don't change embed model status - keep previous state
        }
    };

    const checkEmbedModelStatus = async () => {
        if (!activeThread) return;
        
        setIsEmbedModelValid(null);
        
        const result = await withRetry(
            () => checkEmbedModelReady(activeThread.embed_model),
            {
                maxRetries: 2,
                baseDelay: 1000,
                retryableErrors: (error) => true // All errors are retryable for model checks
            }
        );
        
        if (result.success && result.data !== undefined) {
            setIsEmbedModelValid(result.data);
            if (!result.data) {
                setIndexingStatus('blocked');
            }
        } else {
            console.error('Failed to check embed model status after retries:', result.error);
            setIsEmbedModelValid(false);
            setIndexingStatus('error');
        }
    };

    // Sync chatSentences with parent whenever messages change
    useEffect(() => {
        let globalId = 0;
        const result: { id: number; text: string; messageIndex: number }[] = [];
        messages.forEach((msg, mIdx) => {
            const stripped = stripMarkdown(msg.content);
            const sentences = splitIntoSentences(stripped);
            sentences.forEach((s) => {
                result.push({
                    id: globalId++,
                    text: s,
                    messageIndex: mIdx
                });
            });
        });
        setChatSentences(result);
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
                    intent_agent_ran: useIntentAgent,
                    reasoning_mode: reasoningMode,
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
    }, [settingsDialogOpen, contextWindow, systemRole, effectiveToolInstructions, customInstructions, useWebSearch, useIntentAgent, reasoningMode]);

    const resetAllSettingsToDefault = () => {
        const defaults: Record<string, string> = {};
        toolCatalog.forEach((toolDef) => {
            defaults[toolDef.id] = toolDef.default_prompt;
        });
        setMaxIterations(defaultMaxIterations);
        setSystemRole(defaultSystemRole);
        setToolInstructions(defaults);
        setCustomInstructions(defaultCustomInstructions);
        setUseIntentAgent(true);
        setIntentAgentMaxIterations(defaultIntentAgentMaxIterations);
        setReasoningMode(defaultReasoningMode);
        setUseReranker(defaultUseReranker);
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
        if (autoScroll && activeMessageIndex !== null && messageRefs.current[activeMessageIndex]) {
            messageRefs.current[activeMessageIndex]?.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
            });
        }
    }, [activeMessageIndex, currentChatId, autoScroll]);

    const scrollToBottom = () => {
        if (autoScroll) {
            messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, autoScroll]);

    // Fetch available LLM models
    useEffect(() => {
        fetchAvailableLlmModels()
            .then(setAvailableModels)
            .catch(err => {
                console.error("Failed to fetch models", err);
                setAvailableModels([]);
            });
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
        try {
            const result = await checkLlmModelReady(model);
            setIsLlmModelValid(result.ready);
            setIsLlmToolsSupported(result.ready ? result.supportsTools : null);
        } catch (err) {
            setIsLlmModelValid(false);
            setIsLlmToolsSupported(null);
        }
    };

    const handleContextWindowChange = (val: number) => {
        setContextWindow(val);
        if (val > 0 && typeof window !== 'undefined') {
            localStorage.setItem('last_context_window', val.toString());
        }
    };

    const handleWebSearchToggle = () => {
        setUseWebSearch(prev => {
            const next = !prev;
            if (typeof window !== 'undefined') {
                localStorage.setItem('last_use_web_search', next ? '1' : '0');
            }
            return next;
        });
    };

    // Polling for indexing and embedding model status
    useEffect(() => {
        if (!activeThread) return;
        if (indexingStatus === 'blocked' || isEmbedModelValid === false) return;
        // Keep polling if either indexing is in progress OR embed model is not yet valid/checked
        if (indexingStatus !== 'indexing' && isEmbedModelValid === true) return;

        let intervalId: NodeJS.Timeout | null = null;

        const pollStatus = async () => {
            const result = await withPollingRetry(
                () => getThreadIndexStatus(activeThread.id),
                {
                    maxRetries: 3,
                    interval: 5000,
                    shouldStop: (status) => (
                        (status.status === 'ready' && status.embed_model_ready === true) ||
                        status.status === 'blocked' ||
                        status.embed_model_ready === false
                    ),
                    retryableErrors: (error) => true // All errors are retryable for status checks
                }
            );

            if (result.success && result.data) {
                // Update indexing status
                if (result.data.status === 'ready') {
                    setIndexingStatus('ready');
                } else if (result.data.status === 'blocked') {
                    setIndexingStatus('blocked');
                } else {
                    // 'not_ready' means still indexing
                    setIndexingStatus('indexing');
                }

                // Update embedding model status
                if (result.data.embed_model_ready !== undefined) {
                    setIsEmbedModelValid(result.data.embed_model_ready);
                }
            }

            if (result.stopped || !result.success) {
                // Stop polling on success or max retries exceeded
                if (intervalId) {
                    clearInterval(intervalId);
                    intervalId = null;
                }
                
                if (!result.success) {
                    setIndexingStatus('error');
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
    }, [activeThread?.id, indexingStatus, isEmbedModelValid]);

    // Load browser memory settings (last selected LLM, context window, and web search) on mount
    useEffect(() => {
        if (typeof window === 'undefined') return;

        const savedLlm = localStorage.getItem('last_llm_model');
        if (savedLlm && !llmModel) {
            setLlmModel(savedLlm);
            setIsLlmModelValid(null);
            setIsLlmToolsSupported(null);
            checkLlmModelReady(savedLlm).then((result) => {
                setIsLlmModelValid(result.ready);
                setIsLlmToolsSupported(result.ready ? result.supportsTools : null);
            });
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
    }, []);


    const handleSend = async (overrideInput?: string | React.SyntheticEvent) => {
        const textToSend = typeof overrideInput === 'string' ? overrideInput : input.trim();
        if (!textToSend || !llmModel || !activeThread) return;

        const isClarificationSelection = typeof overrideInput === 'string';
        const priorClarificationIds = isClarificationSelection ? lastClarificationIdsRef.current : null;

        setInput('');
        setClarificationOptions(null);
        setLoading(true);

        const tempUserMsg: ChatMessage = {
            id: 'temp-user-' + Date.now(),
            role: 'user',
            content: textToSend,
            created_at: new Date().toISOString()
        };

        setMessages(prev => {
            let updated = prev;
            // Immediate replacement: If this is a clarification selection, remove the previous "ambiguous" turn
            if (isClarificationSelection) {
                const lastIds = priorClarificationIds;
                updated = updated.filter(m => {
                    if (m.id.startsWith('clarify-')) return false;
                    if (lastIds && (m.id === lastIds.userId || m.id === lastIds.assistantId)) return false;
                    return true;
                });
            }
            return [...updated, tempUserMsg];
        });

        try {
            // Call thread chat endpoint directly without explicit warming probe, retries are handled in api.ts.
            const response = await threadChat(
                activeThread.id,
                textToSend,
                llmModel,
                useWebSearch,
                useReranker,
                contextWindow,
                maxIterations,
                systemRole,
                effectiveToolInstructions,
                customInstructions,
                useIntentAgent,
                useIntentAgent ? intentAgentMaxIterations : undefined,
                reasoningMode,
                isClarificationSelection ? true : undefined
            );

            // Handle ambiguous query / clarification options
            if (response.clarification_options) {
                setClarificationOptions(response.clarification_options);
                lastClarificationIdsRef.current = {
                    userId: response.user_message_id ?? null,
                    assistantId: response.assistant_message_id ?? null
                };

                // Add the clarification request to the message list so it's visible in history.
                setMessages(prev => {
                    const updated = prev.filter(m => m.id !== tempUserMsg.id);
                    return [
                        ...updated,
                        {
                            ...tempUserMsg,
                            id: response.user_message_id || ('clarify-user-' + Date.now())
                        },
                        {
                            id: response.assistant_message_id || ('clarify-asst-' + Date.now()),
                            role: 'assistant',
                            content: "I need a bit more clarification.",
                            created_at: new Date().toISOString()
                        }
                    ];
                });
            } else {
                // Normal flow: update messages with real IDs and add assistant response
                setMessages(prev => {
                    const updated = prev.filter(m => m.id !== tempUserMsg.id);
                    const finalMessages = [...updated];

                    finalMessages.push({
                        id: response.user_message_id || ('final-user-' + Date.now()),
                        role: 'user',
                        content: textToSend, // Keep original input
                        rewritten_query: response.rewritten_query && response.rewritten_query !== textToSend ? response.rewritten_query : undefined,
                        created_at: new Date().toISOString()
                    });

                    if (response.assistant_message_id || response.answer) {
                        finalMessages.push({
                            id: response.assistant_message_id || ('assistant-' + Date.now()),
                            role: 'assistant',
                            content: typeof response.answer === 'string' ? response.answer : String(response.answer ?? ''),
                            reasoning: response.reasoning || '',
                            reasoning_available: !!response.reasoning_available,
                            reasoning_format: response.reasoning_format || 'none',
                            web_sources: response.web_sources || [],
                            created_at: new Date().toISOString()
                        });
                    }

                    return finalMessages;
                });

                // Mark recollected messages
                if (response.used_chat_ids && response.used_chat_ids.length > 0) {
                    setRecollectedIds(new Set(response.used_chat_ids));
                    // Clear recollection highlight after 10 seconds
                    setTimeout(() => setRecollectedIds(new Set()), 10000);
                }

                // Notify parent that thread was updated
                if (onThreadUpdate) {
                    onThreadUpdate();
                }
            }

            if (isClarificationSelection) {
                if (priorClarificationIds && (priorClarificationIds.assistantId || priorClarificationIds.userId)) {
                    const deleteTargetId = priorClarificationIds.assistantId || priorClarificationIds.userId;
                    deleteMessage(deleteTargetId).catch(err => {
                        console.warn('Failed to delete clarification message pair:', err);
                    });
                }
                if (!response.clarification_options) {
                    lastClarificationIdsRef.current = null;
                }
            }
        } catch (err: any) {
            console.error(err);
            // Remove optimistic message and show error
            setMessages(prev => {
                const updated = prev.filter(m => m.id !== tempUserMsg.id);
                return [
                    ...updated,
                    tempUserMsg,
                    {
                        id: 'error-' + Date.now(),
                        role: 'assistant',
                        content: `Error: ${err.message || "Failed to get response."}`,
                        created_at: new Date().toISOString()
                    }
                ];
            });
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteMessage = async (messageId: string, event: React.MouseEvent) => {
        event.stopPropagation();
        if (!confirm('Delete this message?')) return;

        // Frontend-only messages (error responses and their paired user messages) are never
        // persisted to the backend, so we remove them directly from local state.
        const isTempId = messageId.startsWith('error-') ||
            messageId.startsWith('temp-user-') ||
            messageId.startsWith('clarify-user-') ||
            messageId.startsWith('clarify-asst-');
        if (isTempId) {
            setMessages(prev => {
                const idx = prev.findIndex(m => m.id === messageId);
                if (idx === -1) return prev;
                const msg = prev[idx];
                // Deleting an error assistant message → also remove the preceding temp user message
                if (msg.role === 'assistant' && idx > 0) {
                    const prevMsg = prev[idx - 1];
                    if (prevMsg.id.startsWith('temp-user-')) {
                        return prev.filter((_, i) => i !== idx && i !== idx - 1);
                    }
                }
                // Deleting a temp user message → also remove the following error assistant message
                if (msg.role === 'user' && idx < prev.length - 1) {
                    const nextMsg = prev[idx + 1];
                    if (nextMsg.id.startsWith('error-')) {
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
    };

    const handleSaveThreadSettings = async () => {
        if (!activeThread) return;
        try {
            setSavingSettings(true);
            const saved = await updateThreadSettings(activeThread.id, {
                max_iterations: Math.max(minMaxIterations ?? 1, Math.min(maxMaxIterations ?? 30, maxIterations)),
                system_role: systemRole,
                tool_instructions: effectiveToolInstructions,
                custom_instructions: customInstructions,
                use_intent_agent: useIntentAgent,
                intent_agent_max_iterations: Math.max(1, Math.min(10, intentAgentMaxIterations)),
                reasoning_mode: reasoningMode,
                use_reranker: useReranker,
            });
            setMaxIterations(saved.max_iterations);
            setSystemRole(saved.system_role);
            setToolInstructions(saved.tool_instructions || {});
            setCustomInstructions(saved.custom_instructions);
            setUseIntentAgent(saved.use_intent_agent ?? true);
            setIntentAgentMaxIterations(saved.intent_agent_max_iterations ?? defaultIntentAgentMaxIterations);
            setReasoningMode(saved.reasoning_mode ?? defaultReasoningMode);
            setUseReranker(saved.use_reranker ?? defaultUseReranker);
            setSettingsDialogOpen(false);
        } catch (error) {
            console.error('Failed to save thread settings:', error);
        } finally {
            setSavingSettings(false);
        }
    };

    const handleCopy = (text: string, messageId: string) => {
        navigator.clipboard.writeText(text);
        setCopiedId(messageId);
        setTimeout(() => setCopiedId(null), 2000);
    };

    const handleReadAloud = (messageIdx: number) => {
        const firstSentence = chatSentences.find(s => s.messageIndex === messageIdx);
        if (firstSentence) onJump(firstSentence.id);
    };

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

    return (
        <Paper elevation={0} sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 1, bgcolor: theme.palette.background.default, color: theme.palette.text.primary, cursor: 'default' }}>
            {/* Header */}
            <Box sx={{ mb: 1, pt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                    <Tooltip title={
                        isEmbedModelValid === null ? "Checking embedding model status..." :
                            isEmbedModelValid ? `Embedding model: ${activeThread.embed_model}` :
                                `Embedding model ${activeThread.embed_model} not found`
                    }>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            {isEmbedModelValid === true ? (
                                <CheckCircleIcon fontSize="medium" color="primary" />
                            ) : isEmbedModelValid === false ? (
                                <ErrorIcon fontSize="medium" color="error" />
                            ) : (
                                <CircularProgress size={20} />
                            )}
                            {isEmbedModelValid === null && (
                                <Typography variant="caption" color="warning.main" sx={{ ml: 0.5, fontWeight: 'bold' }}>CHECKING...</Typography>
                            )}
                            {isEmbedModelValid === false && <Typography variant="caption" color="error" sx={{ fontWeight: 'bold' }}>OFFLINE</Typography>}
                        </Box>
                    </Tooltip>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, maxWidth: '350px', gap: 1 }}>
                    <Tooltip title="AI prompt settings for this thread" placement="top">
                        <IconButton
                            size="small"
                            onClick={() => setSettingsDialogOpen(true)}
                            sx={{ p: 0.5 }}
                        >
                            <SettingsIcon />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title={useWebSearch ? "Internet Search On" : "Internet Search Off"} placement="top">
                        <IconButton
                            size="small"
                            color={useWebSearch ? "primary" : "default"}
                            onClick={handleWebSearchToggle}
                            sx={{ p: 0.5 }}
                        >
                            {useWebSearch ? <WifiTwoToneIcon /> : <WifiOffTwoToneIcon />}
                        </IconButton>
                    </Tooltip>
                    <Tooltip
                        title={
                            <Box sx={{ p: 0.5 }}>
                                <Typography variant="caption" sx={{ display: "block" }}>
                                    Set context window size for the LLM.
                                </Typography>
                                <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                                    Search for your model here and plug in numbers only from column "Context Len" e.g. 8000 for 8k, 128000 for 128k: <br />
                                    <a
                                        href="https://llm-explorer.com/list/"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        style={{ color: '#90caf9', marginLeft: '4px', textDecoration: 'underline' }}
                                    >
                                        llm-explorer.com
                                    </a>
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
                                    '& fieldset': {
                                        borderColor: showContextHighlight ? 'primary.main' : 'rgba(0, 0, 0, 0.23)',
                                        borderWidth: showContextHighlight ? '2px' : '1px',
                                    },
                                    '&:hover fieldset': {
                                        borderColor: 'primary.main',
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
                        >
                            {availableModels.map(m => (
                                <MenuItem key={m} value={m}>{m}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </Box>
            </Box>

            {/* Messages List */}
            <List sx={{ flexGrow: 1, overflow: 'auto', borderRadius: 1, mb: 1, p: 1 }}>
                {messages.map((msg, idx) => {
                    const isRecollected = recollectedIds.has(msg.id);
                    const isUser = msg.role === 'user';
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
                                    maxWidth: '90%',
                                    boxShadow: activeMessageIndex === idx
                                        ? '0 0 10px rgba(255, 255, 0, 0.4)'
                                        : isRecollected
                                            ? '0 0 10px rgba(156, 39, 176, 0.5)'
                                            : 'none',
                                    border: isRecollected ? '2px solid' : 'none',
                                    borderColor: isRecollected ? 'secondary.main' : 'transparent',
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
                                    <Tooltip title="Delete message">
                                        <IconButton
                                            size="small"
                                            onClick={(e) => handleDeleteMessage(msg.id, e)}
                                            sx={{
                                                color: 'inherit',
                                                p: 0.5,
                                                '&:hover': { color: 'error.main' },
                                                '& .MuiSvgIcon-root': { fontSize: '1.1rem' }
                                            }}
                                        >
                                            <DeleteIcon fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                </Box>

                                <Typography variant="body2" component="div" sx={{
                                    cursor: 'text',
                                    pr: 2, // Add some padding to avoid immediate overlap with icons if possible
                                    '& p': { m: 0, mb: 1 },
                                    '& p:last-child': { mb: 0 },
                                    '& ul, & ol': { pl: 2, m: 0, mb: 1 },
                                    '& li': { mb: 0.5 },
                                    '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1 },
                                    '& code': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace' },
                                    '& pre': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 }
                                }}>
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {typeof msg.content === 'string' ? msg.content : String(msg.content ?? '')}
                                    </ReactMarkdown>
                                </Typography>
                                {msg.role === 'user' && msg.rewritten_query && (
                                    <Box sx={{ mt: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                                <Tooltip
                                                    title="To disable rewriting, turn off Intent Agent in the settings."
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
                                                    bgcolor: 'rgba(255,255,255,0.1)'
                                                }}
                                            >
                                                {msg.rewritten_query}
                                            </Typography>
                                        </details>
                                    </Box>
                                )}
                                {msg.role === 'assistant' && msg.reasoning_available && msg.reasoning && (
                                    <Box sx={{ mt: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                                View reasoning trace
                                            </summary>
                                            <Typography
                                                variant="caption"
                                                component="pre"
                                                sx={{
                                                    mt: 1,
                                                    mb: 0,
                                                    p: 1,
                                                    borderRadius: 1,
                                                    whiteSpace: 'pre-wrap',
                                                    wordBreak: 'break-word',
                                                    bgcolor: 'rgba(0,0,0,0.04)'
                                                }}
                                            >
                                                {msg.reasoning}
                                            </Typography>
                                        </details>
                                    </Box>
                                )}
                                {msg.role === 'assistant' && msg.web_sources && msg.web_sources.length > 0 && (
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
                <div ref={messagesEndRef} />
            </List>

            {/* Input Area */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>


                {clarificationOptions && (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1, justifyContent: 'center', p: 1, bgcolor: 'action.hover', borderRadius: 1 }}>
                        <Typography variant="caption" sx={{ width: '100%', textAlign: 'center', mb: 0.5, color: 'text.secondary', fontWeight: 'bold' }}>
                            I need a bit more clarification. Did you mean one of these?
                        </Typography>
                        {clarificationOptions.map((opt, i) => (
                            <Chip
                                key={i}
                                label={opt}
                                onClick={() => handleSend(opt)}
                                color="primary"
                                variant="outlined"
                                size="medium"
                                sx={{
                                    cursor: 'pointer',
                                    height: 'auto',
                                    maxWidth: '100%',
                                    '& .MuiChip-label': {
                                        whiteSpace: 'normal',
                                        display: 'block',
                                        py: 1,
                                        px: 2
                                    },
                                    '&:hover': { bgcolor: 'primary.main', color: 'white' }
                                }}
                            />
                        ))}
                        <Button
                            size="small"
                            variant="text"
                            onClick={() => {
                                setClarificationOptions(null);
                                const lastIds = lastClarificationIdsRef.current;
                                lastClarificationIdsRef.current = null;
                                if (lastIds) {
                                    setMessages(prev => prev.filter(m => {
                                        if (m.id.startsWith('clarify-')) return false;
                                        if (m.id === lastIds.userId || m.id === lastIds.assistantId) return false;
                                        return true;
                                    }));
                                }
                                if (lastIds && (lastIds.assistantId || lastIds.userId)) {
                                    const deleteTargetId = lastIds.assistantId || lastIds.userId;
                                    deleteMessage(deleteTargetId).catch(err => {
                                        console.warn('Failed to delete clarification message pair:', err);
                                    });
                                }
                            }}
                            sx={{ fontSize: '0.7rem' }}
                        >
                            None of these
                        </Button>
                    </Box>
                )}

                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <TextField
                        fullWidth
                        variant="outlined"
                        multiline
                        maxRows={10}
                        placeholder={
                            (indexingStatus === 'blocked' || isEmbedModelValid === false)
                                ? "Blocked: Selected embedding model is unavailable on server."
                                : indexingStatus === 'error'
                                    ? "Connection error. Please refresh to retry."
                                    : indexingStatus !== 'ready'
                                        ? "Indexing your document. This may take a moment..."
                                        : !llmModel
                                            ? "Select LLM model..."
                                            : "Ask a question about your documents..." + (input ? "\n(Shift+Enter for new line)" : "")
                        }
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        disabled={loading || !llmModel || indexingStatus === 'error' || indexingStatus !== 'ready' || isLlmModelValid === false || isLlmToolsSupported === false || (llmModel !== '' && isLlmModelValid === null) || isEmbedModelValid !== true}
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
                    <IconButton
                        color="primary"
                        onClick={handleSend}
                        disabled={loading || !llmModel || indexingStatus === 'error' || indexingStatus !== 'ready' || isLlmModelValid === false || isLlmToolsSupported === false || (llmModel !== '' && isLlmModelValid === null) || isEmbedModelValid !== true}
                    >
                        {(loading || (llmModel && isLlmModelValid === null) || isEmbedModelValid === null || (indexingStatus !== 'ready' && isEmbedModelValid !== false) || indexingStatus === 'error') ? <CircularProgress size={24} /> : <SendIcon />}
                    </IconButton>
                </Box>
            </Box>

            <ChatSettingsDialog
                open={settingsDialogOpen}
                onClose={() => setSettingsDialogOpen(false)}
                onSave={handleSaveThreadSettings}
                saving={savingSettings}
                maxIterations={maxIterations}
                minMaxIterations={minMaxIterations}
                maxMaxIterations={maxMaxIterations}
                reasoningMode={reasoningMode}
                useIntentAgent={useIntentAgent}
                useReranker={useReranker}
                systemRole={systemRole}
                toolInstructions={toolInstructions}
                customInstructions={customInstructions}
                toolCatalog={toolCatalog}
                effectiveToolInstructions={effectiveToolInstructions}
                promptPreview={promptPreview}
                onMaxIterationsChange={(value) => setMaxIterations(value)}
                onReasoningModeChange={(checked) => {
                    setReasoningMode(checked);
                    if (!checked) {
                        setUseIntentAgent(false);
                    }
                }}
                onIntentAgentChange={(checked) => setUseIntentAgent(checked)}
                onRerankerChange={(checked) => setUseReranker(checked)}
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
        </Paper>
    );
};

export default ChatInterface;
