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
    Divider,
    FormControlLabel,
    Switch,
    Tooltip,
    Chip,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from '@mui/material';
import WifiTwoToneIcon from '@mui/icons-material/WifiTwoTone';
import WifiOffTwoToneIcon from '@mui/icons-material/WifiOffTwoTone';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import MemoryIcon from '@mui/icons-material/Memory';
import LockIcon from '@mui/icons-material/Lock';
import SettingsIcon from '@mui/icons-material/Settings';
import ReplayIcon from '@mui/icons-material/Replay';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
import { fetchAvailableLlmModels, checkLlmModelReady } from '../lib/chat-utils';

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
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    ragApiUrl = "http://localhost:8001",
    activeThread,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump,
    onThreadUpdate,
    onResetChatId,
    darkMode = false
}) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const theme = useTheme();
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isModelWarming, setIsModelWarming] = useState(false);
    const [indexingStatus, setIndexingStatus] = useState<'checking' | 'indexing' | 'ready' | 'error'>('checking');
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [contextWindow, setContextWindow] = useState(4096);
    const [maxIterations, setMaxIterations] = useState(10);
    const [defaultMaxIterations, setDefaultMaxIterations] = useState(10);
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

    // Model selection
    const [llmModel, setLlmModel] = useState('');
    const [availableModels, setAvailableModels] = useState<string[]>([]);
    const [isLlmModelValid, setIsLlmModelValid] = useState<boolean | null>(null);

    const messagesEndRef = useRef<null | HTMLDivElement>(null);
    const messageRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

    // Load messages when thread changes
    useEffect(() => {
        if (activeThread) {
            loadMessages();
            checkIndexStatus();
            loadThreadSettings();
        } else {
            setMessages([]);
            setIndexingStatus('ready');
            setMaxIterations(defaultMaxIterations);
            setSystemRole(defaultSystemRole);
            setToolInstructions({});
            setCustomInstructions(defaultCustomInstructions);
        }
    }, [activeThread?.id, activeThread?.file_count, defaultMaxIterations, defaultSystemRole, defaultCustomInstructions]);

    useEffect(() => {
        const loadTools = async () => {
            try {
                const res = await getPromptTools();
                setToolCatalog(res.tools || []);
                if (res.defaults) {
                    setDefaultMaxIterations(res.defaults.max_iterations ?? 10);
                    setDefaultSystemRole(res.defaults.system_role ?? '');
                    setDefaultCustomInstructions(res.defaults.custom_instructions ?? '');
                    if (!activeThread) {
                        setMaxIterations(res.defaults.max_iterations ?? 10);
                        setSystemRole(res.defaults.system_role ?? '');
                        setCustomInstructions(res.defaults.custom_instructions ?? '');
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
            setMaxIterations(settings.max_iterations ?? 10);
            setSystemRole(settings.system_role ?? defaultSystemRole);
            setToolInstructions(settings.tool_instructions ?? {});
            setCustomInstructions(settings.custom_instructions ?? defaultCustomInstructions);
        } catch (error) {
            console.error('Failed to load thread settings:', error);
            setMaxIterations(defaultMaxIterations);
            setSystemRole(defaultSystemRole);
            setToolInstructions({});
            setCustomInstructions(defaultCustomInstructions);
        }
    };

    const loadMessages = async () => {
        if (!activeThread) return;
        try {
            const response = await getThreadMessages(activeThread.id);
            setMessages(response.messages.map(m => ({ 
                ...m, 
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
            // Map rag-service status ('ready' | 'not_ready') to UI status
            if (status.status === 'ready') {
                setIndexingStatus('ready');
            } else {
                // 'not_ready' means still indexing
                setIndexingStatus('indexing');
            }
        } catch (error) {
            console.error('Failed to check index status:', error);
            setIndexingStatus('ready'); // Assume ready if check fails
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
    }, [settingsDialogOpen, contextWindow, systemRole, effectiveToolInstructions, customInstructions]);

    const resetAllSettingsToDefault = () => {
        const defaults: Record<string, string> = {};
        toolCatalog.forEach((toolDef) => {
            defaults[toolDef.id] = toolDef.default_prompt;
        });
        setMaxIterations(defaultMaxIterations);
        setSystemRole(defaultSystemRole);
        setToolInstructions(defaults);
        setCustomInstructions(defaultCustomInstructions);
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
        if (activeMessageIndex !== null && messageRefs.current[activeMessageIndex]) {
            messageRefs.current[activeMessageIndex]?.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
            });
        }
    }, [activeMessageIndex, currentChatId]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Fetch available LLM models using chat-utils
    useEffect(() => {
        fetchAvailableLlmModels(ragApiUrl)
            .then(setAvailableModels)
            .catch(err => {
                console.error("Failed to fetch models", err);
                setAvailableModels([]);
            });
    }, [ragApiUrl]);

    // Validate LLM model when changed using chat-utils
    const handleLlmModelChange = async (model: string) => {
        setLlmModel(model);
        setIsLlmModelValid(null);
        if (model) {
            setShowContextHighlight(true);
            setTooltipOpen(true);
        }
        if (!model) return;
        try {
            const valid = await checkLlmModelReady(model, ragApiUrl);
            setIsLlmModelValid(valid);
        } catch (err) {
            setIsLlmModelValid(false);
        }
    };

    // Polling for indexing status
    useEffect(() => {
        if (!activeThread || indexingStatus !== 'indexing') return;

        const intervalId = setInterval(async () => {
            try {
                const status = await getThreadIndexStatus(activeThread.id);
                // Map rag-service status ('ready' | 'not_ready') to UI status
                if (status.status === 'ready') {
                    setIndexingStatus('ready');
                    clearInterval(intervalId);
                }
                // If still 'not_ready', keep polling
            } catch (error) {
                console.error('Index status check failed:', error);
            }
        }, 2000);

        return () => clearInterval(intervalId);
    }, [activeThread?.id, indexingStatus]);

    const handleSend = async (overrideInput?: string) => {
        const textToSend = overrideInput || input.trim();
        if (!textToSend || !llmModel || !activeThread) return;

        setInput('');
        setClarificationOptions(null);
        setLoading(true);
        setIsModelWarming(false);

        // Optimistically add user message
        const tempUserMsg: ChatMessage = { 
            id: 'temp-user-' + Date.now(), 
            role: 'user', 
            content: textToSend,
            created_at: new Date().toISOString()
        };
        setMessages(prev => [...prev, tempUserMsg]);

        try {
            // Check model readiness using chat-utils
            const llmReady = await checkLlmModelReady(llmModel, ragApiUrl);
            if (!llmReady) {
                setIsModelWarming(true);
            }

            // Call thread chat endpoint
            const response = await threadChat(
                activeThread.id,
                textToSend,
                llmModel,
                useWebSearch,
                contextWindow,
                maxIterations,
                systemRole,
                effectiveToolInstructions,
                customInstructions
            );

            // Handle ambiguous query / clarification options
            if (response.clarification_options) {
                setClarificationOptions(response.clarification_options);
                // Remove the optimistic user message — nothing was persisted
                setMessages(prev => prev.filter(m => m.id !== tempUserMsg.id));
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
                            content: response.answer,
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
            setIsModelWarming(false);
        }
    };

    const handleDeleteMessage = async (messageId: string, event: React.MouseEvent) => {
        event.stopPropagation();
        if (!confirm('Delete this message?')) return;

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
                max_iterations: Math.max(1, Math.min(30, maxIterations)),
                system_role: systemRole,
                tool_instructions: effectiveToolInstructions,
                custom_instructions: customInstructions,
            });
            setMaxIterations(saved.max_iterations);
            setSystemRole(saved.system_role);
            setToolInstructions(saved.tool_instructions || {});
            setCustomInstructions(saved.custom_instructions);
            setSettingsDialogOpen(false);
        } catch (error) {
            console.error('Failed to save thread settings:', error);
        } finally {
            setSavingSettings(false);
        }
    };

    if (!activeThread) {
        return (
            <Paper elevation={0} sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3, bgcolor: theme.palette.background.default, color: theme.palette.text.primary }}>
                <Box textAlign="center">
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
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Tooltip title={`Embedding model locked: ${activeThread.embed_model}`}>
                        <Chip 
                            icon={<LockIcon fontSize="small" />}
                            label={activeThread.embed_model.split('/').pop()?.split(':')[0] || activeThread.embed_model}
                            size="small"
                            variant="outlined"
                        />
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
                            onClick={() => setUseWebSearch(v => !v)}
                            sx={{ p: 0.5 }}
                        >
                            {useWebSearch ? <WifiTwoToneIcon /> : <WifiOffTwoToneIcon />}
                        </IconButton>
                    </Tooltip>
                    <Tooltip 
                        title={
                            <Box sx={{ p: 0.5 }}>
                                <Typography variant="caption" display="block">
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
                        interactive
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
                            onChange={(e) => setContextWindow(parseInt(e.target.value) || 0)}
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
                                minWidth: 60, 
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
                            inputProps={{ min: 4096, step: 4096, style: { textAlign: 'right' } }}
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
                        {isLlmModelValid === false && (
                            <Typography color="error" variant="caption" sx={{ ml: 2 }}>
                                Selected model is not a valid chat model.
                            </Typography>
                        )}
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
                            ref={el => messageRefs.current[idx] = el} 
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
                                    '&:hover .delete-btn': {
                                        opacity: 1
                                    }
                                }}
                                onDoubleClick={(e) => {
                                    const firstSentence = chatSentences.find(s => s.messageIndex === idx);
                                    if (firstSentence) onJump(firstSentence.id);
                                    e.stopPropagation();
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
                                
                                {/* Delete button */}
                                <IconButton
                                    className="delete-btn"
                                    size="small"
                                    onClick={(e) => handleDeleteMessage(msg.id, e)}
                                    sx={{
                                        position: 'absolute',
                                        top: 4,
                                        right: 4,
                                        opacity: 0,
                                        transition: 'opacity 0.2s',
                                        bgcolor: 'background.paper',
                                        '&:hover': { bgcolor: 'error.light', color: 'white' }
                                    }}
                                >
                                    <DeleteIcon fontSize="small" />
                                </IconButton>

                                <Typography variant="body2" component="div" sx={{
                                    cursor: 'text',
                                    '& p': { m: 0, mb: 1 },
                                    '& p:last-child': { mb: 0 },
                                    '& ul, & ol': { pl: 2, m: 0, mb: 1 },
                                    '& li': { mb: 0.5 },
                                    '& h1, & h2, & h3': { fontSize: '1.1rem', fontWeight: 'bold', mb: 1, mt: 1 },
                                    '& code': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', px: 0.5, borderRadius: '4px', fontFamily: 'monospace' },
                                    '& pre': { bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.05)', p: 1, borderRadius: '4px', overflowX: 'auto', mb: 1 }
                                }}>
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {msg.content}
                                    </ReactMarkdown>
                                </Typography>
                                {msg.role === 'user' && msg.rewritten_query && (
                                    <Box sx={{ mt: 1 }}>
                                        <details>
                                            <summary style={{ cursor: 'pointer', fontSize: '0.75rem', opacity: 0.8 }}>
                                                Rewritten for context
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
                                                    bgcolor: msg.role === 'user' ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.04)'
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
                {isModelWarming && (
                    <Typography variant="caption" sx={{ color: 'info.main', textAlign: 'center', fontStyle: 'italic' }}>
                        Bringing the AI model online, this may take a moment...
                    </Typography>
                )}

                {indexingStatus !== 'ready' && (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                        <CircularProgress size={16} />
                        <Typography variant="caption" color="info.main">
                            Indexing document...
                        </Typography>
                    </Box>
                )}

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
                                    maxWidth: '300px',
                                    '&:hover': { bgcolor: 'primary.main', color: 'white' } 
                                }}
                            />
                        ))}
                        <Button 
                            size="small" 
                            variant="text" 
                            onClick={() => setClarificationOptions(null)}
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
                            indexingStatus !== 'ready'
                                ? "Indexing your document. This may take a moment..."
                                : !llmModel
                                    ? "Select LLM model..."
                                    : "Ask a question..." + (input ? "\n(Shift+Enter for new line)" : "")
                        }
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        disabled={loading || !llmModel || indexingStatus !== 'ready'}
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
                        disabled={loading || !llmModel || indexingStatus !== 'ready'}
                    >
                        {loading ? <CircularProgress size={24} /> : <SendIcon />}
                    </IconButton>
                </Box>
            </Box>

            <Dialog
                open={settingsDialogOpen}
                onClose={() => !savingSettings && setSettingsDialogOpen(false)}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle>AI Prompt Settings</DialogTitle>
                <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '8px !important' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                            These settings are saved per thread and used by default for every message.
                        </Typography>
                        <Tooltip title="Reset all settings to default">
                            <IconButton
                                size="medium"
                                onClick={resetAllSettingsToDefault}
                                sx={{
                                    width: 36,
                                    height: 36,
                                    border: 1,
                                    borderColor: 'divider',
                                }}
                            >
                                <ReplayIcon fontSize="medium" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                    <TextField
                        label="Max tool iterations"
                        type="number"
                        value={maxIterations}
                        onChange={(e) => setMaxIterations(Math.max(1, Math.min(30, parseInt(e.target.value) || 1)))}
                        inputProps={{ min: 1, max: 30 }}
                        helperText="Lower is faster; higher allows deeper research."
                    />
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                        <TextField
                            fullWidth
                            label="System role"
                            value={systemRole}
                            onChange={(e) => setSystemRole(e.target.value)}
                            multiline
                            minRows={2}
                            maxRows={4}
                            helperText="Defines the assistant's role for this thread."
                        />
                        <Tooltip title="Reset System role to default">
                            <IconButton
                                size="small"
                                sx={{ mt: 1 }}
                                onClick={resetSystemRoleToDefault}
                            >
                                <ReplayIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                        These are the tools available in the app. You can configure how the assistant should use each one.
                    </Typography>
                    {toolCatalog.map((toolDef) => (
                        <Box key={toolDef.id} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                            <TextField
                                fullWidth
                                label={toolDef.display_name}
                                value={effectiveToolInstructions[toolDef.id] || ''}
                                onChange={(e) =>
                                    setToolInstructions((prev) => ({
                                        ...prev,
                                        [toolDef.id]: e.target.value,
                                    }))
                                }
                                multiline
                                minRows={2}
                                maxRows={6}
                                helperText={toolDef.description}
                            />
                            <Tooltip title={`Reset ${toolDef.display_name} to default`}>
                                <IconButton
                                    size="small"
                                    sx={{ mt: 1 }}
                                    onClick={() => resetToolInstructionToDefault(toolDef.id)}
                                >
                                    <ReplayIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    ))}
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                        <TextField
                            fullWidth
                            label="Custom instructions"
                            value={customInstructions}
                            onChange={(e) => setCustomInstructions(e.target.value)}
                            multiline
                            minRows={4}
                            maxRows={10}
                            helperText="Locked tool and context constraints still apply."
                        />
                        <Tooltip title="Reset Custom instructions to default">
                            <IconButton
                                size="small"
                                sx={{ mt: 1 }}
                                onClick={resetCustomInstructionsToDefault}
                            >
                                <ReplayIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                    <TextField
                        label="Full Prompt Preview"
                        value={promptPreview}
                        multiline
                        minRows={14}
                        maxRows={24}
                        InputProps={{ readOnly: true }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setSettingsDialogOpen(false)} disabled={savingSettings}>
                        Cancel
                    </Button>
                    <Button onClick={handleSaveThreadSettings} variant="contained" disabled={savingSettings}>
                        {savingSettings ? 'Saving...' : 'Save'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Paper>
    );
};

export default ChatInterface;
