import React, { useState, useEffect, useRef, useMemo } from 'react';
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
    Stack,
    FormControlLabel,
    Switch,
    Tooltip,
    Chip,
    CircularProgress,
} from '@mui/material';
import WifiTwoToneIcon from '@mui/icons-material/WifiTwoTone';
import WifiOffTwoToneIcon from '@mui/icons-material/WifiOffTwoTone';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';
import MemoryIcon from '@mui/icons-material/Memory';
import LockIcon from '@mui/icons-material/Lock';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { splitIntoSentences, stripMarkdown } from '../lib/sentence-utils';
import { 
    Thread, 
    Message,
    threadChat, 
    getThreadMessages, 
    deleteMessage,
    getThreadIndexStatus
} from '../lib/api';

interface ChatMessage extends Message {
    isRecollected?: boolean;
}

interface ChatInterfaceProps {
    ragApiUrl?: string;
    activeThread: Thread | null;
    chatSentences: any[];
    setChatSentences: (sentences: any[]) => void;
    currentChatId: number | null;
    activeSource: 'pdf' | 'chat';
    onJump: (id: number) => void;
    onThreadUpdate?: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
    ragApiUrl = "http://localhost:8001",
    activeThread,
    chatSentences,
    setChatSentences,
    currentChatId,
    activeSource,
    onJump,
    onThreadUpdate
}) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [isModelWarming, setIsModelWarming] = useState(false);
    const [indexingStatus, setIndexingStatus] = useState<'checking' | 'indexing' | 'ready' | 'error'>('checking');
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [recollectedIds, setRecollectedIds] = useState<Set<string>>(new Set());

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
        } else {
            setMessages([]);
            setIndexingStatus('ready');
        }
    }, [activeThread?.id]);

    const loadMessages = async () => {
        if (!activeThread) return;
        try {
            const response = await getThreadMessages(activeThread.id);
            setMessages(response.messages.map(m => ({ ...m, isRecollected: false })));
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

    // Fetch available LLM models
    useEffect(() => {
        fetch(`${ragApiUrl}/models`)
            .then(res => res.json())
            .then(data => {
                if (data.llm_models || data.not_llm_models) {
                    setAvailableModels([...data.llm_models, ...data.not_llm_models]);
                } else if (data.all_models && data.all_models.length > 0) {
                    const ids = data.all_models.map((m: any) => m.id);
                    setAvailableModels(ids);
                } else {
                    throw new Error("No models found");
                }
            })
            .catch(err => {
                console.error("Failed to fetch models", err);
                setAvailableModels([]);
            });
    }, [ragApiUrl]);

    // Validate LLM model when changed
    const handleLlmModelChange = async (model: string) => {
        setLlmModel(model);
        setIsLlmModelValid(null);
        if (!model) return;
        try {
            const res = await fetch(`${ragApiUrl}/health/is_chat_model_ready?model=${encodeURIComponent(model)}`);
            const data = await res.json();
            setIsLlmModelValid(data.ready === true || data.chat_model_ready === true);
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

    const handleSend = async () => {
        if (!input.trim() || !llmModel || !activeThread) return;

        const userContent = input.trim();
        setInput('');
        setLoading(true);
        setIsModelWarming(false);

        // Optimistically add user message
        const tempUserMsg: ChatMessage = { 
            id: 'temp-user-' + Date.now(), 
            role: 'user', 
            content: userContent,
            created_at: new Date().toISOString()
        };
        setMessages(prev => [...prev, tempUserMsg]);

        try {
            // Check model readiness
            const checkModel = async (modelName: string) => {
                try {
                    const res = await fetch(`${ragApiUrl}/health/is_chat_model_ready?model=${encodeURIComponent(modelName)}`);
                    if (!res.ok) return true;
                    const data = await res.json();
                    return data.ready;
                } catch (e) {
                    return true;
                }
            };

            const llmReady = await checkModel(llmModel);
            if (!llmReady) {
                setIsModelWarming(true);
            }

            // Call thread chat endpoint
            const response = await threadChat(
                activeThread.id,
                userContent,
                llmModel,
                useWebSearch
            );

            // Update messages with real IDs and add assistant response
            setMessages(prev => {
                const updated = prev.filter(m => m.id !== tempUserMsg.id);
                return [
                    ...updated,
                    { 
                        id: response.user_message_id, 
                        role: 'user', 
                        content: userContent,
                        created_at: new Date().toISOString()
                    },
                    { 
                        id: response.assistant_message_id, 
                        role: 'assistant', 
                        content: response.answer,
                        created_at: new Date().toISOString()
                    }
                ];
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
            await deleteMessage(messageId);
            setMessages(prev => prev.filter(m => m.id !== messageId));
            if (onThreadUpdate) {
                onThreadUpdate();
            }
        } catch (error) {
            console.error('Failed to delete message:', error);
        }
    };

    if (!activeThread) {
        return (
            <Paper elevation={0} sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 3 }}>
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
        <Paper elevation={0} sx={{ height: '100%', display: 'flex', flexDirection: 'column', p: 1, bgcolor: 'transparent', cursor: 'default' }}>
            {/* Header */}
            <Box sx={{ mb: 1, pt: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold', whiteSpace: 'nowrap' }}>
                        {activeThread.name}
                    </Typography>
                    <Tooltip title={`Embedding model locked: ${activeThread.embed_model}`}>
                        <Chip 
                            icon={<LockIcon fontSize="small" />}
                            label={activeThread.embed_model.split('/').pop()?.split(':')[0] || activeThread.embed_model}
                            size="small"
                            variant="outlined"
                        />
                    </Tooltip>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1, maxWidth: '250px', gap: 1 }}>
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
                    return (
                        <ListItem 
                            key={msg.id} 
                            ref={el => messageRefs.current[idx] = el} 
                            alignItems="flex-start" 
                            sx={{
                                flexDirection: 'column',
                                alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start',
                                px: 0,
                                py: 0.5
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 1.5,
                                    bgcolor: msg.role === 'user' ? 'primary.main' : 'grey.100',
                                    color: msg.role === 'user' ? 'white' : 'text.primary',
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
                                bgcolor: 'white',
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
        </Paper>
    );
};

export default ChatInterface;
