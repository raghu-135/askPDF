import React from 'react';
import {
    Box,
    TextField,
    Button,
    Typography,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Divider,
    FormControlLabel,
    Switch,
    Tooltip,
    IconButton,
} from '@mui/material';
import ReplayIcon from '@mui/icons-material/Replay';
import { PromptToolDefinition } from '../lib/api';

interface ChatSettingsDialogProps {
    open: boolean;
    onClose: () => void;
    onSave: () => void;
    saving: boolean;
    
    // Settings values
    maxIterations: number;
    minMaxIterations: number | null;
    maxMaxIterations: number | null;
    useIntentAgent: boolean;
    useReranker: boolean;
    systemRole: string;
    toolInstructions: Record<string, string>;
    customInstructions: string;
    toolCatalog: PromptToolDefinition[];
    effectiveToolInstructions: Record<string, string>;
    promptPreview: string;
    
    // Change handlers
    onMaxIterationsChange: (value: number) => void;
    onIntentAgentChange: (checked: boolean) => void;
    onRerankerChange: (checked: boolean) => void;
    onSystemRoleChange: (value: string) => void;
    onToolInstructionChange: (toolId: string, value: string) => void;
    onCustomInstructionsChange: (value: string) => void;
    
    // Reset handlers
    onResetAll: () => void;
    onResetSystemRole: () => void;
    onResetToolInstruction: (toolId: string) => void;
    onResetCustomInstructions: () => void;
}

const ChatSettingsDialog: React.FC<ChatSettingsDialogProps> = ({
    open,
    onClose,
    onSave,
    saving,
    maxIterations,
    minMaxIterations,
    maxMaxIterations,
    useIntentAgent,
    useReranker,
    systemRole,
    toolInstructions,
    customInstructions,
    toolCatalog,
    effectiveToolInstructions,
    promptPreview,
    onMaxIterationsChange,
    onIntentAgentChange,
    onRerankerChange,
    onSystemRoleChange,
    onToolInstructionChange,
    onCustomInstructionsChange,
    onResetAll,
    onResetSystemRole,
    onResetToolInstruction,
    onResetCustomInstructions,
}) => {
    return (
        <Dialog
            open={open}
            onClose={() => !saving && onClose()}
            maxWidth="md"
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
                            onClick={onResetAll}
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
                {minMaxIterations !== null && maxMaxIterations !== null ? (
                    <TextField
                        label="Max tool iterations"
                        type="number"
                        value={maxIterations}
                        onChange={(e) => onMaxIterationsChange(Math.max(minMaxIterations, Math.min(maxMaxIterations, parseInt(e.target.value) || minMaxIterations)))}
                        slotProps={{ htmlInput: { min: minMaxIterations, max: maxMaxIterations } }}
                        helperText="Lower is faster; higher allows deeper research."
                    />
                ) : (
                    <Typography variant="caption" color="error">Iteration limits not loaded from server.</Typography>
                )}
                <Divider />
                <Box>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={useIntentAgent}
                                onChange={(e) => onIntentAgentChange(e.target.checked)}
                            />
                        }
                        label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>Intent Agent</Typography>
                            </Box>
                        }
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5, mt: 0.25 }}>
                        Before answering, runs a lightweight LLM pass to detect ambiguity, rewrite follow-up questions
                        into standalone queries, and estimate whether the pre-fetched context is sufficient — reducing
                        unnecessary tool calls.
                    </Typography>
                </Box>
                <Box>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={useReranker}
                                onChange={(e) => onRerankerChange(e.target.checked)}
                            />
                        }
                        label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>Reranker</Typography>
                            </Box>
                        }
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5, mt: 0.25 }}>
                        Reorders retrieved chunks for documents, web results, and chat memory using the reranker model.
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                    <TextField
                        fullWidth
                        label="System role"
                        value={systemRole}
                        onChange={(e) => onSystemRoleChange(e.target.value)}
                        multiline
                        minRows={2}
                        maxRows={4}
                        helperText="Defines the assistant's role for this thread."
                    />
                    <Tooltip title="Reset System role to default">
                        <IconButton
                            size="small"
                            sx={{ mt: 1 }}
                            onClick={onResetSystemRole}
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
                                onToolInstructionChange(toolDef.id, e.target.value)
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
                                onClick={() => onResetToolInstruction(toolDef.id)}
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
                        onChange={(e) => onCustomInstructionsChange(e.target.value)}
                        multiline
                        minRows={4}
                        maxRows={10}
                        helperText="Locked tool and context constraints still apply."
                    />
                    <Tooltip title="Reset Custom instructions to default">
                        <IconButton
                            size="small"
                            sx={{ mt: 1 }}
                            onClick={onResetCustomInstructions}
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
                    slotProps={{ input: { readOnly: true } }}
                />
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose} disabled={saving}>
                    Cancel
                </Button>
                <Button onClick={onSave} variant="contained" disabled={saving}>
                    {saving ? 'Saving...' : 'Save'}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default ChatSettingsDialog;
