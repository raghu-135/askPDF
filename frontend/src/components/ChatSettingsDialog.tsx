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
    MenuItem,
    Chip,
} from '@mui/material';
import ReplayIcon from '@mui/icons-material/Replay';
import { AgentWorkflow, PromptToolDefinition } from '../lib/api';

interface ChatSettingsDialogProps {
    open: boolean;
    onClose: () => void;
    onSave: () => void;
    saving: boolean;
    description?: string;
    saveLabel?: string;
    
    // Settings values
    replans: number;
    replansLimit: number | null;
    useReranker: boolean;
    useMemory: boolean;
    useThreadMemory: boolean;
    useProjectMemory: boolean;
    useGlobalMemory: boolean;
    projectAllowsGlobalMemory: boolean;
    agentWorkflowId: string;
    agentWorkflowIsCustom?: boolean;
    agentWorkflows: AgentWorkflow[];
    systemRole: string;
    toolInstructions: Record<string, string>;
    customInstructions: string;
    toolCatalog: PromptToolDefinition[];
    effectiveToolInstructions: Record<string, string>;
    promptPreview: string;
    
    // Change handlers
    onReplansChange: (value: number) => void;
    onRerankerChange: (checked: boolean) => void;
    onMemoryChange: (checked: boolean) => void;
    onThreadMemoryChange: (checked: boolean) => void;
    onProjectMemoryChange: (checked: boolean) => void;
    onGlobalMemoryChange: (checked: boolean) => void;
    onAgentWorkflowChange: (value: string) => void;
    onAgentWorkflowMenuOpen?: () => void | Promise<void>;
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
    description = 'These settings are saved per thread and used by default for every message. Agent workflows are globally available.',
    saveLabel = 'Save',
    replans,
    replansLimit,
    useReranker,
    useMemory,
    useThreadMemory,
    useProjectMemory,
    useGlobalMemory,
    projectAllowsGlobalMemory,
    agentWorkflowId,
    agentWorkflowIsCustom = false,
    agentWorkflows,
    systemRole,
    toolInstructions,
    customInstructions,
    toolCatalog,
    effectiveToolInstructions,
    promptPreview,
    onReplansChange,
    onRerankerChange,
    onMemoryChange,
    onThreadMemoryChange,
    onProjectMemoryChange,
    onGlobalMemoryChange,
    onAgentWorkflowChange,
    onAgentWorkflowMenuOpen,
    onSystemRoleChange,
    onToolInstructionChange,
    onCustomInstructionsChange,
    onResetAll,
    onResetSystemRole,
    onResetToolInstruction,
    onResetCustomInstructions,
}) => {
    const replansEnabled = Boolean(agentWorkflows.find((workflow) => workflow.id === agentWorkflowId)?.supports_replans);
    const selectedWorkflowListed = agentWorkflows.some((pattern) => pattern.id === agentWorkflowId);

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
                        {description}
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
                <TextField
                    select
                    label="Agent workflow"
                    value={agentWorkflowId}
                    onChange={(e) => onAgentWorkflowChange(e.target.value)}
                    helperText="Router Agent remains the default; advanced workflows are opt-in."
                    SelectProps={{ onOpen: onAgentWorkflowMenuOpen }}
                >
                    {agentWorkflows.map((pattern) => (
                        <MenuItem key={pattern.id} value={pattern.id}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, width: '100%', minWidth: 0 }}>
                                <Typography variant="body2" noWrap>
                                    {pattern.name || pattern.id}
                                </Typography>
                                <Chip
                                    size="small"
                                    variant={pattern.is_builtin ? 'outlined' : 'filled'}
                                    color={pattern.is_builtin ? 'default' : 'primary'}
                                    label={pattern.is_builtin ? 'Built-in' : 'Custom'}
                                    sx={{ flex: '0 0 auto' }}
                                />
                            </Box>
                        </MenuItem>
                    ))}
                    {agentWorkflowIsCustom && !selectedWorkflowListed ? (
                        <MenuItem value={agentWorkflowId}>
                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1, width: '100%', minWidth: 0 }}>
                                <Typography variant="body2" noWrap>
                                    {agentWorkflowId}
                                </Typography>
                                <Chip size="small" color="primary" label="Custom" sx={{ flex: '0 0 auto' }} />
                            </Box>
                        </MenuItem>
                    ) : null}
                </TextField>
                {replansEnabled ? (
                    replansLimit !== null ? (
                        <TextField
                            label="Replans"
                            type="number"
                            value={replans}
                            onChange={(e) => {
                                const parsed = parseInt(e.target.value, 10);
                                onReplansChange(Math.max(1, Math.min(replansLimit, Number.isNaN(parsed) ? 1 : parsed)));
                            }}
                            slotProps={{ htmlInput: { min: 1, max: replansLimit } }}
                            helperText="Allows at least one evaluator-triggered replan, capped by the server limit."
                        />
                    ) : (
                        <Typography variant="caption" color="error">Replan limit not loaded from server.</Typography>
                    )
                ) : null}
                <Divider />
                <Box>
                    <Typography variant="subtitle2">Memory</Typography>
                    <FormControlLabel
                        control={<Switch checked={useMemory} onChange={(event) => onMemoryChange(event.target.checked)} />}
                        label="Use memories"
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5 }}>
                        Recall durable memories for answers in this thread. Turning this off does not delete or stop memory management.
                    </Typography>
                    <FormControlLabel
                        control={<Switch checked={useThreadMemory} disabled={!useMemory} onChange={(event) => onThreadMemoryChange(event.target.checked)} />}
                        label="Use thread memory"
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5 }}>
                        Recall memories saved specifically for this thread.
                    </Typography>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={useProjectMemory}
                                disabled={!useMemory}
                                onChange={(event) => onProjectMemoryChange(event.target.checked)}
                            />
                        }
                        label="Use project memory"
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5 }}>
                        Recall shared memories from this project.
                    </Typography>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={useGlobalMemory}
                                disabled={!useMemory || !projectAllowsGlobalMemory}
                                onChange={(event) => onGlobalMemoryChange(event.target.checked)}
                            />
                        }
                        label="Use global memory"
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5 }}>
                        {projectAllowsGlobalMemory
                            ? 'Recall memories saved for you across projects.'
                            : 'Enable global memory in project settings before this thread can use it.'}
                    </Typography>
                </Box>
                <Divider />
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
                    label="Runtime Prompt Preview"
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
                    {saving ? 'Saving...' : saveLabel}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default ChatSettingsDialog;
