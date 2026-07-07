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
} from '@mui/material';
import ReplayIcon from '@mui/icons-material/Replay';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import { AgentPatternTemplate, PromptToolDefinition } from '../lib/api';

interface ChatSettingsDialogProps {
    open: boolean;
    onClose: () => void;
    onSave: () => void;
    saving: boolean;
    
    // Settings values
    replans: number;
    replansLimit: number | null;
    hitlWebApproval: boolean;
    useReranker: boolean;
    agentPatternId: string;
    agentPatternIsCustom?: boolean;
    agentPatterns: AgentPatternTemplate[];
    systemRole: string;
    toolInstructions: Record<string, string>;
    customInstructions: string;
    toolCatalog: PromptToolDefinition[];
    effectiveToolInstructions: Record<string, string>;
    promptPreview: string;
    
    // Change handlers
    onReplansChange: (value: number) => void;
    onHitlWebApprovalChange: (checked: boolean) => void;
    onRerankerChange: (checked: boolean) => void;
    onAgentPatternChange: (value: string) => void;
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
    replans,
    replansLimit,
    hitlWebApproval,
    useReranker,
    agentPatternId,
    agentPatternIsCustom = false,
    agentPatterns,
    systemRole,
    toolInstructions,
    customInstructions,
    toolCatalog,
    effectiveToolInstructions,
    promptPreview,
    onReplansChange,
    onHitlWebApprovalChange,
    onRerankerChange,
    onAgentPatternChange,
    onSystemRoleChange,
    onToolInstructionChange,
    onCustomInstructionsChange,
    onResetAll,
    onResetSystemRole,
    onResetToolInstruction,
    onResetCustomInstructions,
}) => {
    const replansEnabled = agentPatternId === 'evaluator_replanner_rag_agent';
    const selectedPatternListed = agentPatterns.some((pattern) => pattern.id === agentPatternId);

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
                        These settings are saved per thread and used by default for every message. Agent patterns are globally available.
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
                <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'minmax(0, 1fr) auto' }, gap: 1, alignItems: 'start' }}>
                    <TextField
                        select
                        label="Agent pattern"
                        value={agentPatternId}
                        onChange={(e) => onAgentPatternChange(e.target.value)}
                        helperText="Router RAG remains the default; advanced patterns are opt-in."
                    >
                        {agentPatterns.map((pattern) => (
                            <MenuItem key={pattern.id} value={pattern.id}>
                                {pattern.is_builtin ? pattern.name : `Custom: ${pattern.name || pattern.id}`}
                            </MenuItem>
                        ))}
                        {agentPatternIsCustom && !selectedPatternListed ? (
                            <MenuItem value={agentPatternId}>
                                Custom: {agentPatternId}
                            </MenuItem>
                        ) : null}
                    </TextField>
                    <Button
                        variant="outlined"
                        startIcon={<AccountTreeIcon />}
                        onClick={() => window.open('/agent-pattern-builder', '_blank', 'noopener,noreferrer')}
                        sx={{ borderRadius: 1, minHeight: 40, whiteSpace: 'nowrap' }}
                    >
                        Open Builder
                    </Button>
                </Box>
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
                    <FormControlLabel
                        control={
                            <Switch
                                checked={hitlWebApproval}
                                onChange={(e) => onHitlWebApprovalChange(e.target.checked)}
                            />
                        }
                        label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>Approve web search</Typography>
                            </Box>
                        }
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5, mt: 0.25 }}>
                        Pauses before live web search so you can approve it or continue from existing context.
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
                    {saving ? 'Saving...' : 'Save'}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default ChatSettingsDialog;
