import React from 'react';
import {
  Box,
  TextField,
  Typography,
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
import { ConversationMarkdown } from './conversation/ConversationMarkdown';

export interface ChatSettingsFormProps {
  description?: string;
  replans: number;
  replansLimit: number | null;
  useReranker: boolean;
  hitlCanvasPublish: boolean;
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
  onReplansChange: (value: number) => void;
  onRerankerChange: (checked: boolean) => void;
  onHitlCanvasPublishChange: (checked: boolean) => void;
  onMemoryChange: (checked: boolean) => void;
  onThreadMemoryChange: (checked: boolean) => void;
  onProjectMemoryChange: (checked: boolean) => void;
  onGlobalMemoryChange: (checked: boolean) => void;
  onAgentWorkflowChange: (value: string) => void;
  onAgentWorkflowMenuOpen?: () => void | Promise<void>;
  onSystemRoleChange: (value: string) => void;
  onToolInstructionChange: (toolId: string, value: string) => void;
  onCustomInstructionsChange: (value: string) => void;
  onResetAll: () => void;
  onResetSystemRole: () => void;
  onResetToolInstruction: (toolId: string) => void;
  onResetCustomInstructions: () => void;
}

const ChatSettingsForm: React.FC<ChatSettingsFormProps> = ({
  description = 'These settings are saved per thread and used by default for every message. Agent workflows are globally available.',
  replans,
  replansLimit,
  useReranker,
  hitlCanvasPublish,
  useMemory,
  useThreadMemory,
  useProjectMemory,
  useGlobalMemory,
  projectAllowsGlobalMemory,
  agentWorkflowId,
  agentWorkflowIsCustom = false,
  agentWorkflows,
  systemRole,
  customInstructions,
  toolCatalog,
  effectiveToolInstructions,
  promptPreview,
  onReplansChange,
  onRerankerChange,
  onHitlCanvasPublishChange,
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
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
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
        helperText="Chat workflows become the thread default."
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
      <Box>
        <FormControlLabel
          control={
            <Switch
              checked={hitlCanvasPublish}
              onChange={(e) => onHitlCanvasPublishChange(e.target.checked)}
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>Ask before publishing a canvas</Typography>
            </Box>
          }
        />
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 0.5, mt: 0.25 }}>
          Pause the agent so you can approve or skip publish_canvas before it writes a durable research canvas.
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
      <Box>
        <Typography
          variant="caption"
          color="text.secondary"
          component="label"
          sx={{ display: 'block', mb: 0.75 }}
        >
          Runtime Prompt Preview
        </Typography>
        <Box
          sx={{
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            p: 2,
            minHeight: 280,
            maxHeight: 480,
            overflow: 'auto',
            bgcolor: 'background.paper',
          }}
        >
          {promptPreview ? (
            <ConversationMarkdown content={promptPreview} />
          ) : (
            <Typography variant="body2" color="text.secondary">
              Loading prompt preview…
            </Typography>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default ChatSettingsForm;
