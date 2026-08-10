import React, { useEffect, useMemo, useState } from 'react';
import { Box, Button, Checkbox, FormControlLabel, TextField, Typography } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import type { AgentRunPendingInterrupt, AgentRunResumeAction } from '../../lib/api';
import { AgentRunResumeAction as ResumeAction, HitlSelectionMode } from '../../lib/enums';
import { ResizableDecisionPanel } from './ResizableDecisionPanel';

export type HumanReviewScopeOption = {
  id: string;
  label: string;
  description?: string;
};

export function HumanReviewDecisionPanel({
  interrupt,
  submitting,
  error,
  editText = '',
  scopeOptions = [],
  rootRef,
  onEditTextChange,
  onAction,
}: {
  interrupt: AgentRunPendingInterrupt;
  submitting?: AgentRunResumeAction | null;
  error?: string | null;
  editText?: string;
  scopeOptions?: HumanReviewScopeOption[];
  rootRef?: React.RefObject<HTMLElement | null>;
  onEditTextChange?: (value: string) => void;
  onAction: (
    action: AgentRunResumeAction,
    options?: { selectedOptionIds?: string[]; editedPayload?: Record<string, unknown> },
  ) => void | Promise<void>;
}) {
  const actions = useMemo(
    () => new Set(Array.isArray(interrupt.allowed_actions) ? interrupt.allowed_actions.map(String) : []),
    [interrupt.allowed_actions],
  );
  const interruptOptions = useMemo<HumanReviewScopeOption[]>(
    () => (Array.isArray(interrupt.options) ? interrupt.options : []).map((option) => ({
      id: option.id,
      label: option.label || option.id,
      description: option.description,
    })),
    [interrupt.options],
  );
  const selectableOptions = scopeOptions.length ? scopeOptions : interruptOptions;
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const selectionMode = String(interrupt.selection_mode || HitlSelectionMode.Single);
  const multiSelect = selectionMode === HitlSelectionMode.Multi || selectionMode === HitlSelectionMode.SingleOrMulti || scopeOptions.length > 0;
  const editingScope = actions.has(ResumeAction.Edit) && scopeOptions.length > 0;
  const isWebApproval = interrupt.type === 'external_research_approval'
    || interrupt.node_id === 'web_approval_gate'
    || (typeof interrupt.proposed_tool === 'object' && interrupt.proposed_tool !== null && interrupt.proposed_tool.name === 'search_web');
  const approvalScopeKind = String(interrupt.approval_scope_kind || (interrupt.type === 'external_research_approval' ? 'task' : 'run'));

  useEffect(() => {
    setSelectedIds(selectableOptions.map((option) => option.id));
  }, [interrupt.interrupt_id, selectableOptions.map((option) => option.id).join('|')]);

  const toggleOption = (id: string) => {
    setSelectedIds((current) => {
      if (!multiSelect) return [id];
      return current.includes(id) ? current.filter((value) => value !== id) : [...current, id];
    });
  };

  const submit = (action: AgentRunResumeAction) => {
    const options = action === ResumeAction.ApproveSelected
      ? { selectedOptionIds: selectedIds }
      : action === ResumeAction.Edit && editingScope
        ? { editedPayload: { todo_ids: selectedIds } }
        : undefined;
    return onAction(action, options);
  };

  return (
    <ResizableDecisionPanel
      title={interrupt.title || (interrupt.proposed_tool ? 'Approve tool use?' : 'Human review required')}
      variant="approval"
      rootRef={rootRef}
      horizontalInset={1}
    >
      {(interrupt.prompt || interrupt.body) && (
        <Typography variant="caption" sx={{ color: 'text.secondary', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {interrupt.prompt || interrupt.body}
        </Typography>
      )}
      {selectableOptions.length > 0 && (actions.has(ResumeAction.ApproveSelected) || editingScope) && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25 }}>
          {editingScope && <Typography variant="caption" color="text.secondary">Select the approved research scope.</Typography>}
          {selectableOptions.map((option) => (
            <FormControlLabel
              key={option.id}
              sx={{ m: 0, alignItems: 'flex-start' }}
              control={<Checkbox size="small" checked={selectedIds.includes(option.id)} onChange={() => toggleOption(option.id)} disabled={Boolean(submitting)} />}
              label={<Box><Typography variant="caption">{option.label}</Typography>{option.description && <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>{option.description}</Typography>}</Box>}
            />
          ))}
        </Box>
      )}
      {interrupt.proposed_final_answer && (
        <TextField
          fullWidth
          size="small"
          multiline
          minRows={5}
          maxRows={12}
          label={actions.has(ResumeAction.Edit) ? 'Final answer draft' : 'Proposed final answer'}
          value={editText}
          disabled={!actions.has(ResumeAction.Edit) || Boolean(submitting)}
          onChange={(event) => onEditTextChange?.(event.target.value)}
          sx={{ '& .MuiOutlinedInput-root': { bgcolor: 'action.hover' } }}
        />
      )}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
        {actions.has(ResumeAction.Approve) && <Button size="small" variant="contained" startIcon={<CheckIcon fontSize="inherit" />} disabled={Boolean(submitting)} onClick={() => void submit(ResumeAction.Approve)}>{submitting === ResumeAction.Approve ? 'Approving...' : isWebApproval ? 'Approve once' : 'Approve'}</Button>}
        {actions.has(ResumeAction.ApproveForScope) && <Button size="small" variant="contained" color="secondary" disabled={Boolean(submitting)} onClick={() => void submit(ResumeAction.ApproveForScope)}>{submitting === ResumeAction.ApproveForScope ? 'Approving...' : `Approve for this ${approvalScopeKind}`}</Button>}
        {actions.has(ResumeAction.ApproveSelected) && <Button size="small" variant="contained" disabled={Boolean(submitting) || selectedIds.length === 0} onClick={() => void submit(ResumeAction.ApproveSelected)}>{submitting === ResumeAction.ApproveSelected ? 'Approving...' : 'Approve selected'}</Button>}
        {actions.has(ResumeAction.Edit) && <Button size="small" variant="contained" color="secondary" startIcon={<EditIcon fontSize="inherit" />} disabled={Boolean(submitting) || (editingScope ? selectedIds.length === 0 : !editText.trim())} onClick={() => void submit(ResumeAction.Edit)}>{submitting === ResumeAction.Edit ? 'Saving...' : editingScope ? 'Approve selected scope' : 'Save edit'}</Button>}
        {actions.has(ResumeAction.ContinueWithout) && <Button size="small" variant="outlined" disabled={Boolean(submitting)} onClick={() => void submit(ResumeAction.ContinueWithout)}>{submitting === ResumeAction.ContinueWithout ? 'Continuing...' : isWebApproval ? 'Continue without web research' : 'Continue'}</Button>}
        {actions.has(ResumeAction.Reject) && <Button size="small" variant="outlined" color="error" startIcon={<CloseIcon fontSize="inherit" />} disabled={Boolean(submitting)} onClick={() => void submit(ResumeAction.Reject)}>{submitting === ResumeAction.Reject ? 'Rejecting...' : isWebApproval ? 'Continue without web research' : 'Reject'}</Button>}
      </Box>
      {error && <Typography variant="caption" color="error">{error}</Typography>}
    </ResizableDecisionPanel>
  );
}
