import React, { useMemo } from 'react';
import AddIcon from '@mui/icons-material/Add';
import StickyNote2OutlinedIcon from '@mui/icons-material/StickyNote2Outlined';
import DashboardCustomizeOutlinedIcon from '@mui/icons-material/DashboardCustomizeOutlined';
import {
  Box,
  Chip,
  IconButton,
  Tooltip,
  Typography,
  TextField,
} from '@mui/material';
import type { AgentWorkflowCatalogResponse } from '../../lib/api';
import type { AgentWorkflowBuilderState } from '../../lib/agent-workflow-builder';
import { canAddNodeType } from '../../lib/agent-workflow-builder';

export default function BuilderNodePalette({
  catalog,
  state,
  disabled,
  onAddNodeType,
  onAddNote,
  onAddGroup,
}: {
  catalog: AgentWorkflowCatalogResponse;
  state: AgentWorkflowBuilderState;
  disabled?: boolean;
  onAddNodeType: (nodeType: string) => void;
  onAddNote?: () => void;
  onAddGroup?: () => void;
}) {
  const [query, setQuery] = React.useState('');
  const groupedNodes = useMemo(() => {
    const groups = new Map<string, [string, AgentWorkflowCatalogResponse['node_catalog'][string]][]>();
    Object.entries(catalog.node_catalog || {}).forEach(([nodeType, entry]) => {
      const haystack = [nodeType, entry.display_name, entry.ui?.summary, ...(entry.ui?.keywords || [])].join(' ').toLowerCase();
      if (query.trim() && !haystack.includes(query.trim().toLowerCase())) return;
      const category = entry.category || 'other';
      groups.set(category, [...(groups.get(category) || []), [nodeType, entry]]);
    });
    return Array.from(groups.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([category, entries]) => [
        category,
        entries.sort(([, a], [, b]) => (a.display_name || '').localeCompare(b.display_name || '')),
      ] as const);
  }, [catalog.node_catalog, query]);

  return (
    <Box sx={{ display: 'grid', gridTemplateRows: 'auto minmax(0, 1fr)', gap: 1, minWidth: 0, height: '100%' }}>
      <Box
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: 2,
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) auto auto',
          gap: 0.5,
          alignItems: 'center',
          bgcolor: 'background.paper',
          pb: 0.5,
        }}
      >
        <TextField
          size="small"
          label="Search nodes"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          sx={{ '& .MuiInputBase-root': { height: 36 } }}
        />
        <Tooltip title="Add canvas note">
          <span>
            <IconButton size="small" disabled={disabled} onClick={onAddNote} aria-label="Add canvas note">
              <StickyNote2OutlinedIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Group current nodes">
          <span>
            <IconButton size="small" disabled={disabled || state.nodes.length === 0} onClick={onAddGroup} aria-label="Group current nodes">
              <DashboardCustomizeOutlinedIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      </Box>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, overflow: 'auto', pr: 0.5 }}>
        {groupedNodes.map(([category, entries]) => (
          <Box key={category} sx={{ display: 'flex', flexDirection: 'column', gap: 0.35 }}>
            <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: 0.3, fontSize: '0.68rem' }}>
              {category.replace(/_/g, ' ')}
            </Typography>
            {entries.map(([nodeType, entry]) => {
              const compatibility = canAddNodeType(catalog, state, nodeType);
              const unavailable = disabled || !compatibility.ok;
              const displayName = entry.display_name || nodeType;
              const summary = entry.ui?.summary || nodeType;
              const tooltipTitle = disabled
                ? 'Authoring is disabled.'
                : compatibility.ok
                  ? `${displayName}: ${summary}`
                  : compatibility.reason || 'Not available';
              return (
                <Tooltip key={nodeType} title={tooltipTitle}>
                  <Box
                    role="button"
                    tabIndex={unavailable ? -1 : 0}
                    aria-disabled={unavailable}
                    draggable={!unavailable}
                    onClick={() => !unavailable && onAddNodeType(nodeType)}
                    onKeyDown={(event) => {
                      if (unavailable || (event.key !== 'Enter' && event.key !== ' ')) return;
                      event.preventDefault();
                      onAddNodeType(nodeType);
                    }}
                    onDragStart={(event) => {
                      if (unavailable) return;
                      event.dataTransfer.setData('application/askpdf-node-type', nodeType);
                      event.dataTransfer.effectAllowed = 'move';
                    }}
                    sx={{
                      minHeight: 38,
                      px: 0.75,
                      py: 0.45,
                      display: 'grid',
                      gridTemplateColumns: '22px minmax(0, 1fr) auto',
                      alignItems: 'center',
                      gap: 0.55,
                      borderRadius: 1,
                      color: unavailable ? 'text.disabled' : 'text.primary',
                      cursor: unavailable ? 'not-allowed' : 'grab',
                      opacity: unavailable ? 0.58 : 1,
                      boxShadow: unavailable ? 'inset 0 0 0 1px transparent' : 'inset 0 0 0 1px transparent',
                      '&:hover': unavailable ? {} : { bgcolor: 'action.hover', boxShadow: 'inset 0 0 0 1px currentColor', color: 'primary.main' },
                      '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 1 },
                    }}
                  >
                    <AddIcon fontSize="small" color={unavailable ? 'disabled' : 'primary'} />
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="body2" noWrap sx={{ fontWeight: 700, lineHeight: 1.1 }}>
                        {displayName}
                      </Typography>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{
                          display: '-webkit-box',
                          overflow: 'hidden',
                          WebkitBoxOrient: 'vertical',
                          WebkitLineClamp: 2,
                          lineHeight: 1.18,
                        }}
                      >
                        {summary}
                      </Typography>
                    </Box>
                    {entry.max_instances ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`max ${entry.max_instances}`}
                        sx={{ height: 20, fontSize: '0.68rem' }}
                      />
                    ) : null}
                  </Box>
                </Tooltip>
              );
            })}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
