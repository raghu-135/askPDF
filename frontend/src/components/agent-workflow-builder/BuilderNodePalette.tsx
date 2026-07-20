import React, { useMemo } from 'react';
import AddIcon from '@mui/icons-material/Add';
import StickyNote2OutlinedIcon from '@mui/icons-material/StickyNote2Outlined';
import DashboardCustomizeOutlinedIcon from '@mui/icons-material/DashboardCustomizeOutlined';
import {
  Box,
  Button,
  Chip,
  Divider,
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
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, minWidth: 0 }}>
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
          Node Palette
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Catalog-backed node types
        </Typography>
      </Box>
      <Divider />
      <TextField size="small" label="Search nodes" value={query} onChange={(event) => setQuery(event.target.value)} />
      <Button
        size="small"
        variant="outlined"
        startIcon={<StickyNote2OutlinedIcon />}
        disabled={disabled}
        onClick={onAddNote}
      >
        Add canvas note
      </Button>
      <Button
        size="small"
        variant="outlined"
        startIcon={<DashboardCustomizeOutlinedIcon />}
        disabled={disabled || state.nodes.length === 0}
        onClick={onAddGroup}
      >
        Group current nodes
      </Button>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25, overflow: 'auto', pr: 0.5 }}>
        {groupedNodes.map(([category, entries]) => (
          <Box key={category} sx={{ display: 'flex', flexDirection: 'column', gap: 0.75 }}>
            <Typography variant="caption" sx={{ fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase' }}>
              {category.replace(/_/g, ' ')}
            </Typography>
            {entries.map(([nodeType, entry]) => {
              const compatibility = canAddNodeType(catalog, state, nodeType);
              const button = (
                <span>
                  <Button
                    fullWidth
                    size="small"
                    variant="outlined"
                    startIcon={<AddIcon fontSize="small" />}
                    disabled={disabled || !compatibility.ok}
                    onClick={() => onAddNodeType(nodeType)}
                    draggable={!disabled && compatibility.ok}
                    onDragStart={(event) => {
                      event.dataTransfer.setData('application/askpdf-node-type', nodeType);
                      event.dataTransfer.effectAllowed = 'move';
                    }}
                    sx={{
                      justifyContent: 'flex-start',
                      textAlign: 'left',
                      minHeight: 38,
                      borderRadius: 1,
                      textTransform: 'none',
                    }}
                  >
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="body2" sx={{ fontWeight: 600, lineHeight: 1.2 }}>
                        {entry.display_name || nodeType}
                      </Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.2 }}>
                        {entry.ui?.summary || nodeType}
                      </Typography>
                    </Box>
                  </Button>
                </span>
              );
              return (
                <Tooltip key={nodeType} title={disabled ? 'Authoring is disabled.' : compatibility.ok ? '' : compatibility.reason || 'Not available'}>
                  <Box>
                    {button}
                    {entry.max_instances ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`max ${entry.max_instances}`}
                        sx={{ mt: 0.5, height: 20, fontSize: '0.68rem' }}
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
