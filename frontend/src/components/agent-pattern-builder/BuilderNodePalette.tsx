import React, { useMemo } from 'react';
import AddIcon from '@mui/icons-material/Add';
import {
  Box,
  Button,
  Chip,
  Divider,
  Tooltip,
  Typography,
} from '@mui/material';
import type { AgentPatternCatalogResponse } from '../../lib/api';
import type { AgentPatternBuilderState } from '../../lib/agent-pattern-builder';
import { canAddNodeType } from '../../lib/agent-pattern-builder';

export default function BuilderNodePalette({
  catalog,
  state,
  onAddNodeType,
}: {
  catalog: AgentPatternCatalogResponse;
  state: AgentPatternBuilderState;
  onAddNodeType: (nodeType: string) => void;
}) {
  const groupedNodes = useMemo(() => {
    const groups = new Map<string, [string, AgentPatternCatalogResponse['node_catalog'][string]][]>();
    Object.entries(catalog.node_catalog || {}).forEach(([nodeType, entry]) => {
      const category = entry.category || 'other';
      groups.set(category, [...(groups.get(category) || []), [nodeType, entry]]);
    });
    return Array.from(groups.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([category, entries]) => [
        category,
        entries.sort(([, a], [, b]) => (a.display_name || '').localeCompare(b.display_name || '')),
      ] as const);
  }, [catalog.node_catalog]);

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
                    disabled={!compatibility.ok}
                    onClick={() => onAddNodeType(nodeType)}
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
                        {nodeType}
                      </Typography>
                    </Box>
                  </Button>
                </span>
              );
              return (
                <Tooltip key={nodeType} title={compatibility.ok ? '' : compatibility.reason || 'Not available'}>
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

