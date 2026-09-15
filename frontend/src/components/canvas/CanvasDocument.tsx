import React from 'react';
import {
  Alert,
  Box,
  Button,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import { ConversationMarkdown } from '../conversation/ConversationMarkdown';
import {
  documentCitationTarget,
  type CanvasBlock,
  type CanvasCitation,
  type CanvasSpec,
  type DocumentCanvasCitationTarget,
} from '../../lib/canvas-spec';

const NODE_W = 120;
const NODE_H = 36;

function DagView({ block }: { block: Extract<CanvasBlock, { type: 'dag' }> }) {
  const columns = new Map<string, number>();
  block.nodes.forEach((node, index) => columns.set(node.id, index));
  const width = Math.max(1, block.nodes.length) * (NODE_W + 24);
  const height = NODE_H + 24;
  return (
    <Box>
      {block.title ? <Typography variant="caption" color="text.secondary">{block.title}</Typography> : null}
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={block.title || 'Diagram'} style={{ maxWidth: width }}>
        {block.edges.map((edge) => {
          const from = columns.get(edge.source) ?? 0;
          const to = columns.get(edge.target) ?? 0;
          return (
            <line
              key={`${edge.source}-${edge.target}`}
              x1={from * (NODE_W + 24) + NODE_W / 2}
              y1={NODE_H / 2 + 8}
              x2={to * (NODE_W + 24) + NODE_W / 2}
              y2={NODE_H / 2 + 8}
              stroke="currentColor"
              strokeWidth={1.25}
            />
          );
        })}
        {block.nodes.map((node, index) => (
          <g key={node.id}>
            <rect x={index * (NODE_W + 24)} y={8} width={NODE_W} height={NODE_H} rx={4} fill="none" stroke="currentColor" />
            <text x={index * (NODE_W + 24) + NODE_W / 2} y={8 + NODE_H / 2 + 4} textAnchor="middle" fontSize={11} fill="currentColor">
              {node.label}
            </text>
          </g>
        ))}
      </svg>
    </Box>
  );
}

function CitationButton({
  citation,
  onOpenDocumentCitation,
}: {
  citation: CanvasCitation;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  const documentTarget = documentCitationTarget(citation);
  if (citation.kind === 'web' && citation.url) {
    return (
      <Button size="small" href={citation.url} target="_blank" rel="noopener noreferrer" sx={{ textTransform: 'none', justifyContent: 'flex-start' }}>
        {citation.label}
      </Button>
    );
  }
  if (documentTarget && onOpenDocumentCitation) {
    return (
      <Button size="small" onClick={() => onOpenDocumentCitation(documentTarget)} sx={{ textTransform: 'none', justifyContent: 'flex-start' }}>
        {citation.label}
      </Button>
    );
  }
  return <Typography variant="body2">{citation.label}</Typography>;
}

function BlockView({
  block,
  onOpenDocumentCitation,
}: {
  block: CanvasBlock;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  if (block.type === 'stat') {
    return (
      <Paper variant="outlined" sx={{ p: 1.5, minWidth: 120 }}>
        <Typography variant="h6" component="p">{block.value}</Typography>
        <Typography variant="caption" color="text.secondary">{block.label}</Typography>
      </Paper>
    );
  }
  if (block.type === 'table') {
    return (
      <Box>
        {block.caption ? <Typography variant="caption" color="text.secondary">{block.caption}</Typography> : null}
        <Table size="small" aria-label={block.caption || 'Canvas table'}>
          <TableHead>
            <TableRow>
              {block.headers.map((header) => (
                <TableCell key={header} sx={{ fontWeight: 600 }}>{header}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {block.rows.map((row, index) => (
              <TableRow key={index}>
                {row.map((cell, cellIndex) => (
                  <TableCell key={`${index}:${cellIndex}`}>{cell}</TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Box>
    );
  }
  if (block.type === 'callout') {
    return <Alert severity={block.tone === 'warning' ? 'warning' : 'info'} title={block.title}><strong>{block.title}. </strong>{block.body}</Alert>;
  }
  if (block.type === 'markdown') {
    return <ConversationMarkdown content={block.text} />;
  }
  if (block.type === 'sources') {
    return (
      <Stack spacing={0.5}>
        <Typography variant="caption" color="text.secondary">{block.title || 'Sources'}</Typography>
        {block.citations.map((citation, index) => (
          <CitationButton key={`${citation.kind}:${citation.label}:${index}`} citation={citation} onOpenDocumentCitation={onOpenDocumentCitation} />
        ))}
      </Stack>
    );
  }
  return <DagView block={block} />;
}

export default function CanvasDocument({
  spec,
  onOpenDocumentCitation,
}: {
  spec: CanvasSpec;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  return (
    <Stack spacing={2} sx={{ p: 2 }}>
      <Box>
        <Typography variant="h5">{spec.title}</Typography>
        {spec.summary ? <Typography color="text.secondary" sx={{ mt: 0.5 }}>{spec.summary}</Typography> : null}
      </Box>
      {spec.sections.map((section) => (
        <Stack key={section.title} spacing={1.5}>
          <Typography variant="subtitle1">{section.title}</Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {section.blocks.filter((block) => block.type === 'stat').map((block, index) => (
              <BlockView key={`${section.title}-stat-${index}`} block={block} onOpenDocumentCitation={onOpenDocumentCitation} />
            ))}
          </Box>
          {section.blocks.filter((block) => block.type !== 'stat').map((block, index) => (
            <BlockView key={`${section.title}-block-${index}`} block={block} onOpenDocumentCitation={onOpenDocumentCitation} />
          ))}
        </Stack>
      ))}
    </Stack>
  );
}
