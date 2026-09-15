import React, { useMemo } from 'react';
import {
  Alert,
  Box,
  Chip,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import DescriptionOutlinedIcon from '@mui/icons-material/DescriptionOutlined';
import LanguageIcon from '@mui/icons-material/Language';
import PsychologyOutlinedIcon from '@mui/icons-material/PsychologyOutlined';
import { ConversationMarkdown } from '../conversation/ConversationMarkdown';
import {
  canvasSpecCounts,
  DAG_NODE_HEIGHT,
  DAG_NODE_WIDTH,
  documentCitationTarget,
  layoutCanvasDag,
  type CanvasBlock,
  type CanvasCitation,
  type CanvasCitationKind,
  type CanvasSpec,
  type DocumentCanvasCitationTarget,
} from '../../lib/canvas-spec';

const KIND_LABEL: Record<CanvasCitationKind, string> = {
  document: 'PDF',
  web: 'Web',
  memory: 'Memory',
  conversation: 'Chat',
};

function citationIcon(kind: CanvasCitationKind) {
  if (kind === 'web') return <LanguageIcon sx={{ fontSize: 16 }} />;
  if (kind === 'memory') return <PsychologyOutlinedIcon sx={{ fontSize: 16 }} />;
  if (kind === 'conversation') return <ChatBubbleOutlineIcon sx={{ fontSize: 16 }} />;
  return <DescriptionOutlinedIcon sx={{ fontSize: 16 }} />;
}

function scrollToSection(index: number) {
  document.getElementById(`canvas-section-${index}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function DagView({ block }: { block: Extract<CanvasBlock, { type: 'dag' }> }) {
  const layout = useMemo(() => layoutCanvasDag(block.nodes, block.edges), [block.edges, block.nodes]);
  const markerId = `canvas-dag-arrow-${block.nodes.map((node) => node.id).join('-') || 'empty'}`;
  return (
    <Box>
      {block.title ? (
        <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.8 }}>
          {block.title}
        </Typography>
      ) : null}
      <Box
        sx={{
          mt: 0.5,
          px: 1.5,
          py: 1.5,
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          bgcolor: 'action.hover',
          overflowX: 'auto',
        }}
      >
        <svg
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          role="img"
          aria-label={block.title || 'Evidence diagram'}
        >
          <defs>
            <marker id={markerId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 z" fill="currentColor" />
            </marker>
          </defs>
          {layout.edges.map((edge) => {
            const midX = (edge.x1 + edge.x2) / 2;
            return (
              <path
                key={`${edge.source}-${edge.target}-${edge.y1}-${edge.y2}`}
                d={`M ${edge.x1} ${edge.y1} C ${midX} ${edge.y1}, ${midX} ${edge.y2}, ${edge.x2} ${edge.y2}`}
                fill="none"
                stroke="currentColor"
                strokeOpacity={0.45}
                strokeWidth={1.5}
                markerEnd={`url(#${markerId})`}
              />
            );
          })}
          {layout.nodes.map((node) => (
            <g key={node.id}>
              <rect
                x={node.x}
                y={node.y}
                width={DAG_NODE_WIDTH}
                height={DAG_NODE_HEIGHT}
                rx={10}
                fill="var(--mui-palette-background-paper, #111)"
                stroke="currentColor"
                strokeOpacity={0.35}
              />
              <text
                x={node.x + DAG_NODE_WIDTH / 2}
                y={node.y + DAG_NODE_HEIGHT / 2 + 4}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                fill="currentColor"
              >
                {node.label}
              </text>
            </g>
          ))}
        </svg>
      </Box>
    </Box>
  );
}

function CitationChip({
  citation,
  onOpenDocumentCitation,
}: {
  citation: CanvasCitation;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  const documentTarget = documentCitationTarget(citation);
  const clickable = (citation.kind === 'web' && Boolean(citation.url))
    || Boolean(documentTarget && onOpenDocumentCitation);
  return (
    <Chip
      icon={citationIcon(citation.kind)}
      label={`${KIND_LABEL[citation.kind]} · ${citation.label}`}
      clickable={clickable}
      component={citation.kind === 'web' && citation.url ? 'a' : 'div'}
      href={citation.kind === 'web' && citation.url ? citation.url : undefined}
      target={citation.kind === 'web' && citation.url ? '_blank' : undefined}
      rel={citation.kind === 'web' && citation.url ? 'noopener noreferrer' : undefined}
      onClick={documentTarget && onOpenDocumentCitation ? () => onOpenDocumentCitation(documentTarget) : undefined}
      variant="outlined"
      sx={{
        height: 32,
        borderRadius: 2,
        '& .MuiChip-label': { px: 1 },
        '& .MuiChip-icon': { ml: 0.75 },
      }}
    />
  );
}

function StatCard({ block }: { block: Extract<CanvasBlock, { type: 'stat' }> }) {
  const tone = block.tone || 'neutral';
  const accent = tone === 'warning' ? 'warning.main' : tone === 'info' ? 'info.main' : 'divider';
  const wash = tone === 'warning'
    ? 'rgba(237, 108, 2, 0.08)'
    : tone === 'info'
      ? 'rgba(2, 136, 209, 0.08)'
      : 'transparent';
  return (
    <Paper
      variant="outlined"
      sx={{
        p: 1.75,
        minWidth: 148,
        flex: '1 1 148px',
        borderRadius: 2,
        borderColor: accent,
        bgcolor: wash,
      }}
    >
      <Typography variant="h4" component="p" sx={{ fontWeight: 700, letterSpacing: -0.6, lineHeight: 1.1 }}>
        {block.value}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
        {block.label}
      </Typography>
    </Paper>
  );
}

function BlockView({
  block,
  onOpenDocumentCitation,
}: {
  block: CanvasBlock;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  if (block.type === 'stat') return <StatCard block={block} />;
  if (block.type === 'table') {
    return (
      <Box>
        {block.caption ? (
          <Typography variant="subtitle2" sx={{ mb: 1 }}>{block.caption}</Typography>
        ) : null}
        <TableContainer
          component={Paper}
          variant="outlined"
          sx={{ borderRadius: 2, overflow: 'hidden' }}
        >
          <Table size="small" aria-label={block.caption || 'Canvas table'}>
            <TableHead>
              <TableRow>
                {block.headers.map((header) => (
                  <TableCell key={header} sx={{ fontWeight: 700, bgcolor: 'action.hover' }}>{header}</TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {block.rows.map((row, index) => (
                <TableRow key={index} hover>
                  {row.map((cell, cellIndex) => (
                    <TableCell key={`${index}:${cellIndex}`}>{cell}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  }
  if (block.type === 'callout') {
    return (
      <Alert
        severity={block.tone === 'warning' ? 'warning' : 'info'}
        sx={{ borderRadius: 2, alignItems: 'flex-start' }}
      >
        <Typography variant="subtitle2" sx={{ mb: 0.25 }}>{block.title}</Typography>
        <Typography variant="body2">{block.body}</Typography>
      </Alert>
    );
  }
  if (block.type === 'markdown') {
    return (
      <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
        <ConversationMarkdown content={block.text} />
      </Paper>
    );
  }
  if (block.type === 'sources') {
    return (
      <Box>
        <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.8 }}>
          {block.title || 'Sources'}
        </Typography>
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 0.75 }}>
          {block.citations.map((citation, index) => (
            <CitationChip
              key={`${citation.kind}:${citation.label}:${index}`}
              citation={citation}
              onOpenDocumentCitation={onOpenDocumentCitation}
            />
          ))}
        </Stack>
      </Box>
    );
  }
  return <DagView block={block} />;
}

export default function CanvasDocument({
  spec,
  createdAt,
  onOpenDocumentCitation,
}: {
  spec: CanvasSpec;
  createdAt?: string | null;
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  const counts = canvasSpecCounts(spec);
  const createdLabel = createdAt
    ? new Date(createdAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
    : null;

  return (
    <Box sx={{ minHeight: '100%', bgcolor: 'background.default' }}>
      <Box
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: 2,
          px: 3,
          py: 1,
          borderBottom: '1px solid',
          borderColor: 'divider',
          bgcolor: 'background.paper',
          display: 'flex',
          gap: 1,
          overflowX: 'auto',
        }}
      >
        {spec.sections.map((section, index) => (
          <Chip
            key={`${section.title}:${index}`}
            size="small"
            label={section.title}
            onClick={() => scrollToSection(index)}
            variant="outlined"
            sx={{ borderRadius: 1.5 }}
          />
        ))}
      </Box>
      <Box sx={{ maxWidth: 860, mx: 'auto', px: 3, py: 4 }}>
        <Typography variant="overline" color="primary.main" sx={{ letterSpacing: 1.4 }}>
          Research canvas
        </Typography>
        <Typography variant="h4" sx={{ fontWeight: 700, letterSpacing: -0.8, mt: 0.5 }}>
          {spec.title}
        </Typography>
        {spec.summary ? (
          <Typography color="text.secondary" sx={{ mt: 1.25, fontSize: '1.05rem', lineHeight: 1.55 }}>
            {spec.summary}
          </Typography>
        ) : null}
        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 2 }}>
          <Chip size="small" label={`${counts.sections} section${counts.sections === 1 ? '' : 's'}`} />
          {counts.stats > 0 ? <Chip size="small" label={`${counts.stats} metrics`} /> : null}
          {counts.sources > 0 ? <Chip size="small" label={`${counts.sources} sources`} /> : null}
          {createdLabel ? <Chip size="small" variant="outlined" label={createdLabel} /> : null}
        </Stack>
        <Stack spacing={4} sx={{ mt: 4 }}>
          {spec.sections.map((section, sectionIndex) => {
            const stats = section.blocks.filter((block) => block.type === 'stat');
            const rest = section.blocks.filter((block) => block.type !== 'stat');
            return (
              <Box key={`${section.title}:${sectionIndex}`} id={`canvas-section-${sectionIndex}`} sx={{ scrollMarginTop: 56 }}>
                <Typography variant="h6" sx={{ fontWeight: 650, mb: 1.75 }}>{section.title}</Typography>
                {stats.length ? (
                  <Box sx={{ display: 'flex', gap: 1.5, flexWrap: 'wrap', mb: rest.length ? 2 : 0 }}>
                    {stats.map((block, index) => (
                      <BlockView key={`${section.title}-stat-${index}`} block={block} onOpenDocumentCitation={onOpenDocumentCitation} />
                    ))}
                  </Box>
                ) : null}
                <Stack spacing={2}>
                  {rest.map((block, index) => (
                    <BlockView key={`${section.title}-block-${index}`} block={block} onOpenDocumentCitation={onOpenDocumentCitation} />
                  ))}
                </Stack>
              </Box>
            );
          })}
        </Stack>
      </Box>
    </Box>
  );
}
