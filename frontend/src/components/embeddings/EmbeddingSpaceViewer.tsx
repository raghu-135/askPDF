import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
  useTheme,
} from '@mui/material';
import CenterFocusStrongIcon from '@mui/icons-material/CenterFocusStrong';
import RefreshIcon from '@mui/icons-material/Refresh';
import { JsonPreview } from '../agent-graph/AgentGraphInspectorPrimitives';
import {
  getThreadEmbeddingProjection,
  type EmbeddingPoint3D,
  type EmbeddingProjectionEdge,
} from '../../lib/embedding-projection';
import type { PdfTab } from '../../lib/document-tabs';
import type { DocumentCanvasCitationTarget } from '../../lib/canvas-spec';
import type {
  GraphCanvasRef,
  GraphEdge,
  GraphNode,
  InternalGraphNode,
  Theme,
} from 'reagraph';
import { darkTheme, lightTheme } from 'reagraph';

// Reagraph is WebGL / Three.js based and must not run in Node SSR
const GraphCanvas = dynamic(
  () => import('reagraph').then((mod) => mod.GraphCanvas),
  { ssr: false },
) as React.ComponentType<any>;

type ColorMode = 'document' | 'kind';

function pageLabel(point: EmbeddingPoint3D): string {
  if (point.page_start != null) {
    const end = point.page_end ?? point.page_start;
    return point.page_start === end ? `p. ${point.page_start}` : `p. ${point.page_start}-${end}`;
  }
  if (point.pages) return `pages ${point.pages}`;
  return '';
}

function hashColor(seed: string): string {
  let hash = 0;
  for (let index = 0; index < seed.length; index += 1) {
    hash = seed.charCodeAt(index) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 65% 52%)`;
}

function kindColor(point: EmbeddingPoint3D): string {
  if (point.table_id) return '#f97316';
  if (point.section_id) return '#a855f7';
  return '#3b82f6';
}

export default function EmbeddingSpaceViewer({
  threadId,
  documents,
  onOpenDocumentCitation,
}: {
  threadId: string;
  documents: readonly PdfTab[];
  onOpenDocumentCitation?: (target: DocumentCanvasCitationTarget) => void;
}) {
  const muiTheme = useTheme();
  const graphRef = useRef<GraphCanvasRef | null>(null);

  const [points, setPoints] = useState<EmbeddingPoint3D[]>([]);
  const [edges, setEdges] = useState<EmbeddingProjectionEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | null>(null);
  const [fileHashFilter, setFileHashFilter] = useState<string>('');
  const [colorMode, setColorMode] = useState<ColorMode>('document');
  const [searchFilter, setSearchFilter] = useState('');
  const [showSequenceEdges, setShowSequenceEdges] = useState(true);
  const [showSimilarityEdges, setShowSimilarityEdges] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const documentColors = useMemo(
    () => Object.fromEntries(documents.map((doc) => [doc.fileHash, hashColor(doc.fileHash)])),
    [documents],
  );

  const loadProjection = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getThreadEmbeddingProjection(threadId, {
        fileHash: fileHashFilter || undefined,
        limit: 500,
      });
      setPoints(response.points);
      setEdges(response.edges || []);
      setTruncated(response.truncated);
      setEmbeddingModel(response.embedding_model);
      setSelectedNodeId(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load embedding projection');
      setPoints([]);
      setEdges([]);
    } finally {
      setLoading(false);
    }
  }, [fileHashFilter, threadId]);

  useEffect(() => {
    void loadProjection();
  }, [loadProjection]);

  const visiblePoints = useMemo(() => {
    const needle = searchFilter.trim().toLowerCase();
    if (!needle) return points;
    return points.filter((point) => (
      point.text.toLowerCase().includes(needle)
      || String(point.chunk_id ?? '').includes(needle)
      || (point.file_name || '').toLowerCase().includes(needle)
      || (point.section_id || '').toLowerCase().includes(needle)
      || (point.table_id || '').toLowerCase().includes(needle)
    ));
  }, [points, searchFilter]);

  const visiblePointIds = useMemo(
    () => new Set(visiblePoints.map((point) => point.id)),
    [visiblePoints],
  );

  const nodes: GraphNode[] = useMemo(() => {
    return visiblePoints.map((point) => {
      const fill = colorMode === 'kind'
        ? kindColor(point)
        : (documentColors[point.file_hash] || hashColor(point.file_hash));
      const pageStr = pageLabel(point);
      return {
        id: point.id,
        label: `${point.file_name || point.file_hash.slice(0, 8)} · chunk ${point.chunk_id ?? '?'}`,
        subLabel: pageStr || undefined,
        fill,
        size: 9,
        data: point,
        fx: typeof point.x === 'number' ? point.x * 250 : undefined,
        fy: typeof point.y === 'number' ? point.y * 250 : undefined,
        fz: typeof point.z === 'number' ? point.z * 250 : undefined,
      };
    });
  }, [colorMode, documentColors, visiblePoints]);

  const graphEdges: GraphEdge[] = useMemo(() => {
    return edges
      .filter((edge) => {
        if (!visiblePointIds.has(edge.source) || !visiblePointIds.has(edge.target)) {
          return false;
        }
        if (edge.kind === 'sequence' && !showSequenceEdges) return false;
        if (edge.kind === 'similarity' && !showSimilarityEdges) return false;
        return true;
      })
      .map((edge) => {
        const isSequence = edge.kind === 'sequence';
        const color = isSequence
          ? (muiTheme.palette.mode === 'dark' ? '#94a3b8' : '#64748b')
          : (muiTheme.palette.mode === 'dark' ? '#38bdf8' : '#0284c7');
        return {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          label: isSequence ? 'next' : (edge.label || undefined),
          size: isSequence ? 1.5 : 2.5,
          fill: color,
          arrowPlacement: 'end' as const,
          dashed: !isSequence,
        };
      });
  }, [edges, muiTheme.palette.mode, showSequenceEdges, showSimilarityEdges, visiblePointIds]);

  const selectedPoint = useMemo(() => {
    if (!selectedNodeId) return null;
    return points.find((point) => point.id === selectedNodeId) || null;
  }, [points, selectedNodeId]);

  const handleNodeClick = useCallback((node: InternalGraphNode) => {
    setSelectedNodeId(node.id);
  }, []);

  const handleJumpToDocument = (point: EmbeddingPoint3D) => {
    if (!onOpenDocumentCitation) return;
    const documentTab = documents.find((doc) => doc.fileHash === point.file_hash);
    const sentences = documentTab?.sentences || [];
    const targetPage = point.page_start ?? null;
    const sentence = targetPage == null
      ? sentences[0]
      : sentences.find((item) => item.page === targetPage) || sentences[0];
    onOpenDocumentCitation({
      fileHash: point.file_hash,
      sentenceId: sentence?.id ?? null,
    });
  };

  const handleFitView = useCallback(() => {
    if (graphRef.current) {
      const controls = graphRef.current.getControls?.();
      controls?.reset?.(true);
    }
  }, []);

  const isDark = muiTheme.palette.mode === 'dark';
  const graphTheme: Theme = useMemo(() => {
    const base = isDark ? darkTheme : lightTheme;
    return {
      ...base,
      canvas: {
        ...base.canvas,
        background: isDark ? '#0b0f19' : '#f8fafc',
      },
    };
  }, [isDark]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        spacing={1}
        sx={{
          p: 1.25,
          borderBottom: 1,
          borderColor: 'divider',
          alignItems: { md: 'center' },
          bgcolor: 'background.paper',
          flexWrap: 'wrap',
          gap: 1,
        }}
      >
        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
          Embedding space
        </Typography>

        <FormControl size="small" sx={{ minWidth: 160 }}>
          <InputLabel id="embedding-file-filter-label">Document</InputLabel>
          <Select
            labelId="embedding-file-filter-label"
            label="Document"
            value={fileHashFilter}
            onChange={(event) => setFileHashFilter(event.target.value)}
          >
            <MenuItem value="">All documents</MenuItem>
            {documents.map((doc) => (
              <MenuItem key={doc.fileHash} value={doc.fileHash}>
                {doc.fileName}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel id="embedding-color-mode-label">Color by</InputLabel>
          <Select
            labelId="embedding-color-mode-label"
            label="Color by"
            value={colorMode}
            onChange={(event) => setColorMode(event.target.value as ColorMode)}
          >
            <MenuItem value="document">Document</MenuItem>
            <MenuItem value="kind">Chunk kind</MenuItem>
          </Select>
        </FormControl>

        <TextField
          size="small"
          label="Filter chunks"
          value={searchFilter}
          onChange={(event) => setSearchFilter(event.target.value)}
          sx={{ minWidth: 150, flex: 1 }}
        />

        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={showSequenceEdges}
              onChange={(e) => setShowSequenceEdges(e.target.checked)}
            />
          }
          label={<Typography variant="caption">Sequence</Typography>}
        />

        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={showSimilarityEdges}
              onChange={(e) => setShowSimilarityEdges(e.target.checked)}
            />
          }
          label={<Typography variant="caption">Similarity</Typography>}
        />

        <Button
          size="small"
          variant="outlined"
          startIcon={<CenterFocusStrongIcon fontSize="small" />}
          onClick={handleFitView}
        >
          Fit view
        </Button>

        <Button
          size="small"
          startIcon={<RefreshIcon fontSize="small" />}
          onClick={() => void loadProjection()}
          disabled={loading}
        >
          Refresh
        </Button>
      </Stack>

      {embeddingModel ? (
        <Typography variant="caption" color="text.secondary" sx={{ px: 1.5, py: 0.5, bgcolor: 'background.default' }}>
          Model: <strong>{embeddingModel}</strong>
          {truncated ? ' · Showing first 500 indexed chunks' : ''}
          {` · ${nodes.length} nodes · ${graphEdges.length} relationship arrows`}
        </Typography>
      ) : null}

      <Box sx={{ flex: 1, display: 'flex', minHeight: 0, position: 'relative' }}>
        <Box sx={{ flex: 1, minWidth: 0, height: '100%', position: 'relative', bgcolor: isDark ? '#0f172a' : '#f8fafc' }}>
          {loading ? (
            <Box sx={{ height: '100%', display: 'grid', placeItems: 'center' }}>
              <CircularProgress />
            </Box>
          ) : error ? (
            <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 2 }}>
              <Alert severity="error" sx={{ maxWidth: 480 }}>{error}</Alert>
            </Box>
          ) : nodes.length === 0 ? (
            <Box sx={{ height: '100%', display: 'grid', placeItems: 'center', p: 2 }}>
              <Alert severity="info" sx={{ maxWidth: 480 }}>
                No indexed chunks are available for projection yet.
              </Alert>
            </Box>
          ) : (
            <GraphCanvas
              ref={graphRef}
              nodes={nodes}
              edges={graphEdges}
              layoutType="forceDirected3d"
              cameraMode="rotate"
              selections={selectedNodeId ? [selectedNodeId] : []}
              onNodeClick={handleNodeClick}
              theme={graphTheme}
            />
          )}
        </Box>

        {selectedPoint ? (
          <Card
            variant="outlined"
            sx={{
              width: 340,
              m: 1.5,
              alignSelf: 'stretch',
              overflow: 'auto',
              boxShadow: 2,
              zIndex: 10,
            }}
          >
            <CardContent>
              <Stack spacing={1}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>Chunk details</Typography>
                  <Button size="small" onClick={() => setSelectedNodeId(null)}>Close</Button>
                </Stack>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                  <Chip size="small" label={selectedPoint.file_name || selectedPoint.file_hash.slice(0, 10)} />
                  <Chip size="small" variant="outlined" label={`chunk ${selectedPoint.chunk_id ?? '?'}`} />
                  {pageLabel(selectedPoint) ? (
                    <Chip size="small" variant="outlined" label={pageLabel(selectedPoint)} />
                  ) : null}
                  {selectedPoint.table_id ? (
                    <Chip size="small" variant="outlined" color="warning" label={`table ${selectedPoint.table_id}`} />
                  ) : null}
                  {selectedPoint.section_id ? (
                    <Chip size="small" variant="outlined" color="secondary" label={`section ${selectedPoint.section_id}`} />
                  ) : null}
                </Stack>
                <Typography
                  component="pre"
                  variant="caption"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    maxHeight: 180,
                    overflow: 'auto',
                    bgcolor: 'action.hover',
                    p: 1,
                    borderRadius: 1,
                    fontFamily: 'monospace',
                  }}
                >
                  {selectedPoint.text || '(empty text)'}
                </Typography>
                <JsonPreview
                  value={{
                    source_kind: selectedPoint.source_kind,
                    section_id: selectedPoint.section_id,
                    table_id: selectedPoint.table_id,
                    x: selectedPoint.x,
                    y: selectedPoint.y,
                    z: selectedPoint.z,
                  }}
                  maxHeight={160}
                />
                {onOpenDocumentCitation ? (
                  <Button
                    size="small"
                    variant="contained"
                    onClick={() => handleJumpToDocument(selectedPoint)}
                  >
                    Jump to document
                  </Button>
                ) : null}
              </Stack>
            </CardContent>
          </Card>
        ) : null}
      </Box>
    </Box>
  );
}
