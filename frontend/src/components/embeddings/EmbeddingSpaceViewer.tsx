import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  Tooltip,
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
import {
  GraphCanvas,
  Sphere,
  darkTheme,
  lightTheme,
  useSelection,
  type GraphCanvasRef,
  type GraphEdge,
  type GraphNode,
  type NodeRendererProps,
  type Theme,
} from 'reagraph';
import { assignDocumentColors, chunkGraphLabel } from './document-colors';

type ColorMode = 'document' | 'kind';

function pageLabel(point: EmbeddingPoint3D): string {
  if (point.page_start != null) {
    const end = point.page_end ?? point.page_start;
    return point.page_start === end ? `p. ${point.page_start}` : `p. ${point.page_start}-${end}`;
  }
  if (point.pages) return `pages ${point.pages}`;
  return '';
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
  const [clusterOverride, setClusterOverride] = useState<boolean | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const clearSelectionsRef = useRef<() => void>(() => {});

  const isDark = muiTheme.palette.mode === 'dark';

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
      clearSelectionsRef.current();
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

  const documentColorOrder = useMemo(() => {
    const hashes = [
      ...documents.map((doc) => doc.fileHash),
      ...visiblePoints.map((point) => point.file_hash),
    ];
    return hashes;
  }, [documents, visiblePoints]);

  const documentColors = useMemo(
    () => assignDocumentColors(documentColorOrder, isDark ? 'dark' : 'light'),
    [documentColorOrder, isDark],
  );

  const clusterNames = useMemo(() => {
    const labels = new Map<string, string>();
    const used = new Set<string>();
    const sources: Array<{ hash: string; name: string }> = [
      ...documents.map((doc) => ({ hash: doc.fileHash, name: doc.fileName })),
      ...visiblePoints.map((point) => ({
        hash: point.file_hash,
        name: point.file_name || point.file_hash.slice(0, 8),
      })),
    ];
    for (const source of sources) {
      if (labels.has(source.hash)) continue;
      let label = source.name.trim() || source.hash.slice(0, 8);
      if (label.length > 28) label = `${label.slice(0, 27)}…`;
      if (used.has(label)) {
        label = `${label} · ${source.hash.slice(0, 4)}`;
      }
      used.add(label);
      labels.set(source.hash, label);
    }
    return labels;
  }, [documents, visiblePoints]);

  const uniqueVisibleHashes = useMemo(() => {
    const hashes: string[] = [];
    const seen = new Set<string>();
    for (const point of visiblePoints) {
      if (seen.has(point.file_hash)) continue;
      seen.add(point.file_hash);
      hashes.push(point.file_hash);
    }
    return hashes;
  }, [visiblePoints]);

  const showClusters = clusterOverride ?? uniqueVisibleHashes.length > 1;

  const nodes: GraphNode[] = useMemo(() => {
    return visiblePoints.map((point) => {
      const fill = colorMode === 'kind'
        ? kindColor(point)
        : (documentColors[point.file_hash] || documentColors[uniqueVisibleHashes[0]] || '#3b82f6');
      const pageStr = pageLabel(point);
      const cluster = clusterNames.get(point.file_hash) || point.file_hash;
      return {
        id: point.id,
        label: chunkGraphLabel(point.file_name, point.file_hash, point.chunk_id),
        subLabel: pageStr || undefined,
        fill,
        cluster,
        data: {
          ...point,
          charCount: point.text.length,
          cluster,
        },
        fx: typeof point.x === 'number' ? point.x * 250 : undefined,
        fy: typeof point.y === 'number' ? point.y * 250 : undefined,
        fz: typeof point.z === 'number' ? point.z * 250 : undefined,
      };
    });
  }, [clusterNames, colorMode, documentColors, uniqueVisibleHashes, visiblePoints]);

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
          size: isSequence ? 0.45 : 0.7,
          fill: color,
          arrowPlacement: 'end' as const,
          dashed: !isSequence,
          dashArray: isSequence ? undefined : ([5, 4] as [number, number]),
        };
      });
  }, [edges, muiTheme.palette.mode, showSequenceEdges, showSimilarityEdges, visiblePointIds]);

  const {
    selections,
    actives,
    onNodeClick,
    onCanvasClick,
    clearSelections,
  } = useSelection({
    ref: graphRef,
    nodes,
    edges: graphEdges,
    type: 'single',
    pathHoverType: 'direct',
    pathSelectionType: 'out',
    focusOnSelect: 'singleOnly',
    onSelection: (ids) => {
      const nodeId = ids.find((id) => visiblePointIds.has(id)) ?? null;
      setSelectedNodeId(nodeId);
    },
  });

  clearSelectionsRef.current = clearSelections;

  const handleNodeClick = useCallback((node: GraphNode) => {
    setSelectedNodeId(node.id);
    onNodeClick?.(node);
  }, [onNodeClick]);

  const selectedPoint = useMemo(() => {
    if (!selectedNodeId) return null;
    return points.find((point) => point.id === selectedNodeId) || null;
  }, [points, selectedNodeId]);

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
      play: false,
    });
  };

  const handleFitView = useCallback(() => {
    graphRef.current?.fitNodesInView?.();
  }, []);

  const handleDocumentChipClick = useCallback((hash: string) => {
    setFileHashFilter((current) => (current === hash ? '' : hash));
  }, []);

  const graphTheme: Theme = useMemo(() => {
    const base = isDark ? darkTheme : lightTheme;
    return {
      ...base,
      canvas: {
        ...base.canvas,
        background: muiTheme.palette.background.default,
      },
      node: {
        ...base.node,
        opacity: 0.96,
        selectedOpacity: 1,
        inactiveOpacity: 0.22,
      },
      edge: {
        ...base.edge,
        opacity: 0.42,
        selectedOpacity: 0.95,
        inactiveOpacity: 0.08,
      },
    };
  }, [isDark, muiTheme.palette.background.default]);

  const renderNode = useCallback(
    (props: NodeRendererProps) => <Sphere {...props} selected={false} />,
    [],
  );

  const legendItems = useMemo(() => {
    const hashes: string[] = [];
    const seen = new Set<string>();
    const addHash = (hash: string | undefined) => {
      if (!hash || seen.has(hash)) return;
      seen.add(hash);
      hashes.push(hash);
    };
    documents.forEach((doc) => addHash(doc.fileHash));
    points.forEach((point) => addHash(point.file_hash));
    return hashes.map((hash) => {
      const documentTab = documents.find((doc) => doc.fileHash === hash);
      const sample = points.find((point) => point.file_hash === hash);
      return {
        hash,
        label: documentTab?.fileName || sample?.file_name || hash.slice(0, 8),
        fill: documentColors[hash],
      };
    });
  }, [documentColors, documents, points]);

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

        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={showClusters}
              onChange={(e) => setClusterOverride(e.target.checked)}
            />
          }
          label={<Typography variant="caption">Clusters</Typography>}
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

      {legendItems.length > 0 ? (
        <Stack
          direction="row"
          spacing={0.5}
          useFlexGap
          flexWrap="wrap"
          sx={{ px: 1.5, py: 0.75, bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}
        >
          {legendItems.map((item) => {
            const selected = fileHashFilter === item.hash;
            const dimmed = Boolean(fileHashFilter) && !selected;
            return (
              <Tooltip
                key={item.hash}
                title={selected ? 'Show all documents' : `Show only ${item.label}`}
              >
                <Chip
                  size="small"
                  clickable
                  aria-pressed={selected}
                  label={item.label}
                  onClick={() => handleDocumentChipClick(item.hash)}
                  sx={{
                    bgcolor: item.fill,
                    color: isDark ? '#0b1220' : '#fff',
                    fontWeight: 600,
                    maxWidth: 220,
                    opacity: dimmed ? 0.42 : 1,
                    outline: selected ? '2px solid' : 'none',
                    outlineColor: 'text.primary',
                    outlineOffset: 1,
                  }}
                />
              </Tooltip>
            );
          })}
        </Stack>
      ) : null}

      {embeddingModel ? (
        <Typography variant="caption" color="text.secondary" sx={{ px: 1.5, py: 0.5, bgcolor: 'background.default' }}>
          Model: <strong>{embeddingModel}</strong>
          {truncated ? ' · Showing first 500 indexed chunks' : ''}
          {` · ${nodes.length} nodes · ${graphEdges.length} relationship arrows`}
        </Typography>
      ) : null}

      <Box sx={{ flex: 1, display: 'flex', minHeight: 0, position: 'relative' }}>
        <Box sx={{ flex: 1, minWidth: 0, height: '100%', position: 'relative', bgcolor: 'background.default' }}>
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
              sizingType="attribute"
              sizingAttribute="charCount"
              minNodeSize={5}
              maxNodeSize={14}
              clusterAttribute={showClusters ? 'cluster' : undefined}
              labelType="auto"
              selections={selections}
              actives={actives}
              onNodeClick={handleNodeClick}
              onCanvasClick={onCanvasClick}
              theme={graphTheme}
              renderNode={renderNode}
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
                  <Button
                    size="small"
                    onClick={() => {
                      setSelectedNodeId(null);
                      clearSelections();
                    }}
                  >
                    Close
                  </Button>
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
                    char_count: selectedPoint.text.length,
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
