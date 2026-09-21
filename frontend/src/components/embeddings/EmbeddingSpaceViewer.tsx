import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  FormControlLabel,
  IconButton,
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
import CloseIcon from '@mui/icons-material/Close';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RefreshIcon from '@mui/icons-material/Refresh';
import {
  ChunkIdentityChips,
  ChunkInspectorBody,
  OverlayResizablePanel,
} from '../inspector';
import { useResizablePanelRatio } from '../../hooks/useResizablePanelRatio';
import {
  getThreadEmbeddingProjection,
  type EmbeddingPoint3D,
  type EmbeddingProjectionEdge,
  type EmbeddingSourceFamily,
} from '../../lib/embedding-projection';
import type { PdfTab } from '../../lib/document-tabs';
import type { DocumentCanvasCitationTarget } from '../../lib/canvas-spec';
import { chunkPageLabel } from '../../lib/chunk-page-label';
import {
  Badge,
  GraphCanvas,
  Sphere,
  darkTheme,
  lightTheme,
  useSelection,
  type ClusterRenderer,
  type GraphCanvasRef,
  type GraphEdge,
  type GraphNode,
  type NodeRendererProps,
  type Theme,
} from 'reagraph';
import { DoubleSide } from 'three';
import { assignDocumentColors, chunkGraphLabel } from './document-colors';

type ColorMode = 'document' | 'kind';

const DENSE_NODE_COUNT = 40;
const DOCUMENT_SOURCE_KINDS = new Set(['pdf', 'webpage', 'browser', 'browser_capture']);
const SOURCE_FAMILY_OPTIONS: Array<{ value: EmbeddingSourceFamily; label: string }> = [
  { value: 'all', label: 'All sources' },
  { value: 'documents', label: 'Documents' },
  { value: 'chat', label: 'Chat' },
  { value: 'web_search', label: 'Web search' },
  { value: 'memory', label: 'Memories' },
];

function ClusterHull({
  color,
  innerRadius,
  opacity,
  outerRadius,
  padding,
}: {
  color: string;
  innerRadius: number;
  opacity: number;
  outerRadius: number;
  padding: number;
}) {
  return (
    <>
      <mesh>
        <ringGeometry args={[outerRadius, 0, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          depthTest={false}
          opacity={opacity * 0.1}
          side={DoubleSide}
        />
      </mesh>
      <mesh>
        <ringGeometry args={[outerRadius, innerRadius + padding, 64]} />
        <meshBasicMaterial
          color={color}
          transparent
          depthTest={false}
          opacity={opacity * 0.55}
          side={DoubleSide}
        />
      </mesh>
    </>
  );
}

function kindColor(point: EmbeddingPoint3D): string {
  if (point.source_kind === 'chat') return '#14b8a6';
  if (point.source_kind === 'web_search') return '#22c55e';
  if (point.source_kind === 'memory') return '#eab308';
  if (point.table_id) return '#f97316';
  if (point.section_id) return '#a855f7';
  return '#3b82f6';
}

function isDocumentPoint(point: EmbeddingPoint3D, documents: readonly PdfTab[]): boolean {
  if (documents.some((doc) => doc.fileHash === point.file_hash)) return true;
  return DOCUMENT_SOURCE_KINDS.has(point.source_kind || '');
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
  const graphAreaRef = useRef<HTMLDivElement | null>(null);
  const { ratio: detailsPanelRatio, resizing: detailsPanelResizing, onResizeStart: onDetailsPanelResizeStart } = useResizablePanelRatio(
    graphAreaRef,
    0.42,
    { min: 0.18, max: 0.65 },
  );

  const [points, setPoints] = useState<EmbeddingPoint3D[]>([]);
  const [edges, setEdges] = useState<EmbeddingProjectionEdge[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | null>(null);
  const [fileHashFilter, setFileHashFilter] = useState<string>('');
  const [sourceFamily, setSourceFamily] = useState<EmbeddingSourceFamily>('all');
  const [groupFilter, setGroupFilter] = useState<string>('');
  const [colorMode, setColorMode] = useState<ColorMode>('document');
  const [searchFilter, setSearchFilter] = useState('');
  const [showSequenceEdges, setShowSequenceEdges] = useState(true);
  const [showSimilarityEdges, setShowSimilarityEdges] = useState(false);
  const [showClusters, setShowClusters] = useState(true);
  const [labelOverride, setLabelOverride] = useState<boolean | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const [graphCanvasMounted, setGraphCanvasMounted] = useState(false);
  const [canvasNodes, setCanvasNodes] = useState<GraphNode[]>([]);
  const [canvasEdges, setCanvasEdges] = useState<GraphEdge[]>([]);
  const clearSelectionsRef = useRef<() => void>(() => {});

  const isDark = muiTheme.palette.mode === 'dark';

  const loadProjection = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getThreadEmbeddingProjection(threadId, {
        fileHash: fileHashFilter || undefined,
        sourceFamily,
        limit: 500,
      });
      setPoints(response.points);
      setEdges(response.edges || []);
      setTruncated(response.truncated);
      setEmbeddingModel(response.embedding_model);
      setSelectedNodeId(null);
      setDetailsExpanded(false);
      clearSelectionsRef.current();
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load embedding projection');
      setPoints([]);
      setEdges([]);
    } finally {
      setLoading(false);
    }
  }, [fileHashFilter, sourceFamily, threadId]);

  useEffect(() => {
    void loadProjection();
  }, [loadProjection]);

  const visiblePoints = useMemo(() => {
    const needle = searchFilter.trim().toLowerCase();
    return points.filter((point) => {
      if (groupFilter && point.file_hash !== groupFilter) return false;
      if (!needle) return true;
      return (
        point.text.toLowerCase().includes(needle)
        || String(point.chunk_id ?? '').includes(needle)
        || (point.file_name || '').toLowerCase().includes(needle)
        || (point.section_id || '').toLowerCase().includes(needle)
        || (point.table_id || '').toLowerCase().includes(needle)
        || (point.source_kind || '').toLowerCase().includes(needle)
      );
    });
  }, [groupFilter, points, searchFilter]);

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

  const showLabels = labelOverride ?? visiblePoints.length <= DENSE_NODE_COUNT;

  const nodes: GraphNode[] = useMemo(() => {
    return visiblePoints.map((point) => {
      const fill = colorMode === 'kind'
        ? kindColor(point)
        : (documentColors[point.file_hash] || documentColors[uniqueVisibleHashes[0]] || '#3b82f6');
      const pageStr = chunkPageLabel(point);
      const cluster = clusterNames.get(point.file_hash) || point.file_hash;
      return {
        id: point.id,
        label: showLabels
          ? chunkGraphLabel(point.text)
          : undefined,
        subLabel: showLabels ? (pageStr || undefined) : undefined,
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
  }, [clusterNames, colorMode, documentColors, showLabels, uniqueVisibleHashes, visiblePoints]);

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
          dashed: !isSequence,
          dashArray: isSequence ? undefined : ([5, 4] as [number, number]),
        };
      });
  }, [edges, muiTheme.palette.mode, showSequenceEdges, showSimilarityEdges, visiblePointIds]);

  useEffect(() => {
    if (nodes.length === 0) return;
    setCanvasNodes(nodes);
    setCanvasEdges(graphEdges);
    setGraphCanvasMounted(true);
  }, [graphEdges, nodes]);

  const {
    selections,
    actives,
    onNodeClick,
    onCanvasClick,
    clearSelections,
  } = useSelection({
    ref: graphRef,
    nodes: canvasNodes,
    edges: canvasEdges,
    type: 'single',
    pathHoverType: 'direct',
    pathSelectionType: 'out',
    focusOnSelect: false,
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

  useEffect(() => {
    setDetailsExpanded(Boolean(selectedNodeId));
  }, [selectedNodeId]);

  const handleCloseDetails = useCallback(() => {
    setSelectedNodeId(null);
    setDetailsExpanded(false);
    clearSelections();
  }, [clearSelections]);

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

  const handleGroupChipClick = useCallback((hash: string) => {
    const isDocument = documents.some((doc) => doc.fileHash === hash);
    setGroupFilter((current) => (current === hash ? '' : hash));
    if (isDocument && (sourceFamily === 'all' || sourceFamily === 'documents')) {
      setFileHashFilter((current) => (current === hash ? '' : hash));
    } else {
      setFileHashFilter('');
    }
  }, [documents, sourceFamily]);

  const clusterFillByName = useMemo(() => {
    const fills: Record<string, string> = {};
    clusterNames.forEach((name, hash) => {
      fills[name] = documentColors[hash] || '#64748b';
    });
    return fills;
  }, [clusterNames, documentColors]);

  const renderCluster = useCallback<ClusterRenderer>(({ innerRadius, label, opacity, outerRadius, padding }) => (
    <ClusterHull
      color={clusterFillByName[label?.text || ''] || '#64748b'}
      innerRadius={innerRadius}
      opacity={opacity}
      outerRadius={outerRadius}
      padding={padding}
    />
  ), [clusterFillByName]);

  const renderNode = useCallback((props: NodeRendererProps) => {
    const chunkId = props.node.data?.chunk_id;
    return (
      <>
        <Sphere {...props} />
        <Badge
          {...props}
          label={chunkId == null ? '?' : String(chunkId)}
          backgroundColor={isDark ? '#111827' : '#ffffff'}
          textColor={isDark ? '#f9fafb' : '#111827'}
          strokeColor={typeof props.color === 'string' ? props.color : undefined}
          strokeWidth={0.035}
          badgeSize={0.75}
          fontSize={0.34}
          fontWeight={700}
          position="top-right"
        />
      </>
    );
  }, [isDark]);

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
        opacity: 0.55,
        selectedOpacity: 1,
        inactiveOpacity: 0.18,
      },
      edge: {
        ...base.edge,
        opacity: 0.42,
        selectedOpacity: 0.95,
        inactiveOpacity: 0.08,
      },
      cluster: {
        ...base.cluster,
        opacity: 0.55,
        selectedOpacity: 0.9,
        inactiveOpacity: 0.12,
        label: undefined,
      },
    };
  }, [isDark, muiTheme.palette.background.default]);

  const legendItems = useMemo(() => {
    const hashes: string[] = [];
    const seen = new Set<string>();
    const addHash = (hash: string | undefined) => {
      if (!hash || seen.has(hash)) return;
      seen.add(hash);
      hashes.push(hash);
    };
    if (sourceFamily === 'all' || sourceFamily === 'documents') {
      documents.forEach((doc) => addHash(doc.fileHash));
    }
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
  }, [documentColors, documents, points, sourceFamily]);

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

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel id="embedding-source-family-label">Source</InputLabel>
          <Select
            labelId="embedding-source-family-label"
            label="Source"
            value={sourceFamily}
            onChange={(event) => {
              const next = event.target.value as EmbeddingSourceFamily;
              setSourceFamily(next);
              setGroupFilter('');
              setFileHashFilter('');
            }}
          >
            {SOURCE_FAMILY_OPTIONS.map((option) => (
              <MenuItem key={option.value} value={option.value}>
                {option.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {sourceFamily === 'all' || sourceFamily === 'documents' ? (
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="embedding-file-filter-label">Document</InputLabel>
            <Select
              labelId="embedding-file-filter-label"
              label="Document"
              value={fileHashFilter}
              onChange={(event) => {
                const next = String(event.target.value);
                setFileHashFilter(next);
                setGroupFilter(next);
              }}
            >
              <MenuItem value="">All documents</MenuItem>
              {documents.map((doc) => (
                <MenuItem key={doc.fileHash} value={doc.fileHash}>
                  {doc.fileName}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        ) : null}

        <FormControl size="small" sx={{ minWidth: 130 }}>
          <InputLabel id="embedding-color-mode-label">Color by</InputLabel>
          <Select
            labelId="embedding-color-mode-label"
            label="Color by"
            value={colorMode}
            onChange={(event) => setColorMode(event.target.value as ColorMode)}
          >
            <MenuItem value="document">Group</MenuItem>
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
              onChange={(e) => setShowClusters(e.target.checked)}
            />
          }
          label={<Typography variant="caption">Clusters</Typography>}
        />

        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={showLabels}
              onChange={(e) => setLabelOverride(e.target.checked)}
            />
          }
          label={<Typography variant="caption">Labels</Typography>}
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
            const selected = groupFilter === item.hash || fileHashFilter === item.hash;
            const dimmed = Boolean(groupFilter || fileHashFilter) && !selected;
            return (
              <Tooltip
                key={item.hash}
                title={selected ? 'Show all groups' : `Show only ${item.label}`}
              >
                <Chip
                  size="small"
                  clickable
                  aria-pressed={selected}
                  label={item.label}
                  onClick={() => handleGroupChipClick(item.hash)}
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
          {truncated ? ' · Showing a capped mix of indexed chunks' : ''}
          {` · ${nodes.length} nodes · ${graphEdges.length} relationship arrows`}
        </Typography>
      ) : null}

      <Box sx={{ flex: 1, display: 'flex', minHeight: 0, position: 'relative', overflow: 'hidden' }}>
        <Box ref={graphAreaRef} sx={{ flex: 1, minWidth: 0, height: '100%', position: 'relative', overflow: 'hidden', bgcolor: 'background.default' }}>
          {graphCanvasMounted ? (
            <Box sx={{ position: 'absolute', inset: 0 }}>
              <GraphCanvas
                ref={graphRef}
                nodes={canvasNodes}
                edges={canvasEdges}
                layoutType="forceDirected3d"
                cameraMode="rotate"
                animated={false}
                sizingType="attribute"
                sizingAttribute="charCount"
                minNodeSize={4}
                maxNodeSize={9}
                clusterAttribute={showClusters ? 'cluster' : undefined}
                labelType={showLabels ? 'auto' : 'none'}
                edgeArrowPosition={canvasNodes.length > DENSE_NODE_COUNT ? 'none' : 'end'}
                selections={selections}
                actives={actives}
                onNodeClick={handleNodeClick}
                onCanvasClick={onCanvasClick}
                theme={graphTheme}
                onRenderCluster={showClusters ? renderCluster : undefined}
                renderNode={renderNode}
              />
            </Box>
          ) : null}
          {loading || error || nodes.length === 0 ? (
            <Box
              sx={{
                position: 'absolute',
                inset: 0,
                zIndex: 1,
                display: 'grid',
                placeItems: 'center',
                p: 2,
                bgcolor: 'background.default',
              }}
            >
              {loading ? (
                <CircularProgress />
              ) : error ? (
                <Alert severity="error" sx={{ maxWidth: 480 }}>{error}</Alert>
              ) : (
                <Alert severity="info" sx={{ maxWidth: 480 }}>
                  No indexed chunks are available for these sources yet.
                </Alert>
              )}
            </Box>
          ) : null}
          {selectedPoint ? (
            <OverlayResizablePanel
              ratio={detailsPanelRatio}
              resizing={detailsPanelResizing}
              onResizeStart={onDetailsPanelResizeStart}
              expanded={detailsExpanded}
              resizeLabel="Resize chunk details panel"
              header={(
                <Box sx={{ px: 1.25, py: 0.5 }}>
                  <Stack direction="row" spacing={0.75} alignItems="center" useFlexGap>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      Chunk
                    </Typography>
                    <Box sx={{ flex: 1, minWidth: 8 }} />
                    {onOpenDocumentCitation && isDocumentPoint(selectedPoint, documents) ? (
                      <Button size="small" variant="contained" onClick={() => handleJumpToDocument(selectedPoint)}>
                        Jump to document
                      </Button>
                    ) : null}
                    <Tooltip title={detailsExpanded ? 'Collapse details' : 'Expand details'}>
                      <IconButton
                        size="small"
                        aria-label={detailsExpanded ? 'Collapse details' : 'Expand details'}
                        aria-expanded={detailsExpanded}
                        onClick={() => setDetailsExpanded((open) => !open)}
                      >
                        {detailsExpanded ? <ExpandMoreIcon fontSize="small" /> : <ExpandLessIcon fontSize="small" />}
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Clear selection">
                      <IconButton size="small" aria-label="Clear selection" onClick={handleCloseDetails}>
                        <CloseIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                  <Box sx={{ mt: 0.5 }}>
                    <ChunkIdentityChips chunk={selectedPoint} showFileName />
                  </Box>
                </Box>
              )}
            >
              <ChunkInspectorBody
                text={selectedPoint.text}
                metadata={{
                  source_kind: selectedPoint.source_kind,
                  section_id: selectedPoint.section_id,
                  table_id: selectedPoint.table_id,
                  x: selectedPoint.x,
                  y: selectedPoint.y,
                  z: selectedPoint.z,
                  char_count: selectedPoint.text.length,
                }}
                metadataMaxHeight={false}
              />
            </OverlayResizablePanel>
          ) : null}
        </Box>
      </Box>
    </Box>
  );
}
