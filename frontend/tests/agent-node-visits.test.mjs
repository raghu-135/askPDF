import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applySelectedVisitOverlay,
  agentNodeVisitKey,
  getChronologicalNodeVisits,
  getLatestNodeVisit,
  getNextNodeVisit,
  getNodeVisitRoute,
  getPreviousNodeVisit,
  normalizeVisitIndex,
  toAgentNodeVisitRef,
} from '../src/components/agent-graph/agent-node-visits.ts';

const visit = (id, visitIndex, raw = {}, route) => ({
  id,
  label: id,
  instanceLabel: id,
  visitIndex,
  skipped: false,
  warningCodes: [],
  raw,
  route,
});

test('normalizes visit identities and creates stable keys', () => {
  assert.equal(normalizeVisitIndex(undefined), 1);
  assert.equal(normalizeVisitIndex(0), 1);
  assert.equal(normalizeVisitIndex(2.8), 2);
  assert.deepEqual(toAgentNodeVisitRef(visit('router', undefined)), { nodeId: 'router', visitIndex: 1 });
  assert.equal(agentNodeVisitKey({ nodeId: 'router', visitIndex: 2 }), 'router:2');
  assert.equal(agentNodeVisitKey(visit('router', undefined)), 'router:1');
});

test('preserves timeline order and selects the latest visit for one node', () => {
  const nodes = [
    visit('evaluator', 2),
    visit('retriever', 1),
    visit('evaluator', 1),
  ];
  assert.deepEqual(getChronologicalNodeVisits(nodes, 'evaluator'), [nodes[0], nodes[2]]);
  assert.equal(getLatestNodeVisit(nodes, 'evaluator'), nodes[2]);
  assert.equal(getLatestNodeVisit(nodes, 'missing'), undefined);
});

test('navigates adjacent invocations without wrapping', () => {
  const nodes = [
    visit('evaluator', 1),
    visit('replanner', 1),
    visit('evaluator', 2),
    visit('evaluator', 3),
  ];
  assert.equal(getPreviousNodeVisit(nodes, { nodeId: 'evaluator', visitIndex: 2 }), nodes[0]);
  assert.equal(getNextNodeVisit(nodes, { nodeId: 'evaluator', visitIndex: 2 }), nodes[3]);
  assert.equal(getPreviousNodeVisit(nodes, { nodeId: 'evaluator', visitIndex: 1 }), undefined);
  assert.equal(getNextNodeVisit(nodes, { nodeId: 'evaluator', visitIndex: 3 }), undefined);
  assert.equal(getNextNodeVisit(nodes, { nodeId: 'evaluator', visitIndex: 99 }), undefined);
});

test('resolves ordinary and evaluator-specific routes for the selected invocation', () => {
  assert.equal(getNodeVisitRoute(visit('router', 1, {}, 'document')), 'document');
  assert.equal(getNodeVisitRoute(visit('evaluator', 1, { evaluator_route: 'replan' })), 'replan');
  assert.equal(getNodeVisitRoute(visit('evaluator', 2, {
    detail: { event: { evaluatorRoute: 'answer' } },
    evaluator_route: 'replan',
  }, 'execute')), 'answer');
  assert.equal(getNodeVisitRoute(undefined), undefined);
});

test('overlays only the selected node visit and its conditional route', () => {
  const traceNodes = [
    visit('evaluator', 1, { evaluator_route: 'replan' }),
    visit('retriever', 2),
    visit('evaluator', 2, { evaluator_route: 'answer' }),
  ];
  const graph = {
    nodes: [
      { id: 'evaluator', type: 'evaluator', label: 'Evaluator' },
      { id: 'retriever', type: 'retriever', label: 'Retriever' },
    ],
    edges: [
      { id: 'answer', source: 'evaluator', target: 'finalizer', route: 'answer', conditional: true, selected: true, active: true },
      { id: 'replan', source: 'evaluator', target: 'replanner', route: 'replan', conditional: true, selected: false, active: true },
      { id: 'other', source: 'retriever', target: 'evaluator', conditional: false, selected: false, active: true },
    ],
  };

  const overlaid = applySelectedVisitOverlay(graph, traceNodes, { nodeId: 'evaluator', visitIndex: 1 });

  assert.equal(overlaid.nodes[0].selectedVisitIndex, 1);
  assert.equal(overlaid.nodes[0].selectedVisitPosition, 1);
  assert.equal(overlaid.nodes[1], graph.nodes[1]);
  assert.equal(overlaid.edges.find((edge) => edge.id === 'answer').selected, false);
  assert.equal(overlaid.edges.find((edge) => edge.id === 'replan').selected, true);
  assert.equal(overlaid.edges.find((edge) => edge.id === 'other'), graph.edges[2]);
});

test('overlays an operation linked to topology with a different operation id', () => {
  const operation = {
    ...visit('runtime-step-7', 2, { evaluator_route: 'answer' }),
    topologyRef: { kind: 'graph_node', id: 'evaluator' },
  };
  const graph = {
    nodes: [{ id: 'evaluator', type: 'evaluator', label: 'Evaluator' }],
    edges: [{ id: 'answer', source: 'evaluator', target: 'finalizer', route: 'answer', conditional: true }],
  };

  const overlaid = applySelectedVisitOverlay(graph, [operation], { nodeId: 'evaluator', visitIndex: 2 });

  assert.equal(overlaid.nodes[0].selectedVisitIndex, 2);
  assert.equal(overlaid.nodes[0].selectedVisitPosition, 1);
  assert.equal(overlaid.edges[0].selected, true);
});
