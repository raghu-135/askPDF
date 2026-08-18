import assert from 'node:assert/strict';
import test from 'node:test';

import {
  AgentGraphNodeStatus,
  AgentRunResumeAction,
  AgentRunStatus,
  ChatComposerIndexingStatus,
  ChatComposerStatus,
  EmbeddingReadinessStatus,
  HitlMode,
  HitlPhase,
  HitlSelectionMode,
  InterruptStatus,
  MessageRole,
  ProcessStatus,
  ReasoningFormat,
  RouteFunctionId,
  BuiltinAgentNodeType,
  ThreadFileSourceType,
} from '../src/lib/enums.ts';

test('frontend enum constants preserve API wire values', () => {
  assert.equal(ProcessStatus.Pending, 'pending');
  assert.equal(ProcessStatus.Completed, 'completed');
  assert.equal(MessageRole.User, 'user');
  assert.equal(MessageRole.Assistant, 'assistant');
  assert.equal(ThreadFileSourceType.Pdf, 'pdf');
  assert.equal(ThreadFileSourceType.Browser, 'browser');
  assert.equal(ReasoningFormat.None, 'none');
  assert.equal(EmbeddingReadinessStatus.Blocked, 'blocked');
  assert.equal(AgentRunResumeAction.ContinueWithout, 'continue_without');
  assert.equal(AgentRunResumeAction.ApproveForScope, 'approve_for_scope');
  assert.equal(AgentRunStatus.AwaitingHuman, 'awaiting_human');
  assert.equal(InterruptStatus.Pending, 'pending');
  assert.equal(HitlMode.Choice, 'choice');
  assert.equal(HitlPhase.InsideTool, 'inside_tool');
  assert.equal(HitlSelectionMode.SingleOrMulti, 'single_or_multi');
  assert.equal(ChatComposerIndexingStatus.Checking, 'checking');
  assert.equal(ChatComposerStatus.LlmToolsUnsupported, 'llm_tools_unsupported');
  assert.equal(AgentGraphNodeStatus.Planned, 'planned');
  assert.equal(RouteFunctionId.HitlGate, 'hitl_gate_route');
  assert.equal(RouteFunctionId.CorrectiveRetrieval, 'corrective_retrieval_route');
  assert.equal(RouteFunctionId.GroundedAnswer, 'grounded_answer_route');
  assert.equal(BuiltinAgentNodeType.RetrievalQualityGrader, 'retrieval_quality_grader');
});
