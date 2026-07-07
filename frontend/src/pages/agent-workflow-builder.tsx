import dynamic from 'next/dynamic';

const AgentWorkflowBuilderPage = dynamic(
  () => import('../components/agent-pattern-builder/AgentPatternBuilderPage'),
  { ssr: false },
);

export default AgentWorkflowBuilderPage;
