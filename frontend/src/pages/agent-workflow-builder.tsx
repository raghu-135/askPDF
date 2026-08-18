import dynamic from 'next/dynamic';

const AgentWorkflowBuilderPage = dynamic(
  () => import('../components/agent-workflow-builder/AgentWorkflowBuilderPage'),
  { ssr: false },
);

export default AgentWorkflowBuilderPage;
