import dynamic from 'next/dynamic';

const AgentPatternBuilderPage = dynamic(
  () => import('../components/agent-pattern-builder/AgentPatternBuilderPage'),
  { ssr: false },
);

export default AgentPatternBuilderPage;

