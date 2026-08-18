import React from 'react';
import AgentGraphCanvas from './AgentGraphCanvas';

export default function AgentDebugCanvas(
  props: Omit<React.ComponentProps<typeof AgentGraphCanvas>, 'mode'>,
) {
  return <AgentGraphCanvas {...props} mode="run-debug" />;
}
