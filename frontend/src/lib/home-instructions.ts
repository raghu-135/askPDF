export type HomeInstructionSection = {
  title: string;
  items: string[];
};

export const HOME_INSTRUCTION_SECTIONS: HomeInstructionSection[] = [
  {
    title: 'Projects and Threads',
    items: [
      'Create a project from the right panel, choose its embedding model, and optionally allow it to use Global memory.',
      'Open a project to manage shared project files and create threads inside that project.',
      'Create, rename, fork, clone, or delete threads and projects from the right panel actions.',
      'Fork a thread when you want a new path of conversation; choose how much thread memory to copy.',
    ],
  },
  {
    title: 'Documents and Browser Sources',
    items: [
      'Select a project or thread, then upload PDFs with the toolbar upload button.',
      'Use the Browser tab to capture a web page into the active project or thread as a searchable source.',
      'Promote thread documents to project knowledge when they should be shared across project threads.',
      'Use document tabs to switch sources, retry processing, remove thread files, and inspect PDFs.',
    ],
  },
  {
    title: 'Chat and Retrieval',
    items: [
      'Open a thread to chat with its PDFs, browser captures, project knowledge, and enabled memories.',
      'Use the model picker, context window, web search mode, and settings controls in the chat header.',
      'Open settings to select an agent workflow, tune reranking, memory recall, prompt role, tools, and custom instructions.',
      'Use cited sources, message actions, debug trace links, and thread lineage to inspect how answers were produced.',
    ],
  },
  {
    title: 'Memory & Settings',
    items: [
      'Use Global memory for cross-project preferences, Project memory for shared project facts, and Thread memory for local context.',
      'Open Memory to browse stored memories, recall status, override relationships, indexing status, and web provenance.',
      'Add, edit, delete, retry indexing, review conversation turns, and run consistency reviews from the Memory workspace.',
      'Thread and project settings control whether broader memory is recalled; stored memory can still be inspected administratively.',
    ],
  },
  {
    title: 'Agent Workflows',
    items: [
      'Open the Agent Workflow Builder from the toolbar to customize the assistant route used by chat.',
      'Start from Router RAG, Plan Execute, Evaluator Replanner, or a saved custom workflow.',
      'Edit the graph, add nodes, configure HITL gates and tool permissions, validate, save, and inspect the generated spec.',
      'Use Builder Test to try a workflow against a selected thread before choosing it in chat settings.',
    ],
  },
  {
    title: 'Review, Trace, and Playback',
    items: [
      'Use Debug Trace to inspect agent runs, route choices, tool calls, evidence, and human-review pauses.',
      'Use memory reviews to clean duplicates, conflicts, stale facts, and override relationships.',
      'Use playback controls and text selection to read PDF or chat content aloud from a chosen sentence.',
      'Use dark mode, layout docking, browser capture, and source navigation controls to shape the workspace around the task.',
    ],
  },
];
