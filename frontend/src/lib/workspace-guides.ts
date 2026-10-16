import type { HomeInstructionSection } from './home-instructions';

export const PROJECT_GUIDE_SECTIONS: HomeInstructionSection[] = [
  {
    title: 'What is a project?',
    items: [
      'A project is a shared workspace with a locked embedding model and project-level knowledge.',
      'Threads inside a project inherit the same embedding model and can access shared project files.',
      'Use projects to group related research, documents, and conversations.',
    ],
  },
  {
    title: 'Capabilities',
    items: [
      'Upload PDFs and browser captures as shared project knowledge.',
      'Create threads for separate conversations that can reuse project sources.',
      'Browse and manage project memory from the Memory tab.',
      'Control whether threads in this project may recall global memory.',
    ],
  },
  {
    title: 'Related tabs',
    items: [
      'Memory — browse and curate project-level memories.',
      'Documents — inspect shared PDFs and browser captures attached to the project.',
      'Browser — capture web pages into project knowledge.',
    ],
  },
];

export const THREAD_GUIDE_SECTIONS: HomeInstructionSection[] = [
  {
    title: 'What is a thread?',
    items: [
      'A thread is a conversation scoped to a project, with its own messages, files, and AI prompt settings.',
      'Thread files stay local unless you promote them to project knowledge.',
      'Fork a thread to explore a new path while preserving lineage.',
    ],
  },
  {
    title: 'Capabilities',
    items: [
      'Chat with attached PDFs, browser captures, project knowledge, and enabled memories.',
      'Tune agent workflow, memory recall, reranking, and prompt instructions on this page.',
      'Upload sources or capture pages from the Documents and Browser tabs.',
      'Inspect agent runs from Debug Trace and research output from Canvas.',
    ],
  },
  {
    title: 'Related tabs',
    items: [
      'Documents — switch between thread and project files, upload new sources.',
      'Memory — browse thread, project, and global memories used in answers.',
      'Embeddings — visualize how sources are embedded for retrieval.',
      'Debug Trace — inspect agent routes, tool calls, and evidence.',
    ],
  },
];
