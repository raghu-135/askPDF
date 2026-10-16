import React from 'react';
import { Thread } from '../lib/api';
import ThreadStatsList, { threadDocumentsFromMeta } from './workbench/ThreadStatsList';

interface ThreadLineageTooltipContentProps {
  thread: Thread;
  threadsById: Map<string, Thread>;
  onOpenThread?: (thread: Thread) => void;
}

const ThreadLineageTooltipContent: React.FC<ThreadLineageTooltipContentProps> = ({
  thread,
  threadsById,
  onOpenThread,
}) => (
  <ThreadStatsList
    thread={thread}
    threadsById={threadsById}
    onOpenThread={onOpenThread}
    documents={threadDocumentsFromMeta(thread)}
    compact
    showMessageCounts={false}
  />
);

export default ThreadLineageTooltipContent;
