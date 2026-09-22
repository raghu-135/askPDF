import type { Thread } from '../../lib/api';
import { THREAD_GUIDE_SECTIONS } from '../../lib/workspace-guides';
import type { PdfTab } from '../../lib/document-tabs';
import AddSourcesActions from './AddSourcesActions';
import ThreadStatsList, { threadDocumentsFromMeta } from './ThreadStatsList';
import ThreadChatSettingsPanel from './ThreadChatSettingsPanel';
import WorkspaceOverviewPage from './WorkspaceOverviewPage';
import { useThreadChatSettings } from '../../lib/thread-chat-settings-context';

export default function ThreadOverview({
  thread,
  projectName,
  threadsById,
  documents = [],
  darkMode = false,
  onCapturePage,
  onRequestUpload,
  isBrowserCapturing = false,
  onOpenThread,
  onOpenDocument,
}: {
  thread: Thread;
  projectName?: string | null;
  threadsById?: Map<string, Thread>;
  documents?: readonly PdfTab[];
  darkMode?: boolean;
  onCapturePage?: () => void;
  onRequestUpload?: () => void;
  isBrowserCapturing?: boolean;
  onOpenThread?: (thread: Thread) => void;
  onOpenDocument?: (documentId: string) => void;
}) {
  const threadSettings = useThreadChatSettings();
  const metaDocuments = threadDocumentsFromMeta(thread);
  const documentRows = documents.length > 0
    ? documents.map((document) => ({
      id: document.id,
      fileName: document.fileName,
      pageCount: undefined,
      addedAt: document.addedAt,
      onSelect: onOpenDocument ? () => onOpenDocument(document.id) : undefined,
    }))
    : metaDocuments.map((document) => ({
      ...document,
      onSelect: onOpenDocument ? () => onOpenDocument(document.id) : undefined,
    }));

  return (
    <WorkspaceOverviewPage
      title={thread.name}
      subtitle={projectName ? `Thread in ${projectName}` : 'Conversation workspace for this thread.'}
      darkMode={darkMode}
      guideDefaultExpanded={false}
      actions={(
        <AddSourcesActions
          onRequestUpload={onRequestUpload || (() => undefined)}
          onCapturePage={onCapturePage || (() => undefined)}
          isBrowserCapturing={isBrowserCapturing}
        />
      )}
      guideSections={THREAD_GUIDE_SECTIONS}
      guideTitle="How Threads work"
      stats={(
        <ThreadStatsList
          thread={thread}
          projectName={projectName}
          threadsById={threadsById}
          onOpenThread={onOpenThread}
          documents={documentRows}
          showMessageCounts
        />
      )}
      settings={threadSettings ? <ThreadChatSettingsPanel /> : undefined}
      settingsTitle="AI prompt settings"
    />
  );
}
