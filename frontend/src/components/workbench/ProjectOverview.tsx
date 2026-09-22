import WorkspaceEmptyState from './WorkspaceEmptyState';
import AddSourcesActions from './AddSourcesActions';

export default function ProjectOverview({
  projectName,
  documentCount,
  darkMode = false,
  onCreateThread,
  onCapturePage,
  onRequestUpload,
  isBrowserCapturing = false,
}: {
  projectName: string;
  documentCount: number;
  darkMode?: boolean;
  onCreateThread: () => void;
  onCapturePage: () => void;
  onRequestUpload: () => void;
  isBrowserCapturing?: boolean;
}) {
  return (
    <WorkspaceEmptyState
      darkMode={darkMode}
      title={projectName}
      description={documentCount > 0
        ? 'Open a document tab to inspect project knowledge, or add another source.'
        : 'Add a PDF, open a thread, or capture a page. Browser stays available as a tab when you need it.'}
      actions={(
        <AddSourcesActions
          onRequestUpload={onRequestUpload}
          onCapturePage={onCapturePage}
          onCreateThread={onCreateThread}
          isBrowserCapturing={isBrowserCapturing}
        />
      )}
    />
  );
}
