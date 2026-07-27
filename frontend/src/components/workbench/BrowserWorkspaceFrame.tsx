export default function BrowserWorkspaceFrame() {
  return (
    <iframe
      src="http://localhost:8090"
      style={{ width: '100%', height: '100%', border: 'none' }}
      title="Browser"
      allow="camera; microphone; clipboard-read; clipboard-write"
    />
  );
}
