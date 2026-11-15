'use client';

export default function BrowserWorkspaceFrame() {
  const host = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
  return (
    <iframe
      src={`http://${host}:8090`}
      style={{ width: '100%', height: '100%', border: 'none' }}
      title="Browser"
      allow="camera; microphone; clipboard-read; clipboard-write"
    />
  );
}