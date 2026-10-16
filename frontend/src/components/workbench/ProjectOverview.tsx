import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Stack,
  Typography,
} from '@mui/material';
import { OverviewSeparatedItem } from './OverviewSection';
import {
  getProjectLifecycleSummary,
  updateProject,
  type Project,
  type ProjectLifecycleSummary,
} from '../../lib/api';
import { PROJECT_GUIDE_SECTIONS } from '../../lib/workspace-guides';
import type { PdfTab } from '../../lib/document-tabs';
import AddSourcesActions from './AddSourcesActions';
import ProjectSettingsForm from './ProjectSettingsForm';
import WorkspaceOverviewPage from './WorkspaceOverviewPage';

export default function ProjectOverview({
  project,
  documents = [],
  darkMode = false,
  onCreateThread,
  onCapturePage,
  onRequestUpload,
  isBrowserCapturing = false,
  onProjectUpdated,
  onOpenDocument,
  onCloneProject,
  onCloneProjectWithThreads,
  onDeleteProject,
  projectReady = true,
}: {
  project: Project;
  documents?: readonly PdfTab[];
  darkMode?: boolean;
  onCreateThread: () => void;
  onCapturePage: () => void;
  onRequestUpload: () => void;
  isBrowserCapturing?: boolean;
  onProjectUpdated?: (project: Project) => void;
  onOpenDocument?: (documentId: string) => void;
  onCloneProject?: () => void;
  onCloneProjectWithThreads?: () => void;
  onDeleteProject?: () => void;
  projectReady?: boolean;
}) {
  const [projectName, setProjectName] = useState(project.name);
  const [allowGlobalMemory, setAllowGlobalMemory] = useState(
    project.settings_json?.memory?.project_reads_user_memory === true,
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [lifecycle, setLifecycle] = useState<ProjectLifecycleSummary | null>(null);
  const [lifecycleLoading, setLifecycleLoading] = useState(false);
  const [lifecycleError, setLifecycleError] = useState('');

  useEffect(() => {
    setProjectName(project.name);
    setAllowGlobalMemory(project.settings_json?.memory?.project_reads_user_memory === true);
  }, [project.id, project.name, project.settings_json]);

  useEffect(() => {
    let cancelled = false;
    const loadLifecycle = async () => {
      setLifecycleLoading(true);
      setLifecycleError('');
      try {
        const summary = await getProjectLifecycleSummary(project.id);
        if (!cancelled) setLifecycle(summary);
      } catch (error) {
        if (!cancelled) {
          setLifecycle(null);
          setLifecycleError(error instanceof Error ? error.message : 'Unable to load project stats.');
        }
      } finally {
        if (!cancelled) setLifecycleLoading(false);
      }
    };
    void loadLifecycle();
    return () => {
      cancelled = true;
    };
  }, [project.id]);

  const handleSave = useCallback(async () => {
    const trimmedName = projectName.trim();
    if (!trimmedName) return;
    try {
      setSaving(true);
      setSaveError('');
      const updated = await updateProject(project.id, {
        name: trimmedName,
        settings_json: {
          memory: {
            project_reads_user_memory: allowGlobalMemory,
          },
        },
      });
      onProjectUpdated?.(updated);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to update project settings.');
    } finally {
      setSaving(false);
    }
  }, [allowGlobalMemory, onProjectUpdated, project.id, projectName]);

  const dirty = projectName.trim() !== project.name
    || allowGlobalMemory !== (project.settings_json?.memory?.project_reads_user_memory === true);

  const stats = (
    <>
      <OverviewSeparatedItem label="Created">
        <Typography variant="body2">{new Date(project.created_at).toLocaleString()}</Typography>
      </OverviewSeparatedItem>
      <OverviewSeparatedItem label="Last activity">
        <Typography variant="body2">{new Date(project.last_activity_at).toLocaleString()}</Typography>
      </OverviewSeparatedItem>
      <OverviewSeparatedItem label="Embedding model">
        <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>{project.embeddingModel}</Typography>
      </OverviewSeparatedItem>
      <OverviewSeparatedItem label="Allow global memory">
        <Typography variant="body2">{allowGlobalMemory ? 'Enabled' : 'Disabled'}</Typography>
      </OverviewSeparatedItem>
      {lifecycleLoading ? (
        <OverviewSeparatedItem>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="body2" color="text.secondary">Loading counts...</Typography>
          </Box>
        </OverviewSeparatedItem>
      ) : lifecycle ? (
        <>
          <OverviewSeparatedItem label="Threads">
            <Typography variant="body2">{lifecycle.thread_count}</Typography>
          </OverviewSeparatedItem>
          <OverviewSeparatedItem label="Project files">
            <Typography variant="body2">{lifecycle.project_file_count}</Typography>
          </OverviewSeparatedItem>
          <OverviewSeparatedItem label="Memories">
            <Typography variant="body2">{lifecycle.memory_count}</Typography>
          </OverviewSeparatedItem>
        </>
      ) : null}
      <OverviewSeparatedItem label={`Documents (${documents.length})`}>
        {documents.length > 0 ? (
          <Stack spacing={0.75}>
            {documents.map((document) => (
              <Box
                key={document.id}
                sx={{
                  cursor: onOpenDocument ? 'pointer' : 'default',
                  borderRadius: 1,
                  px: 0.5,
                  mx: -0.5,
                  '&:hover': onOpenDocument ? { bgcolor: 'action.hover' } : undefined,
                }}
                onClick={onOpenDocument ? () => onOpenDocument(document.id) : undefined}
                role={onOpenDocument ? 'button' : undefined}
                tabIndex={onOpenDocument ? 0 : undefined}
                onKeyDown={onOpenDocument ? (event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onOpenDocument(document.id);
                  }
                } : undefined}
              >
                <Typography variant="body2" sx={{ fontWeight: 600, wordBreak: 'break-word' }}>
                  {document.fileName}
                </Typography>
                <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.25 }}>
                  {document.sourceType ? (
                    <Chip size="small" label={document.sourceType} variant="outlined" />
                  ) : null}
                  {document.parsingStatus ? (
                    <Chip size="small" label={document.parsingStatus} color={document.parsingStatus === 'failed' ? 'error' : 'default'} />
                  ) : null}
                  {document.addedAt ? (
                    <Typography variant="caption" color="text.secondary">
                      Added {new Date(document.addedAt).toLocaleString()}
                    </Typography>
                  ) : null}
                </Stack>
              </Box>
            ))}
          </Stack>
        ) : (
          <Typography variant="body2" color="text.secondary">No documents attached yet.</Typography>
        )}
      </OverviewSeparatedItem>
    </>
  );

  const settings = (
    <>
      <ProjectSettingsForm
        projectName={projectName}
        onProjectNameChange={setProjectName}
        allowGlobalMemory={allowGlobalMemory}
        onAllowGlobalMemoryChange={setAllowGlobalMemory}
        embeddingModel={project.embeddingModel}
        disabled={saving}
        lifecycle={lifecycle}
        lifecycleLoading={lifecycleLoading}
        lifecycleError={lifecycleError}
        projectReady={projectReady}
        onCloneProject={onCloneProject}
        onCloneProjectWithThreads={onCloneProjectWithThreads}
        onDeleteProject={onDeleteProject}
      />
      {saveError ? <Alert severity="error" sx={{ mt: 1.5 }}>{saveError}</Alert> : null}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
        <Button
          variant="outlined"
          onClick={() => {
            setProjectName(project.name);
            setAllowGlobalMemory(project.settings_json?.memory?.project_reads_user_memory === true);
            setSaveError('');
          }}
          disabled={saving || !dirty}
        >
          Reset
        </Button>
        <Button
          variant="contained"
          onClick={() => { void handleSave(); }}
          disabled={saving || !dirty || !projectName.trim()}
        >
          {saving ? 'Saving...' : 'Save'}
        </Button>
      </Box>
    </>
  );

  return (
    <WorkspaceOverviewPage
      title={project.name}
      subtitle="Shared project knowledge, memory policy, and thread workspace."
      darkMode={darkMode}
      actions={(
        <AddSourcesActions
          onRequestUpload={onRequestUpload}
          onCapturePage={onCapturePage}
          onCreateThread={onCreateThread}
          isBrowserCapturing={isBrowserCapturing}
        />
      )}
      guideSections={PROJECT_GUIDE_SECTIONS}
      guideTitle="How Projects work"
      guideDefaultExpanded={false}
      stats={stats}
      settings={settings}
    />
  );
}
