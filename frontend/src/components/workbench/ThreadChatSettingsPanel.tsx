import { Box, Button } from '@mui/material';
import ChatSettingsForm from '../ChatSettingsForm';
import { useThreadChatSettings } from '../../lib/thread-chat-settings-context';

export default function ThreadChatSettingsPanel() {
  const settings = useThreadChatSettings();
  if (!settings) return null;

  const {
    saving,
    saveLabel = 'Save',
    onSave,
    onReset,
    ...formProps
  } = settings;

  return (
    <Box>
      <ChatSettingsForm {...formProps} />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 2 }}>
        <Button onClick={onReset} disabled={saving}>
          Reset
        </Button>
        <Button variant="contained" onClick={onSave} disabled={saving}>
          {saving ? 'Saving...' : saveLabel}
        </Button>
      </Box>
    </Box>
  );
}
