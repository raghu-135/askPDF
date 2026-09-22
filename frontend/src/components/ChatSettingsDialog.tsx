import React from 'react';
import {
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from '@mui/material';
import ChatSettingsForm, { type ChatSettingsFormProps } from './ChatSettingsForm';

interface ChatSettingsDialogProps extends ChatSettingsFormProps {
    open: boolean;
    onClose: () => void;
    onSave: () => void;
    saving: boolean;
    saveLabel?: string;
}

const ChatSettingsDialog: React.FC<ChatSettingsDialogProps> = ({
    open,
    onClose,
    onSave,
    saving,
    saveLabel = 'Save',
    ...formProps
}) => (
    <Dialog
        open={open}
        onClose={() => !saving && onClose()}
        maxWidth="md"
        fullWidth
    >
        <DialogTitle>AI Prompt Settings</DialogTitle>
        <DialogContent sx={{ pt: '8px !important' }}>
            <ChatSettingsForm {...formProps} />
        </DialogContent>
        <DialogActions>
            <Button onClick={onClose} disabled={saving}>
                Cancel
            </Button>
            <Button onClick={onSave} variant="contained" disabled={saving}>
                {saving ? 'Saving...' : saveLabel}
            </Button>
        </DialogActions>
    </Dialog>
);

export default ChatSettingsDialog;
