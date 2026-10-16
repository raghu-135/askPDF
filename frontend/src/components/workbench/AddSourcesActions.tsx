import { Button, Stack } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import PublicIcon from '@mui/icons-material/Public';

export default function AddSourcesActions({
  onRequestUpload,
  onCapturePage,
  onCreateThread,
  isBrowserCapturing = false,
}: {
  onRequestUpload: () => void;
  onCapturePage: () => void;
  onCreateThread?: () => void;
  isBrowserCapturing?: boolean;
}) {
  return (
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} justifyContent="center">
      <Button variant="contained" startIcon={<PictureAsPdfIcon />} onClick={onRequestUpload}>
        Upload PDF
      </Button>
      {onCreateThread ? (
        <Button variant="outlined" startIcon={<AddIcon />} onClick={onCreateThread}>
          Create thread
        </Button>
      ) : null}
      <Button
        variant="outlined"
        startIcon={<PublicIcon />}
        onClick={onCapturePage}
        disabled={isBrowserCapturing}
      >
        {isBrowserCapturing ? 'Capturing page…' : 'Capture a page'}
      </Button>
    </Stack>
  );
}
