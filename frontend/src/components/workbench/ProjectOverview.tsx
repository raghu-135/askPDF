import { Box, Button, Stack, Typography } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import PublicIcon from '@mui/icons-material/Public';

export default function ProjectOverview({
  projectName,
  documentCount,
  darkMode = false,
  onCreateThread,
  onCapturePage,
  onRequestUpload,
}: {
  projectName: string;
  documentCount: number;
  darkMode?: boolean;
  onCreateThread: () => void;
  onCapturePage: () => void;
  onRequestUpload: () => void;
}) {
  return (
    <Box
      sx={{
        height: '100%',
        display: 'grid',
        placeItems: 'center',
        bgcolor: darkMode ? '#222' : 'grey.50',
        color: darkMode ? '#eee' : 'inherit',
        p: 4,
      }}
    >
      <Box sx={{ textAlign: 'center', maxWidth: 520 }}>
        <Typography variant="h5" gutterBottom>
          {projectName}
        </Typography>
        <Typography color="text.secondary" sx={{ mb: 2.5 }}>
          {documentCount > 0
            ? 'Open a document tab to inspect project knowledge, or add another source.'
            : 'Add a PDF, open a thread, or capture a page. Browser stays available as a tab when you need it.'}
        </Typography>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} justifyContent="center">
          <Button variant="contained" startIcon={<PictureAsPdfIcon />} onClick={onRequestUpload}>
            Upload PDF
          </Button>
          <Button variant="outlined" startIcon={<AddIcon />} onClick={onCreateThread}>
            Create thread
          </Button>
          <Button variant="outlined" startIcon={<PublicIcon />} onClick={onCapturePage}>
            Capture a page
          </Button>
        </Stack>
      </Box>
    </Box>
  );
}
