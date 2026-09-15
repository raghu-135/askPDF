import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import AutoAwesomeSharpIcon from '@mui/icons-material/AutoAwesomeSharp';
import ChatIcon from '@mui/icons-material/Chat';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import FolderCopyIcon from '@mui/icons-material/FolderCopy';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import PsychologyIcon from '@mui/icons-material/Psychology';
import { HOME_INSTRUCTION_SECTIONS } from '../../lib/home-instructions';

const sectionIcon = (title: string) => {
  if (title === 'Projects and Threads') return <FolderCopyIcon color="primary" />;
  if (title === 'Documents and Browser Sources') return <PictureAsPdfIcon color="primary" />;
  if (title === 'Chat and Retrieval') return <ChatIcon color="primary" />;
  if (title === 'Memory & Settings') return <PsychologyIcon color="primary" />;
  if (title === 'Agent Workflows') return <AutoAwesomeSharpIcon color="primary" />;
  return <FactCheckIcon color="primary" />;
};

function InstructionGuide() {
  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', lg: 'repeat(2, minmax(0, 1fr))' },
        gap: 1.5,
      }}
    >
      {HOME_INSTRUCTION_SECTIONS.map((section) => (
        <Paper
          key={section.title}
          elevation={0}
          sx={{
            border: 1,
            borderColor: 'divider',
            bgcolor: 'background.paper',
            borderRadius: 1,
            minWidth: 0,
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ px: 1.5, py: 1.25 }}>
            {sectionIcon(section.title)}
            <Typography variant="subtitle1" sx={{ fontWeight: 800 }}>
              {section.title}
            </Typography>
          </Stack>
          <Divider />
          <List
            dense
            disablePadding
            sx={{
              listStyleType: 'disc',
              pl: 3,
              pr: 1.5,
            }}
          >
            {section.items.map((item) => (
              <ListItem key={item} alignItems="flex-start" sx={{ display: 'list-item', pl: 0, pr: 0, py: 0.75 }}>
                <ListItemText
                  primary={item}
                  primaryTypographyProps={{ variant: 'body2', sx: { lineHeight: 1.45 } }}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      ))}
    </Box>
  );
}

export default function HomeInstructions({
  darkMode = false,
  hasProjects = false,
  inventoryLoading = false,
  onCreateProject,
}: {
  darkMode?: boolean;
  hasProjects?: boolean;
  inventoryLoading?: boolean;
  onCreateProject?: () => void;
}) {
  return (
    <Box
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: darkMode ? '#222' : 'grey.50',
        color: darkMode ? '#eee' : 'inherit',
        p: { xs: 2, md: 3 },
      }}
    >
      <Box sx={{ maxWidth: 980, mx: 'auto' }}>
        <Stack spacing={1.25} sx={{ mb: 2.5 }}>
          <Stack direction="row" spacing={1} alignItems="center" useFlexGap flexWrap="wrap">
            <Typography variant="h4" sx={{ fontWeight: 800 }}>
              {hasProjects ? 'Continue in AskPDF' : 'Welcome to AskPDF'}
            </Typography>
            <Chip icon={<PsychologyIcon />} label="Projects, chat, memory, workflows" size="small" color="primary" variant="outlined" />
          </Stack>
          <Typography color="text.secondary" sx={{ maxWidth: 760 }}>
            {hasProjects
              ? 'Use the Projects panel to open a workspace, or create a new project.'
              : 'Create a project, add sources, open a thread, and use memory or workflows when the work needs durable context or a more structured agent.'}
          </Typography>
          {onCreateProject && (
            <Box>
              <Button variant="contained" startIcon={<CreateNewFolderIcon />} onClick={onCreateProject}>
                Create project
              </Button>
            </Box>
          )}
        </Stack>

        {inventoryLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
            <CircularProgress />
          </Box>
        ) : hasProjects ? (
          <Accordion
            disableGutters
            elevation={0}
            sx={{
              border: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
              borderRadius: 1,
              '&:before': { display: 'none' },
            }}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ fontWeight: 700 }}>How this works</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <InstructionGuide />
            </AccordionDetails>
          </Accordion>
        ) : (
          <InstructionGuide />
        )}
      </Box>
    </Box>
  );
}
