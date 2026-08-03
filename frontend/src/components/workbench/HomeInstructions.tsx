import {
  Box,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Paper,
  Stack,
  Typography,
} from '@mui/material';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import AutoAwesomeSharpIcon from '@mui/icons-material/AutoAwesomeSharp';
import ChatIcon from '@mui/icons-material/Chat';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import FolderCopyIcon from '@mui/icons-material/FolderCopy';
import LanguageIcon from '@mui/icons-material/Language';
import MemoryIcon from '@mui/icons-material/Memory';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import RouteIcon from '@mui/icons-material/Route';
import SettingsIcon from '@mui/icons-material/Settings';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import { HOME_INSTRUCTION_SECTIONS } from '../../lib/home-instructions';

const sectionIcon = (title: string) => {
  if (title === 'Projects and Threads') return <FolderCopyIcon color="primary" />;
  if (title === 'Documents and Browser Sources') return <PictureAsPdfIcon color="primary" />;
  if (title === 'Chat and Retrieval') return <ChatIcon color="primary" />;
  if (title === 'Memory Workspace') return <MemoryIcon color="primary" />;
  if (title === 'Agent Workflows') return <AutoAwesomeSharpIcon color="primary" />;
  return <FactCheckIcon color="primary" />;
};

export default function HomeInstructions({ darkMode = false }: { darkMode?: boolean }) {
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
              Welcome to AskPDF
            </Typography>
            <Chip icon={<MemoryIcon />} label="Projects, chat, memory, workflows" size="small" color="primary" variant="outlined" />
          </Stack>
          <Typography color="text.secondary" sx={{ maxWidth: 760 }}>
            Start from the Projects panel on the right. Create a project, add sources, open a thread, and use memory or workflows when the work needs durable context or a more structured agent.
          </Typography>
        </Stack>

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
              <List dense disablePadding>
                {section.items.map((item) => (
                  <ListItem key={item} alignItems="flex-start" sx={{ px: 1.5, py: 0.75 }}>
                    <ListItemIcon sx={{ minWidth: 26, mt: 0.25 }}>
                      {section.title === 'Agent Workflows'
                        ? <RouteIcon fontSize="small" color="action" />
                        : section.title === 'Chat and Retrieval'
                          ? <SettingsIcon fontSize="small" color="action" />
                          : section.title === 'Review, Trace, and Playback'
                            ? <VolumeUpIcon fontSize="small" color="action" />
                            : section.title === 'Documents and Browser Sources'
                              ? <LanguageIcon fontSize="small" color="action" />
                              : <AccountTreeIcon fontSize="small" color="action" />}
                    </ListItemIcon>
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
      </Box>
    </Box>
  );
}
