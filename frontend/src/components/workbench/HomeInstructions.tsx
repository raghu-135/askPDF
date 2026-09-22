import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Stack,
  Typography,
} from '@mui/material';
import CreateNewFolderIcon from '@mui/icons-material/CreateNewFolder';
import PsychologyIcon from '@mui/icons-material/Psychology';
import { HOME_INSTRUCTION_SECTIONS } from '../../lib/home-instructions';
import InstructionGuide from './InstructionGuide';
import OverviewSection from './OverviewSection';

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
        ) : (
          <OverviewSection
            title="How this works"
            collapsible={hasProjects}
            defaultExpanded
          >
            <InstructionGuide sections={HOME_INSTRUCTION_SECTIONS} />
          </OverviewSection>
        )}
      </Box>
    </Box>
  );
}
