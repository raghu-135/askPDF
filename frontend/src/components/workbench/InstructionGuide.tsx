import {
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import type { HomeInstructionSection } from '../../lib/home-instructions';
import { OverviewSeparatedItem } from './OverviewSection';

export default function InstructionGuide({
  sections,
}: {
  sections: HomeInstructionSection[];
}) {
  return (
    <>
      {sections.map((section) => (
        <OverviewSeparatedItem key={section.title} label={section.title}>
          <List
            dense
            disablePadding
            sx={{
              listStyleType: 'disc',
              pl: 2.5,
              m: 0,
            }}
          >
            {section.items.map((item) => (
              <ListItem
                key={item}
                alignItems="flex-start"
                sx={{ display: 'list-item', pl: 0, pr: 0, py: 0.5 }}
              >
                <ListItemText
                  primary={item}
                  primaryTypographyProps={{ variant: 'body2', sx: { lineHeight: 1.45 } }}
                />
              </ListItem>
            ))}
          </List>
        </OverviewSeparatedItem>
      ))}
    </>
  );
}
