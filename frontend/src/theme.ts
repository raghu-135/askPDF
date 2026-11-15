import { createTheme } from '@mui/material/styles';

/** Shared variant for native selects and select/autocomplete fields only (not plain text fields). */
export const selectFieldVariant = 'filled' as const;

export const getTheme = (darkMode: boolean) =>
  createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
    },
    components: {
      MuiSelect: {
        defaultProps: {
          variant: selectFieldVariant,
        },
      },
    },
  });