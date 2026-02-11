import { createTheme } from '@mui/material/styles';

export const getTheme = (darkMode: boolean) =>
  createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
    },
  });
