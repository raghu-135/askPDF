import { createTheme } from '@mui/material/styles';

export const getTheme = (darkMode: boolean) =>
  createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
    },
    components: {
      MuiTextField: {
        defaultProps: {
          variant: 'filled',
        },
      },
      MuiSelect: {
        defaultProps: {
          variant: 'filled',
        },
      },
      MuiFormControl: {
        defaultProps: {
          variant: 'filled',
        },
      },
    },
  });