import assert from 'node:assert/strict';
import test from 'node:test';

import { getTheme } from '../src/theme.ts';

test('theme defaults form controls to filled variant', () => {
  const theme = getTheme(false);
  assert.equal(theme.components?.MuiTextField?.defaultProps?.variant, 'filled');
  assert.equal(theme.components?.MuiSelect?.defaultProps?.variant, 'filled');
  assert.equal(theme.components?.MuiFormControl?.defaultProps?.variant, 'filled');
});