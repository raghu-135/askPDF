import assert from 'node:assert/strict';
import test from 'node:test';

import { getTheme } from '../src/theme.ts';

test('theme defaults only MuiSelect to filled (plain text fields stay outlined)', () => {
  const theme = getTheme(false);
  assert.equal(theme.components?.MuiSelect?.defaultProps?.variant, 'filled');
  assert.equal(theme.components?.MuiTextField?.defaultProps?.variant, undefined);
  assert.equal(theme.components?.MuiFormControl?.defaultProps?.variant, undefined);
});