import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

test('development Compose installs frontend dev dependencies', async () => {
  const compose = await readFile('../docker-compose.dev.yml', 'utf8');

  assert.match(compose, /NODE_ENV:\s*development/);
  assert.match(compose, /node_modules\/@next\/bundle-analyzer/);
});
