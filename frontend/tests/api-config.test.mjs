import assert from 'node:assert/strict';
import test from 'node:test';

// The module exports a resolved build-time constant, so seed the environment
// before importing it and test missing values through the pure resolver.
process.env.NEXT_PUBLIC_API_URL = 'http://test-api.example';
const { API_BASE, resolveApiBase } = await import('../src/lib/api-config.ts');

test('module API base is resolved from required configuration', () => {
  assert.equal(API_BASE, 'http://test-api.example');
});

test('explicit API URL wins and trailing slashes are removed', () => {
  assert.equal(resolveApiBase(' https://api.example.test/// '), 'https://api.example.test');
});

test('all environments reject missing or blank API configuration', () => {
  const configured = process.env.NEXT_PUBLIC_API_URL;
  delete process.env.NEXT_PUBLIC_API_URL;
  try {
    assert.throws(() => resolveApiBase(), /NEXT_PUBLIC_API_URL is required/);
  } finally {
    process.env.NEXT_PUBLIC_API_URL = configured;
  }
  assert.throws(() => resolveApiBase('  '), /NEXT_PUBLIC_API_URL is required/);
});
