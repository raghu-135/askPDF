import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeLlmMathDelimiters } from '../src/lib/normalize-llm-math-delimiters.ts';
import {
  renderConversationMarkdownHtml,
  renderConversationMarkdownHtmlRaw,
} from '../src/lib/render-conversation-markdown-html.ts';

test('normalize converts bare bracket display delimiters to $$', () => {
  const input = '[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V ]';
  const normalized = normalizeLlmMathDelimiters(input);
  assert.match(normalized, /\$\$/);
  assert.doesNotMatch(normalized, /^\[/);
});

test('normalize converts bare parenthesis inline delimiters with LaTeX', () => {
  const input = 'Scale by ( \\sqrt{d_k} ) and dimension ( d_k ).';
  const normalized = normalizeLlmMathDelimiters(input);
  assert.match(normalized, /\$\\sqrt\{d_k\}\$/);
  assert.match(normalized, /\$d_k\$/);
});

test('normalize preserves escaped TeX delimiters', () => {
  const input = String.raw`Inline \(x^2\) and display \[E = mc^2\]`;
  const normalized = normalizeLlmMathDelimiters(input);
  assert.match(normalized, /\$x\^2\$/);
  assert.match(normalized, /\$\$/);
});

test('normalize leaves code fences and inline code untouched', () => {
  const input = '`( d_k )` and\n```\n[ \\frac{a}{b} ]\n```\nand prose (see Section 3).';
  const normalized = normalizeLlmMathDelimiters(input);
  assert.match(normalized, /`\( d_k \)`/);
  assert.match(normalized, /\[ \\frac\{a\}\{b\} \]/);
  assert.match(normalized, /\(see Section 3\)/);
});

test('normalize escapes currency dollars without turning them into math', () => {
  const input = 'Prices are $5 and $10 today.';
  const normalized = normalizeLlmMathDelimiters(input);
  assert.match(normalized, /\\\$5/);
  assert.match(normalized, /\\\$10/);
  const html = renderConversationMarkdownHtml(input);
  assert.doesNotMatch(html, /class="katex"/);
  assert.match(html, /\$5/);
});

const ATTENTION_DISPLAY =
  '[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V ]';

test('raw bracket LaTeX without normalization shows no KaTeX markup', () => {
  const html = renderConversationMarkdownHtmlRaw(ATTENTION_DISPLAY);
  assert.doesNotMatch(html, /class="katex"/);
  assert.match(html, /\\text\{Attention\}/);
});

test('normalized delimiters render KaTeX HTML for display math', () => {
  const html = renderConversationMarkdownHtml(ATTENTION_DISPLAY);
  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex-display"/);
  assert.doesNotMatch(html, /\\text\{Attention\}\(Q, K, V\)=/);
});

test('standard dollar delimiters render inline and display math', () => {
  const html = renderConversationMarkdownHtml('Inline $d_k$ and $$\sum_{i=1}^n x_i$$');
  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex"/);
});

test('edge cases: brackets with prose and doc citations are not math', () => {
  const input = '[Source: Attention.pdf] and (doc:abc:c-12) remain plain.';
  const normalized = normalizeLlmMathDelimiters(input);
  assert.equal(normalized, input);
  const html = renderConversationMarkdownHtml(input);
  assert.doesNotMatch(html, /class="katex"/);
});