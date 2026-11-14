import type { PluggableList } from 'unified';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { normalizeLlmMathDelimiters } from './normalize-llm-math-delimiters.ts';

export const conversationRemarkPlugins: PluggableList = [
  remarkGfm,
  [remarkMath, { singleDollarTextMath: true }],
];
export const conversationRehypePlugins: PluggableList = [rehypeKatex];

export function prepareConversationMarkdown(content: string): string {
  return normalizeLlmMathDelimiters(content);
}