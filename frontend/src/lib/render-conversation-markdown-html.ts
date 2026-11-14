import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import ReactMarkdown from 'react-markdown';
import {
  conversationRehypePlugins,
  conversationRemarkPlugins,
  prepareConversationMarkdown,
} from './conversation-markdown-plugins.ts';

function renderMarkdownHtml(content: string, prepared: string): string {
  return renderToStaticMarkup(
    React.createElement(ReactMarkdown, {
      remarkPlugins: conversationRemarkPlugins,
      rehypePlugins: conversationRehypePlugins,
      children: prepared,
    }),
  );
}

/** Render assistant markdown the same way ConversationMarkdown does (for node tests). */
export function renderConversationMarkdownHtml(content: string): string {
  return renderMarkdownHtml(content, prepareConversationMarkdown(content));
}

/** Markdown + KaTeX pipeline without LLM delimiter normalization (for contrast tests). */
export function renderConversationMarkdownHtmlRaw(content: string): string {
  return renderMarkdownHtml(content, content);
}