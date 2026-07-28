import { splitIntoSentences, stripMarkdown } from './sentence-utils.ts';

export type ChatSentence = {
  id: number;
  text: string;
  messageIndex: number;
};

type MessageLike = {
  id: string;
  content: unknown;
};

export type ChatSentenceCache = Map<string, {
  content: string;
  sentences: string[];
}>;

export function deriveChatSentences(
  messages: readonly MessageLike[],
  cache: ChatSentenceCache,
): ChatSentence[] {
  let globalId = 0;
  const result: ChatSentence[] = [];
  const activeKeys = new Set<string>();

  messages.forEach((message, messageIndex) => {
    const content = typeof message.content === 'string'
      ? message.content
      : String(message.content ?? '');
    activeKeys.add(message.id);

    let cached = cache.get(message.id);
    if (!cached || cached.content !== content) {
      cached = {
        content,
        sentences: splitIntoSentences(stripMarkdown(content)),
      };
      cache.set(message.id, cached);
    }

    cached.sentences.forEach((text) => {
      result.push({
        id: globalId++,
        text,
        messageIndex,
      });
    });
  });

  for (const key of cache.keys()) {
    if (!activeKeys.has(key)) {
      cache.delete(key);
    }
  }

  return result;
}
