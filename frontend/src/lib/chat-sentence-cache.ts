import { splitIntoSentences, stripMarkdown } from './sentence-utils.ts';

export type ConversationSentence = {
  id: number;
  text: string;
  itemIndex: number;
  itemId: string;
};

type MessageLike = {
  id: string;
  content: unknown;
};

export type ConversationSentenceCache = Map<string, {
  content: string;
  sentences: string[];
}>;

export function deriveConversationSentences(
  items: readonly MessageLike[],
  cache: ConversationSentenceCache,
): ConversationSentence[] {
  let globalId = 0;
  const result: ConversationSentence[] = [];
  const activeKeys = new Set<string>();

  items.forEach((item, itemIndex) => {
    const content = typeof item.content === 'string'
      ? item.content
      : String(item.content ?? '');
    activeKeys.add(item.id);

    let cached = cache.get(item.id);
    if (!cached || cached.content !== content) {
      cached = {
        content,
        sentences: splitIntoSentences(stripMarkdown(content)),
      };
      cache.set(item.id, cached);
    }

    cached.sentences.forEach((text) => {
      result.push({
        id: globalId++,
        text,
        itemIndex,
        itemId: item.id,
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
