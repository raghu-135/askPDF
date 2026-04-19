import { useCallback, useEffect, useRef } from "react";

type Sentence = {
  id: number;
  text: string;
  label?: string;
  page?: number;
  bbox?: [number, number, number, number];
  page_width?: number;
  page_height?: number;
  bboxes?: any[];
  words?: any[];
};
type TtsResult = { audioUrl: string };

type UseTtsPrefetchCacheParams = {
  sentences: Sentence[];
  prefetchAheadCount?: number;
  synthesize: (text: string, voice: string, speed: number) => Promise<TtsResult>;
};

/**
 * Client-side cache for sentence-level TTS requests with lookahead prefetch.
 * Cache key includes sentence content, voice, and speed to avoid stale audio reuse.
 */
export function useTtsPrefetchCache({
  sentences,
  prefetchAheadCount = 5,
  synthesize,
}: UseTtsPrefetchCacheParams) {
  const cacheRef = useRef<Map<string, Promise<TtsResult>>>(new Map());

  const buildCacheKey = useCallback(
    (sentenceIndex: number, voice: string, speed: number): string | null => {
      const sentence = sentences[sentenceIndex];
      if (!sentence) return null;
      return `${sentence.id}:${voice}:${speed}:${sentence.text}`;
    },
    [sentences],
  );

  const getOrCreateSentenceAudio = useCallback(
    (sentenceIndex: number, voice: string, speed: number): Promise<TtsResult> | null => {
      const sentence = sentences[sentenceIndex];
      if (!sentence) return null;
      const key = buildCacheKey(sentenceIndex, voice, speed);
      if (!key) return null;

      const cached = cacheRef.current.get(key);
      if (cached) return cached;

      const request = synthesize(sentence.text, voice, speed).catch((error) => {
        cacheRef.current.delete(key);
        throw error;
      });
      cacheRef.current.set(key, request);
      return request;
    },
    [buildCacheKey, sentences, synthesize],
  );

  const prefetchAhead = useCallback(
    (fromIndex: number, voice: string, speed: number) => {
      for (let i = 1; i <= prefetchAheadCount; i++) {
        const nextIndex = fromIndex + i;
        if (nextIndex >= sentences.length) break;
        void getOrCreateSentenceAudio(nextIndex, voice, speed);
      }
    },
    [getOrCreateSentenceAudio, prefetchAheadCount, sentences.length],
  );

  const clearCache = useCallback(() => {
    cacheRef.current.clear();
  }, []);

  useEffect(() => {
    clearCache();
  }, [sentences, clearCache]);

  useEffect(() => clearCache, [clearCache]);

  return {
    getOrCreateSentenceAudio,
    prefetchAhead,
    clearCache,
  };
}
