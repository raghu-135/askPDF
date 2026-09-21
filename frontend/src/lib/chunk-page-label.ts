export type ChunkPageFields = {
  page_start?: number | null;
  page_end?: number | null;
  pages?: string | null;
};

export function chunkPageLabel(chunk: ChunkPageFields): string {
  if (chunk.page_start != null) {
    const end = chunk.page_end ?? chunk.page_start;
    return chunk.page_start === end ? `p. ${chunk.page_start}` : `p. ${chunk.page_start}-${end}`;
  }
  if (chunk.pages) return `pages ${chunk.pages}`;
  return '';
}
