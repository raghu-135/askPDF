export const GRAPH_VIEWPORT_MIN_PX = 40;

export function isGraphViewportReady(width: number, height: number): boolean {
  return width >= GRAPH_VIEWPORT_MIN_PX && height >= GRAPH_VIEWPORT_MIN_PX;
}