// Annotation tool constants and configuration

export const MARKUP_TOOLS = ["highlight", "underline", "strikeout", "squiggly"] as const;
export const SHAPE_TOOLS = ["ink", "line", "square", "circle"] as const;
export const VIEW_ONLY_MODE_ID = "view-only";

export const STANDARD_COLORS = [
  "#ffeb3b", // Yellow
  "#4caf50", // Green
  "#2196f3", // Blue
  "#f44336", // Red
  "#e91e63", // Pink
  "#ff9800", // Orange
  "#00bcd4", // Cyan
  "#9c27b0", // Purple
] as const;

export type MarkupTool = typeof MARKUP_TOOLS[number];
export type ShapeTool = typeof SHAPE_TOOLS[number];
export type AnnotationTool = MarkupTool | ShapeTool;

export interface MarkupSettings {
  strokeColor: string;
  opacity: number;
}

export interface ShapeSettings {
  strokeColor: string;
  strokeWidth: number;
  opacity: number;
}

export interface CurrentSettings {
  strokeColor?: string;
  strokeWidth?: number;
  opacity?: number;
}

export const DEFAULT_MARKUP_SETTINGS: MarkupSettings = {
  strokeColor: "#ffeb3b",
  opacity: 0.3,
};

export const DEFAULT_SHAPE_SETTINGS: ShapeSettings = {
  strokeColor: "#f44336",
  strokeWidth: 2,
  opacity: 1,
};
