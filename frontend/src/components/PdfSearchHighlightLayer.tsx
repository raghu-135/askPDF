import React from "react";
import type { SearchResultByPage } from "../lib/pdf-utils";

type PdfSearchHighlightLayerProps = {
  results: SearchResultByPage[];
  activeIndex: number;
  scale: number;
};

export const PdfSearchHighlightLayer = React.memo(function PdfSearchHighlightLayer({
  results,
  activeIndex,
  scale,
}: PdfSearchHighlightLayerProps) {
  if (results.length === 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        zIndex: 22,
      }}
    >
      {results.flatMap(({ result, index }) =>
        result.rects.map((rect, rectIndex) => {
          const isActive = index === activeIndex;
          return (
            <div
              key={`${index}-${rectIndex}`}
              style={{
                position: "absolute",
                left: rect.origin.x * scale,
                top: rect.origin.y * scale,
                width: rect.size.width * scale,
                height: rect.size.height * scale,
                backgroundColor: isActive
                  ? "rgba(255, 152, 0, 0.38)"
                  : "rgba(255, 235, 59, 0.32)",
                outline: isActive
                  ? `${Math.max(1, 1.5 * scale)}px solid rgba(245, 124, 0, 0.85)`
                  : "none",
                boxSizing: "border-box",
              }}
            />
          );
        }),
      )}
    </div>
  );
});
