import React from "react";
import { Stack } from "@mui/material";
import { AnnotationChromeItems } from "./AnnotationChromeItems";

export interface AnnotationToolbarProps {
  documentId: string;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  isHistoryProcessingRef: React.MutableRefObject<boolean>;
  searchControls?: React.ReactNode;
}

/**
 * Simplified annotation toolbar using EmbedPDF native patterns.
 * Properties moved to contextual selection menu.
 */
export const AnnotationToolbar: React.FC<AnnotationToolbarProps> = React.memo(function AnnotationToolbar({
  documentId,
  showSidebar,
  onToggleSidebar,
  isHistoryProcessingRef,
  searchControls,
}) {
  return (
    <Stack
      direction="row"
      spacing={0.5}
      sx={{
        alignItems: "center",
        flexShrink: 0,
        px: 1,
        py: 0.5,
        borderBottom: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        width: "100%",
        justifyContent: "space-between",
      }}
    >
      <AnnotationChromeItems
        documentId={documentId}
        showSidebar={showSidebar}
        onToggleSidebar={onToggleSidebar}
        isHistoryProcessingRef={isHistoryProcessingRef}
      />
      {searchControls}
    </Stack>
  );
});
