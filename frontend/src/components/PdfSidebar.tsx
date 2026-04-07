import React, { useState } from "react";
import { Box, Typography, Tooltip, IconButton } from "@mui/material";
import { ThumbnailsPane, ThumbImg } from "@embedpdf/plugin-thumbnail/react";
import { useScroll } from "@embedpdf/plugin-scroll/react";
import CollectionsIcon from "@mui/icons-material/Collections";

interface PdfSidebarProps {
  documentId: string;
}

export const PdfSidebar: React.FC<PdfSidebarProps> = ({ documentId }) => {
  const { provides: scrollPlugin } = useScroll(documentId);

  const handleThumbnailClick = (pageIndex: number) => {
    if (scrollPlugin) {
      scrollPlugin.scrollToPage({
        pageNumber: pageIndex + 1,
        behavior: "smooth",
        alignY: 0
      });
    }
  };

  return (
    <Box
      sx={{
        width: 140,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        borderRight: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
      }}
    >
      <Box sx={{ borderBottom: 1, borderColor: "divider", px: 2, py: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
        <CollectionsIcon fontSize="small" color="primary" />
        <Typography variant="subtitle2" fontWeight="600">Thumbnails</Typography>
      </Box>
      <Box sx={{ flex: 1, overflow: "hidden", position: "relative" }}>
        <Box sx={{ height: "100%", overflow: "hidden" }}>
          <ThumbnailsPane documentId={documentId}>
            {(thumbnail) => (
              <Box
                key={thumbnail.pageIndex}
                onClick={() => handleThumbnailClick(thumbnail.pageIndex)}
                sx={{
                  position: "absolute",
                  top: thumbnail.top,
                  left: 0,
                  right: 0,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  cursor: "pointer",
                  py: 1,
                  "&:hover": {
                    bgcolor: "action.hover",
                  },
                }}
              >
                <Box
                  sx={{
                    width: thumbnail.width,
                    height: thumbnail.height,
                    boxShadow: 2,
                    bgcolor: "white",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    overflow: "hidden"
                  }}
                >
                  <ThumbImg
                    documentId={documentId}
                    meta={thumbnail}
                    style={{ maxWidth: '100%', maxHeight: '100%' }}
                  />
                </Box>
                <Typography
                  variant="caption"
                  sx={{ mt: 0.5, color: "text.secondary" }}
                >
                  {thumbnail.pageIndex + 1}
                </Typography>
              </Box>
            )}
          </ThumbnailsPane>
        </Box>
      </Box>
    </Box>
  );
};
