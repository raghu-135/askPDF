import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  IconButton,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import { useScroll } from "@embedpdf/plugin-scroll/react";
import { ThumbnailsPane, ThumbImg } from "@embedpdf/plugin-thumbnail/react";
import CollectionsIcon from "@mui/icons-material/Collections";
import CommentIcon from "@mui/icons-material/Comment";
import { PdfCommentsPane } from "./PdfCommentsPane";
import { clampPanelRatio, getHorizontalDragRatio, PANEL_RATIOS } from "../lib/panel-ratio";

export type SidebarTab = "thumbnails" | "comments";

interface PdfSidebarProps {
  documentId: string;
  widthRatio: number;
  activeTab: SidebarTab;
  onTabChange: (tab: SidebarTab) => void;
  onWidthRatioChange: (ratio: number) => void;
  onToggleSidebar: () => void;
  commentComposerRequest: number;
}

export const PdfSidebar = React.memo(function PdfSidebar({
  documentId,
  widthRatio,
  activeTab,
  onTabChange,
  onWidthRatioChange,
  onToggleSidebar,
  commentComposerRequest,
}: PdfSidebarProps) {
  const { provides: scrollPlugin } = useScroll(documentId);
  const dragRef = useRef<{ startX: number; startRatio: number; containerWidth: number } | null>(null);
  const [isResizing, setIsResizing] = useState(false);

  const handleThumbnailClick = (pageIndex: number) => {
    scrollPlugin?.scrollToPage({
      pageNumber: pageIndex + 1,
      behavior: "smooth",
      alignY: 0,
    });
  };

  useEffect(() => {
    if (!isResizing) return;

    const handlePointerMove = (event: PointerEvent) => {
      if (!dragRef.current) return;
      const deltaRatio = getHorizontalDragRatio(
        event.clientX,
        dragRef.current.startX,
        dragRef.current.containerWidth
      );
      onWidthRatioChange(clampPanelRatio(
        dragRef.current.startRatio + deltaRatio,
        PANEL_RATIOS.pdfSidebar
      ));
    };

    const handlePointerUp = () => {
      dragRef.current = null;
      setIsResizing(false);
    };

    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [isResizing, onWidthRatioChange]);

  const tabs = useMemo(
    () => [
      { value: "thumbnails" as const, label: "Thumbnails", icon: <CollectionsIcon fontSize="small" /> },
      { value: "comments" as const, label: "Comments", icon: <CommentIcon fontSize="small" /> },
    ],
    []
  );

  return (
    <Box
      sx={{
        flexBasis: `${widthRatio * 100}%`,
        flexShrink: 0,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        borderRight: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        position: "relative",
        minWidth: 0,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: 1,
          borderColor: "divider",
          px: 1,
          pr: 0.5,
          minHeight: 52,
        }}
      >
        <Tabs
          value={activeTab}
          onChange={(_, next) => onTabChange(next as SidebarTab)}
          variant="scrollable"
          scrollButtons={false}
          sx={{
            minHeight: 52,
            "& .MuiTabs-flexContainer": { alignItems: "stretch" },
          }}
        >
          {tabs.map((tab) => (
            <Tab
              key={tab.value}
              value={tab.value}
              label={tab.label}
              icon={tab.icon}
              iconPosition="start"
              sx={{
                minHeight: 52,
                textTransform: "none",
                fontWeight: 600,
                minWidth: 0,
              }}
            />
          ))}
        </Tabs>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
        {activeTab === "thumbnails" ? (
          <Box sx={{ height: "100%", overflow: "hidden", position: "relative" }}>
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
                    px: 1,
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
                      overflow: "hidden",
                      borderRadius: 1,
                    }}
                  >
                    <ThumbImg
                      documentId={documentId}
                      meta={thumbnail}
                      style={{ maxWidth: "100%", maxHeight: "100%" }}
                    />
                  </Box>
                  <Typography variant="caption" sx={{ mt: 0.5, color: "text.secondary" }}>
                    {thumbnail.pageIndex + 1}
                  </Typography>
                </Box>
              )}
            </ThumbnailsPane>
          </Box>
        ) : (
          <PdfCommentsPane
            documentId={documentId}
            focusComposerRequest={commentComposerRequest}
          />
        )}
      </Box>

      <Tooltip title="Resize sidebar">
        <Box
          onPointerDown={(event) => {
            event.preventDefault();
            const containerWidth = event.currentTarget.parentElement?.parentElement?.getBoundingClientRect().width || window.innerWidth;
            dragRef.current = {
              startX: event.clientX,
              startRatio: widthRatio,
              containerWidth,
            };
            setIsResizing(true);
          }}
          sx={{
            position: "absolute",
            top: 0,
            right: -6,
            width: 12,
            height: "100%",
            cursor: "col-resize",
            zIndex: 2,
            touchAction: "none",
            "&::after": {
              content: '""',
              position: "absolute",
              top: 0,
              right: 5,
              width: 2,
              height: "100%",
              bgcolor: isResizing ? "primary.main" : "divider",
              opacity: isResizing ? 0.9 : 0.5,
            },
          }}
        />
      </Tooltip>
    </Box>
  );
});
