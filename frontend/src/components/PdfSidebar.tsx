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
import MenuOpenIcon from "@mui/icons-material/MenuOpen";
import { PdfCommentsPane } from "./PdfCommentsPane";

export type SidebarTab = "thumbnails" | "comments";

const MIN_SIDEBAR_WIDTH = 220;
const MAX_SIDEBAR_WIDTH = 520;

interface PdfSidebarProps {
  documentId: string;
  width: number;
  activeTab: SidebarTab;
  onTabChange: (tab: SidebarTab) => void;
  onWidthChange: (width: number) => void;
  onToggleSidebar: () => void;
  commentComposerRequest: number;
}

export const PdfSidebar: React.FC<PdfSidebarProps> = ({
  documentId,
  width,
  activeTab,
  onTabChange,
  onWidthChange,
  onToggleSidebar,
  commentComposerRequest,
}) => {
  const { provides: scrollPlugin } = useScroll(documentId);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);
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
      const nextWidth = dragRef.current.startWidth + (event.clientX - dragRef.current.startX);
      onWidthChange(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, nextWidth)));
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
  }, [isResizing, onWidthChange]);

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
        width,
        flexShrink: 0,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        borderRight: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        position: "relative",
        minWidth: MIN_SIDEBAR_WIDTH,
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

        <Tooltip title="Hide sidebar">
          <IconButton size="small" onClick={onToggleSidebar}>
            <MenuOpenIcon fontSize="small" />
          </IconButton>
        </Tooltip>
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
            dragRef.current = {
              startX: event.clientX,
              startWidth: width,
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
};
