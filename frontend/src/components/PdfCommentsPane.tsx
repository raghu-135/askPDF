import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Button,
  Chip,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import CheckIcon from "@mui/icons-material/Check";
import CloseIcon from "@mui/icons-material/Close";
import BorderColorIcon from "@mui/icons-material/BorderColor";
import AddCommentIcon from "@mui/icons-material/AddComment";
import DeleteIcon from "@mui/icons-material/Delete";
import CommentIcon from "@mui/icons-material/Comment";
import CropSquareIcon from "@mui/icons-material/CropSquare";
import DrawIcon from "@mui/icons-material/Draw";
import FormatUnderlinedIcon from "@mui/icons-material/FormatUnderlined";
import GestureIcon from "@mui/icons-material/Gesture";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import StrikethroughSIcon from "@mui/icons-material/StrikethroughS";
import {
  getSidebarAnnotationsWithRepliesGroupedByPage,
  isText,
  useAnnotation,
  type SidebarAnnotationEntry,
  type TrackedAnnotation,
} from "@embedpdf/plugin-annotation/react";
import { useScroll } from "@embedpdf/plugin-scroll/react";
import {
  PdfAnnotationReplyType,
  PdfAnnotationSubtype,
  PdfAnnotationSubtypeName,
  uuidV4,
  type PdfAnnotationObject,
  type PdfTextAnnoObject,
  type Rect,
} from "@embedpdf/models";

type PdfCommentsPaneProps = {
  documentId: string;
  focusComposerRequest: number;
};

type CommentThreadEntry = {
  pageIndex: number;
  entry: SidebarAnnotationEntry;
};

type PageSection = {
  pageIndex: number;
  threads: CommentThreadEntry[];
  commentCount: number;
  replyCount: number;
};

function buildCommentRect(anchorRect: Rect, ordinal = 0): Rect {
  const size = 24;
  const offset = Math.min(ordinal * 8, 24);
  return {
    origin: {
      x: Math.max(anchorRect.origin.x + anchorRect.size.width - size - offset, anchorRect.origin.x),
      y: Math.max(anchorRect.origin.y + offset, anchorRect.origin.y),
    },
    size: {
      width: size,
      height: size,
    },
  };
}

function labelForAnnotation(annotation: PdfAnnotationObject) {
  return PdfAnnotationSubtypeName[annotation.type] ?? "Annotation";
}

function AnnotationBadge({
  annotation,
  selected,
  pageLabel = false,
}: {
  annotation: PdfAnnotationObject;
  selected?: boolean;
  pageLabel?: boolean;
}) {
  const pageIcon = <AddCommentIcon sx={{ fontSize: 17 }} />;
  const iconByType: Partial<Record<PdfAnnotationSubtype, React.ReactNode>> = {
    [PdfAnnotationSubtype.HIGHLIGHT]: <BorderColorIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.UNDERLINE]: <FormatUnderlinedIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.STRIKEOUT]: <StrikethroughSIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.SQUIGGLY]: <GestureIcon sx={{ fontSize: 17, transform: "rotate(90deg)" }} />,
    [PdfAnnotationSubtype.INK]: <DrawIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.LINE]: <GestureIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.SQUARE]: <CropSquareIcon sx={{ fontSize: 17 }} />,
    [PdfAnnotationSubtype.CIRCLE]: <RadioButtonUncheckedIcon sx={{ fontSize: 17 }} />,
  };

  const icon = pageLabel ? pageIcon : iconByType[annotation.type] ?? <CommentIcon sx={{ fontSize: 17 }} />;

  return (
    <Box
      sx={{
        width: 28,
        height: 28,
        flexShrink: 0,
        borderRadius: "50%",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        color: selected ? "primary.main" : "text.secondary",
        bgcolor: selected ? "rgba(66, 133, 244, 0.08)" : "background.paper",
      }}
    >
      {icon}
    </Box>
  );
}

function summarizeAnnotation(annotation: PdfAnnotationObject) {
  const contents = (annotation.contents || "").trim();

  if (contents) {
    return contents.length > 120 ? `${contents.slice(0, 117)}...` : contents;
  }

  return "";
}

function SaveActionButton({
  onClick,
  disabled,
  title = "Save",
}: {
  onClick: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <Tooltip title={title}>
      <span>
        <IconButton
          size="small"
          onClick={onClick}
          disabled={disabled}
          sx={{
            width: 30,
            height: 30,
            borderRadius: "50%",
            border: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
            "&:hover": {
              bgcolor: "action.hover",
            },
          }}
        >
          <CheckIcon fontSize="small" />
        </IconButton>
      </span>
    </Tooltip>
  );
}

function InlineEditableText({
  value,
  placeholder,
  onSave,
  onDelete,
  autoFocus,
  minRows = 2,
  emptyLabel,
  clickable = true,
  emphasized = false,
}: {
  value: string;
  placeholder: string;
  onSave: (next: string) => void;
  onDelete?: () => void;
  autoFocus?: boolean;
  minRows?: number;
  emptyLabel?: string;
  clickable?: boolean;
  emphasized?: boolean;
}) {
  const [draft, setDraft] = useState(value);
  const [editing, setEditing] = useState(Boolean(autoFocus));
  const editorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setDraft(value);
  }, [value]);

  useEffect(() => {
    if (autoFocus) setEditing(true);
  }, [autoFocus]);

  useEffect(() => {
    if (!editing) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && editorRef.current?.contains(target)) return;
      setDraft(value);
      setEditing(false);
    };

    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [editing, value]);

  const commit = useCallback(() => {
    onSave(draft.trim());
    setEditing(false);
  }, [draft, onSave]);

  const cancel = useCallback(() => {
    setDraft(value);
    setEditing(false);
  }, [value]);

  const canSave = draft.trim().length > 0;

  if (!editing) {
    return (
      <Box
        onClick={clickable ? () => setEditing(true) : undefined}
        sx={{ cursor: clickable ? "text" : "default", minWidth: 0 }}
        data-comment-interactive="true"
      >
        <Typography
          variant={emphasized ? "subtitle2" : "body2"}
          sx={{ 
            fontWeight: emphasized ? 700 : 400,
            color: value.trim() ? "text.primary" : "text.secondary",
            lineHeight: 1.45, 
            whiteSpace: "pre-wrap", 
            wordBreak: "break-word" 
          }}
        >
          {value.trim() || emptyLabel || placeholder}
        </Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={0.5} data-comment-interactive="true" ref={editorRef}>
      <TextField
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        placeholder={placeholder}
        multiline
        minRows={minRows}
        fullWidth
        size="small"
        variant="standard"
        autoFocus={autoFocus}
        InputProps={{ disableUnderline: true }}
        sx={{
          "& .MuiInputBase-root": {
            alignItems: "flex-start",
            px: 0,
            py: 0.5,
            borderRadius: 1,
            bgcolor: "transparent",
          },
          "& textarea": {
            resize: "none",
          },
        }}
      />
      <Stack direction="row" sx={{ justifyContent: "flex-end" }} spacing={0.5}>
        <SaveActionButton onClick={commit} disabled={!canSave} />
        <Tooltip title="Cancel">
          <span>
            <IconButton size="small" onClick={cancel} disabled={!editing}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        {onDelete ? (
          <Tooltip title="Delete">
            <IconButton
              size="small"
              onClick={() => {
                onDelete();
                setEditing(false);
              }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        ) : null}
      </Stack>
    </Stack>
  );
}

function CommentHeader({
  annotation,
  selected,
  title,
  onSaveTitle,
}: {
  annotation: PdfAnnotationObject;
  selected?: boolean;
  title: string;
  onSaveTitle: (next: string) => void;
}) {
  return (
    <Stack
      direction="row"
      spacing={1}
      sx={{ alignItems: "center", minWidth: 0 }}
    >
      <AnnotationBadge annotation={annotation} selected={selected} />
      <InlineEditableText
        value={title}
        placeholder={`${labelForAnnotation(annotation)} title`}
        onSave={onSaveTitle}
        minRows={1}
        emptyLabel={`Page ${annotation.pageIndex + 1}`}
        clickable
        emphasized
      />
    </Stack>
  );
}

function CommentRow({
  children,
  selected,
}: {
  children: React.ReactNode;
  selected?: boolean;
}) {
  return (
    <Box
      sx={{
        px: 1,
        py: 0.35,
        bgcolor: "transparent",
        transition: "background-color 120ms ease",
        "&:hover": {
          bgcolor: "rgba(66, 133, 244, 0.12)",
        },
      }}
    >
      {children}
    </Box>
  );
}

function CommentThreadCard({
  thread,
  selectedAnnotationId,
  onJump,
  onCreateReply,
  onUpdateText,
  onUpdateTitle,
  onDeleteText,
  threadRef,
}: {
  thread: CommentThreadEntry;
  selectedAnnotationId: string | null;
  onJump: (annotation: PdfAnnotationObject) => void;
  onCreateReply: (parent: PdfAnnotationObject, text: string) => void;
  onUpdateText: (annotation: TrackedAnnotation<PdfTextAnnoObject>, text: string) => void;
  onUpdateTitle: (annotation: PdfAnnotationObject, text: string) => void;
  onDeleteText: (annotation: TrackedAnnotation<PdfTextAnnoObject>) => void;
  threadRef?: React.Ref<HTMLDivElement>;
}) {
  const { entry } = thread;
  const root = entry.annotation;
  const isRootText = isText(root);
  const selected =
    selectedAnnotationId === root.object.id ||
    entry.replies.some((reply) => reply.object.id === selectedAnnotationId);
  const replySelected = entry.replies.some((reply) => reply.object.id === selectedAnnotationId);
  const [replyDraft, setReplyDraft] = useState("");

  const handleCreateReply = useCallback(() => {
    const next = replyDraft.trim();
    if (!next) return;
    onCreateReply(root.object, next);
    setReplyDraft("");
  }, [onCreateReply, replyDraft, root.object]);

  return (
    <Box
      ref={threadRef}
      sx={{
        py: 0.5,
        bgcolor: selected
          ? replySelected
            ? "rgba(66, 133, 244, 0.22)"
            : "rgba(66, 133, 244, 0.18)"
          : "transparent",
        transition: "background-color 120ms ease",
        "&:hover": {
          bgcolor: selected
            ? replySelected
              ? "rgba(66, 133, 244, 0.30)"
              : "rgba(66, 133, 244, 0.26)"
            : "rgba(66, 133, 244, 0.14)",
        },
      }}
      onClick={(event) => {
        const target = event.target as HTMLElement | null;
        if (target?.closest('[data-comment-interactive="true"],button,input,textarea')) return;
        onJump(root.object);
      }}
    >
      <CommentRow>
        <CommentHeader
          annotation={root.object}
          selected={selected}
          title={
            (root.object.subject && root.object.subject.trim())
              ? root.object.subject.trim()
              : `${labelForAnnotation(root.object)} · Page ${root.object.pageIndex + 1}`
          }
          onSaveTitle={(next) => onUpdateTitle(root.object, next)}
        />
      </CommentRow>

      <Box sx={{ mt: 0.25 }}>
        {isRootText ? (
          <CommentRow>
            <InlineEditableText
              value={root.object.contents || ""}
              placeholder="Write a comment..."
              onSave={(next) => onUpdateText(root as TrackedAnnotation<PdfTextAnnoObject>, next)}
              minRows={3}
              emptyLabel={`Page ${root.object.pageIndex + 1}`}
            />
          </CommentRow>
        ) : (
          summarizeAnnotation(root.object) ? (
            <CommentRow>
              <Typography
                variant="body2"
                color="text.primary"
                sx={{ lineHeight: 1.45, whiteSpace: "pre-wrap", wordBreak: "break-word" }}
              >
                {summarizeAnnotation(root.object)}
              </Typography>
            </CommentRow>
          ) : null
        )}
      </Box>

      {entry.replies.length > 0 ? (
        <Box sx={{ mt: 0.35 }}>
          {Array.from(new Map(entry.replies.map(r => [r.object.id, r])).values()).map((reply, index) => (
            <React.Fragment key={reply.object.id}>
              {index > 0 ? <Box sx={{ height: 1, mx: 1, bgcolor: "divider", opacity: 0.35 }} /> : null}
              <CommentRow>
                <InlineEditableText
                  value={reply.object.contents || ""}
                  placeholder="Write a reply..."
                  onSave={(next) => onUpdateText(reply, next)}
                  onDelete={() => onDeleteText(reply)}
                  minRows={2}
                  emptyLabel="Reply"
                />
              </CommentRow>
            </React.Fragment>
          ))}
        </Box>
      ) : null}

      <Box sx={{ pt: 0.5, px: 1 }} data-comment-interactive="true">
        <Stack direction="row" spacing={1} alignItems="flex-end">
          <TextField
            value={replyDraft}
            onChange={(event) => setReplyDraft(event.target.value)}
            placeholder="Add reply..."
            multiline
            minRows={2}
            fullWidth
            size="small"
            variant="standard"
            InputProps={{ disableUnderline: true }}
            sx={{
              "& .MuiInputBase-root": {
                alignItems: "flex-start",
                px: 0,
                py: 0.5,
                borderRadius: 1,
                bgcolor: "transparent",
              },
              "& textarea": {
                resize: "none",
              },
            }}
          />
          {replyDraft.trim() ? (
            <Box sx={{ pb: 0.15 }}>
              <SaveActionButton onClick={handleCreateReply} title="Save reply" />
            </Box>
          ) : null}
        </Stack>
      </Box>
    </Box>
  );
}

export function PdfCommentsPane({
  documentId,
  focusComposerRequest,
}: PdfCommentsPaneProps) {
  const theme = useTheme();
  const { state, provides: annotationApi } = useAnnotation(documentId);
  const { provides: scrollApi } = useScroll(documentId);
  const composerRef = useRef<HTMLTextAreaElement | HTMLInputElement | null>(null);
  const threadRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [composerText, setComposerText] = useState("");
  const [composerToken, setComposerToken] = useState(0);

  const grouped = useMemo(
    () => (state ? getSidebarAnnotationsWithRepliesGroupedByPage(state) : {}),
    [state]
  );

  const selectedAnnotation = annotationApi?.getSelectedAnnotations()?.[0] ?? null;
  const selectedAnnotationId = selectedAnnotation?.object.id ?? null;

  const findThreadRootId = useCallback(
    (annotationId: string) => {
      for (const pageThreads of Object.values(grouped)) {
        for (const entry of pageThreads || []) {
          if (entry.annotation.object.id === annotationId) return entry.annotation.object.id;
          if (entry.replies.some((reply) => reply.object.id === annotationId)) {
            return entry.annotation.object.id;
          }
        }
      }
      return null;
    },
    [grouped]
  );

  const threads = useMemo(() => {
    const pageNumbers = Object.keys(grouped)
      .map((page) => Number(page))
      .sort((a, b) => a - b);

    return pageNumbers.flatMap((pageNumber) =>
      (grouped[pageNumber] || [])
        .filter((entry) => isText(entry.annotation) || entry.replies.length > 0)
        .map((entry) => ({ pageIndex: pageNumber, entry }))
    );
  }, [grouped]);

  const groupedThreads = useMemo(() => {
    const pageNumbers = Object.keys(grouped)
      .map((page) => Number(page))
      .sort((a, b) => a - b);

    return pageNumbers
      .map((pageIndex) => {
        const threads = (grouped[pageIndex] || [])
          .filter((entry) => isText(entry.annotation) || entry.replies.length > 0)
          .map((entry) => ({ pageIndex, entry }));

        return {
          pageIndex,
          threads,
          commentCount: threads.length,
          replyCount: threads.reduce((count, thread) => count + thread.entry.replies.length, 0),
        };
      })
      .filter((section) => section.threads.length > 0);
  }, [grouped]);

  useEffect(() => {
    if (focusComposerRequest === 0) return;
    setComposerToken(focusComposerRequest);
  }, [focusComposerRequest]);

  useEffect(() => {
    if (composerToken === 0) return;
    composerRef.current?.focus();
    composerRef.current?.select?.();
  }, [composerToken]);

  useEffect(() => {
    if (!selectedAnnotationId) return;
    const rootId = findThreadRootId(selectedAnnotationId);
    if (!rootId) return;

    const node = threadRefs.current[rootId];
    node?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [findThreadRootId, selectedAnnotationId]);

  const selectAndScroll = useCallback(
    (annotation: PdfAnnotationObject) => {
      annotationApi?.selectAnnotation(annotation.pageIndex, annotation.id);
      scrollApi?.scrollToPage({
        pageNumber: annotation.pageIndex + 1,
        pageCoordinates: {
          x: annotation.rect.origin.x,
          y: annotation.rect.origin.y,
        },
        alignY: 50,
        behavior: "smooth",
      });
    },
    [annotationApi, scrollApi]
  );

  const createTextComment = useCallback(
    (parent: PdfAnnotationObject, text: string) => {
      const nextText = text.trim();
      if (!annotationApi || !nextText) return;

      const existingThread =
        threads.find((thread) => thread.entry.annotation.object.id === parent.id) ??
        threads.find((thread) =>
          thread.entry.replies.some((reply) => reply.object.id === parent.id)
        );
      const ordinal = existingThread?.entry.replies.length ?? 0;

      const annotation: PdfTextAnnoObject = {
        type: PdfAnnotationSubtype.TEXT,
        id: uuidV4(),
        pageIndex: parent.pageIndex,
        rect: buildCommentRect(parent.rect, ordinal),
        contents: nextText,
        author: "AskPDF",
        strokeColor: theme.palette.primary.main,
        opacity: 1,
        subject: "Comment",
        inReplyToId: parent.id,
        replyType: PdfAnnotationReplyType.Reply,
        created: new Date(),
        modified: new Date(),
      };

      annotationApi.createAnnotation(parent.pageIndex, annotation);
      setComposerText("");
    },
    [annotationApi, theme.palette.primary.main, threads]
  );

  const updateTextComment = useCallback(
    (annotation: TrackedAnnotation<PdfTextAnnoObject>, nextText: string) => {
      annotationApi?.updateAnnotation(annotation.object.pageIndex, annotation.object.id, {
        contents: nextText,
        modified: new Date(),
      });
    },
    [annotationApi]
  );

  const updateAnnotationTitle = useCallback(
    (annotation: PdfAnnotationObject, nextTitle: string) => {
      annotationApi?.updateAnnotation(annotation.pageIndex, annotation.id, {
        subject: nextTitle || labelForAnnotation(annotation),
        modified: new Date(),
      });
    },
    [annotationApi]
  );

  const deleteTextComment = useCallback(
    (annotation: TrackedAnnotation<PdfTextAnnoObject>) => {
      annotationApi?.deleteAnnotation(annotation.object.pageIndex, annotation.object.id);
    },
    [annotationApi]
  );

  const canCreateTopLevelComment = Boolean(selectedAnnotation && composerText.trim());

  return (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        overflow: "hidden",
        bgcolor: "background.paper",
      }}
    >
      <Box sx={{ px: 1, py: 1.25, borderBottom: 1, borderColor: "divider" }}>
        <Stack spacing={1}>
          <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
            <CommentIcon fontSize="small" color="primary" />
            <Typography variant="subtitle2" fontWeight={700}>
              New comment
            </Typography>
          </Stack>

          {selectedAnnotation ? (
            <Chip
              size="small"
              label={`On ${labelForAnnotation(selectedAnnotation.object)} · Page ${selectedAnnotation.object.pageIndex + 1}`}
              variant="outlined"
              sx={{ alignSelf: "flex-start" }}
            />
          ) : (
            <Typography variant="caption" color="text.secondary">
              Select an annotation in the document to attach a comment.
            </Typography>
          )}

          <TextField
            inputRef={composerRef}
            value={composerText}
            onChange={(event) => setComposerText(event.target.value)}
            placeholder={
              selectedAnnotation
                ? "Write a comment for the selected annotation..."
                : "Select an annotation first"
            }
            multiline
            minRows={3}
            fullWidth
            size="small"
            variant="standard"
            InputProps={{ disableUnderline: true }}
            sx={{
              "& .MuiInputBase-root": {
                alignItems: "flex-start",
                px: 0,
                py: 0.5,
                borderRadius: 1,
                bgcolor: "transparent",
              },
              "& textarea": {
                resize: "none",
              },
            }}
            disabled={!selectedAnnotation}
          />

          <Button
            variant="contained"
            size="small"
            startIcon={<CheckIcon />}
            onClick={() => {
              if (!selectedAnnotation) return;
              createTextComment(selectedAnnotation.object, composerText);
            }}
            disabled={!canCreateTopLevelComment}
            sx={{ alignSelf: "flex-start" }}
          >
            Add comment
          </Button>
        </Stack>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflowY: "auto", px: 0, py: 1 }}>
        {threads.length === 0 ? (
          <Box sx={{ px: 1, py: 1.5, color: "text.secondary" }}>
            <Typography variant="body2">
              No comments yet. Select an annotation and add a note to start a thread.
            </Typography>
          </Box>
        ) : (
          <Stack spacing={1.25}>
            {groupedThreads.map((section: PageSection) => {
              const commentLabel = section.commentCount === 1 ? "1 comment" : `${section.commentCount} comments`;
              const replyLabel = section.replyCount === 1 ? "1 reply" : `${section.replyCount} replies`;

              return (
                <Box key={section.pageIndex} sx={{ px: 0 }}>
                  <Stack direction="row" alignItems="flex-start" spacing={1} sx={{ px: 1, mb: 0.5 }}>
                    <AnnotationBadge
                      annotation={section.threads[0].entry.annotation.object}
                      selected={selectedAnnotationId === section.threads[0].entry.annotation.object.id}
                      pageLabel
                    />
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="subtitle2" fontWeight={700}>
                        Page {section.pageIndex + 1}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {commentLabel}
                        {section.replyCount > 0 ? ` · ${replyLabel}` : ""}
                      </Typography>
                    </Box>
                  </Stack>

                  <Box sx={{ mx: 0, mb: 0.35, height: 2, bgcolor: "divider", opacity: 0.6 }} />

                  <Stack spacing={0}>
                    {Array.from(new Map(section.threads.map(t => [t.entry.annotation.object.id, t])).values()).map((thread, threadIndex) => (
                      <Box key={thread.entry.annotation.object.id} sx={{ px: 0 }}>
                        <CommentThreadCard
                          thread={thread}
                          selectedAnnotationId={selectedAnnotationId}
                          onJump={selectAndScroll}
                          onCreateReply={createTextComment}
                          onUpdateText={updateTextComment}
                          onUpdateTitle={updateAnnotationTitle}
                          onDeleteText={deleteTextComment}
                          threadRef={(node) => {
                            threadRefs.current[thread.entry.annotation.object.id] = node;
                          }}
                        />
                        {threadIndex < section.threads.length - 1 ? (
                          <Box sx={{ height: 1, bgcolor: "divider", opacity: 0.24 }} />
                        ) : null}
                      </Box>
                    ))}
                  </Stack>
                </Box>
              );
            })}
          </Stack>
        )}
      </Box>
    </Box>
  );
}
