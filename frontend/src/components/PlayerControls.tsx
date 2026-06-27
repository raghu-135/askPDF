import React, { useEffect, useRef, useState } from "react";
import { Stack, Select, MenuItem, Slider, Typography, FormControl, InputLabel, IconButton, Popover, Box, Tooltip, CircularProgress } from "@mui/material";
import { PlayArrow, Pause, SkipPrevious, SkipNext } from '@mui/icons-material';
import RecordVoiceOverIcon from '@mui/icons-material/RecordVoiceOver';
import AutoStoriesIcon from '@mui/icons-material/AutoStories';
import EditNoteIcon from '@mui/icons-material/EditNote';

import { ttsSentence, getVoices } from "../lib/tts-api";
import { useTtsPrefetchCache } from "../hooks/useTtsPrefetchCache";

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


type Props = {
  sentences: Sentence[] | null;
  currentId: number | null;                // highlight only
  onCurrentChange: (id: number | null) => void;
  playRequestId: number | null;            // explicit command to play now
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
  highlightEnabled: boolean;
  onHighlightEnabledChange: (value: boolean) => void;
};

const PlayerControls = React.memo(function PlayerControls({ sentences, currentId, onCurrentChange, playRequestId, autoScroll, onAutoScrollChange, highlightEnabled, onHighlightEnabledChange }: Props) {
  // Ref to the audio element for playback control
  const audioRef = useRef<HTMLAudioElement>(null);
  const playRequestTokenRef = useRef(0);
  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPreparingAudio, setIsPreparingAudio] = useState(false);
  // Available TTS voices
  const [voices, setVoices] = useState<string[]>([]);
  // Currently selected TTS voice
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  // Playback speed
  const [speed, setSpeed] = useState<number>(1.0);
  // Track paused position for resume
  const [pausedAt, setPausedAt] = useState<number | null>(null);
  // Popover anchor
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

  // Disable controls when sentences are null (parsing in progress)
  const isDisabled = sentences === null;

  const handleOpenSettings = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseSettings = () => {
    setAnchorEl(null);
  };

  const open = Boolean(anchorEl);
  const id_popover = open ? 'voice-settings-popover' : undefined;
  const voiceOptions = voices.length > 0
    ? voices
    : (selectedVoice ? [selectedVoice] : []);
  const effectiveVoice = selectedVoice || "af_heart";
  const { getOrCreateSentenceAudio, prefetchAhead, clearCache } = useTtsPrefetchCache({
    sentences,
    prefetchAheadCount: 3,
    synthesize: ttsSentence,
  });

  // Fetch available TTS voices on mount
  useEffect(() => {
    async function fetchVoices() {
      try {
        const voicesData = await getVoices();
        setVoices(voicesData);
        if (voicesData.length > 0) {
          // Prefer af_heart when available; otherwise keep current if valid, else first voice.
          if (voicesData.includes('af_heart')) {
            setSelectedVoice('af_heart');
          } else if (!voicesData.includes(selectedVoice)) {
            setSelectedVoice(voicesData[0]);
          }
        }
      } catch (err) {
        console.error("Failed to fetch voices", err);
      }
    }
    fetchVoices();
  }, []);

  // Cleanup on unmount: stop audio
  useEffect(() => {
    return () => {
      playRequestTokenRef.current += 1;
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }
      clearCache();
    };
  }, [clearCache]);

  // Play a sentence when an external play request is received, such as the PDF selection menu.
  useEffect(() => {
    if (playRequestId == null) return;
    void playSentence(playRequestId);
  }, [playRequestId]);

  // Restart playback with new voice if changed during playback
  useEffect(() => {
    if (isPlaying && currentId !== null && selectedVoice !== "") {
      void playSentence(currentId);
    }
  }, [selectedVoice]);

  // Stop playback and reset state when sentences change (e.g., new PDF uploaded)
  useEffect(() => {
    // If we're jumping to a new source, the playRequestId effect will handle it.
    // Only auto-stop for passive changes
    if (playRequestId !== null) return;

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    setIsPlaying(false);
    setIsPreparingAudio(false);
    setPausedAt(null);
    clearCache();
  }, [sentences]);

  // Voice/speed changes alter synthesis output, so old cache entries are invalid.
  useEffect(() => {
    clearCache();
  }, [effectiveVoice, speed, clearCache]);

  /**
   * Play the sentence at the given index. If resumeFrom is provided, resumes from that time.
   * Handles TTS audio fetching and playback, and auto-advances to next sentence on end.
   */
  async function playSentence(id: number, resumeFrom?: number) {
    const audio = audioRef.current;
    if (!audio) return;
    const requestToken = playRequestTokenRef.current + 1;
    playRequestTokenRef.current = requestToken;

    // Stop any current playback
    audio.pause();
    audio.currentTime = 0;
    audio.onended = null;

    const s = sentences[id];
    if (!s) {
      return;
    }
    onCurrentChange(id);
    setIsPlaying(false);
    setIsPreparingAudio(true);

    try {
      const cached = getOrCreateSentenceAudio(id, effectiveVoice, speed);
      if (!cached) return;
      const { audioUrl } = await cached;
      if (playRequestTokenRef.current !== requestToken) return;
      audio.src = audioUrl;
      await audio.play();
      if (playRequestTokenRef.current !== requestToken) return;
      prefetchAhead(id, effectiveVoice, speed);
      if (resumeFrom) {
        audio.currentTime = resumeFrom;
      }
      setIsPlaying(true);
      setPausedAt(null);

      // Auto-advance to next sentence on playback end
      audio.onended = () => {
        const next = id + 1;
        if (next < sentences.length) {
          void playSentence(next);
        } else {
          setIsPlaying(false);
          onCurrentChange(null);
        }
      };
    } catch (e) {
      // Ignore expected abort errors from interrupted play() calls (pause/skip)
      if (e instanceof Error && e.name === "AbortError") {
        return;
      }
      console.error("Playback failed", e);
      setIsPlaying(false);
    } finally {
      if (playRequestTokenRef.current === requestToken) {
        setIsPreparingAudio(false);
      }
    }
  }

  /**
   * Toggle play/pause for the current sentence.
   * If paused, resumes from last position; otherwise, starts from current or first sentence.
   */
  function handlePlayPause() {
    const audio = audioRef.current;
    if (isPreparingAudio) {
      playRequestTokenRef.current += 1;
      setIsPreparingAudio(false);
      audio?.pause();
      return;
    }

    if (!isPlaying) {
      if (pausedAt !== null && currentId !== null) {
        audio!.play();
        setIsPlaying(true);
        setPausedAt(null);
      } else {
        void playSentence(currentId ?? 0);
      }
    } else {
      if (audio) {
        audio.pause();
        setPausedAt(audio.currentTime);
        setIsPlaying(false);
      }
    }
  }

  return (
    <Stack direction="row" spacing={1} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
      <Tooltip title={isDisabled ? "Parsing in progress" : ""}>
        <Stack direction="row" spacing={0.5} alignItems="center">
          <IconButton
            color="primary"
            onClick={handlePlayPause}
            size="small"
            disabled={isDisabled}
            aria-label={isPreparingAudio ? "Preparing audio" : isPlaying ? "Pause" : "Play"}
          >
            {isPlaying ? (
              <Pause fontSize="small" />
            ) : isPreparingAudio ? (
              <CircularProgress size={18} thickness={5} />
            ) : (
              <PlayArrow fontSize="small" />
            )}
          </IconButton>
          <IconButton onClick={() => currentId !== null && currentId > 0 && playSentence(currentId - 1)} disabled={isDisabled || currentId === null || currentId <= 0} size="small">
            <SkipPrevious fontSize="small" />
          </IconButton>
          <IconButton onClick={() => currentId !== null && currentId < (sentences?.length ?? 0) - 1 && playSentence(currentId + 1)} disabled={isDisabled || currentId === null || currentId >= (sentences?.length ?? 0) - 1} size="small">
            <SkipNext fontSize="small" />
          </IconButton>

          <Tooltip title={autoScroll ? "Disable Auto-Scroll" : "Enable Auto-Scroll"}>
            <IconButton
              color={autoScroll ? "primary" : "default"}
              onClick={() => onAutoScrollChange(!autoScroll)}
              size="small"
              disabled={isDisabled}
            >
              <AutoStoriesIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          <Tooltip title={highlightEnabled ? "Disable TTS Highlighting" : "Enable TTS Highlighting"}>
            <IconButton
              color={highlightEnabled ? "primary" : "default"}
              onClick={() => onHighlightEnabledChange(!highlightEnabled)}
              size="small"
              disabled={isDisabled}
            >
              <EditNoteIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Stack>
      </Tooltip>

      <Tooltip title={isDisabled ? "Parsing in progress" : ""}>
        <IconButton
          aria-describedby={id_popover}
          size="small"
          onClick={handleOpenSettings}
          color={open ? "primary" : "default"}
          disabled={isDisabled}
        >
          <RecordVoiceOverIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      <Popover
        id={id_popover}
        open={open}
        anchorEl={anchorEl}
        onClose={handleCloseSettings}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'center',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'center',
        }}
        PaperProps={{
          sx: { p: 2, minWidth: 200 }
        }}
      >
        <Stack spacing={2}>
          <FormControl size="small" fullWidth>
            <InputLabel>Voice</InputLabel>
            <Select
              value={voiceOptions.includes(selectedVoice) ? selectedVoice : ""}
              label="Voice"
              onChange={(e: any) => setSelectedVoice(e.target.value as string)}
            >
              {voiceOptions.length === 0 && (
                <MenuItem value="" disabled>
                  No voices available
                </MenuItem>
              )}
              {voiceOptions.map((v: string) => (
                <MenuItem key={v} value={v}>
                  {v.replace(".json", "")}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
              <Typography variant="caption">Speed: {speed.toFixed(1)}x</Typography>
            </Stack>
            <Slider
              value={speed}
              min={0.5}
              max={2.0}
              step={0.1}
              onChange={(_: Event, val: number | number[]) => setSpeed(val as number)}
              onChangeCommitted={() => {
                if (isPlaying && currentId !== null) {
                  void playSentence(currentId);
                }
              }}
              valueLabelDisplay="auto"
              size="small"
            />
          </Box>
        </Stack>
      </Popover>

      <audio ref={audioRef} />
    </Stack>
  );
});

export default PlayerControls;
