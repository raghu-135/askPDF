import React, { createContext, useContext, useEffect, useRef, useState } from "react";
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

export type PlayerControlsProps = {
  sentences: Sentence[] | null;
  sourceKey: string;
  currentId: number | null;
  onCurrentChange: (id: number | null) => void;
  playRequestId: number | null;
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
  highlightEnabled: boolean;
  onHighlightEnabledChange: (value: boolean) => void;
};

type PlayerControlsContextValue = {
  sentences: Sentence[] | null;
  currentId: number | null;
  isDisabled: boolean;
  disabledTooltip: string;
  isPlaying: boolean;
  isPreparingAudio: boolean;
  autoScroll: boolean;
  onAutoScrollChange: (value: boolean) => void;
  highlightEnabled: boolean;
  onHighlightEnabledChange: (value: boolean) => void;
  handlePlayPause: () => void;
  playSentence: (id: number, resumeFrom?: number) => Promise<void>;
  open: boolean;
  idPopover: string | undefined;
  handleOpenSettings: (event: React.MouseEvent<HTMLButtonElement>) => void;
  handleCloseSettings: () => void;
  anchorEl: HTMLButtonElement | null;
  voiceOptions: string[];
  selectedVoice: string;
  setSelectedVoice: (voice: string) => void;
  speed: number;
  setSpeed: (speed: number) => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
};

const PlayerControlsContext = createContext<PlayerControlsContextValue | null>(null);

function usePlayerControlsContext() {
  const context = useContext(PlayerControlsContext);
  if (!context) {
    throw new Error('Player controls must be rendered within PlayerControlsProvider');
  }
  return context;
}

function usePlayerControlsState({
  sentences,
  sourceKey,
  currentId,
  onCurrentChange,
  playRequestId,
  autoScroll,
  onAutoScrollChange,
  highlightEnabled,
  onHighlightEnabledChange,
}: PlayerControlsProps): PlayerControlsContextValue {
  const audioRef = useRef<HTMLAudioElement>(null);
  const playRequestTokenRef = useRef(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isPreparingAudio, setIsPreparingAudio] = useState(false);
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [speed, setSpeed] = useState<number>(1.0);
  const [pausedAt, setPausedAt] = useState<number | null>(null);
  const [anchorEl, setAnchorEl] = useState<HTMLButtonElement | null>(null);

  const isDisabled = !sentences || sentences.length === 0;
  const disabledTooltip = isDisabled ? "Sentences are still processing" : "";
  const open = Boolean(anchorEl);
  const idPopover = open ? 'voice-settings-popover' : undefined;
  const voiceOptions = voices.length > 0
    ? voices
    : (selectedVoice ? [selectedVoice] : []);
  const effectiveVoice = selectedVoice || "af_heart";
  const { getOrCreateSentenceAudio, prefetchAhead, clearCache } = useTtsPrefetchCache({
    sentences,
    prefetchAheadCount: 3,
    synthesize: ttsSentence,
  });

  useEffect(() => {
    async function fetchVoices() {
      try {
        const voicesData = await getVoices();
        setVoices(voicesData);
        if (voicesData.length > 0) {
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
    void fetchVoices();
  }, []);

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

  useEffect(() => {
    playRequestTokenRef.current += 1;
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.onended = null;
    }
    setIsPlaying(false);
    setIsPreparingAudio(false);
    setPausedAt(null);
    clearCache();
    onCurrentChange(null);
  }, [clearCache, onCurrentChange, sourceKey]);

  useEffect(() => {
    if (playRequestId == null) return;
    void playSentence(playRequestId);
  }, [playRequestId]);

  useEffect(() => {
    if (isPlaying && currentId !== null && selectedVoice !== "") {
      void playSentence(currentId);
    }
  }, [selectedVoice]);

  useEffect(() => {
    clearCache();
  }, [clearCache, effectiveVoice, speed]);

  async function playSentence(id: number, resumeFrom?: number) {
    const audio = audioRef.current;
    if (!audio || isDisabled) return;
    const requestToken = playRequestTokenRef.current + 1;
    playRequestTokenRef.current = requestToken;

    audio.pause();
    audio.currentTime = 0;
    audio.onended = null;

    const s = sentences?.[id];
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

      audio.onended = () => {
        const next = id + 1;
        if (sentences && next < sentences.length) {
          void playSentence(next);
        } else {
          setIsPlaying(false);
          onCurrentChange(null);
        }
      };
    } catch (e) {
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
        void audio?.play();
        setIsPlaying(true);
        setPausedAt(null);
      } else {
        void playSentence(currentId ?? 0);
      }
    } else if (audio) {
      audio.pause();
      setPausedAt(audio.currentTime);
      setIsPlaying(false);
    }
  }

  return {
    sentences,
    currentId,
    isDisabled,
    disabledTooltip,
    isPlaying,
    isPreparingAudio,
    autoScroll,
    onAutoScrollChange,
    highlightEnabled,
    onHighlightEnabledChange,
    handlePlayPause,
    playSentence,
    open,
    idPopover,
    handleOpenSettings: (event) => setAnchorEl(event.currentTarget),
    handleCloseSettings: () => setAnchorEl(null),
    anchorEl,
    voiceOptions,
    selectedVoice,
    setSelectedVoice,
    speed,
    setSpeed,
    audioRef,
  };
}

export function PlayerControlsProvider({
  children,
  ...props
}: PlayerControlsProps & { children: React.ReactNode }) {
  const value = usePlayerControlsState(props);
  return (
    <PlayerControlsContext.Provider value={value}>
      {children}
      <audio ref={value.audioRef} />
    </PlayerControlsContext.Provider>
  );
}

export const PlayerPlaybackChrome = React.memo(function PlayerPlaybackChrome() {
  const {
    currentId,
    disabledTooltip,
    handlePlayPause,
    isDisabled,
    isPlaying,
    isPreparingAudio,
    playSentence,
    sentences,
  } = usePlayerControlsContext();

  return (
    <Tooltip title={disabledTooltip}>
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
        <IconButton
          onClick={() => currentId !== null && currentId > 0 && void playSentence(currentId - 1)}
          disabled={isDisabled || currentId === null || currentId <= 0}
          size="small"
        >
          <SkipPrevious fontSize="small" />
        </IconButton>
        <IconButton
          onClick={() => currentId !== null && currentId < (sentences?.length ?? 0) - 1 && void playSentence(currentId + 1)}
          disabled={isDisabled || currentId === null || currentId >= (sentences?.length ?? 0) - 1}
          size="small"
        >
          <SkipNext fontSize="small" />
        </IconButton>
      </Stack>
    </Tooltip>
  );
});

export const PlayerExtrasChrome = React.memo(function PlayerExtrasChrome() {
  const {
    anchorEl,
    autoScroll,
    currentId,
    disabledTooltip,
    handleCloseSettings,
    handleOpenSettings,
    highlightEnabled,
    idPopover,
    isDisabled,
    isPlaying,
    onAutoScrollChange,
    onHighlightEnabledChange,
    open,
    playSentence,
    selectedVoice,
    setSelectedVoice,
    setSpeed,
    speed,
    voiceOptions,
  } = usePlayerControlsContext();

  return (
    <>
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

      <Tooltip title={disabledTooltip}>
        <IconButton
          aria-describedby={idPopover}
          size="small"
          onClick={handleOpenSettings}
          color={open ? "primary" : "default"}
          disabled={isDisabled}
        >
          <RecordVoiceOverIcon fontSize="small" />
        </IconButton>
      </Tooltip>

      <Popover
        id={idPopover}
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
          sx: { p: 2, minWidth: 200 },
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
    </>
  );
});

const PlayerControls = React.memo(function PlayerControls(props: PlayerControlsProps) {
  return (
    <PlayerControlsProvider {...props}>
      <Stack direction="row" spacing={1} alignItems="center" useFlexGap sx={{ flexWrap: "wrap" }}>
        <PlayerPlaybackChrome />
        <PlayerExtrasChrome />
      </Stack>
    </PlayerControlsProvider>
  );
});

export default PlayerControls;
