import React, { useState, useEffect } from "react";
import { Container, Stack, Typography, Box, Button } from "@mui/material";
import PdfUploader from "../components/PdfUploader";
import PdfViewer from "../components/PdfViewer";
import PlayerControls from "../components/PlayerControls";

type Sentence = { id: number; text: string; bboxes: any[] };

export default function Home() {
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [playRequestId, setPlayRequestId] = useState<number | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // Clean up object URL if we were using one (not needed here as we use server URL)

  return (
    <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <Container sx={{ py: 2 }}>
        <Typography variant="h4" gutterBottom>
          PDF to Speech
        </Typography>
        <Stack direction="row" spacing={2} alignItems="center">
          <PdfUploader onUploaded={(data) => {
            setSentences(data.sentences);
            setPdfUrl(data.pdfUrl);
          }} />
          <Button
            variant="outlined"
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? "Disable Auto‑Scroll" : "Enable Auto‑Scroll"}
          </Button>
        </Stack>
      </Container>

      {sentences.length > 0 && pdfUrl && (
        <>
          <Box sx={{ flex: 1, overflow: "hidden", px: 2, position: 'relative' }}>
            <PdfViewer
              pdfUrl={pdfUrl}
              sentences={sentences}
              currentId={currentId}
              onJump={(id) => {
                setCurrentId(id);
                setPlayRequestId(id);
              }}
              autoScroll={autoScroll}
            />
          </Box>

          <Box sx={{ p: 2 }}>
            <PlayerControls
              sentences={sentences}
              currentId={currentId}
              onCurrentChange={(id) => {
                setCurrentId(id);
                setPlayRequestId(null);
              }}
              playRequestId={playRequestId}
            />
          </Box>
        </>
      )}
    </Box>
  );
}
