import React, { useState, useEffect, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { Box, Typography } from '@mui/material';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

type BBox = {
    page: number;
    x: number;
    y: number;
    width: number;
    height: number;
    page_height: number;
    page_width: number;
};

type Sentence = {
    id: number;
    text: string;
    bboxes: BBox[];
};

type Props = {
    pdfUrl: string;
    sentences: Sentence[];
    currentId: number | null;
    onJump: (id: number) => void;
    autoScroll: boolean;
};

export default function PdfViewer({ pdfUrl, sentences, currentId, onJump, autoScroll }: Props) {
    const [numPages, setNumPages] = useState<number | null>(null);
    const [pageWidth, setPageWidth] = useState<number>(600); // Default width
    const containerRef = useRef<HTMLDivElement>(null);
    const sentenceRefs = useRef<{ [key: number]: HTMLDivElement | null }>({});

    function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
        setNumPages(numPages);
    }

    // Handle resizing to fit container
    useEffect(() => {
        const updateWidth = () => {
            if (containerRef.current) {
                setPageWidth(containerRef.current.clientWidth - 40); // padding
            }
        };

        window.addEventListener('resize', updateWidth);
        updateWidth();

        return () => window.removeEventListener('resize', updateWidth);
    }, []);

    // Auto-scroll to active sentence
    useEffect(() => {
        if (autoScroll && currentId !== null && sentenceRefs.current[currentId]) {
            sentenceRefs.current[currentId]?.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [currentId, autoScroll]);

    // Group bboxes by page for rendering overlays
    const getPageOverlays = (pageNumber: number) => {
        return sentences.map((sentence) => {
            // Filter bboxes for this page
            const pageBBoxes = sentence.bboxes.filter(b => b.page === pageNumber);
            if (pageBBoxes.length === 0) return null;

            return (
                <div
                    key={sentence.id}
                    ref={el => {
                        if (sentence.id === currentId) {
                            sentenceRefs.current[sentence.id] = el;
                        }
                    }}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        pointerEvents: 'none', // Let clicks pass through to the click handler layer
                    }}
                >
                    {pageBBoxes.map((bbox, idx) => (
                        <div
                            key={idx}
                            style={{
                                position: 'absolute',
                                left: `${(bbox.x / bbox.page_width) * 100}%`,
                                // PDF coordinates are usually from bottom-left, but pdfminer might give them differently.
                                // If y is from bottom: top = 100 - (y + height)/page_height * 100
                                // If y is from top: top = y/page_height * 100
                                // pdfminer.six LTChar.y0 is from bottom-left.
                                // So top = (page_height - (y0 + height)) / page_height * 100
                                // Wait, bbox.y is y0 (bottom). y1 = y0 + height.
                                // So top corresponds to y1 (top edge of char).
                                // top % = (page_height - (bbox.y + bbox.height)) / page_height * 100
                                top: `${((bbox.page_height - (bbox.y + bbox.height)) / bbox.page_height) * 100}%`,
                                width: `${(bbox.width / bbox.page_width) * 100}%`,
                                height: `${(bbox.height / bbox.page_height) * 100}%`,
                                backgroundColor: sentence.id === currentId ? 'rgba(255, 255, 0, 0.4)' : 'transparent',
                                cursor: 'pointer',
                                pointerEvents: 'auto', // Enable clicks on the highlight box
                            }}
                            onDoubleClick={(e) => {
                                e.stopPropagation();
                                onJump(sentence.id);
                            }}
                            title={sentence.text}
                        />
                    ))}
                </div>
            );
        });
    };

    return (
        <Box
            ref={containerRef}
            sx={{
                height: '100%',
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                bgcolor: '#f5f5f5',
                p: 2
            }}
        >
            <Document
                file={pdfUrl}
                onLoadSuccess={onDocumentLoadSuccess}
                loading={<Typography>Loading PDF...</Typography>}
                error={<Typography color="error">Failed to load PDF.</Typography>}
            >
                {Array.from(new Array(numPages), (el, index) => (
                    <Box key={`page_${index + 1}`} sx={{ position: 'relative', mb: 2, boxShadow: 3 }}>
                        <Page
                            pageNumber={index + 1}
                            width={pageWidth}
                            renderAnnotationLayer={true}
                            renderTextLayer={true}
                        />
                        {/* Overlay Layer */}
                        <div
                            style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                zIndex: 10
                            }}
                        >
                            {getPageOverlays(index + 1)}
                        </div>
                    </Box>
                ))}
            </Document>
        </Box>
    );
}
