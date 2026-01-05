import { Button } from "@mui/material";
import React from "react";

type Props = {
  embedModel: string;
  onUploaded: (data: { sentences: any[]; pdfUrl: string; fileHash: string }) => void;
};

export default function PdfUploader({ embedModel, onUploaded }: Props) {
  const inputId = "pdf-upload-input";
  const [isUploading, setIsUploading] = React.useState(false);

  const handleChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("embedding_model", embedModel);
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiBase}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      onUploaded(data);
    } catch (error) {
      console.error("Upload failed", error);
    } finally {
      setIsUploading(false);
      e.target.value = ""; // reset
    }
  };

  return (
    <>
      <input
        id={inputId}
        type="file"
        accept="application/pdf"
        onChange={handleChange}
        style={{ display: "none" }}
        disabled={!embedModel || isUploading}
      />
      <label htmlFor={inputId}>
        <Button
          variant="contained"
          component="span"
          disabled={!embedModel || isUploading}
        >
          {isUploading ? "Uploading..." : "Upload PDF"}
        </Button>
      </label>
    </>
  );
}
