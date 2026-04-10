import type { AnnotationTransferItem } from "@embedpdf/plugin-annotation/react";

const ANNOTATION_SERIALIZED_TAG = "__askpdf_serialized_type";
const ANNOTATION_SERIALIZED_VALUE = "value";

/** Convert raw bytes into a base64 string for JSON-safe storage. */
function bytesToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(bytes).toString("base64");
  }

  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    for (let j = 0; j < chunk.length; j += 1) {
      binary += String.fromCharCode(chunk[j]);
    }
  }
  return globalThis.btoa(binary);
}

/** Convert a base64 string back into raw bytes. */
function base64ToBytes(base64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    return new Uint8Array(Buffer.from(base64, "base64"));
  }

  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/**
 * Recursively encode annotation payload values so SQLite JSON can preserve dates and bytes.
 */
function encodeAnnotationValue(value: any): any {
  if (value instanceof Date) {
    return {
      [ANNOTATION_SERIALIZED_TAG]: "date",
      [ANNOTATION_SERIALIZED_VALUE]: value.toISOString(),
    };
  }

  if (value instanceof ArrayBuffer) {
    return {
      [ANNOTATION_SERIALIZED_TAG]: "arraybuffer",
      [ANNOTATION_SERIALIZED_VALUE]: bytesToBase64(new Uint8Array(value)),
    };
  }

  if (ArrayBuffer.isView(value)) {
    const bytes = new Uint8Array(
      value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)
    );
    return {
      [ANNOTATION_SERIALIZED_TAG]: "arraybuffer",
      [ANNOTATION_SERIALIZED_VALUE]: bytesToBase64(bytes),
      subtype: value.constructor?.name,
    };
  }

  if (Array.isArray(value)) {
    return value.map((entry) => encodeAnnotationValue(entry));
  }

  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, encodeAnnotationValue(entry)])
    );
  }

  return value;
}

/**
 * Recursively decode annotation payload values back into their runtime types.
 */
function decodeAnnotationValue(value: any): any {
  if (Array.isArray(value)) {
    return value.map((entry) => decodeAnnotationValue(entry));
  }

  if (!value || typeof value !== "object") {
    return value;
  }

  const tag = value[ANNOTATION_SERIALIZED_TAG];
  if (tag === "date" && typeof value[ANNOTATION_SERIALIZED_VALUE] === "string") {
    return new Date(value[ANNOTATION_SERIALIZED_VALUE]);
  }

  if (tag === "arraybuffer" && typeof value[ANNOTATION_SERIALIZED_VALUE] === "string") {
    return base64ToBytes(value[ANNOTATION_SERIALIZED_VALUE]).buffer;
  }

  return Object.fromEntries(
    Object.entries(value).map(([key, entry]) => [key, decodeAnnotationValue(entry)])
  );
}

/**
 * Serialize EmbedPDF annotation items into a JSON-safe shape for persistence.
 */
export function serializeAnnotationItems(items: AnnotationTransferItem[]): AnnotationTransferItem[] {
  return encodeAnnotationValue(items) as AnnotationTransferItem[];
}

/**
 * Restore EmbedPDF annotation items from the JSON-safe persistence shape.
 */
export function deserializeAnnotationItems(items: AnnotationTransferItem[]): AnnotationTransferItem[] {
  return decodeAnnotationValue(items) as AnnotationTransferItem[];
}

export type { AnnotationTransferItem };
